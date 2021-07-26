import torch
import math
import copy
import numpy as np
import sklearn.covariance
from torch import nn
from typing import Optional
from torch import Tensor
from torch.autograd import Variable, grad
from utils import l2_normalize


def BertLinear(i_dim, o_dim, bias=True):
    m = nn.Linear(i_dim, o_dim, bias)
    nn.init.normal_(m.weight, std=0.02)
    if bias:
        nn.init.constant_(m.bias, 0.)
    return m


class Classifier(nn.Module):
    def __init__(self, backbone,
                 hidden_size=768,
                 num_classes=3,
                 layer_idx=None,
                 device='cuda'):
        super(Classifier, self).__init__()
        if layer_idx is None:
            layer_idx = [0, 3, 7, 11]
        self.backbone = backbone
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.output_fc = BertLinear(self.hidden_size, self.num_classes)
        self.num_hidden_layers = len(self.backbone.encoder.layer)
        self.device = device
        self.layer_idx = [0, 3, 7, 11] if layer_idx is None else layer_idx

    def get_generative_output(self, out_features, sample_mean, precision):
        # compute Mahalanobis score
        gaussian_score = 0
        for i in range(self.num_classes):
            batch_sample_mean = sample_mean[i]
            zero_f = out_features.data - batch_sample_mean
            term_gau = torch.mm(torch.mm(zero_f, precision), zero_f.t()).diag()
            if i == 0:
                gaussian_score = term_gau.view(-1, 1)
            else:
                gaussian_score = torch.cat((gaussian_score, term_gau.view(-1, 1)), 1)
        return torch.softmax(gaussian_score, dim=-1)

    def input_processing(self, gaussian_score, embedding_output, out_features, sample_mean, precision, layer_index, magnitude):
        gaussian_score, sample_pred = torch.max(gaussian_score, dim=1)
        batch_sample_mean = sample_mean[layer_index].index_select(0, sample_pred)
        zero_f = out_features - Variable(batch_sample_mean)
        pure_gau = -0.5 * torch.mm(torch.mm(zero_f, Variable(precision[layer_index])), zero_f.t()).diag()
        loss = torch.mean(-pure_gau)
        loss.backward()

        gradient = torch.ge(embedding_output.grad.data, 0)
        gradient = (gradient.float() - 0.5) * 2
        tempInputs = torch.add(embedding_output.data, gradient, alpha=-magnitude)
        return tempInputs

    def get_gaussian_score(self, features, sample_mean, precision, layer_index):
        out_features = features
        gaussian_score = 0
        for i in range(self.num_classes):
            batch_sample_mean = sample_mean[layer_index][i]
            zero_f = out_features.data - batch_sample_mean
            term_gau = -0.5 * torch.mm(torch.mm(zero_f, precision[layer_index]), zero_f.t()).diag()
            if i == 0:
                gaussian_score = term_gau.view(-1, 1)
            else:
                gaussian_score = torch.cat((gaussian_score, term_gau.view(-1, 1)), 1)

        return gaussian_score

    def get_mahalanobis_score(self, data_loader, num_classes,
                              sample_mean, precision, layer_index, magnitude):
        '''
            Compute the proposed Mahalanobis confidence score on input dataset
            return: Mahalanobis score from layer_index
        '''
        indomain_Mahalanobis = []
        for batch in data_loader:
            if self.device == 'cuda':
                batch = [x.to('cuda') for x in batch]
            x_ids, x_segs, x_attns, target = batch

            if x_attns is None:
                x_attns = torch.ones_like(x_ids)

            extended_attention_mask = x_attns.unsqueeze(1).unsqueeze(2)
            extended_attention_mask = extended_attention_mask.to(torch.float32)
            extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
            head_mask = [None] * self.num_hidden_layers

            # in-domain
            out_features, embedding_output = self.extract_feature_lists(x_ids, x_segs, x_attns, extract_layers=True,
                                                                        input_processing=True)
            out_features = out_features[layer_index]
            indomain_out_features = out_features[:, 0, :]

            # compute in-domain Mahalanobis score
            indomain_gaussian_score = self.get_gaussian_score(indomain_out_features, sample_mean, precision, layer_index)

            # # Input_processing
            new_embeddings = self.input_processing(indomain_gaussian_score, embedding_output,
                                                   indomain_out_features, sample_mean, precision,
                                                   layer_index, magnitude)
            with torch.no_grad():
                encoded_layers = self.backbone.encoder(new_embeddings,
                                                       extended_attention_mask,
                                                       head_mask=head_mask)
            out_features = encoded_layers[self.layer_idx[layer_index]]
            indomain_out_features = out_features[:, 0, :]
            indomain_gaussian_score = self.get_gaussian_score(indomain_out_features, sample_mean, precision,
                                                              layer_index)
            indomain_gaussian_score, _ = torch.max(indomain_gaussian_score, dim=1)
            indomain_Mahalanobis.extend(indomain_gaussian_score.cpu().numpy())
        return indomain_Mahalanobis

    def extract_mahalanobis_score(self, data_loader, magnitude):
        num_output = len(self.layer_idx)
        for i in range(num_output):
            M_in = self.get_mahalanobis_score(data_loader, self.num_classes,
                                                     self.sample_class_mean, self.precision, i, magnitude)

            M_in = np.asarray(M_in, dtype=np.float32).flatten()
            if i == 0:
                Mahalanobis_in = M_in
            else:
                Mahalanobis_in += M_in
        Mahalanobis_in = Mahalanobis_in / num_output
        return (Mahalanobis_in - np.min(Mahalanobis_in)) / np.max(Mahalanobis_in)

    @torch.no_grad()
    def get_sample_mean_precision(self, data_loader):
        # model, args.num_classes, feature_list, train_loader
        """
            compute sample mean and precision (inverse of covariance)
            return: sample_class_mean: list of class mean
                     precision: list of precisions
        """

        group_lasso = sklearn.covariance.EmpiricalCovariance(assume_centered=False)
        num_output = len(self.layer_idx)
        num_sample_per_class = np.empty(self.num_classes)
        num_sample_per_class.fill(0)
        list_features = np.zeros([num_output, self.num_classes]).tolist()

        for batch in data_loader:
            if self.device == 'cuda':
                batch = [x.to('cuda') for x in batch]
            x_ids, x_segs, x_attns, target = batch
            out_features, _ = self.extract_feature_lists(x_ids, x_segs, x_attns, extract_layers=True,
                                                         input_processing=False)

            # get hidden features
            for i in range(num_output):
                mean_feature = out_features[i].data[:, 0, :]
                out_features[i] = mean_feature

            # construct the sample matrix
            for i in range(x_ids.size(0)):
                label = target[i]
                if num_sample_per_class[label] == 0:
                    out_count = 0
                    for out in out_features:
                        list_features[out_count][label] = out[i].view(1, -1)
                        out_count += 1
                else:
                    out_count = 0
                    for out in out_features:
                        list_features[out_count][label] \
                            = torch.cat((list_features[out_count][label], out[i].view(1, -1)), 0)
                        out_count += 1
                num_sample_per_class[label] += 1

        sample_class_mean = []
        for out_count in range(len(self.layer_idx)):
            temp_list = torch.Tensor(self.num_classes, int(self.hidden_size)).to(self.device)
            for j in range(self.num_classes):
                temp_list[j] = torch.mean(list_features[out_count][j], 0)
            sample_class_mean.append(temp_list)

        precision = []
        for k in range(num_output):
            X = 0
            for i in range(self.num_classes):
                if i == 0:
                    X = list_features[k][i] - sample_class_mean[k][i]
                else:
                    X = torch.cat((X, list_features[k][i] - sample_class_mean[k][i]), 0)

            # find inverse
            group_lasso.fit(X.cpu().numpy())
            temp_precision = group_lasso.precision_
            temp_precision = torch.from_numpy(temp_precision).float().to(self.device)
            precision.append(temp_precision)

        self.sample_class_mean = sample_class_mean
        self.precision = precision
        return sample_class_mean, precision

    def extract_feature_lists(self, input_ids, token_type_ids: Optional[Tensor] = None,
                              attention_mask: Optional[Tensor] = None, adversarial_inputs: Optional[Tensor] = None,
                              extract_layers=False, input_processing=False):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(torch.float32)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        head_mask = [None] * self.num_hidden_layers

        embedding_output = self.backbone.embeddings(input_ids, token_type_ids)
        if input_processing:
            embedding_output = Variable(embedding_output, requires_grad=True)

        if adversarial_inputs is not None:
            embedding_output += adversarial_inputs

        encoded_layers = self.backbone.encoder(embedding_output,
                                               extended_attention_mask,
                                               head_mask=head_mask)

        if extract_layers:
            sequence_output = []
            for idx in self.layer_idx:
                sequence_output.append(encoded_layers[idx])

        else:
            sequence_output = encoded_layers[-1]

        return sequence_output, embedding_output

    def get_adv(self, embedding_outputs, loss, magnitude, use_normalize=False, allow_unused=False):
        device = self.device
        emb_grad = grad(loss, embedding_outputs, retain_graph=True)

        # Normalizing the gradient to binary in {0, 1}
        gradient = torch.ge(emb_grad[0].data, 0)
        gradient = (gradient.float() - 0.5) * 2
        if use_normalize:
            gradient = l2_normalize(gradient)
        p_adv = -1 * magnitude * gradient
        p_adv = Variable(p_adv).to(device)
        return p_adv

    def forward(self, input_ids, token_type_ids: Optional[Tensor] = None,
                attention_mask: Optional[Tensor] = None, adversarial_inputs: Optional[Tensor] = None):

        sequence_output, embedding_output = self.extract_feature_lists(input_ids, token_type_ids, attention_mask,
                                                                       adversarial_inputs=adversarial_inputs)
        cls_feature = sequence_output[:, 0, :]
        cls_output = self.output_fc(cls_feature)
        return cls_output, embedding_output


if __name__ == "__main__":
    import numpy as np
    from bert import BertModel, Config