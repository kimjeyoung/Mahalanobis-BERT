import torch
import os
import numpy as np
from tqdm import tqdm
from models.bert import BertModel, Config
from models.mahalanobis import Classifier
from data_loader import DatasetLoader
from torch.utils.data import DataLoader
from models.optimizers import BertAdam
from utils import to_numpy, Accumulator
from sklearn.metrics import accuracy_score, roc_auc_score, precision_recall_curve, auc, mean_absolute_error
from sklearn.linear_model import LogisticRegressionCV
from sklearn.utils import shuffle
import pickle


class Trainer:
    def __init__(self, args):
        t_total = -1
        self.epochs = args.epochs
        self.device = args.device
        self.batch_size = args.batch_size
        self.save_path = args.save_path
        self.L = args.L

        if args.train_or_test == 'train':
            self.val_dataset = DatasetLoader(data_dir=args.data_dir, vocab_path=args.bert_vocab,
                                             max_len=args.max_len, train_or_test='val')
            self.val_loader = DataLoader(self.val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=8)

        self.train_dataset = DatasetLoader(data_dir=args.data_dir, vocab_path=args.bert_vocab,
                                           max_len=args.max_len, train_or_test='train')
        self.train_loader = DataLoader(self.train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8,
                                       drop_last=True)
        t_total = len(self.train_loader) * args.epochs

        self.test_dataset = DatasetLoader(data_dir=args.data_dir, vocab_path=args.bert_vocab,
                                          max_len=args.max_len, train_or_test='test')
        self.test_loader = DataLoader(self.test_dataset, batch_size=8, shuffle=False, num_workers=8)

        self.outdomain_idx = np.where(self.test_dataset.y_data == 150)[0][0]
        self.num_classes = self.test_dataset.num_classes

        # default config is bert-base
        self.bert_config = Config()
        self.backbone = BertModel(self.bert_config)
        self.backbone.load_pretrain_huggingface(torch.load(args.bert_ckpt))
        self.model = Classifier(self.backbone,
                                hidden_size=self.bert_config.hidden_size,
                                num_classes=self.num_classes,
                                device="cuda" if self.device == 'gpu' else 'cpu')
        self.criterion = torch.nn.CrossEntropyLoss()

        if args.device == 'gpu':
            self.model = self.model.to("cuda")
        self.optimizer = BertAdam(self.model.parameters(), lr=args.lr,
                                  warmup=args.warmup, weight_decay=args.weight_decay, t_total=t_total)

        if args.train_or_test == 'test' and os.path.isfile(os.path.join(args.save_path, "bestmodel.bin")):
            self.model.load_state_dict(torch.load(os.path.join(args.save_path, "bestmodel.bin")))

    def train(self):
        best_acc = 0.
        for epoch in range(self.epochs):
            cnt = 0
            self.model.train()
            metrics = Accumulator()
            loader = tqdm(self.train_loader, disable=False)
            loader.set_description('[%s %04d/%04d]' % ('train', epoch, self.epochs))
            for i, batch in enumerate(loader):
                cnt += self.batch_size
                if self.device == 'gpu':
                    batch = [x.to('cuda') for x in batch]
                self.optimizer.zero_grad()
                x_ids, x_segs, x_attns, label = batch
                pred, _ = self.model(x_ids, x_segs, x_attns)
                loss = self.criterion(pred, label)
                metrics.add_dict({
                    'loss': loss.item() * self.batch_size,
                })
                postfix = metrics / cnt
                loader.set_postfix(postfix)
                loss.backward()
                self.optimizer.step()

            # update sample mean, precison
            _ = self.model.get_sample_mean_precision(self.train_loader)

            val_acc = self.eval()
            print(f'\t Val dataset --> metric score : {val_acc:.3f}')
            if val_acc > best_acc:
                best_acc = val_acc
                test_auroc, test_auprc, test_acc = self.test()
                print(f'\t Test dataset --> AUROC : {test_auroc:.3f} | AUPRC: {test_auprc:.3f} | ACC: {test_acc:.3f}')
                torch.save(self.model.state_dict(), os.path.join(self.save_path, "bestmodel.bin"))

    def eval(self):
        self.model.eval()
        y_true = []
        y_pred = []
        for i, batch in enumerate(self.val_loader):
            if self.device == 'gpu':
                batch = [x.to('cuda') for x in batch]
            x_ids, x_segs, x_attns, label = batch

            out_features, embedding_output = self.model.extract_feature_lists(x_ids, x_segs, x_attns, extract_layers=True,
                                                                              input_processing=False)
            sample_mean = self.model.sample_class_mean[-1]
            precision = self.model.precision[-1]

            cls_feature = out_features[-1][:, 0, :]
            cls_output = self.model.output_fc(cls_feature)
            # softmax_output = torch.softmax(cls_output, dim=-1)

            # penultimate features
            out_features = out_features[-1][:, 0, :]  # torch.mean(out_features[-1], dim=1)
            generative_output = self.model.get_generative_output(out_features, sample_mean, precision)
            y_hat = self.L * cls_output + (1. - self.L) * generative_output
            pred = torch.softmax(y_hat, dim=-1)

            pred = to_numpy(torch.argmax(pred, dim=-1)).flatten().tolist()
            true = to_numpy(label).flatten().tolist()
            y_true.extend(true)
            y_pred.extend(pred)

        value = accuracy_score(y_true, y_pred)
        return value

    def test(self, training=True):
        if not training:
            _ = self.model.get_sample_mean_precision(self.train_loader)

        self.model.eval()

        y_trues = []
        y_preds = []
        prob_preds = []
        for i, batch in enumerate(self.test_loader):
            if self.device == 'gpu':
                batch = [x.to('cuda') for x in batch]
            x_ids, x_segs, x_attns, label = batch

            out_features, embedding_output = self.model.extract_feature_lists(x_ids, x_segs, x_attns,
                                                                              extract_layers=True,
                                                                              input_processing=False)
            sample_mean = self.model.sample_class_mean[-1]
            precision = self.model.precision[-1]

            cls_feature = out_features[-1][:, 0, :]
            cls_output = self.model.output_fc(cls_feature)
            # softmax_output = torch.softmax(cls_output, dim=-1)

            # penultimate features
            out_features = out_features[-1][:, 0, :]  # torch.mean(out_features[-1], dim=1)
            generative_output = self.model.get_generative_output(out_features, sample_mean, precision)
            y_pred = self.L * cls_output + (1. - self.L) * generative_output
            pred = torch.softmax(y_pred, dim=-1)
            prob_preds.append(to_numpy(pred))

            pred = to_numpy(torch.argmax(pred, dim=-1)).flatten().tolist()
            true = to_numpy(label).flatten().tolist()
            y_trues.extend(true)
            y_preds.extend(pred)

        ood_trues = (np.array(y_trues) == 150).astype(np.uint8).tolist()
        indomain_true = np.array(y_trues)[np.where(np.array(ood_trues) == 0)[0]]
        indomain_pred = np.array(y_preds)[np.where(np.array(ood_trues) == 0)[0]]

        test_acc = accuracy_score(indomain_true, indomain_pred)
        best_auroc = 0
        best_auprc = 0

        m_list = [0.0, 0.01, 0.005, 0.002, 0.0014, 0.001, 0.0005]
        for magnitude in m_list:
            M_in = self.model.extract_mahalanobis_score(self.test_loader, magnitude)
            test_auroc = roc_auc_score(ood_trues, M_in)
            # calculate precision-recall curve
            precision, recall, thresholds = precision_recall_curve(ood_trues, M_in)
            test_auprc = auc(recall, precision)

            if test_auroc > best_auroc:
                best_auroc = test_auroc
            if test_auprc > best_auprc:
                best_auprc = test_auprc

        if training:
            return best_auroc, best_auprc, test_acc
        else:
            print(f'\t Test dataset --> AUROC : {best_auroc:.3f} | AUPRC: {best_auprc:.3f} | ACC: {test_acc:.3f}')
