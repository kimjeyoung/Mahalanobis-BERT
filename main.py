import os
import argparse
from trainer import Trainer


def main(args):
    if args.device == 'gpu':
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    sngp = Trainer(args)

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    if args.train_or_test == 'train':
        sngp.train()
    else:
        sngp.test(training=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='gpu')
    parser.add_argument('--gpu', type=str, default="1")
    parser.add_argument('--train_or_test', type=str, default='train', choices=['train', 'test'])
    parser.add_argument('--data_dir', type=str, default='./dataset/clinc_oos')
    parser.add_argument('--bert_ckpt', type=str, default='./ckpt/bert-base-uncased-pytorch_model.bin')
    parser.add_argument('--bert_vocab', type=str, default='./dataset/vocab/bert-base-uncased-vocab.txt')
    parser.add_argument('--save_path', type=str, default='./mahalanobis_ckpt')

    # default setting
    parser.add_argument('--max_len', type=int, default=32)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--warmup', type=float, default=0.1)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--L', default=0.5, type=float)
    main(parser.parse_args())