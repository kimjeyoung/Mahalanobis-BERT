# :fire: Mahalanobis-BERT :fire:
Reimplementation of "A Simple Unified Framework for Detecting Out-of-Distribution Samples and Adversarial Attacks"

The codes are based on [official repo (Pytorch)](https://github.com/pokaxpoka/deep_Mahalanobis_detector) and [huggingface](https://huggingface.co/).

Original Paper : [Link](https://papers.nips.cc/paper/2018/file/abdeb6f575ac5c6676b747bca8d09cc2-Paper.pdf)

## Installation :coffee:

Training environment : Ubuntu 18.04, python 3.6
```bash
pip3 install torch torchvision torchaudio
pip install scikit-learn
```

Download `bert-base-uncased` checkpoint from [hugginface-ckpt](https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-pytorch_model.bin)  
Download `bert-base-uncased` vocab file from [hugginface-vocab](https://huggingface.co/bert-base-uncased/resolve/main/vocab.txt)  
Download CLINC OOS intent detection benchmark dataset from [tensorflow-dataset](https://github.com/jereliu/datasets/raw/master/clinc_oos.zip)

The downloaded files' directory should be:

```bash
Mahalanobis-BERT
ㄴckpt
  ㄴbert-base-uncased-pytorch_model.bin
ㄴdataset
  ㄴclinc_oos
    ㄴtrain.csv
    ㄴval.csv
    ㄴtest.csv
    ㄴtest_ood.csv
  ㄴvocab
    ㄴbert-base-uncased-vocab.txt
ㄴmodels
...
```


## Dataset Info :book:

In their paper, the authors conducted OOD experiment for NLP using CLINC OOS intent detection benchmark dataset, the OOS dataset contains data for 150 in-domain services with 150 training
sentences in each domain, and also 1500 natural out-of-domain utterances.
You can download the dataset at [Link](https://github.com/jereliu/datasets/raw/master/clinc_oos.zip).

Original dataset paper, and Github : [Paper Link](https://aclanthology.org/D19-1131/), [Git Link](https://github.com/clinc/oos-eval)

## Run :star2:

#### Train
```bash
python main.py --train_or_test train --device gpu --gpu 0
```

#### Test

```bash
python main.py --train_or_test test --device gpu --gpu 0
```


## References

[1] https://arxiv.org/pdf/1807.03888.pdf  
[2] https://github.com/pokaxpoka/deep_Mahalanobis_detector  