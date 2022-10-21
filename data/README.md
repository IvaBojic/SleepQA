## agreement 

This folder contains two files:
1. *labels_agreement.csv*
2. *open_agreement.csv*

*labels_agreement.csv* file contains 750 Q/A pairs with 5-ways annotations where the first column is paragraph id, the second is short span answer and the third one indicates whether this is an initial answer. This file is used in *inter_agreement.py* to calculate inter-annotator agreement for label collection process.

*open_agreement.csv* file contains 150 5-ways scores where the first column is question, the second one is the answer from pipeline 1, the third one is the answer from pipeline 2, and the subsequent five columns are scores from five annotators. This file is used in *inter_agreement.py* for calculating inter-annotator agreement for extrinsic evaluation.

## bm25_indexes & bm25_json

These folders contain files created by *indexes.py* script. Files from *bm25_indexes* folder are used in Lucerne BM25 retrieval model.

## processed

This folder contains files and folders created using fine-tuned domain-specific BERT models.

## training

This folder contains files needed to train different domain-specific BERT retrieval models.

``` 
python train_dense_encoder.py \
train_datasets=[sleep_train] \
dev_datasets=[sleep_dev] \
output_dir="PubMedBERT_full/retrieval/"
``` 

### oracle

This folder contains files needed to train different domain-specific BERT reader models.

``` 
python train_extractive_reader.py \
encoder.sequence_length=300 \
train_files="../../../../data/training/oracle/sleep-train.json" \
dev_files="../../../../data/training/oracle/sleep-dev.json"  \
output_dir="PubMedBERT_full/reader/"
```
