# Convert checkpoints to pytorch

## Changing the script for conversion

After fine-tuning your own model, you would need to convert the created checkpoint to a format that is supported by pytorch. There is a script available within transformers library[^1] that allows the conversion. Since we are not using *bert-base-uncased* as the pretrained model, we need to change:

`
model = DPRContextEncoder(DPRConfig(**BertConfig.get_config_dict("bert-base-uncased")[0]))
`

with:

`
model = DPRContextEncoder(DPRConfig(**BertConfig.get_config_dict("microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext")[0]))
`

Similarly, we need to change code for *DPRQuestionEncoder*:

`
model = DPRQuestionEncoder(DPRConfig(**BertConfig.get_config_dict("microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext")[0]))
`

and for *DPRReader* as:

`
model = DPRReader(DPRConfig(**BertConfig.get_config_dict("gdario/biobert_bioasq")[0]))
`

*convert_dpr_original_checkpoint_to_pytorch.py* script is already changed in the above-mentioned way.

## Running scripts for conversion

After editing the script in the above-mentioned way, we need to run three scripts to first convert *ctx_encoder*:

```
python convert_dpr_original_checkpoint_to_pytorch.py --type ctx_encoder --src pipeline1_baseline/cp_models/dpr_biencoder.29 --dest SleepQA/models/pytorch/ctx_encoder
```

then *question_encoder*:

```
python convert_dpr_original_checkpoint_to_pytorch.py --type question_encoder --src pipeline1_baseline/cp_models/dpr_biencoder.29 --dest SleepQA/models/pytorch/question_encoder
```

and finally *reader*:

```
python convert_dpr_original_checkpoint_to_pytorch.py --type reader --src pipeline1_baseline/cp_models/dpr_extractive_reader.1.250 --dest SleepQA/models/pytorch/reader
```

## Adding missing files after the conversion

After running three above mentioned scripts, we need to download *tokenizer_config.json* and *vocab.txt* files from respective Hugging Face repositories: PubMedBERT[^2] for *ctx_encoder* and *question_encoder*, and BioBERT BioASQ[^3] for *reader*.


# Building QA pipeline

*qa_system.py* script allows us to use fine-tuned models in a QA pipeline:
1. *generate_dense_encodings* function generates encodings for text corpus,
2. *dense_retriever* function retrieves the most relevant passage for the given question, and
3. *extractive_reader* function retrieves the most relevant text span for the given question.


[^1]: [Convert DPR original checkpoint to pytorch](https://github.com/huggingface/transformers/blob/main/src/transformers/models/dpr/convert_dpr_original_checkpoint_to_pytorch.py)
[^2]: [PubMedBERT Hugging Face](https://huggingface.co/microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext/tree/main)
[^3]: [BioBERT BioASQ Hugging Face](https://huggingface.co/gdario/biobert_bioasq/tree/main)
