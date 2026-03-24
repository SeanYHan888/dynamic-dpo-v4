---
library_name: transformers
base_model: princeton-nlp/Llama-3-Base-8B-SFT
tags:
- alignment-handbook
- beta-dpo
- generated_from_trainer
datasets:
- HuggingFaceH4/ultrafeedback_binarized
model-index:
- name: llama-3-8b-base-beta-dpo-4xh100-test
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# llama-3-8b-base-beta-dpo-4xh100-test

This model is a fine-tuned version of [princeton-nlp/Llama-3-Base-8B-SFT](https://huggingface.co/princeton-nlp/Llama-3-Base-8B-SFT) on the HuggingFaceH4/ultrafeedback_binarized dataset.

## Model description

More information needed

## Intended uses & limitations

More information needed

## Training and evaluation data

More information needed

## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 5e-07
- train_batch_size: 1
- eval_batch_size: 1
- seed: 42
- distributed_type: multi-GPU
- num_devices: 2
- total_train_batch_size: 2
- total_eval_batch_size: 2
- optimizer: Adam with betas=(0.9,0.999) and epsilon=1e-08
- lr_scheduler_type: linear
- lr_scheduler_warmup_ratio: 0.05
- training_steps: 50

### Training results



### Framework versions

- Transformers 4.44.2
- Pytorch 2.3.1+cu121
- Datasets 2.21.0
- Tokenizers 0.19.1
