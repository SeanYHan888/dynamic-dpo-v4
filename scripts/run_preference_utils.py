import logging
import random
import sys
from typing import Iterable

import torch
import transformers
from save_utils import (
    maybe_push_margin_dataset_summary,
    push_prevalidated_hf_artifacts,
    save_hf_compatible_training_artifacts,
)
from torch_dtype_utils import normalize_torch_dtype
from utils.preprocessing_cache import (
    attach_prompt_preprocessing_metadata,
    build_prompt_preprocessing_metadata,
    configure_persistent_hf_cache,
)
from transformers import set_seed

from alignment import get_checkpoint, get_datasets, get_kbit_device_map, get_quantization_config, get_tokenizer
from alignment.data import is_openai_format, maybe_convert_hh_to_openai_format, maybe_insert_system_message

logger = logging.getLogger(__name__)

MISTRAL_CHAT_TEMPLATE = "{% if messages[0]['role'] == 'system' %}{% set loop_messages = messages[1:] %}{% set system_message = messages[0]['content'].strip() + '\n\n' %}{% else %}{% set loop_messages = messages %}{% set system_message = '' %}{% endif %}{% for message in loop_messages %}{% if loop.index0 == 0 %}{% set content = system_message + message['content'] %}{% else %}{% set content = message['content'] %}{% endif %}{% if message['role'] == 'user' %}{{ '[INST] ' + content.strip() + ' [/INST]' }}{% elif message['role'] == 'assistant' %}{{ ' '  + content.strip() + ' ' + eos_token }}{% endif %}{% endfor %}"
PREFERENCE_COLUMNS = ["messages", "chosen", "rejected", "prompt", "completion", "label"]


def apply_preference_chat_template(
    example,
    tokenizer,
    auto_insert_empty_system_msg: bool = True,
    change_template=None,
):
    if change_template == "mistral":
        tokenizer.chat_template = MISTRAL_CHAT_TEMPLATE

    example = maybe_convert_hh_to_openai_format(example)

    if not all(key in example.keys() for key in ("chosen", "rejected")):
        raise ValueError(
            "Could not format example as dialogue for preference training; expected either "
            "`[chosen, rejected]` or `[prompt, chosen, rejected]`."
        )
    if not is_openai_format(example["chosen"]) or not is_openai_format(example["rejected"]):
        raise ValueError("Preference training expects OpenAI-style message dictionaries.")

    if "prompt" in example and is_openai_format(example["prompt"]):
        prompt_messages = example["prompt"]
        chosen_messages = example["chosen"]
        rejected_messages = example["rejected"]
    else:
        prompt_messages = example["chosen"][:-1]
        chosen_messages = example["chosen"][-1:]
        rejected_messages = example["rejected"][-1:]

    if auto_insert_empty_system_msg:
        maybe_insert_system_message(prompt_messages, tokenizer)

    example["text_prompt"] = tokenizer.apply_chat_template(prompt_messages, tokenize=False)
    example["text_chosen"] = tokenizer.apply_chat_template(chosen_messages, tokenize=False)
    if tokenizer.bos_token and example["text_chosen"].startswith(tokenizer.bos_token):
        example["text_chosen"] = example["text_chosen"][len(tokenizer.bos_token) :]
    example["text_rejected"] = tokenizer.apply_chat_template(rejected_messages, tokenize=False)
    if tokenizer.bos_token and example["text_rejected"].startswith(tokenizer.bos_token):
        example["text_rejected"] = example["text_rejected"][len(tokenizer.bos_token) :]
    return example


def setup_run(model_args, data_args, training_args, run_logger):
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()
    run_logger.setLevel(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    run_logger.info(f"Model parameters {model_args}")
    run_logger.info(f"Data parameters {data_args}")
    run_logger.info(f"Training/evaluation parameters {training_args}")

    last_checkpoint = get_checkpoint(training_args)
    if last_checkpoint is not None and training_args.resume_from_checkpoint is None:
        run_logger.info(f"Checkpoint detected, resuming training at {last_checkpoint=}.")

    set_seed(training_args.seed)
    return last_checkpoint


def prepare_preference_datasets(model_args, data_args, training_args, run_logger):
    configure_persistent_hf_cache(data_args, run_logger)
    raw_datasets = get_datasets(
        data_args,
        splits=data_args.dataset_splits,
        configs=data_args.dataset_configs,
        columns_to_keep=PREFERENCE_COLUMNS,
    )
    run_logger.info(
        f"Training on the following splits: {[split + ' : ' + str(dset.num_rows) for split, dset in raw_datasets.items()]}"
    )
    column_names = list(raw_datasets["train"].features)

    data_args.truncation_side = "left"
    tokenizer = get_tokenizer(model_args, data_args)
    change_template = "mistral" if "mistral" in model_args.model_name_or_path.lower() else None
    if change_template == "mistral":
        tokenizer.chat_template = MISTRAL_CHAT_TEMPLATE
    attach_prompt_preprocessing_metadata(
        training_args,
        build_prompt_preprocessing_metadata(tokenizer, data_args, apply_preference_chat_template),
    )

    raw_datasets = raw_datasets.map(
        apply_preference_chat_template,
        fn_kwargs={
            "tokenizer": tokenizer,
            "auto_insert_empty_system_msg": data_args.auto_insert_empty_system_msg,
            "change_template": change_template,
        },
        num_proc=data_args.preprocessing_num_workers,
        remove_columns=column_names,
        desc="Formatting comparisons with prompt template",
    )

    for split in raw_datasets.keys():
        raw_datasets[split] = raw_datasets[split].rename_columns(
            {"text_prompt": "prompt", "text_chosen": "chosen", "text_rejected": "rejected"}
        )

    if "train" in raw_datasets:
        sample_count = min(3, len(raw_datasets["train"]))
        for index in random.sample(range(len(raw_datasets["train"])), sample_count):
            run_logger.info(f"Prompt sample {index} of the raw training set:\n\n{raw_datasets['train'][index]['prompt']}")
            run_logger.info(f"Chosen sample {index} of the raw training set:\n\n{raw_datasets['train'][index]['chosen']}")
            run_logger.info(f"Rejected sample {index} of the raw training set:\n\n{raw_datasets['train'][index]['rejected']}")

    training_args.model_init_kwargs = build_model_init_kwargs(model_args, training_args)
    return raw_datasets, tokenizer


def build_model_init_kwargs(model_args, training_args):
    torch_dtype = normalize_torch_dtype(model_args.torch_dtype)
    quantization_config = get_quantization_config(model_args)
    return {
        "revision": model_args.model_revision,
        "trust_remote_code": model_args.trust_remote_code,
        "torch_dtype": torch_dtype,
        "use_cache": False if training_args.gradient_checkpointing else True,
        "device_map": get_kbit_device_map() if quantization_config is not None else None,
        "quantization_config": quantization_config,
        "attn_implementation": model_args.attn_implementation,
    }


def finalize_training(trainer, training_args, model_args, data_args, raw_datasets, run_logger, tags: Iterable[str]):
    checkpoint = None
    last_checkpoint = get_checkpoint(training_args)
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    elif last_checkpoint is not None:
        checkpoint = last_checkpoint

    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    metrics = train_result.metrics
    metrics["train_samples"] = len(raw_datasets["train"])
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    run_logger.info("*** Training complete ***")
    run_logger.info("*** Save model ***")
    save_hf_compatible_training_artifacts(trainer, training_args.output_dir, run_logger)

    kwargs = {
        "finetuned_from": model_args.model_name_or_path,
        "dataset": list(data_args.dataset_mixer.keys()),
        "dataset_tags": list(data_args.dataset_mixer.keys()),
        "tags": list(tags),
    }
    if trainer.accelerator.is_main_process:
        trainer.create_model_card(**kwargs)
        trainer.model.config.use_cache = True
        trainer.model.config.save_pretrained(training_args.output_dir)

    if training_args.do_eval:
        run_logger.info("*** Evaluate ***")
        metrics = trainer.evaluate()
        metrics["eval_samples"] = len(raw_datasets["test"])
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    if training_args.push_to_hub is True:
        run_logger.info("Pushing to hub...")
        push_prevalidated_hf_artifacts(
            trainer,
            training_args.output_dir,
            run_logger,
        )
        maybe_push_margin_dataset_summary(
            trainer,
            run_logger,
            model_args=model_args,
            data_args=data_args,
        )

    run_logger.info("*** Training complete! ***")
