import logging
import json
import random
import sys
from pathlib import Path
from typing import Iterable

import torch
import transformers
from transformers import set_seed
from huggingface_hub import HfFolder, hf_hub_download
from huggingface_hub.errors import GatedRepoError, HfHubHTTPError, RepositoryNotFoundError

from alignment import get_checkpoint, get_datasets, get_kbit_device_map, get_quantization_config, get_tokenizer
from alignment.data import is_openai_format, maybe_convert_hh_to_openai_format, maybe_insert_system_message
from utils.checkpoint_io import (
    maybe_push_margin_dataset_summary,
    push_prevalidated_hf_artifacts,
    save_hf_compatible_training_artifacts,
)
from utils.dtypes import normalize_torch_dtype
from utils.preprocessing_cache import (
    attach_prompt_preprocessing_metadata,
    build_prompt_preprocessing_metadata,
    configure_persistent_hf_cache,
)
logger = logging.getLogger(__name__)

MISTRAL_CHAT_TEMPLATE = "{% if messages[0]['role'] == 'system' %}{% set loop_messages = messages[1:] %}{% set system_message = messages[0]['content'].strip() + '\n\n' %}{% else %}{% set loop_messages = messages %}{% set system_message = '' %}{% endif %}{% for message in loop_messages %}{% if loop.index0 == 0 %}{% set content = system_message + message['content'] %}{% else %}{% set content = message['content'] %}{% endif %}{% if message['role'] == 'user' %}{{ '[INST] ' + content.strip() + ' [/INST]' }}{% elif message['role'] == 'assistant' %}{{ ' '  + content.strip() + ' ' + eos_token }}{% endif %}{% endfor %}"
PREFERENCE_COLUMNS = ["messages", "chosen", "rejected", "prompt", "completion", "label"]


def _is_local_repo_path(repo_id_or_path: str | None) -> bool:
    return bool(repo_id_or_path) and Path(repo_id_or_path).expanduser().exists()


def _build_hf_access_error(
    *,
    repo_id: str,
    revision: str,
    filename: str,
    token_present: bool,
    error: Exception,
) -> str:
    lines = [
        f"Hugging Face preflight failed for `{repo_id}` (revision `{revision}`).",
        f"The run needs `{filename}` from that repo before training can start.",
    ]

    if isinstance(error, GatedRepoError):
        lines.append("This repo is gated, and the current shell is not authenticated with an account that can read it.")
    elif isinstance(error, RepositoryNotFoundError):
        lines.append("The repo could not be found from the current shell. This usually means the repo name is wrong or the repo is private and this shell is not authenticated.")
    elif isinstance(error, HfHubHTTPError):
        lines.append(f"Hugging Face Hub returned HTTP {error.response.status_code}.")
    else:
        lines.append(f"Underlying error: {error}")

    if token_present:
        lines.append("A Hugging Face token is present, so the most likely issue is that this account has not been granted access to the repo.")
    else:
        lines.append("No Hugging Face token was found in this shell.")

    lines.extend(
        [
            "Next steps:",
            f"1. Confirm access at https://huggingface.co/{repo_id}",
            "2. Authenticate in this shell with `hf auth login` or `huggingface-cli login`",
            "3. Relaunch the training command after `huggingface-cli whoami` shows the expected account",
            "4. Or point `model_name_or_path` at a local checkpoint or another accessible repo",
        ]
    )
    return "\n".join(lines)


def ensure_hf_model_access(model_args, run_logger) -> None:
    repo_id = getattr(model_args, "model_name_or_path", None)
    revision = getattr(model_args, "model_revision", "main")
    token = HfFolder.get_token()

    if repo_id is None or _is_local_repo_path(repo_id):
        return

    try:
        hf_hub_download(
            repo_id=repo_id,
            filename="config.json",
            revision=revision,
            token=token,
        )
    except (GatedRepoError, RepositoryNotFoundError, HfHubHTTPError) as error:
        raise RuntimeError(
            _build_hf_access_error(
                repo_id=repo_id,
                revision=revision,
                filename="config.json",
                token_present=bool(token),
                error=error,
            )
        ) from error

    tokenizer_repo_id = getattr(model_args, "tokenizer_name_or_path", None)
    if tokenizer_repo_id is None or tokenizer_repo_id == repo_id or _is_local_repo_path(tokenizer_repo_id):
        return

    try:
        hf_hub_download(
            repo_id=tokenizer_repo_id,
            filename="tokenizer_config.json",
            revision=revision,
            token=token,
        )
    except (GatedRepoError, RepositoryNotFoundError, HfHubHTTPError) as error:
        raise RuntimeError(
            _build_hf_access_error(
                repo_id=tokenizer_repo_id,
                revision=revision,
                filename="tokenizer_config.json",
                token_present=bool(token),
                error=error,
            )
        ) from error


def apply_preference_chat_template(
    example,
    tokenizer,
    auto_insert_empty_system_msg: bool = True,
    change_template=None,
):
    # Keep the legacy kwarg for call-site compatibility, but do not override
    # tokenizer.chat_template based on the model name anymore.
    del change_template

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
    ensure_hf_model_access(model_args, run_logger)

    last_checkpoint = get_checkpoint(training_args)
    if last_checkpoint is not None and training_args.resume_from_checkpoint is None:
        run_logger.info(f"Checkpoint detected, resuming training at {last_checkpoint=}.")

    set_seed(training_args.seed)
    return last_checkpoint


def _is_main_process(training_args) -> bool:
    return getattr(training_args, "process_index", 0) == 0


def _resolve_preprocessing_log_dir(data_args, training_args) -> Path:
    if data_args.preprocessing_log_dir:
        return Path(data_args.preprocessing_log_dir).expanduser()
    return Path(training_args.output_dir) / "preprocessing_logs"


def _select_preprocessing_sample_indices(dataset_size: int, sample_count: int, seed: int) -> list[int]:
    if dataset_size <= 0 or sample_count <= 0:
        return []
    effective_count = min(dataset_size, sample_count)
    return random.Random(seed).sample(range(dataset_size), k=effective_count)


def _build_processed_sample_record(dataset, index: int, split_name: str) -> dict:
    sample = dataset[index]
    return {
        "sample_index": index,
        "split": split_name,
        "prompt": sample["prompt"],
        "chosen": sample["chosen"],
        "rejected": sample["rejected"],
    }


def _log_processed_train_samples(raw_datasets, data_args, training_args, run_logger) -> None:
    if "train" not in raw_datasets or len(raw_datasets["train"]) == 0 or not _is_main_process(training_args):
        return

    train_dataset = raw_datasets["train"]
    sample_indices = _select_preprocessing_sample_indices(
        dataset_size=len(train_dataset),
        sample_count=max(1, data_args.preprocessing_log_samples),
        seed=training_args.seed,
    )
    terminal_record = _build_processed_sample_record(train_dataset, sample_indices[0], split_name="train")
    run_logger.info(
        "Processed train sample %(sample_index)s:\n\nPrompt:\n%(prompt)s\n\nChosen:\n%(chosen)s\n\nRejected:\n%(rejected)s",
        terminal_record,
    )

    if data_args.preprocessing_log_samples <= 0:
        return

    log_dir = _resolve_preprocessing_log_dir(data_args, training_args)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "train_samples.jsonl"
    records = [
        _build_processed_sample_record(train_dataset, index, split_name="train")
        for index in sample_indices[: data_args.preprocessing_log_samples]
    ]
    with log_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    run_logger.info(f"Saved {len(records)} processed train samples to {log_path}")


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
    # Preserve the formatting metadata path, but keep the Mistral-specific
    # template override disabled.
    change_template = None
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

    _log_processed_train_samples(raw_datasets, data_args, training_args, run_logger)

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

    if training_args.do_eval and "test" not in raw_datasets:
        run_logger.warning("Skipping evaluation because no `test` split was loaded.")
    elif training_args.do_eval:
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
