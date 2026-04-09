import hashlib
import inspect
import json
import logging
import os
import shutil
import time
from pathlib import Path
from typing import Any, Dict, Optional, Union

from datasets import Dataset, DatasetDict, config as datasets_config, load_from_disk
from utils.inspection_logging import (
    is_main_process,
    resolve_inspection_log_dir,
    select_sample_indices,
    write_jsonl_records,
)

DEFAULT_HF_CACHE_SUBDIR = Path(".cache") / "hf_datasets"
DEFAULT_TOKENIZED_CACHE_SUBDIR = Path(".cache") / "tokenized_preferences"
TOKENIZED_DATASET_SUBDIR = "dataset"
TOKENIZED_MANIFEST_FILENAME = "manifest.json"
REQUIRED_DECODER_ONLY_COLUMNS = (
    "chosen_input_ids",
    "chosen_attention_mask",
    "chosen_labels",
    "rejected_input_ids",
    "rejected_attention_mask",
    "rejected_labels",
    "prompt_input_ids",
    "prompt_attention_mask",
)
LOGGER = logging.getLogger(__name__)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _resolve_repo_cache_dir(explicit_path: Optional[str], default_subdir: Path) -> Path:
    if explicit_path:
        path = Path(explicit_path).expanduser()
        if not path.is_absolute():
            path = (_repo_root() / path).resolve()
        return path
    return (_repo_root() / default_subdir).resolve()


def _coerce_token_sequence(value: Any) -> Optional[list[int]]:
    if value is None:
        return None
    return [int(token) for token in value]


def _decode_tokens(tokenizer, token_ids: Optional[list[int]]) -> Optional[str]:
    if token_ids is None:
        return None

    decode = getattr(tokenizer, "decode", None)
    if callable(decode):
        try:
            return decode(token_ids, skip_special_tokens=False)
        except TypeError:
            return decode(token_ids)
    return None


def _build_post_tokenization_sample_record(
    dataset: Dataset,
    index: int,
    split_name: str,
    tokenization_source: str,
    tokenizer,
    label_pad_token_id: int,
) -> dict[str, Any]:
    sample = dataset[index]
    prompt_input_ids = _coerce_token_sequence(sample.get("prompt_input_ids"))
    prompt_attention_mask = _coerce_token_sequence(sample.get("prompt_attention_mask"))
    chosen_input_ids = _coerce_token_sequence(sample.get("chosen_input_ids"))
    chosen_attention_mask = _coerce_token_sequence(sample.get("chosen_attention_mask"))
    chosen_labels = _coerce_token_sequence(sample.get("chosen_labels"))
    rejected_input_ids = _coerce_token_sequence(sample.get("rejected_input_ids"))
    rejected_attention_mask = _coerce_token_sequence(sample.get("rejected_attention_mask"))
    rejected_labels = _coerce_token_sequence(sample.get("rejected_labels"))
    chosen_target_ids = None if chosen_labels is None else [
        token for token in chosen_labels if token != label_pad_token_id
    ]
    rejected_target_ids = None if rejected_labels is None else [
        token for token in rejected_labels if token != label_pad_token_id
    ]

    return {
        "sample_index": index,
        "split": split_name,
        "tokenization_source": tokenization_source,
        "prompt": sample.get("prompt"),
        "chosen": sample.get("chosen"),
        "rejected": sample.get("rejected"),
        "prompt_input_ids": prompt_input_ids,
        "prompt_attention_mask": prompt_attention_mask,
        "chosen_input_ids": chosen_input_ids,
        "chosen_attention_mask": chosen_attention_mask,
        "chosen_labels": chosen_labels,
        "rejected_input_ids": rejected_input_ids,
        "rejected_attention_mask": rejected_attention_mask,
        "rejected_labels": rejected_labels,
        "decoded_prompt": _decode_tokens(tokenizer, prompt_input_ids),
        "decoded_chosen_full": _decode_tokens(tokenizer, chosen_input_ids),
        "decoded_rejected_full": _decode_tokens(tokenizer, rejected_input_ids),
        "decoded_chosen_target": _decode_tokens(tokenizer, chosen_target_ids),
        "decoded_rejected_target": _decode_tokens(tokenizer, rejected_target_ids),
        "prompt_length": None if prompt_input_ids is None else len(prompt_input_ids),
        "chosen_length": None if chosen_input_ids is None else len(chosen_input_ids),
        "rejected_length": None if rejected_input_ids is None else len(rejected_input_ids),
        "chosen_target_length": None if chosen_target_ids is None else len(chosen_target_ids),
        "rejected_target_length": None if rejected_target_ids is None else len(rejected_target_ids),
    }


def _log_post_tokenization_samples(
    dataset: Dataset,
    args,
    split_name: str,
    tokenization_source: str,
    tokenizer,
    label_pad_token_id: int,
    logger: logging.Logger = LOGGER,
) -> None:
    if split_name != "train" or len(dataset) == 0 or not is_main_process(getattr(args, "process_index", 0)):
        return

    sample_count = getattr(args, "post_tokenization_log_samples", 0)
    sample_indices = select_sample_indices(
        dataset_size=len(dataset),
        sample_count=max(1, sample_count),
        seed=getattr(args, "seed", 0),
    )
    terminal_record = _build_post_tokenization_sample_record(
        dataset,
        sample_indices[0],
        split_name=split_name,
        tokenization_source=tokenization_source,
        tokenizer=tokenizer,
        label_pad_token_id=label_pad_token_id,
    )
    logger.info(
        "Post-tokenization train sample %(sample_index)s (%(tokenization_source)s):\n\n"
        "Prompt ids:\n%(prompt_input_ids)s\n\n"
        "Chosen ids:\n%(chosen_input_ids)s\n\n"
        "Rejected ids:\n%(rejected_input_ids)s\n\n"
        "Decoded prompt:\n%(decoded_prompt)s\n\n"
        "Decoded chosen target:\n%(decoded_chosen_target)s\n\n"
        "Decoded rejected target:\n%(decoded_rejected_target)s",
        terminal_record,
    )

    if sample_count <= 0:
        return

    records = [
        _build_post_tokenization_sample_record(
            dataset,
            index,
            split_name=split_name,
            tokenization_source=tokenization_source,
            tokenizer=tokenizer,
            label_pad_token_id=label_pad_token_id,
        )
        for index in sample_indices[:sample_count]
    ]
    log_dir = resolve_inspection_log_dir(
        getattr(args, "post_tokenization_log_dir", None),
        getattr(args, "output_dir", None),
    )
    log_path = write_jsonl_records(log_dir, "post_tokenization_train_samples.jsonl", records)
    logger.info(f"Saved {len(records)} post-tokenization train samples to {log_path}")


def _callable_source_hash(fn: Any) -> str:
    target = getattr(fn, "__func__", fn)
    try:
        source = inspect.getsource(target)
    except (OSError, TypeError):
        source = repr(target)
    return hashlib.sha256(source.encode("utf-8")).hexdigest()


def _stable_hash(payload: Dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def configure_persistent_hf_cache(data_args, run_logger) -> Optional[str]:
    if not getattr(data_args, "use_persistent_hf_cache", False):
        return None

    cache_dir = _resolve_repo_cache_dir(getattr(data_args, "hf_cache_dir", None), DEFAULT_HF_CACHE_SUBDIR)
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["HF_DATASETS_CACHE"] = str(cache_dir)
    datasets_config.HF_DATASETS_CACHE = str(cache_dir)
    run_logger.info(f"Using persistent HF datasets cache at {cache_dir}")
    return str(cache_dir)


def build_prompt_preprocessing_metadata(tokenizer, data_args, prompt_format_fn) -> Dict[str, Any]:
    active_chat_template = getattr(tokenizer, "chat_template", None)
    if active_chat_template is None:
        active_chat_template = getattr(tokenizer, "default_chat_template", None)

    return {
        "tokenizer_name_or_path": getattr(tokenizer, "name_or_path", None),
        "tokenizer_class": tokenizer.__class__.__name__,
        "active_chat_template": active_chat_template,
        "auto_insert_empty_system_msg": getattr(data_args, "auto_insert_empty_system_msg", True),
        "prompt_format_function_hash": _callable_source_hash(prompt_format_fn),
    }


def attach_prompt_preprocessing_metadata(training_args, metadata: Dict[str, Any]) -> None:
    setattr(training_args, "prompt_preprocessing_metadata", metadata)


def maybe_prepare_tokenized_datasets(trainer, args, train_dataset, eval_dataset):
    tokenization_mode = getattr(args, "tokenization_mode", "online")
    reuse_tokenized_dataset = bool(getattr(args, "reuse_tokenized_dataset", False))

    if tokenization_mode == "online" and not reuse_tokenized_dataset:
        train_dataset = _tokenize_dataset(trainer, args, train_dataset, "train")
        if eval_dataset is not None:
            if isinstance(eval_dataset, dict):
                eval_dataset = {
                    split_name: _tokenize_dataset(trainer, args, split_dataset, split_name)
                    for split_name, split_dataset in eval_dataset.items()
                }
            else:
                eval_dataset = _tokenize_dataset(trainer, args, eval_dataset, "test")
        return train_dataset, eval_dataset

    cache_root = _resolve_repo_cache_dir(getattr(args, "tokenized_dataset_cache_dir", None), DEFAULT_TOKENIZED_CACHE_SUBDIR)
    cache_root.mkdir(parents=True, exist_ok=True)

    train_dataset = _maybe_prepare_single_tokenized_split(
        trainer=trainer,
        args=args,
        dataset=train_dataset,
        split_name="train",
        cache_root=cache_root,
        tokenization_mode=tokenization_mode,
    )
    if eval_dataset is not None:
        if isinstance(eval_dataset, dict):
            eval_dataset = {
                split_name: _maybe_prepare_single_tokenized_split(
                    trainer=trainer,
                    args=args,
                    dataset=split_dataset,
                    split_name=split_name,
                    cache_root=cache_root,
                    tokenization_mode=tokenization_mode,
                )
                for split_name, split_dataset in eval_dataset.items()
            }
        else:
            eval_dataset = _maybe_prepare_single_tokenized_split(
                trainer=trainer,
                args=args,
                dataset=eval_dataset,
                split_name="test",
                cache_root=cache_root,
                tokenization_mode=tokenization_mode,
            )
    return train_dataset, eval_dataset


def _maybe_prepare_single_tokenized_split(
    trainer,
    args,
    dataset: Dataset,
    split_name: str,
    cache_root: Path,
    tokenization_mode: str,
) -> Dataset:
    manifest = _build_tokenized_manifest(trainer=trainer, args=args, dataset=dataset, split_name=split_name)
    manifest_hash = _stable_hash(manifest)
    split_root = cache_root / manifest["trainer_type"] / split_name
    split_dir = split_root / manifest_hash
    dataset_dir = split_dir / TOKENIZED_DATASET_SUBDIR
    manifest_path = split_dir / TOKENIZED_MANIFEST_FILENAME

    if dataset_dir.exists() and manifest_path.exists():
        cached_manifest = _load_manifest(manifest_path)
        if cached_manifest == manifest:
            cached_dataset = load_from_disk(str(dataset_dir))
            if _has_required_columns(cached_dataset, trainer.is_encoder_decoder):
                LOGGER.info(f"tokenized cache hit for {split_name} at {split_dir}")
                _log_post_tokenization_samples(
                    cached_dataset,
                    args=args,
                    split_name=split_name,
                    tokenization_source="cache_hit",
                    tokenizer=trainer.tokenizer,
                    label_pad_token_id=trainer.label_pad_token_id,
                )
                return cached_dataset
            LOGGER.warning(f"rebuild after invalid cache for {split_name}: missing required tokenized columns at {split_dir}")
        else:
            LOGGER.info(f"tokenized manifest mismatch for {split_name}; rebuilding cache at {split_dir}")
        _clear_cache_dir(split_dir)
    elif any(path.is_dir() for path in split_root.iterdir()) if split_root.exists() else False:
        LOGGER.info(f"tokenized manifest mismatch for {split_name}; expected cache key {manifest_hash}")
    else:
        LOGGER.info(f"tokenized cache miss for {split_name}; building {split_dir}")

    if tokenization_mode == "reuse_only":
        raise FileNotFoundError(
            f"Tokenized cache for split '{split_name}' was not found or was stale at {split_dir}. "
            "Run the offline pretokenization job first or switch tokenization_mode back to 'online'."
        )

    tokenized_dataset = _tokenize_dataset(trainer, args, dataset, split_name)
    split_dir.mkdir(parents=True, exist_ok=True)
    tokenized_dataset.save_to_disk(str(dataset_dir))
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    return tokenized_dataset


def _build_tokenized_manifest(trainer, args, dataset: Dataset, split_name: str) -> Dict[str, Any]:
    prompt_metadata = getattr(args, "prompt_preprocessing_metadata", {})
    tokenizer = trainer.tokenizer
    active_chat_template = prompt_metadata.get("active_chat_template")
    if active_chat_template is None:
        active_chat_template = getattr(tokenizer, "chat_template", None)
    if active_chat_template is None:
        active_chat_template = getattr(tokenizer, "default_chat_template", None)

    tokenization_code_hashes = getattr(trainer, "get_tokenization_code_hashes", None)
    if callable(tokenization_code_hashes):
        raw_code_hashes = tokenization_code_hashes()
        code_hashes = {name: _stable_hash({"source": source}) for name, source in raw_code_hashes.items()}
    else:
        code_hashes = {
            "tokenize_row": _callable_source_hash(trainer.tokenize_row),
            "build_tokenized_answer": _callable_source_hash(trainer.build_tokenized_answer),
        }
    code_hashes["prompt_formatting"] = prompt_metadata.get("prompt_format_function_hash")

    manifest = {
        "trainer_type": getattr(args, "trainer_type", trainer.__class__.__name__),
        "tokenizer_name_or_path": prompt_metadata.get("tokenizer_name_or_path", getattr(tokenizer, "name_or_path", None)),
        "tokenizer_class": prompt_metadata.get("tokenizer_class", tokenizer.__class__.__name__),
        "tokenizer_special_token_ids": {
            "bos_token_id": getattr(tokenizer, "bos_token_id", None),
            "eos_token_id": getattr(tokenizer, "eos_token_id", None),
            "pad_token_id": getattr(tokenizer, "pad_token_id", None),
        },
        "active_chat_template": active_chat_template,
        "auto_insert_empty_system_msg": prompt_metadata.get("auto_insert_empty_system_msg"),
        "max_length": trainer.max_length,
        "max_prompt_length": trainer.max_prompt_length,
        "max_target_length": trainer.max_target_length,
        "truncation_mode": trainer.truncation_mode,
        "label_pad_token_id": trainer.label_pad_token_id,
        "padding_value": trainer.padding_value,
        "is_encoder_decoder": trainer.is_encoder_decoder,
        "split_name": split_name,
        "dataset_size": len(dataset),
        "dataset_fingerprint": getattr(dataset, "_fingerprint", None),
        "dataset_columns": list(dataset.column_names),
        "code_hashes": code_hashes,
        "effective_tokenization_code_hash": _stable_hash(code_hashes),
    }
    return manifest


def _tokenize_dataset(trainer, args, dataset: Dataset, split_name: str) -> Dataset:
    start_time = time.perf_counter()
    tokenized_dataset = dataset.map(
        trainer.tokenize_batch,
        batched=True,
        batch_size=getattr(args, "tokenization_batch_size", 128),
        num_proc=args.dataset_num_proc,
        desc=f"Tokenizing {split_name}",
    )
    elapsed = time.perf_counter() - start_time
    LOGGER.info(f"tokenized split {split_name} in {elapsed:.2f}s")
    _log_post_tokenization_samples(
        tokenized_dataset,
        args=args,
        split_name=split_name,
        tokenization_source="fresh",
        tokenizer=trainer.tokenizer,
        label_pad_token_id=trainer.label_pad_token_id,
    )
    return tokenized_dataset


def _load_manifest(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text())


def _has_required_columns(dataset: Union[Dataset, DatasetDict], is_encoder_decoder: bool) -> bool:
    column_names = set(dataset.column_names)
    if is_encoder_decoder:
        required_columns = {"chosen_labels", "rejected_labels", "prompt_input_ids", "prompt_attention_mask"}
    else:
        required_columns = set(REQUIRED_DECODER_ONLY_COLUMNS)
    return required_columns.issubset(column_names)


def _clear_cache_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
