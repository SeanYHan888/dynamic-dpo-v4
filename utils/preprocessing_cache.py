import hashlib
import inspect
import json
import logging
import os
import shutil
from pathlib import Path
from typing import Any, Dict, Optional, Union

from datasets import Dataset, DatasetDict, config as datasets_config, load_from_disk

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
    active_chat_template = tokenizer.chat_template
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
    if not getattr(args, "reuse_tokenized_dataset", False):
        train_dataset = train_dataset.map(trainer.tokenize_row, num_proc=args.dataset_num_proc)
        if eval_dataset is not None:
            if isinstance(eval_dataset, dict):
                eval_dataset = {
                    split_name: split_dataset.map(trainer.tokenize_row, num_proc=args.dataset_num_proc)
                    for split_name, split_dataset in eval_dataset.items()
                }
            else:
                eval_dataset = eval_dataset.map(trainer.tokenize_row, num_proc=args.dataset_num_proc)
        return train_dataset, eval_dataset

    cache_root = _resolve_repo_cache_dir(getattr(args, "tokenized_dataset_cache_dir", None), DEFAULT_TOKENIZED_CACHE_SUBDIR)
    cache_root.mkdir(parents=True, exist_ok=True)

    train_dataset = _maybe_prepare_single_tokenized_split(
        trainer=trainer,
        args=args,
        dataset=train_dataset,
        split_name="train",
        cache_root=cache_root,
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
            )
    return train_dataset, eval_dataset


def _maybe_prepare_single_tokenized_split(trainer, args, dataset: Dataset, split_name: str, cache_root: Path) -> Dataset:
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
                return cached_dataset
            LOGGER.warning(f"rebuild after invalid cache for {split_name}: missing required tokenized columns at {split_dir}")
        else:
            LOGGER.info(f"tokenized manifest mismatch for {split_name}; rebuilding cache at {split_dir}")
        _clear_cache_dir(split_dir)
    elif any(path.is_dir() for path in split_root.iterdir()) if split_root.exists() else False:
        LOGGER.info(f"tokenized manifest mismatch for {split_name}; expected cache key {manifest_hash}")
    else:
        LOGGER.info(f"tokenized cache miss for {split_name}; building {split_dir}")

    tokenized_dataset = dataset.map(trainer.tokenize_row, num_proc=args.dataset_num_proc)
    split_dir.mkdir(parents=True, exist_ok=True)
    tokenized_dataset.save_to_disk(str(dataset_dir))
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    return tokenized_dataset


def _build_tokenized_manifest(trainer, args, dataset: Dataset, split_name: str) -> Dict[str, Any]:
    prompt_metadata = getattr(args, "prompt_preprocessing_metadata", {})
    tokenizer = trainer.tokenizer
    active_chat_template = prompt_metadata.get("active_chat_template")
    if active_chat_template is None:
        active_chat_template = tokenizer.chat_template
    if active_chat_template is None:
        active_chat_template = getattr(tokenizer, "default_chat_template", None)

    code_hashes = {
        "tokenize_row": _callable_source_hash(trainer.tokenize_row),
        "build_tokenized_answer": _callable_source_hash(trainer.build_tokenized_answer),
        "prompt_formatting": prompt_metadata.get("prompt_format_function_hash"),
    }

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
