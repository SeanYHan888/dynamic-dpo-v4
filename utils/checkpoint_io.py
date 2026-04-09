import logging
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from accelerate.utils import DistributedType
from datasets import Dataset
from huggingface_hub import HfApi, upload_folder
from transformers.trainer import PREFIX_CHECKPOINT_DIR

MARGIN_DATASET_COLUMNS = (
    "epoch",
    "step",
    "batch_size",
    "mean",
    "std",
    "min",
    "p10",
    "median",
    "p90",
    "max",
    "pos_frac",
    "sample",
    "npy",
)


def _find_state_tensor(state_dict: Mapping[str, torch.Tensor], suffix: str) -> torch.Tensor | None:
    tensor = state_dict.get(suffix)
    if tensor is not None:
        return tensor
    for key, value in state_dict.items():
        if key.endswith(suffix):
            return value
    return None


def validate_causal_lm_export(
    model: Any,
    state_dict: Mapping[str, torch.Tensor],
) -> None:
    config = getattr(model, "config", None)
    if config is None:
        return

    expected_vocab_size = getattr(config, "vocab_size", None)
    hidden_size = getattr(config, "hidden_size", None)
    if expected_vocab_size is None or hidden_size is None:
        return

    embed_tokens = _find_state_tensor(state_dict, "model.embed_tokens.weight")
    if embed_tokens is not None:
        expected_shape = (expected_vocab_size, hidden_size)
        actual_shape = tuple(embed_tokens.shape)
        if actual_shape != expected_shape:
            raise RuntimeError(
                "Refusing to save an invalid checkpoint: "
                f"expected model.embed_tokens.weight shape {expected_shape}, got {actual_shape}. "
                "This usually means the save path exported a sharded/flattened FSDP tensor instead of a full HF checkpoint."
            )

    lm_head = _find_state_tensor(state_dict, "lm_head.weight")
    if lm_head is not None and lm_head.numel() > 0:
        expected_shape = (expected_vocab_size, hidden_size)
        actual_shape = tuple(lm_head.shape)
        if actual_shape != expected_shape:
            raise RuntimeError(
                "Refusing to save an invalid checkpoint: "
                f"expected lm_head.weight shape {expected_shape}, got {actual_shape}. "
                "This checkpoint would not load cleanly for inference."
            )


def _get_processing_artifact(trainer: Any) -> Any | None:
    for attr_name in ("processing_class", "tokenizer"):
        artifact = getattr(trainer, attr_name, None)
        if artifact is not None and hasattr(artifact, "save_pretrained"):
            return artifact
    return None


def save_hf_compatible_model_artifacts(
    accelerator: Any,
    model: Any,
    output_dir: str | Path,
    logger: logging.Logger,
    *,
    processing_artifact: Any | None = None,
    safe_serialization: bool = True,
) -> None:
    output_path = Path(output_dir)

    accelerator.wait_for_everyone()
    # Under FSDP, Accelerate needs the wrapped model to materialize a proper FULL_STATE_DICT.
    # Using unwrap=True there can expose flattened inner parameters instead of gathered HF weights.
    unwrap_for_state_dict = accelerator.distributed_type != DistributedType.FSDP
    state_dict = accelerator.get_state_dict(model, unwrap=unwrap_for_state_dict)
    unwrapped_model = accelerator.unwrap_model(model)

    if accelerator.is_main_process:
        validate_causal_lm_export(unwrapped_model, state_dict)
        unwrapped_model.save_pretrained(
            output_path,
            state_dict=state_dict,
            safe_serialization=safe_serialization,
            save_function=accelerator.save,
        )

        if processing_artifact is not None:
            processing_artifact.save_pretrained(output_path)

        logger.info(f"Saved HF-compatible model artifacts to {output_path}")

    accelerator.wait_for_everyone()


def save_hf_compatible_training_artifacts(
    trainer: Any,
    output_dir: str | Path,
    logger: logging.Logger,
) -> None:
    save_hf_compatible_model_artifacts(
        trainer.accelerator,
        trainer.model,
        output_dir,
        logger,
        processing_artifact=_get_processing_artifact(trainer),
        safe_serialization=bool(getattr(trainer.args, "save_safetensors", True)),
    )


def push_prevalidated_hf_artifacts(
    trainer: Any,
    output_dir: str | Path,
    logger: logging.Logger,
    *,
    commit_message: str = "End of training",
) -> str | Any | None:
    accelerator = trainer.accelerator
    accelerator.wait_for_everyone()

    if not accelerator.is_main_process:
        return None

    token = getattr(trainer.args, "hub_token", None)
    trainer.init_hf_repo(token=token)
    trainer._finish_current_push()

    output_path = Path(output_dir)
    logger.info(f"Uploading validated model artifacts from {output_path} to {trainer.hub_model_id}")
    return upload_folder(
        repo_id=trainer.hub_model_id,
        folder_path=str(output_path),
        commit_message=commit_message,
        token=token,
        run_as_future=False,
        ignore_patterns=["_*", f"{PREFIX_CHECKPOINT_DIR}-*"],
    )


def _resolve_margin_dataset_repo_id(trainer: Any) -> str:
    args = trainer.args
    explicit_repo_id = getattr(args, "hub_margin_dataset_id", None)
    if explicit_repo_id:
        return explicit_repo_id

    model_repo_id = getattr(trainer, "hub_model_id", None) or getattr(args, "hub_model_id", None)
    if not model_repo_id:
        raise ValueError("Cannot resolve margin dataset repo id because hub_model_id is not set.")

    return f"{model_repo_id}-margin"


def _load_margin_summary_rows(margin_jsonl_path: Path) -> list[dict[str, Any]]:
    if not margin_jsonl_path.exists():
        raise FileNotFoundError(f"Margin summary log file does not exist: {margin_jsonl_path}")

    rows = []
    with margin_jsonl_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            payload = line.strip()
            if not payload:
                continue
            raw_row = json.loads(payload)
            rows.append({column: raw_row.get(column) for column in MARGIN_DATASET_COLUMNS})

    if not rows:
        raise ValueError(f"Margin summary log file is empty: {margin_jsonl_path}")

    return rows


def _build_margin_dataset_card(
    *,
    dataset_repo_id: str,
    model_repo_id: str | None,
    model_args: Any | None,
    data_args: Any | None,
    training_args: Any,
    row_count: int,
) -> str:
    dataset_mixer = getattr(data_args, "dataset_mixer", None)
    dataset_mixer_json = json.dumps(dataset_mixer, indent=2, sort_keys=True) if dataset_mixer is not None else "null"
    model_name_or_path = getattr(model_args, "model_name_or_path", None)

    return (
        f"# {dataset_repo_id}\n\n"
        "Per-step margin summary statistics exported from a margin-DPO training run.\n\n"
        "## Source Run\n\n"
        f"- Model repo id: `{model_repo_id}`\n"
        f"- Base model: `{model_name_or_path}`\n"
        f"- Run name: `{getattr(training_args, 'run_name', None)}`\n"
        f"- Margin log path: `{getattr(training_args, 'margin_log_path', None)}`\n"
        f"- Published split: `{getattr(training_args, 'margin_dataset_split', 'train')}`\n"
        f"- Rows: `{row_count}`\n\n"
        "## Columns\n\n"
        "- `epoch`\n"
        "- `step`\n"
        "- `batch_size`\n"
        "- `mean`\n"
        "- `std`\n"
        "- `min`\n"
        "- `p10`\n"
        "- `median`\n"
        "- `p90`\n"
        "- `max`\n"
        "- `pos_frac`\n"
        "- `sample` (per-example margins for the effective batch on that logged step)\n"
        "- `npy` (optional path to the saved full margin array when `margin_save_full=true`)\n\n"
        "## Dataset Mixer\n\n"
        "```json\n"
        f"{dataset_mixer_json}\n"
        "```\n"
    )


def push_margin_dataset_summary(
    trainer: Any,
    logger: logging.Logger,
    *,
    model_args: Any | None = None,
    data_args: Any | None = None,
    commit_message: str = "Upload margin summary dataset",
) -> str | Any:
    training_args = trainer.args
    token = getattr(training_args, "hub_token", None)
    dataset_repo_id = _resolve_margin_dataset_repo_id(trainer)
    model_repo_id = getattr(trainer, "hub_model_id", None) or getattr(training_args, "hub_model_id", None)
    margin_jsonl_path = Path(getattr(training_args, "margin_log_path")).expanduser().resolve() / "margins.jsonl"
    rows = _load_margin_summary_rows(margin_jsonl_path)
    private = getattr(training_args, "margin_dataset_private", None)
    split = getattr(training_args, "margin_dataset_split", "train")

    logger.info(f"Resolved margin dataset repo id: {dataset_repo_id}")
    logger.info(f"Preparing {len(rows)} margin summary rows from {margin_jsonl_path}")

    api = HfApi()
    create_repo_kwargs = {
        "repo_id": dataset_repo_id,
        "repo_type": "dataset",
        "exist_ok": True,
        "token": token,
    }
    if private is not None:
        create_repo_kwargs["private"] = private
    api.create_repo(**create_repo_kwargs)

    dataset = Dataset.from_list(rows)
    dataset.push_to_hub(
        dataset_repo_id,
        split=split,
        private=False if private is None else private,
        token=token,
        commit_message=commit_message,
    )

    dataset_card = _build_margin_dataset_card(
        dataset_repo_id=dataset_repo_id,
        model_repo_id=model_repo_id,
        model_args=model_args,
        data_args=data_args,
        training_args=training_args,
        row_count=len(rows),
    )
    api.upload_file(
        path_or_fileobj=dataset_card.encode("utf-8"),
        path_in_repo="README.md",
        repo_id=dataset_repo_id,
        repo_type="dataset",
        token=token,
        commit_message="Add margin dataset card",
        run_as_future=False,
    )
    return dataset_repo_id


def maybe_push_margin_dataset_summary(
    trainer: Any,
    logger: logging.Logger,
    *,
    model_args: Any | None = None,
    data_args: Any | None = None,
) -> str | Any | None:
    training_args = getattr(trainer, "args", None)
    if training_args is None:
        return None
    if getattr(training_args, "trainer_type", None) != "margin_dpo":
        return None
    if not bool(getattr(training_args, "push_margin_dataset", True)):
        logger.info("Skipping margin dataset upload because push_margin_dataset is false.")
        return None

    try:
        return push_margin_dataset_summary(
            trainer,
            logger,
            model_args=model_args,
            data_args=data_args,
        )
    except (FileNotFoundError, ValueError) as exc:
        logger.warning(f"Skipping margin dataset upload: {exc}")
        return None
    except Exception as exc:  # pragma: no cover - defensive best-effort wrapper
        logger.warning(f"Margin dataset upload failed: {exc}")
        return None
