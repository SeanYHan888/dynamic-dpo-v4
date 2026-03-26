import logging
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from accelerate.utils import DistributedType
from huggingface_hub import upload_folder
from transformers.trainer import PREFIX_CHECKPOINT_DIR


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
