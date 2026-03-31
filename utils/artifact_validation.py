"""Checkpoint preflight validation for local and Hub-hosted HF causal LM exports."""

from __future__ import annotations

import json
import struct
from functools import lru_cache
from pathlib import Path
from typing import Any, BinaryIO, Dict

from huggingface_hub import HfFileSystem, hf_hub_download


def _is_local_model_path(model_name_or_path: str) -> bool:
    return Path(model_name_or_path).expanduser().exists()


@lru_cache(maxsize=32)
def _load_json(model_name_or_path: str, filename: str) -> Dict[str, Any]:
    if _is_local_model_path(model_name_or_path):
        path = Path(model_name_or_path).expanduser() / filename
    else:
        path = Path(hf_hub_download(model_name_or_path, filename=filename))
    return json.loads(path.read_text(encoding="utf-8"))


def _open_remote_binary(model_name_or_path: str, filename: str) -> BinaryIO:
    fs = HfFileSystem()
    return fs.open(f"{model_name_or_path}/{filename}", "rb")


def _open_binary(model_name_or_path: str, filename: str) -> BinaryIO:
    if _is_local_model_path(model_name_or_path):
        return (Path(model_name_or_path).expanduser() / filename).open("rb")
    return _open_remote_binary(model_name_or_path, filename)


@lru_cache(maxsize=32)
def _get_tensor_header(model_name_or_path: str, tensor_name: str) -> Dict[str, Any]:
    index = _load_json(model_name_or_path, "model.safetensors.index.json")
    weight_map = index.get("weight_map", {})
    shard_name = weight_map.get(tensor_name)
    if shard_name is None:
        raise RuntimeError(
            f"Checkpoint validation failed for {model_name_or_path}: missing {tensor_name} in model.safetensors.index.json."
        )

    with _open_binary(model_name_or_path, shard_name) as handle:
        header_len = struct.unpack("<Q", handle.read(8))[0]
        header = json.loads(handle.read(header_len))

    tensor_header = header.get(tensor_name)
    if tensor_header is None:
        raise RuntimeError(
            f"Checkpoint validation failed for {model_name_or_path}: shard {shard_name} does not contain {tensor_name}."
        )
    return tensor_header


def _shape_tuple(tensor_header: Dict[str, Any]) -> tuple[int, ...]:
    shape = tensor_header.get("shape")
    if not isinstance(shape, list):
        return ()
    return tuple(int(dim) for dim in shape)


@lru_cache(maxsize=32)
def preflight_validate_causal_lm_artifact(model_name_or_path: str) -> None:
    config = _load_json(model_name_or_path, "config.json")
    expected_vocab_size = config.get("vocab_size")
    hidden_size = config.get("hidden_size")
    if not isinstance(expected_vocab_size, int) or not isinstance(hidden_size, int):
        return

    embed_header = _get_tensor_header(model_name_or_path, "model.embed_tokens.weight")
    embed_shape = _shape_tuple(embed_header)
    expected_embed_shape = (expected_vocab_size, hidden_size)
    if embed_shape != expected_embed_shape:
        raise RuntimeError(
            "Invalid HF checkpoint export detected before inference startup. "
            f"{model_name_or_path} declares vocab_size={expected_vocab_size} and hidden_size={hidden_size}, "
            f"but model.embed_tokens.weight is stored as shape {embed_shape} with dtype={embed_header.get('dtype')}. "
            "This checkpoint was exported incorrectly and vLLM will fail while loading embeddings. "
            "Republish the model from a valid local training output; the current training code already uses "
            "utils/checkpoint_io.py to guard against flattened or sharded embedding exports."
        )

    lm_head_header = _get_tensor_header(model_name_or_path, "lm_head.weight")
    lm_head_shape = _shape_tuple(lm_head_header)
    expected_lm_head_shape = (expected_vocab_size, hidden_size)
    if lm_head_shape != expected_lm_head_shape:
        raise RuntimeError(
            "Invalid HF checkpoint export detected before inference startup. "
            f"{model_name_or_path} declares lm_head.weight shape {expected_lm_head_shape}, "
            f"but the saved tensor header is {lm_head_shape} with dtype={lm_head_header.get('dtype')}. "
            "This checkpoint was exported incorrectly and must be re-saved or re-published before benchmarking."
        )

    norm_header = _get_tensor_header(model_name_or_path, "model.norm.weight")
    norm_shape = _shape_tuple(norm_header)
    expected_norm_shape = (hidden_size,)
    if norm_shape != expected_norm_shape:
        raise RuntimeError(
            "Invalid HF checkpoint export detected before inference startup. "
            f"{model_name_or_path} declares hidden_size={hidden_size}, but model.norm.weight is stored as {norm_shape}. "
            "The checkpoint appears truncated or otherwise corrupted and must be re-exported."
        )
