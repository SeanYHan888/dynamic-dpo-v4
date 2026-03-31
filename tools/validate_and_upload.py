#!/usr/bin/env python
"""Validate a local HF checkpoint folder and optionally upload it to the Hub."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable

from huggingface_hub import HfApi, upload_folder
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.artifact_validation import preflight_validate_causal_lm_artifact


REQUIRED_FILES = (
    "config.json",
    "model.safetensors.index.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
)

REMOTE_ARTIFACT_SUFFIXES = (
    ".safetensors",
    ".bin",
    ".json",
    ".md",
)

REMOTE_ARTIFACT_NAMES = {
    "README.md",
    "config.json",
    "generation_config.json",
    "model.safetensors.index.json",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "added_tokens.json",
    "trainer_state.json",
    "train_results.json",
    "all_results.json",
    "training_args.bin",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint-dir",
        required=True,
        help="Absolute or relative path to a local checkpoint folder.",
    )
    parser.add_argument(
        "--repo-id",
        help="Target Hugging Face repo id, for example W-61/my-model.",
    )
    parser.add_argument(
        "--commit-message",
        default="Upload validated checkpoint",
        help="Commit message used for the Hub upload.",
    )
    parser.add_argument(
        "--token",
        default=None,
        help="Optional Hugging Face token. If omitted, the normal HF auth resolution is used.",
    )
    parser.add_argument(
        "--skip-upload",
        action="store_true",
        help="Validate only. Do not upload anything.",
    )
    parser.add_argument(
        "--full-model-load",
        action="store_true",
        help="Also load the full model with transformers. This can require a lot of RAM.",
    )
    parser.add_argument(
        "--delete-stale-remote-artifacts",
        action="store_true",
        help="Delete known model artifact files in the remote repo that are not present locally before upload.",
    )
    return parser.parse_args()


def _print_step(message: str) -> None:
    print(f"[validate_and_upload] {message}")


def _iter_local_files(checkpoint_dir: Path) -> Iterable[str]:
    for path in checkpoint_dir.rglob("*"):
        if path.is_file():
            yield str(path.relative_to(checkpoint_dir))


def _require_checkpoint_files(checkpoint_dir: Path) -> None:
    missing = [name for name in REQUIRED_FILES if not (checkpoint_dir / name).exists()]
    if missing:
        raise RuntimeError(
            "Checkpoint folder is missing required files: "
            + ", ".join(missing)
        )

    shard_files = sorted(checkpoint_dir.glob("model-*.safetensors"))
    if not shard_files:
        raise RuntimeError("Checkpoint folder does not contain any model-*.safetensors shard files.")


def _load_tokenizer(checkpoint_dir: Path):
    tokenizer_kwargs = {"use_fast": True}
    try:
        return AutoTokenizer.from_pretrained(checkpoint_dir, **tokenizer_kwargs)
    except AttributeError as exc:
        if "'list' object has no attribute 'keys'" not in str(exc):
            raise
        return AutoTokenizer.from_pretrained(
            checkpoint_dir,
            **tokenizer_kwargs,
            extra_special_tokens={},
        )


def validate_checkpoint(checkpoint_dir: Path, *, full_model_load: bool) -> None:
    _print_step(f"Checking required files in {checkpoint_dir}")
    _require_checkpoint_files(checkpoint_dir)

    _print_step("Validating safetensor headers against config")
    preflight_validate_causal_lm_artifact(str(checkpoint_dir))

    _print_step("Loading config")
    AutoConfig.from_pretrained(checkpoint_dir)

    _print_step("Loading tokenizer")
    _load_tokenizer(checkpoint_dir)

    if full_model_load:
        _print_step("Loading full model with transformers")
        AutoModelForCausalLM.from_pretrained(checkpoint_dir, low_cpu_mem_usage=True)

    _print_step("Local checkpoint validation passed")


def _should_delete_remote_artifact(path_in_repo: str) -> bool:
    name = Path(path_in_repo).name
    if name in REMOTE_ARTIFACT_NAMES:
        return True
    if name.startswith("model-") and name.endswith(".safetensors"):
        return True
    return name.endswith(REMOTE_ARTIFACT_SUFFIXES) and name.startswith("tokenizer")


def delete_stale_remote_artifacts(
    api: HfApi,
    *,
    repo_id: str,
    checkpoint_dir: Path,
    token: str | None,
) -> None:
    local_files = set(_iter_local_files(checkpoint_dir))
    remote_files = api.list_repo_files(repo_id=repo_id, token=token)
    stale_files = [
        path_in_repo
        for path_in_repo in remote_files
        if _should_delete_remote_artifact(path_in_repo) and path_in_repo not in local_files
    ]

    if not stale_files:
        _print_step("No stale remote artifact files to delete")
        return

    _print_step("Deleting stale remote artifact files:")
    for path_in_repo in stale_files:
        _print_step(f"  delete {path_in_repo}")
        api.delete_file(
            path_in_repo=path_in_repo,
            repo_id=repo_id,
            repo_type="model",
            commit_message=f"Delete stale artifact {path_in_repo}",
            token=token,
        )


def upload_checkpoint(
    checkpoint_dir: Path,
    *,
    repo_id: str,
    commit_message: str,
    token: str | None,
    delete_stale_remote_artifacts_first: bool,
) -> None:
    api = HfApi()
    _print_step(f"Ensuring remote repo exists: {repo_id}")
    api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True, token=token)

    if delete_stale_remote_artifacts_first:
        delete_stale_remote_artifacts(
            api,
            repo_id=repo_id,
            checkpoint_dir=checkpoint_dir,
            token=token,
        )

    _print_step(f"Uploading validated checkpoint to {repo_id}")
    upload_folder(
        repo_id=repo_id,
        folder_path=str(checkpoint_dir),
        repo_type="model",
        commit_message=commit_message,
        token=token,
        ignore_patterns=["_*", "checkpoint-*"],
    )
    _print_step("Upload completed")


def main() -> int:
    args = parse_args()
    checkpoint_dir = Path(args.checkpoint_dir).expanduser().resolve()
    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory does not exist: {checkpoint_dir}")
    if not checkpoint_dir.is_dir():
        raise NotADirectoryError(f"Checkpoint path is not a directory: {checkpoint_dir}")

    validate_checkpoint(checkpoint_dir, full_model_load=args.full_model_load)

    if args.skip_upload:
        _print_step("Skipping upload by request")
        return 0

    if not args.repo_id:
        raise ValueError("--repo-id is required unless --skip-upload is set.")

    upload_checkpoint(
        checkpoint_dir,
        repo_id=args.repo_id,
        commit_message=args.commit_message,
        token=args.token,
        delete_stale_remote_artifacts_first=args.delete_stale_remote_artifacts,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
