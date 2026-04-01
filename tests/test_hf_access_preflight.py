from pathlib import Path
from types import SimpleNamespace

import pytest
from huggingface_hub.utils import GatedRepoError
from requests import Response

from utils import runtime


def test_ensure_hf_model_access_skips_local_model_path(monkeypatch, tmp_path):
    local_model_dir = tmp_path / "local-model"
    local_model_dir.mkdir()
    model_args = SimpleNamespace(
        model_name_or_path=str(local_model_dir),
        tokenizer_name_or_path=None,
        model_revision="main",
    )

    called = False

    def fake_download(**kwargs):
        nonlocal called
        called = True
        raise AssertionError(f"hf_hub_download should not be called for local paths: {kwargs}")

    monkeypatch.setattr(runtime, "hf_hub_download", fake_download)

    runtime.ensure_hf_model_access(model_args, run_logger=None)

    assert called is False


def test_ensure_hf_model_access_raises_actionable_error_for_gated_repo(monkeypatch):
    model_args = SimpleNamespace(
        model_name_or_path="meta-llama/Meta-Llama-3-8B",
        tokenizer_name_or_path=None,
        model_revision="main",
    )

    response = Response()
    response.status_code = 401
    response.url = "https://huggingface.co/meta-llama/Meta-Llama-3-8B/resolve/main/config.json"
    response.headers["X-Request-Id"] = "test-request-id"

    def fake_download(**kwargs):
        raise GatedRepoError("gated", response=response)

    monkeypatch.setattr(runtime, "hf_hub_download", fake_download)
    monkeypatch.setattr(runtime.HfFolder, "get_token", lambda: None)

    with pytest.raises(RuntimeError) as exc_info:
        runtime.ensure_hf_model_access(model_args, run_logger=None)

    message = str(exc_info.value)
    assert "meta-llama/Meta-Llama-3-8B" in message
    assert "No Hugging Face token was found in this shell." in message
    assert "hf auth login" in message
