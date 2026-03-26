import json
import logging
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import run_preference_utils
import save_utils


def _write_margin_jsonl(margin_dir: Path, rows: list[dict]) -> Path:
    margin_dir.mkdir(parents=True, exist_ok=True)
    path = margin_dir / "margins.jsonl"
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    return path


def test_resolve_margin_dataset_repo_id_derives_from_model_repo():
    trainer = SimpleNamespace(
        args=SimpleNamespace(hub_margin_dataset_id=None, hub_model_id="org/model"),
        hub_model_id="org/model",
    )

    assert save_utils._resolve_margin_dataset_repo_id(trainer) == "org/model-margin"


def test_maybe_push_margin_dataset_summary_skips_non_margin_trainer():
    trainer = SimpleNamespace(
        args=SimpleNamespace(trainer_type="beta_dpo", push_margin_dataset=True),
    )

    result = save_utils.maybe_push_margin_dataset_summary(trainer, logging.getLogger("test_non_margin"))

    assert result is None


def test_maybe_push_margin_dataset_summary_warns_when_margin_log_missing(caplog, tmp_path):
    trainer = SimpleNamespace(
        args=SimpleNamespace(
            trainer_type="margin_dpo",
            push_margin_dataset=True,
            hub_margin_dataset_id=None,
            hub_model_id="org/model",
            margin_log_path=str(tmp_path / "missing_margin_logs"),
            margin_dataset_private=None,
            margin_dataset_split="train",
            run_name="missing-log-run",
        ),
        hub_model_id="org/model",
    )

    caplog.set_level(logging.WARNING)
    result = save_utils.maybe_push_margin_dataset_summary(
        trainer,
        logging.getLogger("test_missing_margin_log"),
        model_args=SimpleNamespace(model_name_or_path="base-model"),
        data_args=SimpleNamespace(dataset_mixer={"dataset": 0.1}),
    )

    assert result is None
    assert "Skipping margin dataset upload" in caplog.text


def test_maybe_push_margin_dataset_summary_uses_override_repo_id(monkeypatch, tmp_path):
    calls = []

    class FakeDatasetObject:
        def __init__(self, rows):
            self.rows = rows

        def push_to_hub(self, repo_id, **kwargs):
            calls.append(("dataset_push", repo_id, kwargs, self.rows))
            return "commit-info"

    class FakeDatasetModule:
        @staticmethod
        def from_list(rows):
            calls.append(("from_list", rows))
            return FakeDatasetObject(rows)

    class FakeApi:
        def create_repo(self, **kwargs):
            calls.append(("create_repo", kwargs))

        def upload_file(self, **kwargs):
            calls.append(("upload_file", kwargs))
            return "upload-info"

    margin_dir = tmp_path / "margin_logs"
    _write_margin_jsonl(
        margin_dir,
        [
            {
                "epoch": 0.0,
                "step": 10,
                "batch_size": 4,
                "mean": 0.25,
                "std": 0.1,
                "min": -0.2,
                "p10": 0.0,
                "median": 0.2,
                "p90": 0.4,
                "max": 0.5,
                "pos_frac": 0.75,
            }
        ],
    )

    monkeypatch.setattr(save_utils, "Dataset", FakeDatasetModule)
    monkeypatch.setattr(save_utils, "HfApi", lambda: FakeApi())

    trainer = SimpleNamespace(
        args=SimpleNamespace(
            trainer_type="margin_dpo",
            push_margin_dataset=True,
            hub_margin_dataset_id="org/custom-margin",
            hub_model_id="org/model",
            hub_token="token-123",
            margin_log_path=str(margin_dir),
            margin_dataset_private=None,
            margin_dataset_split="eval",
            run_name="margin-upload-test",
        ),
        hub_model_id="org/model",
    )

    result = save_utils.maybe_push_margin_dataset_summary(
        trainer,
        logging.getLogger("test_margin_upload"),
        model_args=SimpleNamespace(model_name_or_path="base-model"),
        data_args=SimpleNamespace(dataset_mixer={"dataset": 0.1}),
    )

    assert result == "org/custom-margin"
    assert calls[0][0] == "create_repo"
    assert calls[0][1]["repo_id"] == "org/custom-margin"
    assert calls[0][1]["repo_type"] == "dataset"
    assert calls[1][0] == "from_list"
    assert calls[2][0] == "dataset_push"
    assert calls[2][1] == "org/custom-margin"
    assert calls[2][2]["split"] == "eval"
    assert calls[3][0] == "upload_file"
    assert calls[3][1]["repo_id"] == "org/custom-margin"
    assert calls[3][1]["repo_type"] == "dataset"
    assert b"org/model" in calls[3][1]["path_or_fileobj"]


def test_finalize_training_pushes_margin_dataset_after_model_upload(monkeypatch, tmp_path):
    events = []

    class FakeConfig:
        def __init__(self):
            self.use_cache = False

        def save_pretrained(self, output_dir):
            events.append(("save_pretrained", output_dir))

    class FakeTrainer:
        def __init__(self):
            self.accelerator = SimpleNamespace(is_main_process=True)
            self.model = SimpleNamespace(config=FakeConfig())
            self.args = SimpleNamespace(trainer_type="margin_dpo")

        def train(self, resume_from_checkpoint=None):
            events.append(("train", resume_from_checkpoint))
            return SimpleNamespace(metrics={"loss": 1.0})

        def log_metrics(self, split, metrics):
            events.append(("log_metrics", split, metrics))

        def save_metrics(self, split, metrics):
            events.append(("save_metrics", split, metrics))

        def save_state(self):
            events.append(("save_state",))

        def create_model_card(self, **kwargs):
            events.append(("create_model_card", kwargs))

    monkeypatch.setattr(run_preference_utils, "get_checkpoint", lambda _: None)
    monkeypatch.setattr(run_preference_utils, "save_hf_compatible_training_artifacts", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        run_preference_utils,
        "push_prevalidated_hf_artifacts",
        lambda *args, **kwargs: events.append(("push_model",)),
    )
    monkeypatch.setattr(
        run_preference_utils,
        "maybe_push_margin_dataset_summary",
        lambda *args, **kwargs: events.append(("push_margin_dataset",)),
    )

    trainer = FakeTrainer()
    training_args = SimpleNamespace(
        output_dir=str(tmp_path / "output"),
        resume_from_checkpoint=None,
        do_eval=False,
        push_to_hub=True,
    )
    model_args = SimpleNamespace(model_name_or_path="base-model")
    data_args = SimpleNamespace(dataset_mixer={"dataset": 0.1})
    raw_datasets = {"train": [1, 2, 3]}

    run_preference_utils.finalize_training(
        trainer=trainer,
        training_args=training_args,
        model_args=model_args,
        data_args=data_args,
        raw_datasets=raw_datasets,
        run_logger=logging.getLogger("test_finalize_training"),
        tags=["margin-dpo"],
    )

    push_model_index = events.index(("push_model",))
    push_margin_index = events.index(("push_margin_dataset",))
    assert push_model_index < push_margin_index
