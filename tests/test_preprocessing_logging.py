import json
import logging
from types import SimpleNamespace

from datasets import Dataset, DatasetDict

from alignment import DataArguments

from utils import runtime as run_preference_utils


class DummyTokenizer:
    def __init__(self):
        self.chat_template = "system"
        self.default_chat_template = None
        self.bos_token = "<s>"

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        del tokenize, add_generation_prompt
        rendered = " || ".join(f"{message['role']}:{message['content']}" for message in messages)
        return f"{self.bos_token}{rendered}"


def _processed_train_dataset():
    return Dataset.from_list(
        [
            {"prompt": "prompt-0", "chosen": "chosen-0", "rejected": "rejected-0"},
            {"prompt": "prompt-1", "chosen": "chosen-1", "rejected": "rejected-1"},
            {"prompt": "prompt-2", "chosen": "chosen-2", "rejected": "rejected-2"},
        ]
    )


def _raw_preference_dataset():
    return Dataset.from_list(
        [
            {
                "prompt": [{"role": "user", "content": "Question 0"}],
                "chosen": [{"role": "assistant", "content": "Answer 0"}],
                "rejected": [{"role": "assistant", "content": "Bad 0"}],
            },
            {
                "prompt": [{"role": "user", "content": "Question 1"}],
                "chosen": [{"role": "assistant", "content": "Answer 1"}],
                "rejected": [{"role": "assistant", "content": "Bad 1"}],
            },
        ]
    )


def test_log_processed_train_samples_logs_terminal_sample_without_file(caplog, tmp_path):
    caplog.set_level(logging.INFO)
    raw_datasets = {"train": _processed_train_dataset()}
    data_args = DataArguments(preprocessing_log_samples=0)
    training_args = SimpleNamespace(output_dir=str(tmp_path / "output"), seed=13, process_index=0)

    run_preference_utils._log_processed_train_samples(
        raw_datasets,
        data_args,
        training_args,
        logging.getLogger("test_preprocessing_terminal_only"),
    )

    assert "Processed train sample" in caplog.text
    assert not (tmp_path / "output" / "preprocessing_logs" / "train_samples.jsonl").exists()


def test_log_processed_train_samples_writes_default_jsonl_and_clamps_count(tmp_path):
    raw_datasets = {"train": _processed_train_dataset()}
    data_args = DataArguments(preprocessing_log_samples=10)
    training_args = SimpleNamespace(output_dir=str(tmp_path / "output"), seed=7, process_index=0)

    run_preference_utils._log_processed_train_samples(
        raw_datasets,
        data_args,
        training_args,
        logging.getLogger("test_preprocessing_default_path"),
    )

    log_path = tmp_path / "output" / "preprocessing_logs" / "train_samples.jsonl"
    assert log_path.exists()
    rows = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 3
    assert set(rows[0].keys()) == {"sample_index", "split", "prompt", "chosen", "rejected"}
    assert all(row["split"] == "train" for row in rows)


def test_log_processed_train_samples_honors_explicit_log_dir(tmp_path):
    raw_datasets = {"train": _processed_train_dataset()}
    explicit_dir = tmp_path / "custom-preprocessing-logs"
    data_args = DataArguments(preprocessing_log_samples=2, preprocessing_log_dir=str(explicit_dir))
    training_args = SimpleNamespace(output_dir=str(tmp_path / "output"), seed=11, process_index=0)

    run_preference_utils._log_processed_train_samples(
        raw_datasets,
        data_args,
        training_args,
        logging.getLogger("test_preprocessing_explicit_path"),
    )

    log_path = explicit_dir / "train_samples.jsonl"
    assert log_path.exists()
    rows = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 2


def test_log_processed_train_samples_skips_non_main_process(caplog, tmp_path):
    caplog.set_level(logging.INFO)
    raw_datasets = {"train": _processed_train_dataset()}
    data_args = DataArguments(preprocessing_log_samples=2)
    training_args = SimpleNamespace(output_dir=str(tmp_path / "output"), seed=5, process_index=1)

    run_preference_utils._log_processed_train_samples(
        raw_datasets,
        data_args,
        training_args,
        logging.getLogger("test_preprocessing_non_main"),
    )

    assert "Processed train sample" not in caplog.text
    assert not (tmp_path / "output" / "preprocessing_logs" / "train_samples.jsonl").exists()


def test_prepare_preference_datasets_logs_processed_train_samples(monkeypatch, caplog, tmp_path):
    caplog.set_level(logging.INFO)

    monkeypatch.setattr(
        run_preference_utils,
        "get_datasets",
        lambda *args, **kwargs: DatasetDict({"train": _raw_preference_dataset()}),
    )
    monkeypatch.setattr(run_preference_utils, "get_tokenizer", lambda *args, **kwargs: DummyTokenizer())
    monkeypatch.setattr(run_preference_utils, "build_model_init_kwargs", lambda *args, **kwargs: {"ok": True})
    monkeypatch.setattr(run_preference_utils, "configure_persistent_hf_cache", lambda *args, **kwargs: None)

    model_args = SimpleNamespace(
        model_name_or_path="test-model",
        torch_dtype=None,
        model_revision="main",
        trust_remote_code=False,
        attn_implementation=None,
    )
    data_args = DataArguments(preprocessing_log_samples=2)
    training_args = SimpleNamespace(
        output_dir=str(tmp_path / "output"),
        seed=17,
        process_index=0,
        gradient_checkpointing=False,
    )

    raw_datasets, _ = run_preference_utils.prepare_preference_datasets(
        model_args=model_args,
        data_args=data_args,
        training_args=training_args,
        run_logger=logging.getLogger("test_prepare_preference_datasets_logging"),
    )

    assert "Processed train sample" in caplog.text
    assert set(raw_datasets["train"].column_names) == {"prompt", "chosen", "rejected"}
    log_path = tmp_path / "output" / "preprocessing_logs" / "train_samples.jsonl"
    rows = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 2
    assert rows[0]["prompt"].startswith("<s>system: || user:Question")
    assert rows[0]["chosen"].startswith("assistant:Answer")
    assert rows[0]["rejected"].startswith("assistant:Bad")
