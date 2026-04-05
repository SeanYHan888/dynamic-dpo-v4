import json
import logging
from types import SimpleNamespace

from datasets import Dataset, DatasetDict

from alignment import DataArguments

from utils import preprocessing_cache as preprocessing_cache_utils
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

    def decode(self, token_ids, skip_special_tokens=False):
        parts = []
        for token_id in token_ids:
            if token_id == 1:
                if not skip_special_tokens:
                    parts.append("<bos>")
            elif token_id == 2:
                if not skip_special_tokens:
                    parts.append("<eos>")
            else:
                parts.append(chr(token_id))
        return "".join(parts)


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


def _post_tokenized_train_dataset():
    return Dataset.from_list(
        [
            {
                "prompt": "prompt-0",
                "chosen": "chosen-0",
                "rejected": "rejected-0",
                "prompt_input_ids": [1, 112, 48],
                "prompt_attention_mask": [1, 1, 1],
                "chosen_input_ids": [1, 112, 48, 99, 48, 2],
                "chosen_attention_mask": [1, 1, 1, 1, 1, 1],
                "chosen_labels": [-100, -100, -100, 99, 48, 2],
                "rejected_input_ids": [1, 112, 48, 114, 48, 2],
                "rejected_attention_mask": [1, 1, 1, 1, 1, 1],
                "rejected_labels": [-100, -100, -100, 114, 48, 2],
            },
            {
                "prompt": "prompt-1",
                "chosen": "chosen-1",
                "rejected": "rejected-1",
                "prompt_input_ids": [1, 112, 49],
                "prompt_attention_mask": [1, 1, 1],
                "chosen_input_ids": [1, 112, 49, 99, 49, 2],
                "chosen_attention_mask": [1, 1, 1, 1, 1, 1],
                "chosen_labels": [-100, -100, -100, 99, 49, 2],
                "rejected_input_ids": [1, 112, 49, 114, 49, 2],
                "rejected_attention_mask": [1, 1, 1, 1, 1, 1],
                "rejected_labels": [-100, -100, -100, 114, 49, 2],
            },
            {
                "prompt": "prompt-2",
                "chosen": "chosen-2",
                "rejected": "rejected-2",
                "prompt_input_ids": [1, 112, 50],
                "prompt_attention_mask": [1, 1, 1],
                "chosen_input_ids": [1, 112, 50, 99, 50, 2],
                "chosen_attention_mask": [1, 1, 1, 1, 1, 1],
                "chosen_labels": [-100, -100, -100, 99, 50, 2],
                "rejected_input_ids": [1, 112, 50, 114, 50, 2],
                "rejected_attention_mask": [1, 1, 1, 1, 1, 1],
                "rejected_labels": [-100, -100, -100, 114, 50, 2],
            },
        ]
    )


class DummyTokenizationTrainer:
    def __init__(self):
        self.tokenizer = DummyTokenizer()
        self.is_encoder_decoder = False
        self.max_length = 32
        self.max_prompt_length = 16
        self.max_target_length = None
        self.truncation_mode = "keep_end"
        self.label_pad_token_id = -100
        self.padding_value = 0
        self.tokenization_calls = 0

    def _encode(self, text):
        return [ord(char) for char in text]

    def tokenize_batch(self, features, model=None):
        del model
        self.tokenization_calls += 1
        output = {
            "chosen_input_ids": [],
            "chosen_attention_mask": [],
            "chosen_labels": [],
            "rejected_input_ids": [],
            "rejected_attention_mask": [],
            "rejected_labels": [],
            "prompt_input_ids": [],
            "prompt_attention_mask": [],
        }

        for prompt, chosen, rejected in zip(features["prompt"], features["chosen"], features["rejected"]):
            prompt_ids = [1] + self._encode(prompt)
            chosen_target = self._encode(chosen) + [2]
            rejected_target = self._encode(rejected) + [2]

            output["prompt_input_ids"].append(prompt_ids)
            output["prompt_attention_mask"].append([1] * len(prompt_ids))
            output["chosen_input_ids"].append(prompt_ids + chosen_target)
            output["chosen_attention_mask"].append([1] * (len(prompt_ids) + len(chosen_target)))
            output["chosen_labels"].append([self.label_pad_token_id] * len(prompt_ids) + chosen_target)
            output["rejected_input_ids"].append(prompt_ids + rejected_target)
            output["rejected_attention_mask"].append([1] * (len(prompt_ids) + len(rejected_target)))
            output["rejected_labels"].append([self.label_pad_token_id] * len(prompt_ids) + rejected_target)

        return output

    def tokenize_row(self, feature, model=None):
        tokenized = self.tokenize_batch(
            {"prompt": [feature["prompt"]], "chosen": [feature["chosen"]], "rejected": [feature["rejected"]]},
            model=model,
        )
        return {key: value[0] for key, value in tokenized.items()}

    def build_tokenized_answer(self, prompt, answer):
        return {"prompt": prompt, "answer": answer}

    def get_tokenization_code_hashes(self):
        return {
            "tokenize_row": "dummy-tokenize-row",
            "build_tokenized_answer": "dummy-build-tokenized-answer",
            "tokenize_batch": "dummy-tokenize-batch",
        }


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


def test_log_post_tokenization_samples_logs_terminal_sample_without_file(caplog, tmp_path):
    caplog.set_level(logging.INFO)
    dataset = _post_tokenized_train_dataset()
    training_args = SimpleNamespace(
        output_dir=str(tmp_path / "output"),
        seed=13,
        process_index=0,
        post_tokenization_log_samples=0,
        post_tokenization_log_dir=None,
    )

    preprocessing_cache_utils._log_post_tokenization_samples(
        dataset,
        training_args,
        split_name="train",
        tokenization_source="fresh",
        tokenizer=DummyTokenizer(),
        label_pad_token_id=-100,
        logger=logging.getLogger("test_post_tokenization_terminal_only"),
    )

    assert "Post-tokenization train sample" in caplog.text
    assert not (tmp_path / "output" / "preprocessing_logs" / "post_tokenization_train_samples.jsonl").exists()


def test_log_post_tokenization_samples_writes_default_jsonl_and_clamps_count(tmp_path):
    dataset = _post_tokenized_train_dataset()
    training_args = SimpleNamespace(
        output_dir=str(tmp_path / "output"),
        seed=7,
        process_index=0,
        post_tokenization_log_samples=10,
        post_tokenization_log_dir=None,
    )

    preprocessing_cache_utils._log_post_tokenization_samples(
        dataset,
        training_args,
        split_name="train",
        tokenization_source="fresh",
        tokenizer=DummyTokenizer(),
        label_pad_token_id=-100,
    )

    log_path = tmp_path / "output" / "preprocessing_logs" / "post_tokenization_train_samples.jsonl"
    assert log_path.exists()
    rows = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 3
    assert rows[0]["tokenization_source"] == "fresh"
    assert rows[0]["decoded_prompt"].startswith("<bos>")
    assert rows[0]["decoded_chosen_target"].endswith("<eos>")
    assert rows[0]["chosen_target_length"] == 3


def test_log_post_tokenization_samples_honors_explicit_log_dir(tmp_path):
    dataset = _post_tokenized_train_dataset()
    explicit_dir = tmp_path / "custom-post-tokenization-logs"
    training_args = SimpleNamespace(
        output_dir=str(tmp_path / "output"),
        seed=11,
        process_index=0,
        post_tokenization_log_samples=2,
        post_tokenization_log_dir=str(explicit_dir),
    )

    preprocessing_cache_utils._log_post_tokenization_samples(
        dataset,
        training_args,
        split_name="train",
        tokenization_source="fresh",
        tokenizer=DummyTokenizer(),
        label_pad_token_id=-100,
    )

    log_path = explicit_dir / "post_tokenization_train_samples.jsonl"
    assert log_path.exists()
    rows = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 2


def test_log_post_tokenization_samples_skips_non_main_process(caplog, tmp_path):
    caplog.set_level(logging.INFO)
    dataset = _post_tokenized_train_dataset()
    training_args = SimpleNamespace(
        output_dir=str(tmp_path / "output"),
        seed=5,
        process_index=1,
        post_tokenization_log_samples=2,
        post_tokenization_log_dir=None,
    )

    preprocessing_cache_utils._log_post_tokenization_samples(
        dataset,
        training_args,
        split_name="train",
        tokenization_source="fresh",
        tokenizer=DummyTokenizer(),
        label_pad_token_id=-100,
        logger=logging.getLogger("test_post_tokenization_non_main"),
    )

    assert "Post-tokenization train sample" not in caplog.text
    assert not (tmp_path / "output" / "preprocessing_logs" / "post_tokenization_train_samples.jsonl").exists()


def test_pre_and_post_tokenization_logs_can_coexist(tmp_path):
    processed_datasets = {"train": _processed_train_dataset()}
    data_args = DataArguments(preprocessing_log_samples=2)
    pre_training_args = SimpleNamespace(output_dir=str(tmp_path / "output"), seed=19, process_index=0)

    run_preference_utils._log_processed_train_samples(
        processed_datasets,
        data_args,
        pre_training_args,
        logging.getLogger("test_pre_and_post_pre"),
    )

    post_training_args = SimpleNamespace(
        output_dir=str(tmp_path / "output"),
        seed=19,
        process_index=0,
        post_tokenization_log_samples=2,
        post_tokenization_log_dir=None,
    )
    preprocessing_cache_utils._log_post_tokenization_samples(
        _post_tokenized_train_dataset(),
        post_training_args,
        split_name="train",
        tokenization_source="fresh",
        tokenizer=DummyTokenizer(),
        label_pad_token_id=-100,
    )

    log_dir = tmp_path / "output" / "preprocessing_logs"
    assert (log_dir / "train_samples.jsonl").exists()
    assert (log_dir / "post_tokenization_train_samples.jsonl").exists()


def test_maybe_prepare_tokenized_datasets_logs_post_tokenization_samples(tmp_path):
    trainer = DummyTokenizationTrainer()
    dataset = Dataset.from_list(
        [
            {"prompt": "Prompt 0", "chosen": " chosen-0", "rejected": " rejected-0"},
            {"prompt": "Prompt 1", "chosen": " chosen-1", "rejected": " rejected-1"},
        ]
    )
    training_args = SimpleNamespace(
        tokenization_mode="online",
        reuse_tokenized_dataset=False,
        tokenization_batch_size=2,
        dataset_num_proc=None,
        tokenized_dataset_cache_dir=None,
        post_tokenization_log_samples=2,
        post_tokenization_log_dir=None,
        output_dir=str(tmp_path / "output"),
        seed=23,
        process_index=0,
        trainer_type="margin_dpo",
    )

    tokenized_train, eval_dataset = preprocessing_cache_utils.maybe_prepare_tokenized_datasets(
        trainer,
        training_args,
        dataset,
        None,
    )

    assert eval_dataset is None
    assert trainer.tokenization_calls == 1
    assert "chosen_input_ids" in tokenized_train.column_names
    log_path = tmp_path / "output" / "preprocessing_logs" / "post_tokenization_train_samples.jsonl"
    rows = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 2
    assert {row["prompt"] for row in rows} == {"Prompt 0", "Prompt 1"}
    assert all(row["tokenization_source"] == "fresh" for row in rows)
    assert all(row["decoded_prompt"].startswith("<bos>Prompt ") for row in rows)
    assert all(row["decoded_chosen_full"].endswith("<eos>") for row in rows)
    assert all(row["prompt_length"] == len(row["prompt_input_ids"]) for row in rows)
    assert all(row["chosen_length"] == len(row["chosen_input_ids"]) for row in rows)
    assert all(row["rejected_length"] == len(row["rejected_input_ids"]) for row in rows)


def test_maybe_prepare_tokenized_datasets_logs_cache_hits_without_retokenizing(tmp_path):
    trainer = DummyTokenizationTrainer()
    dataset = Dataset.from_list(
        [{"prompt": "Prompt 0", "chosen": " chosen-0", "rejected": " rejected-0"}]
    )
    cache_dir = tmp_path / "tokenized-cache"
    training_args = SimpleNamespace(
        tokenization_mode="online",
        reuse_tokenized_dataset=True,
        tokenization_batch_size=2,
        dataset_num_proc=None,
        tokenized_dataset_cache_dir=str(cache_dir),
        post_tokenization_log_samples=1,
        post_tokenization_log_dir=None,
        output_dir=str(tmp_path / "output"),
        seed=31,
        process_index=0,
        trainer_type="margin_dpo",
    )

    preprocessing_cache_utils.maybe_prepare_tokenized_datasets(trainer, training_args, dataset, None)
    first_call_count = trainer.tokenization_calls
    assert first_call_count > 0

    preprocessing_cache_utils.maybe_prepare_tokenized_datasets(trainer, training_args, dataset, None)
    assert trainer.tokenization_calls == first_call_count

    log_path = tmp_path / "output" / "preprocessing_logs" / "post_tokenization_train_samples.jsonl"
    rows = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 1
    assert rows[0]["tokenization_source"] == "cache_hit"
