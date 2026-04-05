# coding=utf-8
import pytest
from datasets import Dataset

from alignment import DataArguments, get_datasets
from alignment.data import (
    apply_chat_template,
    get_hh_preference_validation_error,
    maybe_convert_hh_to_openai_format,
    maybe_normalize_preference_dataset,
    normalize_raw_hh_preference_dataset,
    parse_hh_transcript,
)
from utils.runtime import apply_preference_chat_template


class DummyTokenizer:
    def __init__(self):
        self.chat_template = "system"
        self.default_chat_template = None
        self.bos_token = "<s>"

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        del tokenize, add_generation_prompt
        rendered = " || ".join(f"{message['role']}:{message['content']}" for message in messages)
        return f"{self.bos_token}{rendered}"


def _build_hh_example():
    return {
        "chosen": (
            "Human: Tell me a joke.\n\n"
            "Assistant: Sure.\n\n"
            "Human: Make it about penguins.\n\n"
            "Assistant: Why don't penguins fly? Because they're not tall enough to be pilots."
        ),
        "rejected": (
            "Human: Tell me a joke.\n\n"
            "Assistant: Sure.\n\n"
            "Human: Make it about penguins.\n\n"
            "Assistant: Penguins are bad at jokes."
        ),
    }


def test_parse_hh_transcript_preserves_multiline_content():
    transcript = (
        "Human: Explain merge sort.\n\n"
        "Assistant: It is a divide-and-conquer algorithm.\n\n"
        "Human: Include complexity.\n\n"
        "Please be concise.\n\n"
        "Assistant: It runs in O(n log n) time.\n\n"
        "It usually needs extra memory."
    )

    messages = parse_hh_transcript(transcript)

    assert messages == [
        {"role": "user", "content": "Explain merge sort."},
        {"role": "assistant", "content": "It is a divide-and-conquer algorithm."},
        {"role": "user", "content": "Include complexity.\n\nPlease be concise."},
        {"role": "assistant", "content": "It runs in O(n log n) time.\n\nIt usually needs extra memory."},
    ]


def test_convert_hh_example_to_openai_preference_format():
    converted = maybe_convert_hh_to_openai_format(_build_hh_example())

    assert converted["prompt"] == [
        {"role": "user", "content": "Tell me a joke."},
        {"role": "assistant", "content": "Sure."},
        {"role": "user", "content": "Make it about penguins."},
    ]
    assert converted["chosen"] == [
        {"role": "assistant", "content": "Why don't penguins fly? Because they're not tall enough to be pilots."}
    ]
    assert converted["rejected"] == [
        {"role": "assistant", "content": "Penguins are bad at jokes."}
    ]


def test_non_hh_examples_are_left_unchanged():
    example = {
        "prompt": [{"role": "user", "content": "Hi"}],
        "chosen": [{"role": "assistant", "content": "Hello"}],
        "rejected": [{"role": "assistant", "content": "No"}],
    }

    assert maybe_convert_hh_to_openai_format(example) is example
    assert example["chosen"] == [{"role": "assistant", "content": "Hello"}]


def test_invalid_hh_examples_raise_clear_error():
    example = {
        "chosen": "Human: Hi\n\nAssistant: Final",
        "rejected": "Human: Different opener\n\nAssistant: Other",
    }

    with pytest.raises(ValueError, match="common prefix"):
        maybe_convert_hh_to_openai_format(example)


def test_hh_validation_reports_missing_divergent_assistant_response():
    example = {
        "chosen": "Human: Can you provide private info?\n\nAssistant: ",
        "rejected": "Human: Can you provide private info?\n\nAssistant: No.",
    }

    assert (
        get_hh_preference_validation_error(example)
        == "HH chosen/rejected transcripts must each contain a divergent assistant response."
    )


def test_hh_validation_reports_extra_turns_after_compared_response():
    example = {
        "chosen": (
            "Human: Ask me something.\n\n"
            "Assistant: Okay.\n\n"
            "Human: Keep going.\n\n"
            "Assistant: Final answer."
        ),
        "rejected": (
            "Human: Ask me something.\n\n"
            "Assistant: Okay.\n\n"
            "Human: Keep going.\n\n"
            "Assistant: Different answer.\n\n"
            "Human: One more thing.\n\n"
            "Assistant: Extra turn."
        ),
    }

    assert (
        get_hh_preference_validation_error(example)
        == "HH preprocessing expects exactly one final assistant response in chosen/rejected suffixes."
    )


def test_apply_preference_chat_template_accepts_raw_hh_examples():
    tokenizer = DummyTokenizer()

    normalized = maybe_convert_hh_to_openai_format(_build_hh_example())
    formatted = apply_preference_chat_template(
        normalized,
        tokenizer=tokenizer,
        auto_insert_empty_system_msg=True,
    )

    assert formatted["text_prompt"].startswith("<s>system: || user:Tell me a joke.")
    assert formatted["text_chosen"] == (
        "assistant:Why don't penguins fly? Because they're not tall enough to be pilots."
    )
    assert formatted["text_rejected"] == "assistant:Penguins are bad at jokes."
    assert list(formatted.keys()) == ["text_prompt", "text_chosen", "text_rejected"]


def test_apply_chat_template_accepts_raw_hh_examples_for_sft():
    tokenizer = DummyTokenizer()

    formatted = apply_chat_template(
        _build_hh_example(),
        tokenizer=tokenizer,
        task="sft",
        auto_insert_empty_system_msg=True,
    )

    assert formatted["text"].startswith("<s>system: || user:Tell me a joke.")
    assert "assistant:Why don't penguins fly? Because they're not tall enough to be pilots." in formatted["text"]
    assert "assistant:Penguins are bad at jokes." not in formatted["text"]


def test_get_datasets_uses_data_dir_for_anthropic_hh(monkeypatch):
    calls = []

    def fake_load_dataset(dataset_name, *args, split=None, **kwargs):
        calls.append((dataset_name, args, split, kwargs))
        return Dataset.from_dict(
            {
                "chosen": ["Human: Hi\n\nAssistant: Hello\n\nHuman: Why?\n\nAssistant: Because."],
                "rejected": ["Human: Hi\n\nAssistant: Hello\n\nHuman: Why?\n\nAssistant: No idea."],
                "unused": [1],
            }
        )

    monkeypatch.setattr("alignment.data.load_dataset", fake_load_dataset)

    data_args = DataArguments(dataset_mixer={"Anthropic/hh-rlhf": 1.0})
    datasets = get_datasets(
        data_args,
        splits=["train", "test"],
        configs=["helpful-base"],
        columns_to_keep=["chosen", "rejected"],
        shuffle=False,
    )

    assert list(datasets.keys()) == ["train", "test"]
    assert all(dataset_name == "Anthropic/hh-rlhf" for dataset_name, _, _, _ in calls)
    assert all(args == () for _, args, _, _ in calls)
    assert all(kwargs["data_dir"] == "helpful-base" for _, _, _, kwargs in calls)
    train_row = datasets["train"][0]
    assert train_row["prompt"] == [
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hello"},
        {"role": "user", "content": "Why?"},
    ]
    assert train_row["chosen"] == [{"role": "assistant", "content": "Because."}]
    assert train_row["rejected"] == [{"role": "assistant", "content": "No idea."}]


def test_get_datasets_keeps_existing_non_hh_config_behavior(monkeypatch):
    calls = []

    def fake_load_dataset(dataset_name, *args, split=None, **kwargs):
        calls.append((dataset_name, args, split, kwargs))
        return Dataset.from_dict({"chosen": [[{"role": "assistant", "content": "ok"}]], "rejected": [[{"role": "assistant", "content": "no"}]]})

    monkeypatch.setattr("alignment.data.load_dataset", fake_load_dataset)

    data_args = DataArguments(dataset_mixer={"some/dataset": 1.0})
    get_datasets(
        data_args,
        splits=["train"],
        configs=["config-name"],
        columns_to_keep=["chosen", "rejected"],
        shuffle=False,
    )

    assert calls == [("some/dataset", ("config-name",), "train", {})]


def test_normalize_raw_hh_preference_dataset_filters_only_invalid_rows(caplog):
    caplog.set_level("WARNING")
    dataset = Dataset.from_dict(
        {
            "chosen": [
                (
                    "Human: Tell me a joke.\n\n"
                    "Assistant: Sure.\n\n"
                    "Human: Make it about penguins.\n\n"
                    "Assistant: Why don't penguins fly? Because they're not tall enough to be pilots."
                ),
                "Human: Can you provide private info?\n\nAssistant: ",
                (
                    "Human: Ask me something.\n\n"
                    "Assistant: Okay.\n\n"
                    "Human: Keep going.\n\n"
                    "Assistant: Final answer."
                ),
            ],
            "rejected": [
                (
                    "Human: Tell me a joke.\n\n"
                    "Assistant: Sure.\n\n"
                    "Human: Make it about penguins.\n\n"
                    "Assistant: Penguins are bad at jokes."
                ),
                "Human: Can you provide private info?\n\nAssistant: No.",
                (
                    "Human: Ask me something.\n\n"
                    "Assistant: Okay.\n\n"
                    "Human: Keep going.\n\n"
                    "Assistant: Different answer.\n\n"
                    "Human: One more thing.\n\n"
                    "Assistant: Extra turn."
                ),
            ],
        }
    )

    normalized = normalize_raw_hh_preference_dataset(dataset, split_name="train")

    assert len(normalized) == 1
    assert normalized[0]["prompt"][-1] == {"role": "user", "content": "Make it about penguins."}
    assert normalized[0]["chosen"] == [
        {"role": "assistant", "content": "Why don't penguins fly? Because they're not tall enough to be pilots."}
    ]
    assert "Dropped 2 non-canonical HH preference examples" in caplog.text


def test_maybe_normalize_preference_dataset_leaves_non_hh_data_unchanged():
    dataset = Dataset.from_dict(
        {
            "prompt": [[{"role": "user", "content": "Hi"}]],
            "chosen": [[{"role": "assistant", "content": "Hello"}]],
            "rejected": [[{"role": "assistant", "content": "No"}]],
        }
    )

    normalized = maybe_normalize_preference_dataset(dataset, dataset_name="some/dataset", split_name="train")

    assert normalized.column_names == dataset.column_names
    assert normalized[0] == dataset[0]
