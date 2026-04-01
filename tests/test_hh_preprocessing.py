# coding=utf-8
import sys
from pathlib import Path

import pytest
from datasets import Dataset

from alignment import DataArguments, get_datasets
from alignment.data import maybe_convert_hh_to_openai_format, parse_hh_transcript


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from run_preference_utils import apply_preference_chat_template


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


def test_apply_preference_chat_template_accepts_raw_hh_examples():
    tokenizer = DummyTokenizer()

    formatted = apply_preference_chat_template(
        _build_hh_example(),
        tokenizer=tokenizer,
        auto_insert_empty_system_msg=True,
    )

    assert formatted["text_prompt"].startswith("<s>system: || user:Tell me a joke.")
    assert formatted["text_chosen"] == (
        "assistant:Why don't penguins fly? Because they're not tall enough to be pilots."
    )
    assert formatted["text_rejected"] == "assistant:Penguins are bad at jokes."


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
