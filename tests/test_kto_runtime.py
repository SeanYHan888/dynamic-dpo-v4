import logging

from datasets import Dataset

from utils.runtime import _preference_dataset_to_kto


def test_preference_dataset_to_kto_doubles_rows_and_preserves_prompt_and_labels():
    dataset = Dataset.from_list(
        [
            {"prompt": "prompt-0", "chosen": "chosen-0", "rejected": "rejected-0"},
            {"prompt": "prompt-1", "chosen": "chosen-1", "rejected": "rejected-1"},
        ]
    )

    converted = _preference_dataset_to_kto(dataset)

    assert len(converted) == 4
    assert converted[0] == {"prompt": "prompt-0", "completion": "chosen-0", "label": True}
    assert converted[1] == {"prompt": "prompt-0", "completion": "rejected-0", "label": False}
    assert converted[2] == {"prompt": "prompt-1", "completion": "chosen-1", "label": True}
    assert converted[3] == {"prompt": "prompt-1", "completion": "rejected-1", "label": False}
