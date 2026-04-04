import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.skipif(
    not os.getenv("EDPO_REFERENCE_PYTHON") or not os.getenv("EDPO_PARITY_MODEL"),
    reason=(
        "Set EDPO_REFERENCE_PYTHON to a Python interpreter with the original reference-codes/e-dpo stack "
        "(TRL 0.13 compatible) and EDPO_PARITY_MODEL to a shared model checkpoint to run parity."
    ),
)
def test_frozen_batch_parity_against_reference_stack():
    examples = [
        {
            "prompt": "User: name a prime number.\nAssistant:",
            "chosen": " 13",
            "rejected": " 12",
        },
        {
            "prompt": "User: reply with yes or no.\nAssistant:",
            "chosen": " yes",
            "rejected": " maybe",
        },
    ]

    subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "tools" / "epsilon_dpo_parity.py"),
            "--mode",
            "compare",
            "--reference-python",
            os.environ["EDPO_REFERENCE_PYTHON"],
            "--model-name-or-path",
            os.environ["EDPO_PARITY_MODEL"],
            "--examples-json",
            json.dumps(examples),
            "--beta",
            "0.1",
            "--epsilon",
            "0.01",
            "--gradient-accumulation-steps",
            "2",
        ],
        check=True,
    )
