import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PARITY_CONFIG = REPO_ROOT / "training_configs" / "llama3-8b-base" / "e-dpo" / "llama-3-8b-base-epsilon-dpo-helpful.yaml"


@pytest.mark.skipif(
    not os.getenv("EDPO_REFERENCE_PYTHON") or not os.getenv("EDPO_PARITY_MODEL"),
    reason=(
        "Set EDPO_REFERENCE_PYTHON to a Python interpreter with the original reference-codes/e-dpo stack "
        "(TRL 0.13 compatible) and EDPO_PARITY_MODEL to a shared model checkpoint to run full trainer parity."
    ),
)
def test_full_trainer_frozen_batch_parity_against_reference_stack():
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


@pytest.mark.skipif(
    not os.getenv("EDPO_REFERENCE_PYTHON") or not os.getenv("EDPO_PARITY_MODEL"),
    reason=(
        "Set EDPO_REFERENCE_PYTHON to a Python interpreter with the original reference-codes/e-dpo stack "
        "(TRL 0.13 compatible) and EDPO_PARITY_MODEL to a shared model checkpoint to run full trainer parity."
    ),
)
def test_full_trainer_short_config_trajectory_parity_against_reference_stack():
    config_path = os.getenv("EDPO_PARITY_CONFIG", str(DEFAULT_PARITY_CONFIG))

    subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "tools" / "epsilon_dpo_parity.py"),
            "--mode",
            "compare_trajectory",
            "--reference-python",
            os.environ["EDPO_REFERENCE_PYTHON"],
            "--config-path",
            config_path,
            "--model-name-or-path",
            os.environ["EDPO_PARITY_MODEL"],
            "--max-examples",
            "4",
            "--train-batch-size",
            "2",
            "--gradient-accumulation-steps",
            "1",
            "--trajectory-optimizer-steps",
            "2",
            "--learning-rate",
            "1e-6",
        ],
        check=True,
    )
