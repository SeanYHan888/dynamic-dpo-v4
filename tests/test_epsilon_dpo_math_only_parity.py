import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from epsilon_dpo_trainer import EpsilonDPOTrainer


class _FakeAccelerator:
    def __init__(self, sync_gradients: bool):
        self.device = torch.device("cpu")
        self.gradient_state = SimpleNamespace(sync_gradients=sync_gradients)

    def gather(self, tensor):
        return tensor

    def gather_for_metrics(self, tensor):
        return tensor


def _math_inputs():
    return {
        "beta": 0.3,
        "epsilon": 0.2,
        "label_smoothing": 0.1,
        "chosen_logps": [1.5, 0.9],
        "rejected_logps": [0.2, 0.4],
        "ref_chosen_logps": [0.7, 0.5],
        "ref_rejected_logps": [0.1, 0.1],
        "steps": [1, -1],
        "gradient_accumulation_steps": 2,
        "microbatch_steps": [[1, 1], [1, 1]],
    }


def _current_math_payload():
    inputs = _math_inputs()
    trainer = object.__new__(EpsilonDPOTrainer)
    trainer.beta = inputs["beta"]
    trainer.epsilon = inputs["epsilon"]
    trainer.label_smoothing = inputs["label_smoothing"]

    losses, chosen_rewards, rejected_rewards = trainer.dpo_loss(
        torch.tensor(inputs["chosen_logps"], dtype=torch.float32),
        torch.tensor(inputs["rejected_logps"], dtype=torch.float32),
        torch.tensor(inputs["ref_chosen_logps"], dtype=torch.float32),
        torch.tensor(inputs["ref_rejected_logps"], dtype=torch.float32),
        torch.tensor(inputs["steps"], dtype=torch.long),
    )

    trainer.args = SimpleNamespace(gradient_accumulation_steps=inputs["gradient_accumulation_steps"])
    trainer.accelerator = _FakeAccelerator(sync_gradients=False)
    trainer._pending_steps = torch.zeros(())
    trainer.beta = inputs["beta"]
    for microbatch_steps in inputs["microbatch_steps"]:
        trainer._accumulate_microbatch_steps(torch.tensor(microbatch_steps, dtype=torch.long))
    trainer.accelerator = _FakeAccelerator(sync_gradients=True)
    metrics = {}
    trainer._apply_pending_beta_update(metrics, prefix="")

    return {
        "losses": losses.detach().cpu().tolist(),
        "chosen_rewards": chosen_rewards.detach().cpu().tolist(),
        "rejected_rewards": rejected_rewards.detach().cpu().tolist(),
        "beta_before_boundary": inputs["beta"],
        "beta_after_boundary": float(trainer.beta),
        "avg_steps": float(metrics["kl/avg_steps"].item()),
    }


def test_math_only_current_matches_closed_form():
    payload = _current_math_payload()
    inputs = _math_inputs()

    chosen_logps = torch.tensor(inputs["chosen_logps"], dtype=torch.float32)
    rejected_logps = torch.tensor(inputs["rejected_logps"], dtype=torch.float32)
    ref_chosen_logps = torch.tensor(inputs["ref_chosen_logps"], dtype=torch.float32)
    ref_rejected_logps = torch.tensor(inputs["ref_rejected_logps"], dtype=torch.float32)
    steps = torch.tensor(inputs["steps"], dtype=torch.float32)

    logits = (chosen_logps - rejected_logps) - (ref_chosen_logps - ref_rejected_logps)
    updated_beta = inputs["beta"] / (1 + inputs["epsilon"] * steps)
    expected_losses = (
        -torch.nn.functional.logsigmoid(updated_beta * logits) * (1 - inputs["label_smoothing"])
        - torch.nn.functional.logsigmoid(updated_beta * logits) * inputs["label_smoothing"]
    )
    expected_chosen_rewards = updated_beta * (chosen_logps - ref_chosen_logps)
    expected_rejected_rewards = updated_beta * (rejected_logps - ref_rejected_logps)

    assert torch.allclose(torch.tensor(payload["losses"]), expected_losses)
    assert torch.allclose(torch.tensor(payload["chosen_rewards"]), expected_chosen_rewards)
    assert torch.allclose(torch.tensor(payload["rejected_rewards"]), expected_rejected_rewards)
    assert payload["avg_steps"] == 1.0
    assert payload["beta_after_boundary"] == inputs["beta"] / 1.2


@pytest.mark.skipif(
    not os.getenv("EDPO_REFERENCE_PYTHON"),
    reason="Set EDPO_REFERENCE_PYTHON to a Python interpreter with the original reference-codes/e-dpo stack.",
)
def test_math_only_cross_stack_loss_and_beta_parity():
    current_payload = _current_math_payload()
    inputs = _math_inputs()

    reference_script = f"""
import json
from pathlib import Path
from types import SimpleNamespace
import sys
import torch

repo_root = Path({str(REPO_ROOT)!r})
sys.path.insert(0, str(repo_root / "reference-codes" / "e-dpo"))
from trainer import EpsilonDPOTrainer

class _FakeAccelerator:
    def __init__(self, sync_gradients: bool):
        self.device = torch.device("cpu")
        self.gradient_state = SimpleNamespace(sync_gradients=sync_gradients)
    def gather(self, tensor):
        return tensor
    def gather_for_metrics(self, tensor):
        return tensor

inputs = json.loads({json.dumps(json.dumps(inputs))!r})
trainer = object.__new__(EpsilonDPOTrainer)
trainer.beta = inputs["beta"]
trainer.epsilon = inputs["epsilon"]
trainer.label_smoothing = inputs["label_smoothing"]

losses, chosen_rewards, rejected_rewards = trainer.dpo_loss(
    torch.tensor(inputs["chosen_logps"], dtype=torch.float32),
    torch.tensor(inputs["rejected_logps"], dtype=torch.float32),
    torch.tensor(inputs["ref_chosen_logps"], dtype=torch.float32),
    torch.tensor(inputs["ref_rejected_logps"], dtype=torch.float32),
    torch.tensor(inputs["steps"], dtype=torch.long),
)

trainer.args = SimpleNamespace(gradient_accumulation_steps=inputs["gradient_accumulation_steps"])
trainer.accelerator = _FakeAccelerator(sync_gradients=False)
trainer.steps = 0.0
for microbatch_steps in inputs["microbatch_steps"]:
    trainer.steps += torch.tensor(microbatch_steps, dtype=torch.float32).mean() / inputs["gradient_accumulation_steps"]
trainer.accelerator = _FakeAccelerator(sync_gradients=True)
mean_steps = trainer.accelerator.gather(trainer.steps).mean()
beta_after_boundary = trainer.beta / (1 + mean_steps * trainer.epsilon)

print(json.dumps({{
    "losses": losses.detach().cpu().tolist(),
    "chosen_rewards": chosen_rewards.detach().cpu().tolist(),
    "rejected_rewards": rejected_rewards.detach().cpu().tolist(),
    "beta_before_boundary": inputs["beta"],
    "beta_after_boundary": float(beta_after_boundary),
    "avg_steps": float(mean_steps.item()),
}}))
"""

    result = subprocess.run(
        [os.environ["EDPO_REFERENCE_PYTHON"], "-c", reference_script],
        check=True,
        capture_output=True,
        text=True,
    )
    reference_payload = json.loads(result.stdout)

    for key in ("losses", "chosen_rewards", "rejected_rewards"):
        assert current_payload[key] == pytest.approx(reference_payload[key], abs=1e-6, rel=1e-6)
    for key in ("beta_before_boundary", "beta_after_boundary", "avg_steps"):
        assert current_payload[key] == pytest.approx(reference_payload[key], abs=1e-6, rel=1e-6)
