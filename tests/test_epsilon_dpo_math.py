from pathlib import Path
from types import SimpleNamespace
import sys

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


def test_score_sequence_logits_ignores_masked_tokens_and_shifts_decoder_only():
    logits = torch.tensor(
        [
            [
                [2.0, 0.0],
                [0.0, 3.0],
                [4.0, 1.0],
            ]
        ],
        dtype=torch.float32,
    )
    labels = torch.tensor([[99, 1, -100]], dtype=torch.long)

    scored = EpsilonDPOTrainer._score_sequence_logits(
        logits,
        labels,
        label_pad_token_id=-100,
        is_encoder_decoder=False,
    )

    shifted_logits = logits[:, :-1, :]
    shifted_labels = labels[:, 1:]
    loss_mask = shifted_labels != -100
    safe_labels = shifted_labels.masked_fill(~loss_mask, 0)
    manual = torch.gather(shifted_logits.log_softmax(-1), 2, safe_labels.unsqueeze(2)).squeeze(2)
    expected = (manual * loss_mask).sum(-1)

    assert torch.equal(scored["safe_labels"], safe_labels)
    assert torch.equal(scored["loss_mask"], loss_mask)
    assert torch.allclose(scored["sequence_logps"], expected)


def test_policy_and_reference_sequence_scoring_use_identical_masked_tokens():
    labels = torch.tensor([[1, -100, 4], [3, 2, -100]], dtype=torch.long)
    policy_logits = torch.randn(2, 3, 5, dtype=torch.float32)
    reference_logits = torch.randn(2, 3, 5, dtype=torch.float32)

    policy_scored = EpsilonDPOTrainer._score_sequence_logits(policy_logits, labels, is_encoder_decoder=True)
    reference_scored = EpsilonDPOTrainer._score_sequence_logits(reference_logits, labels, is_encoder_decoder=True)

    assert torch.equal(policy_scored["safe_labels"], reference_scored["safe_labels"])
    assert torch.equal(policy_scored["loss_mask"], reference_scored["loss_mask"])


def test_compute_steps_matches_frozen_perturbed_logit_example():
    epsilon = 0.1
    policy_logits = torch.tensor(
        [
            [[1.0, 0.0]],
            [[1.0, 0.0]],
            [[1.0, 0.0]],
            [[1.0, 0.0]],
        ],
        dtype=torch.float32,
    )
    reference_logits = torch.tensor(
        [
            [[0.0, 0.0]],
            [[2.0, 0.0]],
            [[2.0, 0.0]],
            [[0.0, 0.0]],
        ],
        dtype=torch.float32,
    )
    safe_labels = torch.zeros((4, 1), dtype=torch.long)
    loss_mask = torch.ones((4, 1), dtype=torch.bool)

    logps = EpsilonDPOTrainer._score_aligned_logits(policy_logits, safe_labels, loss_mask)
    p_epsilon_logits = ((1 + epsilon) * policy_logits) - (epsilon * reference_logits)
    n_epsilon_logits = ((1 - epsilon) * policy_logits) + (epsilon * reference_logits)
    p_epsilon_logps = EpsilonDPOTrainer._score_aligned_logits(p_epsilon_logits, safe_labels, loss_mask)
    n_epsilon_logps = EpsilonDPOTrainer._score_aligned_logits(n_epsilon_logits, safe_labels, loss_mask)

    logratios = logps[:2] - logps[2:]
    p_epsilon_logratios = p_epsilon_logps[:2] - p_epsilon_logps[2:]
    n_epsilon_logratios = n_epsilon_logps[:2] - n_epsilon_logps[2:]
    steps = EpsilonDPOTrainer._compute_steps(logratios, p_epsilon_logratios, n_epsilon_logratios)

    assert torch.equal(steps, torch.tensor([1, -1], dtype=torch.long))


def test_dpo_loss_matches_original_updated_beta_formula():
    trainer = object.__new__(EpsilonDPOTrainer)
    trainer.beta = 0.3
    trainer.epsilon = 0.2
    trainer.label_smoothing = 0.1

    chosen_logps = torch.tensor([1.5, 0.9], dtype=torch.float32)
    rejected_logps = torch.tensor([0.2, 0.4], dtype=torch.float32)
    ref_chosen_logps = torch.tensor([0.7, 0.5], dtype=torch.float32)
    ref_rejected_logps = torch.tensor([0.1, 0.1], dtype=torch.float32)
    steps = torch.tensor([1, -1], dtype=torch.long)

    losses, chosen_rewards, rejected_rewards = trainer.dpo_loss(
        chosen_logps,
        rejected_logps,
        ref_chosen_logps,
        ref_rejected_logps,
        steps,
    )

    logratios = chosen_logps - rejected_logps
    ref_logratios = ref_chosen_logps - ref_rejected_logps
    logits = logratios - ref_logratios
    updated_beta = trainer.beta / (1 + trainer.epsilon * steps.to(torch.float32))
    expected_losses = (
        -torch.nn.functional.logsigmoid(updated_beta * logits) * (1 - trainer.label_smoothing)
        - torch.nn.functional.logsigmoid(updated_beta * logits) * trainer.label_smoothing
    )
    expected_chosen_rewards = updated_beta * (chosen_logps - ref_chosen_logps)
    expected_rejected_rewards = updated_beta * (rejected_logps - ref_rejected_logps)

    assert torch.allclose(losses, expected_losses)
    assert torch.allclose(chosen_rewards, expected_chosen_rewards)
    assert torch.allclose(rejected_rewards, expected_rejected_rewards)


def test_beta_updates_only_on_gradient_sync_boundary():
    trainer = object.__new__(EpsilonDPOTrainer)
    trainer.args = SimpleNamespace(gradient_accumulation_steps=2)
    trainer.accelerator = _FakeAccelerator(sync_gradients=False)
    trainer.beta = 0.5
    trainer.epsilon = 0.1
    trainer._pending_steps = torch.zeros(())

    trainer._accumulate_microbatch_steps(torch.tensor([1, 1], dtype=torch.long))
    beta_before_sync = trainer.beta
    pending_before_sync = trainer._pending_steps.item()

    assert beta_before_sync == 0.5
    assert pending_before_sync == 0.5

    trainer._accumulate_microbatch_steps(torch.tensor([1, 1], dtype=torch.long))
    trainer.accelerator = _FakeAccelerator(sync_gradients=True)
    metrics = {}
    trainer._apply_pending_beta_update(metrics, prefix="")

    assert metrics["kl/beta"] == 0.5
    assert torch.isclose(metrics["kl/avg_steps"], torch.tensor(1.0))
    assert trainer.beta == 0.5 / 1.1
    assert trainer._pending_steps.item() == 0.0
