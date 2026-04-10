from pathlib import Path
import sys

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from slic_hf_trainer import SLiCHFTrainer
from tokenized_dpo_trainer import TokenizedDPOTrainer


class _FakeAccelerator:
    def gather_for_metrics(self, tensor):
        return tensor


def test_slic_hf_loss_has_zero_rank_term_when_margin_is_met():
    trainer = object.__new__(SLiCHFTrainer)
    trainer.slic_margin = 1.0
    trainer.slic_lambda = 0.5

    chosen_logps = torch.tensor([-3.0], dtype=torch.float32)
    rejected_logps = torch.tensor([-4.5], dtype=torch.float32)

    losses, rank_losses, ce_losses = trainer.slic_hf_loss(chosen_logps, rejected_logps)

    assert torch.allclose(rank_losses, torch.tensor([0.0]))
    assert torch.allclose(ce_losses, torch.tensor([1.5]))
    assert torch.allclose(losses, torch.tensor([1.5]))


def test_slic_hf_loss_adds_positive_hinge_when_margin_is_missed():
    trainer = object.__new__(SLiCHFTrainer)
    trainer.slic_margin = 1.0
    trainer.slic_lambda = 0.25

    chosen_logps = torch.tensor([-2.0], dtype=torch.float32)
    rejected_logps = torch.tensor([-2.4], dtype=torch.float32)

    losses, rank_losses, ce_losses = trainer.slic_hf_loss(chosen_logps, rejected_logps)

    assert torch.allclose(rank_losses, torch.tensor([0.6]))
    assert torch.allclose(ce_losses, torch.tensor([0.5]))
    assert torch.allclose(losses, torch.tensor([1.1]))


def test_slic_hf_get_batch_loss_metrics_uses_summed_logps_not_average(monkeypatch):
    trainer = object.__new__(SLiCHFTrainer)
    trainer.slic_margin = 0.5
    trainer.slic_lambda = 0.1
    trainer.accelerator = _FakeAccelerator()

    def _fake_concatenated_forward(model, batch, average_log_prob, logit_source):
        del model, batch, logit_source
        assert average_log_prob is False
        chosen_logits = torch.ones((2, 3, 5), dtype=torch.float32)
        rejected_logits = torch.zeros((2, 3, 5), dtype=torch.float32)
        return (
            torch.tensor([-4.0, -3.0], dtype=torch.float32),
            torch.tensor([-5.0, -4.0], dtype=torch.float32),
            chosen_logits,
            rejected_logits,
            torch.zeros((2, 3), dtype=torch.long),
        )

    trainer.concatenated_forward = _fake_concatenated_forward

    loss, metrics = trainer.get_batch_loss_metrics(model=None, batch={}, train_eval="train")

    expected_losses = torch.tensor([0.4, 0.3], dtype=torch.float32)
    assert torch.allclose(loss, expected_losses.mean())
    assert torch.allclose(metrics["slic/ce_loss"], torch.tensor(0.35))


def test_completion_only_masking_and_logp_sums_follow_tokenized_dpo_convention():
    logits = torch.tensor(
        [
            [
                [2.0, 0.0, 0.0],
                [0.0, 3.0, 0.0],
                [0.0, 0.0, 4.0],
                [1.0, 1.0, 1.0],
            ]
        ],
        dtype=torch.float32,
    )
    labels = torch.tensor([[-100, -100, 1, 2]], dtype=torch.long)

    summed = TokenizedDPOTrainer.get_batch_logps(
        logits,
        labels,
        average_log_prob=False,
        is_encoder_decoder=False,
    )
    averaged = TokenizedDPOTrainer.get_batch_logps(
        logits,
        labels,
        average_log_prob=True,
        is_encoder_decoder=False,
    )

    shifted_logits = logits[:, :-1, :]
    shifted_labels = labels[:, 1:]
    loss_mask = shifted_labels != -100
    safe_labels = shifted_labels.masked_fill(~loss_mask, 0)
    manual = torch.gather(shifted_logits.log_softmax(-1), 2, safe_labels.unsqueeze(2)).squeeze(2)
    expected_sum = (manual * loss_mask).sum(-1)

    assert torch.allclose(summed, expected_sum)
    assert torch.allclose(averaged, expected_sum / loss_mask.sum(-1))
