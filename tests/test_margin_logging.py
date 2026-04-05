import json
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from margin_dpo_trainer import MarginDPOTrainer


class FakeAccelerator:
    def __init__(self):
        self.sync_gradients = False

    def gather_for_metrics(self, tensor):
        return tensor


def _build_trainer(tmp_path: Path, *, margin_log_steps: int) -> MarginDPOTrainer:
    trainer = MarginDPOTrainer.__new__(MarginDPOTrainer)
    trainer.model = SimpleNamespace(training=True)
    trainer.accelerator = FakeAccelerator()
    trainer.state = SimpleNamespace(global_step=0, epoch=0.25)
    trainer.margin_log_steps = margin_log_steps
    trainer.margin_log_path = str(tmp_path)
    trainer.margin_save_full = False
    trainer._pending_margin_tensors = []
    trainer.is_world_process_zero = lambda: True
    return trainer


def _read_margin_rows(tmp_path: Path) -> list[dict]:
    path = tmp_path / "margins.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def test_margin_logging_uses_effective_batch_across_gradient_accumulation(tmp_path):
    trainer = _build_trainer(tmp_path, margin_log_steps=1)

    trainer.accelerator.sync_gradients = False
    trainer._maybe_log_margin(torch.tensor([1.0, 2.0]))
    trainer._maybe_log_margin(torch.tensor([3.0, 4.0]))

    assert _read_margin_rows(tmp_path) == []

    trainer.accelerator.sync_gradients = True
    trainer._maybe_log_margin(torch.tensor([5.0, 6.0]))

    rows = _read_margin_rows(tmp_path)
    assert len(rows) == 1
    assert rows[0]["step"] == 1
    assert rows[0]["batch_size"] == 6
    assert rows[0]["sample"] == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]


def test_margin_logging_clears_buffer_between_optimizer_steps(tmp_path):
    trainer = _build_trainer(tmp_path, margin_log_steps=2)

    trainer.accelerator.sync_gradients = False
    trainer._maybe_log_margin(torch.tensor([10.0]))
    trainer.accelerator.sync_gradients = True
    trainer._maybe_log_margin(torch.tensor([20.0]))

    assert _read_margin_rows(tmp_path) == []

    trainer.state.global_step = 1
    trainer.accelerator.sync_gradients = False
    trainer._maybe_log_margin(torch.tensor([30.0]))
    trainer.accelerator.sync_gradients = True
    trainer._maybe_log_margin(torch.tensor([40.0]))

    rows = _read_margin_rows(tmp_path)
    assert len(rows) == 1
    assert rows[0]["step"] == 2
    assert rows[0]["batch_size"] == 2
    assert rows[0]["sample"] == [30.0, 40.0]
