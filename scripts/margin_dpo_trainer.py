import json
import os
from typing import Dict, Literal, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from tokenized_dpo_trainer import TokenizedDPOTrainer
from trainer_configs import MarginDPOConfig


def log_margin(
    margin: torch.Tensor,
    log_dir: str,
    epoch: float,
    step: int,
    save_full: bool = False,
) -> None:
    os.makedirs(log_dir, exist_ok=True)

    margin_np = margin.detach().float().cpu().numpy()
    p10, p50, p90 = np.percentile(margin_np, [10, 50, 90])

    record = {
        "epoch": float(epoch),
        "step": int(step),
        "mean": float(margin_np.mean()),
        "std": float(margin_np.std(ddof=0)),
        "min": float(margin_np.min()),
        "p10": float(p10),
        "median": float(p50),
        "p90": float(p90),
        "max": float(margin_np.max()),
        "pos_frac": float((margin_np > 0).mean()),
        "sample": [float(x) for x in margin_np[:]],
    }

    if save_full:
        npy_path = os.path.join(log_dir, f"step_{step:07d}.npy")
        np.save(npy_path, margin_np)
        record["npy"] = npy_path

    jsonl_path = os.path.join(log_dir, "margins.jsonl")
    with open(jsonl_path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


class MarginDPOTrainer(TokenizedDPOTrainer):
    _tag_names = ["trl", "margin-dpo"]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if not isinstance(self.args, MarginDPOConfig):
            raise TypeError("MarginDPOTrainer requires args=MarginDPOConfig")

        self.margin_log_path = self.args.margin_log_path
        self.margin_log_steps = int(self.args.margin_log_steps)
        self.margin_save_full = bool(self.args.margin_save_full)
        self.require_explicit_ref_model = bool(self.args.require_explicit_ref_model)
        self.f_divergence_type = self.args.f_divergence_type
        self.f_alpha_divergence_coef = float(self.args.f_alpha_divergence_coef)
        os.makedirs(self.margin_log_path, exist_ok=True)

    def _get_reference_logps(
        self,
        batch: Dict[str, Union[torch.Tensor, str]],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if getattr(self, "precompute_ref_logps", False):
            if "ref_chosen_logps" not in batch or "ref_rejected_logps" not in batch:
                raise KeyError(
                    "precompute_ref_logps=True requires both 'ref_chosen_logps' and 'ref_rejected_logps' in the batch."
                )
            if (
                not torch.isfinite(batch["ref_chosen_logps"]).all()
                or not torch.isfinite(batch["ref_rejected_logps"]).all()
            ):
                raise ValueError(
                    "Encountered non-finite precomputed reference log-probs in the training batch."
                )
            return batch["ref_chosen_logps"], batch["ref_rejected_logps"]

        if self.ref_model is None and self.require_explicit_ref_model:
            raise ValueError(
                "This trainer is configured to require an explicit ref_model, but self.ref_model is None."
            )
        if self.ref_model is None:
            raise ValueError(
                "ref_model is None. Pass an explicit frozen reference model or enable precomputed reference log-probs."
            )

        with torch.no_grad():
            ref_chosen_logps, ref_rejected_logps, _, _, _ = self.concatenated_forward(
                self.ref_model,
                batch,
                average_log_prob=False,
                logit_source="reference",
            )
        return ref_chosen_logps, ref_rejected_logps

    def _stable_alpha_scores(
        self, logratios: torch.Tensor, alpha: float, coef: float
    ) -> torch.Tensor:
        t = (alpha - 1.0) * logratios
        clamp_max_map = {
            torch.float16: 11.0,
            torch.bfloat16: 80.0,
            torch.float32: 80.0,
            torch.float64: 80.0,
        }
        clamp_max = clamp_max_map.get(t.dtype, 80.0)
        return torch.exp(torch.clamp(t.float(), max=clamp_max)).to(t.dtype) * coef

    def _project_scores(
        self,
        chosen_logratios: torch.Tensor,
        rejected_logratios: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.f_divergence_type == "reverse_kl":
            return chosen_logratios, rejected_logratios
        if self.f_divergence_type == "forward_kl":
            return -torch.exp(-chosen_logratios), -torch.exp(-rejected_logratios)
        if self.f_divergence_type == "js_divergence":
            return F.logsigmoid(chosen_logratios), F.logsigmoid(rejected_logratios)
        if self.f_divergence_type == "alpha_divergence":
            alpha = float(self.f_alpha_divergence_coef)
            if abs(alpha - 1.0) < 1e-6:
                return chosen_logratios, rejected_logratios
            coef = 1.0 / (alpha - 1.0)
            return (
                self._stable_alpha_scores(chosen_logratios, alpha, coef),
                self._stable_alpha_scores(rejected_logratios, alpha, coef),
            )
        raise ValueError(f"Unknown f_divergence_type: {self.f_divergence_type}")

    def _maybe_log_margin(self, margin_tensor: torch.Tensor) -> None:
        if not self.model.training:
            return
        if not self.accelerator.sync_gradients:
            return

        step = int(self.state.global_step)
        if self.margin_log_steps <= 0 or step % self.margin_log_steps != 0:
            return

        margin_all = self.accelerator.gather_for_metrics(margin_tensor.detach())
        if not self.is_world_process_zero():
            return

        epoch = self.state.epoch if self.state.epoch is not None else 0.0
        log_margin(
            margin=margin_all,
            log_dir=self.margin_log_path,
            epoch=float(epoch),
            step=step,
            save_full=self.margin_save_full,
        )

    def get_batch_loss_metrics(
        self,
        model,
        batch: Dict[str, Union[torch.Tensor, str]],
        train_eval: Literal["train", "eval"] = "train",
    ):
        prefix = "eval_" if train_eval == "eval" else ""
        metrics = {}

        chosen_logps, rejected_logps, chosen_logits, rejected_logits, _ = (
            self.concatenated_forward(
                model,
                batch,
                average_log_prob=False,
                logit_source="policy",
            )
        )
        ref_chosen_logps, ref_rejected_logps = self._get_reference_logps(batch)
        ref_chosen_logps = ref_chosen_logps.to(chosen_logps.device)
        ref_rejected_logps = ref_rejected_logps.to(chosen_logps.device)

        margin = (chosen_logps - rejected_logps) - (
            ref_chosen_logps - ref_rejected_logps
        )
        self._maybe_log_margin(margin.detach())

        chosen_logratios = chosen_logps - ref_chosen_logps
        rejected_logratios = rejected_logps - ref_rejected_logps
        chosen_scores, rejected_scores = self._project_scores(
            chosen_logratios, rejected_logratios
        )

        delta_score = chosen_scores - rejected_scores
        per_seq_loss = -F.logsigmoid(float(self.beta) * delta_score)
        loss = per_seq_loss.mean()

        metrics[f"{prefix}margin_dpo/margin_mean"] = margin.detach().mean().cpu()
        metrics[f"{prefix}margin_dpo/margin_std"] = (
            margin.detach().std(unbiased=False).cpu()
        )
        metrics[f"{prefix}logps/chosen"] = chosen_logps.detach().mean().cpu()
        metrics[f"{prefix}logps/rejected"] = rejected_logps.detach().mean().cpu()
        metrics[f"{prefix}logps/ref_chosen"] = ref_chosen_logps.detach().mean().cpu()
        metrics[f"{prefix}logps/ref_rejected"] = (
            ref_rejected_logps.detach().mean().cpu()
        )
        metrics["eval_logits/chosen" if prefix else "logits/chosen"] = (
            chosen_logits.detach().mean().cpu()
        )
        metrics["eval_logits/rejected" if prefix else "logits/rejected"] = (
            rejected_logits.detach().mean().cpu()
        )
        return loss, metrics
