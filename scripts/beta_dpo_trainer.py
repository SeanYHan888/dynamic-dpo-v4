from typing import Dict, Literal, Tuple, Union

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from tokenized_dpo_trainer import TokenizedDPOTrainer
from trainer_configs import BetaDPOConfig


class BetaDPOTrainer(TokenizedDPOTrainer):
    _tag_names = ["trl", "beta-dpo"]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if not isinstance(self.args, BetaDPOConfig):
            raise TypeError("BetaDPOTrainer requires args=BetaDPOConfig")

        device = self.accelerator.device
        self.r_gap_mean = torch.zeros((), device=device)
        self.r_gap_std = torch.ones((), device=device)
        self._gap_std_eps = torch.tensor(1e-6, device=device)

    def _is_distributed(self) -> bool:
        return dist.is_available() and dist.is_initialized() and self.accelerator.num_processes > 1

    def _gather_training_tensor(self, local_tensor: torch.Tensor) -> torch.Tensor:
        if not self._is_distributed():
            return local_tensor
        return self.accelerator.gather(local_tensor)

    def _assert_equal_local_batch_size(self, local_bsz: int, device: torch.device) -> None:
        if not self._is_distributed():
            return
        gathered_sizes = self.accelerator.gather(torch.tensor([local_bsz], device=device, dtype=torch.long))
        if not torch.all(gathered_sizes == gathered_sizes[0]):
            raise ValueError(
                "BetaDPOTrainer currently requires equal per-rank local batch size for global mask slicing. "
                "Use drop_last=True or implement variable-size slicing."
            )

    def _slice_global_to_local(self, global_vec: torch.Tensor, local_bsz: int) -> torch.Tensor:
        if not self._is_distributed():
            if global_vec.numel() != local_bsz:
                raise ValueError(f"Expected local-only vector of size {local_bsz}, got {global_vec.numel()}")
            return global_vec

        if self.args.require_equal_local_batch_size:
            self._assert_equal_local_batch_size(local_bsz, global_vec.device)

        rank = self.accelerator.process_index
        start = rank * local_bsz
        end = start + local_bsz
        if end > global_vec.numel():
            raise ValueError(
                f"Global vector too short for slicing: rank={rank}, local_bsz={local_bsz}, "
                f"start={start}, end={end}, global_numel={global_vec.numel()}"
            )
        return global_vec[start:end]

    def _get_reference_logps(
        self,
        batch: Dict[str, Union[torch.Tensor, str]],
        model: nn.Module,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if getattr(self, "precompute_ref_logps", False):
            if "ref_chosen_logps" not in batch or "ref_rejected_logps" not in batch:
                raise KeyError(
                    "precompute_ref_logps=True requires both 'ref_chosen_logps' and 'ref_rejected_logps' in the batch."
                )
            return batch["ref_chosen_logps"], batch["ref_rejected_logps"]

        if self.ref_model is None:
            raise ValueError("BetaDPOTrainer requires an explicit ref_model when precompute_ref_logps=False.")

        with torch.no_grad():
            ref_chosen_logps, ref_rejected_logps, _, _, _ = self.concatenated_forward(
                self.ref_model,
                batch,
                average_log_prob=False,
            )
        return ref_chosen_logps, ref_rejected_logps

    @torch.no_grad()
    def ema_update_gap_mean_and_std(self, r_gap_global: torch.Tensor) -> None:
        momentum = float(self.args.ema_momentum)
        batch_r_mean = r_gap_global.mean()
        batch_r_std = r_gap_global.std(unbiased=False)

        self.r_gap_mean.mul_(momentum).add_(batch_r_mean, alpha=1.0 - momentum)
        self.r_gap_std.mul_(momentum).add_(batch_r_std, alpha=1.0 - momentum)

        if self._is_distributed():
            dist.all_reduce(self.r_gap_mean, op=dist.ReduceOp.SUM)
            dist.all_reduce(self.r_gap_std, op=dist.ReduceOp.SUM)
            self.r_gap_mean.div_(self.accelerator.num_processes)
            self.r_gap_std.div_(self.accelerator.num_processes)

    @torch.no_grad()
    def _sample_global_mask(self, r_gap_global: torch.Tensor, mode: str) -> torch.Tensor:
        if mode != "train" and self.args.deterministic_eval:
            return torch.ones_like(r_gap_global, dtype=torch.bool)

        r_mean = self.r_gap_mean
        r_std = torch.clamp(self.r_gap_std, min=self._gap_std_eps)
        weight = torch.exp(-0.5 * ((r_gap_global - r_mean) / r_std).pow(2))
        weight = torch.nan_to_num(weight, nan=0.0, posinf=0.0, neginf=0.0)
        if float(weight.sum()) <= 0.0:
            weight = torch.ones_like(weight)

        total = weight.numel()
        keep = max(1, min(total, int(total * float(self.args.rho))))

        if self.args.sync_global_mask and self._is_distributed():
            if self.accelerator.is_main_process:
                idx = torch.multinomial(weight, keep, replacement=False)
                mask = torch.zeros_like(weight, dtype=torch.bool)
                mask[idx] = True
            else:
                mask = torch.zeros_like(weight, dtype=torch.bool)
            dist.broadcast(mask, src=0)
            return mask.detach()

        idx = torch.multinomial(weight, keep, replacement=False)
        mask = torch.zeros_like(weight, dtype=torch.bool)
        mask[idx] = True
        return mask.detach()

    @torch.no_grad()
    def _compute_adaptive_beta(
        self,
        r_gap_global: torch.Tensor,
        global_mask: torch.Tensor,
        beta: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        selected = global_mask.bool()
        if bool(selected.any()):
            r_used = r_gap_global[selected].mean()
        else:
            r_used = r_gap_global.mean()

        beta_used_raw = beta * (1.0 + float(self.args.alpha) * (r_used - self.r_gap_mean))
        beta_used = torch.clamp(beta_used_raw, min=float(self.args.beta_min))
        return beta_used_raw, beta_used

    def get_batch_loss_metrics(
        self,
        model,
        batch: Dict[str, Union[torch.Tensor, str]],
        train_eval: Literal["train", "eval"] = "train",
    ):
        prefix = "eval_" if train_eval == "eval" else ""
        metrics = {}

        chosen_logps, rejected_logps, chosen_logits, rejected_logits, _ = self.concatenated_forward(
            model,
            batch,
            average_log_prob=False,
        )
        ref_chosen_logps, ref_rejected_logps = self._get_reference_logps(batch, model)

        logits = (chosen_logps - rejected_logps) - (ref_chosen_logps - ref_rejected_logps)
        r_gap_local = (chosen_logps - ref_chosen_logps - rejected_logps + ref_rejected_logps).detach()
        r_gap_global = self._gather_training_tensor(r_gap_local)

        if train_eval == "train":
            self.ema_update_gap_mean_and_std(r_gap_global)

        global_mask = self._sample_global_mask(r_gap_global, mode=train_eval)
        beta_used_raw, beta_used = self._compute_adaptive_beta(
            r_gap_global=r_gap_global,
            global_mask=global_mask,
            beta=float(self.beta),
        )

        local_bsz = chosen_logps.size(0)
        mask_local = self._slice_global_to_local(global_mask, local_bsz).to(logits.device)
        mask_local_f = mask_local.to(dtype=logits.dtype)
        losses_local = -F.logsigmoid(beta_used.detach() * logits)
        loss = (losses_local * mask_local_f).sum() / mask_local_f.sum().clamp(min=1.0)

        metrics[f"{prefix}beta_dpo/gap_mean"] = self.r_gap_mean.detach().cpu()
        metrics[f"{prefix}beta_dpo/gap_std"] = self.r_gap_std.detach().cpu()
        metrics[f"{prefix}beta_dpo/beta_used_raw"] = beta_used_raw.detach().cpu()
        metrics[f"{prefix}beta_dpo/beta_used"] = beta_used.detach().cpu()
        metrics[f"{prefix}beta_dpo/mask_keep_frac"] = mask_local_f.mean().detach().cpu()
        metrics["eval_logits/chosen" if prefix else "logits/chosen"] = chosen_logits.detach().mean().cpu()
        metrics["eval_logits/rejected" if prefix else "logits/rejected"] = rejected_logits.detach().mean().cpu()
        return loss, metrics
