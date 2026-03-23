from dataclasses import dataclass, field
from typing import Any, Dict, Tuple

import torch
import torch.distributed as dist
import torch.nn.functional as F
from trl import DPOConfig, DPOTrainer
from trl.trainer.utils import selective_log_softmax


@dataclass
class BetaDPOConfig(DPOConfig):
    """
    TRL-style config for beta-DPO.

    Parameters
    ----------
    rho:
        Fraction of the global batch to keep after Gaussian-weighted sampling.
    alpha:
        Scaling factor used to adapt beta from the selected subset.
    ema_momentum:
        EMA momentum for reward-gap mean/std updates.
    beta_min:
        Lower bound for the adaptive beta.
    sync_global_mask:
        If True, sample one global mask on rank 0 and broadcast it to all ranks.
    deterministic_eval:
        If True, disable stochastic masking during eval and keep all samples.
    require_equal_local_batch_size:
        If True, distributed global->local slicing requires equal local batch sizes.
        This is the safest option for TRL DPO concatenated batches.
    """

    rho: float = field(default=0.8)
    alpha: float = field(default=1.0)
    ema_momentum: float = field(default=0.9)
    beta_min: float = field(default=1e-3)
    sync_global_mask: bool = field(default=True)
    deterministic_eval: bool = field(default=True)
    require_equal_local_batch_size: bool = field(default=True)

    def __post_init__(self):
        if not (0.0 < self.rho <= 1.0):
            raise ValueError("rho must be in (0, 1].")
        if self.alpha < 0:
            raise ValueError("alpha must be >= 0.")
        if not (0.0 <= self.ema_momentum < 1.0):
            raise ValueError("ema_momentum must be in [0, 1).")
        if self.beta_min <= 0.0:
            raise ValueError("beta_min must be > 0.")
        super().__post_init__()


class BetaDPOTrainer(DPOTrainer):
    """
    TRL DPOTrainer + beta-DPO.

    Training flow
    -------------
    1. Compute policy and reference completion log-probs.
    2. Build DPO logits and reward-gap signal.
    3. Gather reward gap globally across ranks.
    4. Update EMA reward-gap mean/std during training only.
    5. Sample a global subset mask using Gaussian weights.
    6. Compute adaptive beta from the selected subset.
    7. Apply masked DPO loss locally.

    Notes
    -----
    - This trainer expects TRL-style concatenated chosen/rejected batches.
    - For safety, pass an explicit frozen ref_model unless using precomputed ref log-probs.
    - The distributed global->local mask slicing assumes equal local batch size by default.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if not isinstance(self.args, BetaDPOConfig):
            raise TypeError("BetaDPOTrainer requires args=BetaDPOConfig")

        device = self.accelerator.device
        self.r_gap_mean = torch.zeros((), device=device)
        self.r_gap_std = torch.ones((), device=device)
        self._gap_std_eps = torch.tensor(1e-6, device=device)

    # -----------------------------
    # validation helpers
    # -----------------------------
    def _validate_inputs(self, inputs: Dict[str, torch.Tensor]) -> None:
        required = ("input_ids", "attention_mask", "completion_mask")
        missing = [k for k in required if k not in inputs]
        if missing:
            raise KeyError(
                f"Missing required keys for BetaDPOTrainer: {missing}. "
                f"Available keys: {sorted(list(inputs.keys()))}"
            )

        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        completion_mask = inputs["completion_mask"]

        if input_ids.size(0) % 2 != 0:
            raise ValueError(
                "BetaDPOTrainer expects chosen/rejected examples concatenated into "
                f"one batch with even size, but got batch size {input_ids.size(0)}."
            )

        if attention_mask.shape != input_ids.shape:
            raise ValueError(
                "attention_mask shape must match input_ids shape, got "
                f"{tuple(attention_mask.shape)} vs {tuple(input_ids.shape)}"
            )

        if completion_mask.shape != input_ids.shape:
            raise ValueError(
                "completion_mask shape must match input_ids shape, got "
                f"{tuple(completion_mask.shape)} vs {tuple(input_ids.shape)}"
            )

        if getattr(self, "precompute_ref_logps", False):
            if "ref_chosen_logps" not in inputs or "ref_rejected_logps" not in inputs:
                raise KeyError(
                    "precompute_ref_logps=True requires both 'ref_chosen_logps' and "
                    "'ref_rejected_logps' in the input batch."
                )

    def _split_chosen_rejected(
        self, x: torch.Tensor, name: str
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if x.size(0) % 2 != 0:
            raise ValueError(
                f"{name} must have even first dimension for chosen/rejected split, "
                f"but got shape {tuple(x.shape)}"
            )
        return x.chunk(2, dim=0)

    # -----------------------------
    # distributed helpers
    # -----------------------------
    def _is_distributed(self) -> bool:
        return (
            dist.is_available()
            and dist.is_initialized()
            and self.accelerator.num_processes > 1
        )

    def _gather_training_tensor(self, local_tensor: torch.Tensor) -> torch.Tensor:
        """
        Gather the full per-rank tensor without metric-time truncation.
        """
        if not self._is_distributed():
            return local_tensor
        return self.accelerator.gather(local_tensor)

    def _assert_equal_local_batch_size(
        self, local_bsz: int, device: torch.device
    ) -> None:
        if not self._is_distributed():
            return
        gathered_sizes = self.accelerator.gather(
            torch.tensor([local_bsz], device=device, dtype=torch.long)
        )
        if not torch.all(gathered_sizes == gathered_sizes[0]):
            raise ValueError(
                "BetaDPOTrainer currently requires equal per-rank local batch size "
                "for global mask slicing. Use drop_last=True or implement variable-size slicing."
            )

    def _slice_global_to_local(
        self, global_vec: torch.Tensor, local_bsz: int
    ) -> torch.Tensor:
        """
        Slice a global vector [N_global] into the local chunk for the current rank.

        By default this requires equal local batch size on each rank.
        """
        if not self._is_distributed():
            if global_vec.numel() != local_bsz:
                raise ValueError(
                    f"Expected local-only vector of size {local_bsz}, got {global_vec.numel()}"
                )
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

    # -----------------------------
    # model helpers
    # -----------------------------
    def _build_model_kwargs(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        inputs: Dict[str, torch.Tensor],
    ) -> Dict[str, Any]:
        model_kwargs: Dict[str, Any] = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "use_cache": False,
        }

        for key in (
            "pixel_values",
            "pixel_attention_mask",
            "image_grid_thw",
            "image_sizes",
            "token_type_ids",
        ):
            if key in inputs:
                model_kwargs[key] = inputs[key]

        return model_kwargs

    def _sequence_logps(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        completion_mask: torch.Tensor,
    ) -> torch.Tensor:
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        shift_completion_mask = completion_mask[:, 1:].contiguous()

        per_token_logps = selective_log_softmax(shift_logits, shift_labels)
        per_token_logps = per_token_logps.masked_fill(shift_completion_mask == 0, 0.0)
        return per_token_logps.sum(dim=1)

    def _compute_policy_logps(
        self,
        model,
        model_kwargs: Dict[str, Any],
        input_ids: torch.Tensor,
        completion_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, Any]:
        outputs = model(**model_kwargs)
        seq_logps = self._sequence_logps(outputs.logits, input_ids, completion_mask)
        chosen_logps, rejected_logps = self._split_chosen_rejected(
            seq_logps, "policy_logps"
        )
        return chosen_logps, rejected_logps, outputs

    def _compute_ref_logps(
        self,
        inputs: Dict[str, torch.Tensor],
        model_kwargs: Dict[str, Any],
        input_ids: torch.Tensor,
        completion_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if getattr(self, "precompute_ref_logps", False):
            return inputs["ref_chosen_logps"], inputs["ref_rejected_logps"]

        ref_model = getattr(self, "ref_model", None)
        if ref_model is None:
            raise ValueError(
                "BetaDPOTrainer requires an explicit ref_model when precompute_ref_logps=False."
            )

        with torch.no_grad():
            ref_outputs = ref_model(**model_kwargs)

        ref_seq_logps = self._sequence_logps(
            ref_outputs.logits, input_ids, completion_mask
        )
        ref_chosen_logps, ref_rejected_logps = self._split_chosen_rejected(
            ref_seq_logps, "ref_logps"
        )
        return ref_chosen_logps, ref_rejected_logps

    # -----------------------------
    # beta-DPO helpers
    # -----------------------------
    @torch.no_grad()
    def ema_update_gap_mean_and_std(self, r_gap_global: torch.Tensor) -> None:
        """
        EMA update for reward-gap mean/std.

        Uses unbiased=False for consistency with your original trainer.
        """
        m = float(self.args.ema_momentum)
        batch_r_mean = r_gap_global.mean()
        batch_r_std = r_gap_global.std(unbiased=False)

        self.r_gap_mean.mul_(m).add_(batch_r_mean, alpha=1.0 - m)
        self.r_gap_std.mul_(m).add_(batch_r_std, alpha=1.0 - m)

        if self._is_distributed():
            dist.all_reduce(self.r_gap_mean, op=dist.ReduceOp.SUM)
            dist.all_reduce(self.r_gap_std, op=dist.ReduceOp.SUM)
            self.r_gap_mean.div_(self.accelerator.num_processes)
            self.r_gap_std.div_(self.accelerator.num_processes)

    @torch.no_grad()
    def _sample_global_mask(
        self, r_gap_global: torch.Tensor, mode: str
    ) -> torch.Tensor:
        """
        Gaussian-weighted sampling without replacement.

        In eval mode, can optionally return an all-ones mask for deterministic evaluation.
        """
        if mode != "train" and self.args.deterministic_eval:
            return torch.ones_like(r_gap_global, dtype=torch.bool)

        r_mean = self.r_gap_mean
        r_std = torch.clamp(self.r_gap_std, min=self._gap_std_eps)

        weight = torch.exp(-0.5 * ((r_gap_global - r_mean) / r_std).pow(2))
        weight = torch.nan_to_num(weight, nan=0.0, posinf=0.0, neginf=0.0)

        if float(weight.sum()) <= 0.0:
            weight = torch.ones_like(weight)

        N = weight.numel()
        k = max(1, min(N, int(N * float(self.args.rho))))

        if self.args.sync_global_mask and self._is_distributed():
            if self.accelerator.is_main_process:
                idx = torch.multinomial(weight, k, replacement=False)
                mask = torch.zeros_like(weight, dtype=torch.bool)
                mask[idx] = True
            else:
                mask = torch.zeros_like(weight, dtype=torch.bool)

            dist.broadcast(mask, src=0)
            return mask.detach()

        idx = torch.multinomial(weight, k, replacement=False)
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
        """
        beta_used = beta * (1 + alpha * (r_used - r_gap_mean))
        where r_used = mean(r[selected by global_mask])

        Returns
        -------
        beta_used_raw, beta_used_clamped
        """
        select = global_mask.bool()
        if bool(select.any()):
            r_used = r_gap_global[select].mean()
        else:
            r_used = r_gap_global.mean()

        beta_used_raw = beta * (
            1.0 + float(self.args.alpha) * (r_used - self.r_gap_mean)
        )
        beta_used = torch.clamp(beta_used_raw, min=float(self.args.beta_min))
        return beta_used_raw, beta_used

    def _record_metric(self, mode: str, key: str, value: float) -> None:
        self._metrics[mode].setdefault(key, []).append(value)

    # -----------------------------
    # main loss override
    # -----------------------------
    def _compute_loss(self, model, inputs, return_outputs: bool = False):
        """
        Core TRL hook. Override to implement beta-DPO.
        """
        mode = "train" if self.model.training else "eval"

        self._validate_inputs(inputs)

        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        completion_mask = inputs["completion_mask"]

        input_ids, attention_mask, completion_mask = self._truncate_inputs(
            input_ids, attention_mask, completion_mask
        )

        model_kwargs = self._build_model_kwargs(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs=inputs,
        )

        # 1) policy and reference completion log-probs
        chosen_logps, rejected_logps, outputs = self._compute_policy_logps(
            model=model,
            model_kwargs=model_kwargs,
            input_ids=input_ids,
            completion_mask=completion_mask,
        )

        ref_chosen_logps, ref_rejected_logps = self._compute_ref_logps(
            inputs=inputs,
            model_kwargs=model_kwargs,
            input_ids=input_ids,
            completion_mask=completion_mask,
        )

        # Standard DPO logits
        logits = (chosen_logps - rejected_logps) - (
            ref_chosen_logps - ref_rejected_logps
        )

        # 2) reward discrepancy / reward gap
        r_gap_local = (
            chosen_logps - ref_chosen_logps - rejected_logps + ref_rejected_logps
        ).detach()
        r_gap_global = self._gather_training_tensor(r_gap_local)

        # 3) EMA update in train only
        if mode == "train":
            self.ema_update_gap_mean_and_std(r_gap_global)

        # 4) global subset mask
        global_mask = self._sample_global_mask(r_gap_global, mode=mode)

        # 5) adaptive beta from selected subset
        base_beta = float(self.beta)
        beta_used_raw, beta_used = self._compute_adaptive_beta(
            r_gap_global=r_gap_global,
            global_mask=global_mask,
            beta=base_beta,
        )

        # 6) local masked loss
        local_bsz = chosen_logps.size(0)
        mask_local = self._slice_global_to_local(global_mask, local_bsz).to(
            logits.device
        )
        mask_local_f = mask_local.to(dtype=logits.dtype)

        losses_local = -F.logsigmoid(beta_used.detach() * logits)
        loss = (losses_local * mask_local_f).sum() / mask_local_f.sum().clamp(min=1.0)

        # 7) metrics
        self._record_metric(mode, "beta_dpo/gap_mean", float(self.r_gap_mean.item()))
        self._record_metric(mode, "beta_dpo/gap_std", float(self.r_gap_std.item()))
        self._record_metric(mode, "beta_dpo/beta_used_raw", float(beta_used_raw.item()))
        self._record_metric(mode, "beta_dpo/beta_used", float(beta_used.item()))
        self._record_metric(
            mode, "beta_dpo/mask_keep_frac", float(mask_local_f.mean().item())
        )

        if return_outputs:
            aux = {
                "logits": outputs.logits,
                "dpo_logits": logits.detach(),
                "r_gap_local": r_gap_local.detach(),
                "r_gap_global": r_gap_global.detach(),
                "global_mask": global_mask.detach(),
                "mask_local": mask_local.detach(),
                "beta_used_raw": beta_used_raw.detach(),
                "beta_used": beta_used.detach(),
                "chosen_logps": chosen_logps.detach(),
                "rejected_logps": rejected_logps.detach(),
                "ref_chosen_logps": ref_chosen_logps.detach(),
                "ref_rejected_logps": ref_rejected_logps.detach(),
                "losses_local": losses_local.detach(),
            }
            return loss, aux

        return loss
