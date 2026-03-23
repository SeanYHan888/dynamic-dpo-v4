import json
import os
from typing import Any, Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from trl import DPOTrainer
from trl.trainer.utils import selective_log_softmax


def log_margin(
    margin: torch.Tensor,
    log_dir: str,
    epoch: float,
    step: int,
    save_full: bool = False,
) -> None:
    """
    Log summary stats for a batch of margins.
    Optionally save the full per-sample margin array as a .npy file.
    """
    os.makedirs(log_dir, exist_ok=True)

    m = margin.detach().float().cpu().numpy()
    p10, p50, p90 = np.percentile(m, [10, 50, 90])

    record = {
        "epoch": float(epoch),
        "step": int(step),
        "batch_size": int(m.shape[0]),
        "mean": float(m.mean()),
        "std": float(m.std(ddof=0)),
        "min": float(m.min()),
        "p10": float(p10),
        "median": float(p50),
        "p90": float(p90),
        "max": float(m.max()),
        "pos_frac": float((m > 0).mean()),
    }

    if save_full:
        npy_path = os.path.join(log_dir, f"step_{step:07d}.npy")
        np.save(npy_path, m)
        record["npy"] = npy_path

    jsonl_path = os.path.join(log_dir, "margins.jsonl")
    with open(jsonl_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


class MarginDPOTrainer(DPOTrainer):
    """
    A stricter, cleaner DPOTrainer variant that:

    1. Computes policy and reference sequence log-probs on completion tokens only
    2. Logs DPO margin statistics during training
    3. Supports a few custom f-divergence score transforms
    """

    def __init__(
        self,
        *args,
        margin_log_path: str = "./margin_logs",
        margin_log_steps: int = 50,
        margin_save_full: bool = False,
        require_explicit_ref_model: bool = True,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.margin_log_path = margin_log_path
        self.margin_log_steps = int(margin_log_steps)
        self.margin_save_full = bool(margin_save_full)
        self.require_explicit_ref_model = bool(require_explicit_ref_model)

        os.makedirs(self.margin_log_path, exist_ok=True)

    def _validate_inputs(self, inputs: Dict[str, torch.Tensor]) -> None:
        required = ("input_ids", "attention_mask", "completion_mask")
        missing = [k for k in required if k not in inputs]
        if missing:
            raise KeyError(
                f"Missing required keys for MarginDPOTrainer: {missing}. "
                f"Available keys: {sorted(list(inputs.keys()))}"
            )

        batch_size = inputs["input_ids"].size(0)
        if batch_size % 2 != 0:
            raise ValueError(
                "MarginDPOTrainer expects chosen/rejected examples to be concatenated "
                f"into a single batch with even size, but got batch size {batch_size}."
            )

        if inputs["attention_mask"].shape != inputs["input_ids"].shape:
            raise ValueError(
                "attention_mask shape must match input_ids shape, got "
                f"{tuple(inputs['attention_mask'].shape)} vs {tuple(inputs['input_ids'].shape)}"
            )

        if inputs["completion_mask"].shape != inputs["input_ids"].shape:
            raise ValueError(
                "completion_mask shape must match input_ids shape, got "
                f"{tuple(inputs['completion_mask'].shape)} vs {tuple(inputs['input_ids'].shape)}"
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
            if "ref_chosen_logps" not in inputs or "ref_rejected_logps" not in inputs:
                raise KeyError(
                    "precompute_ref_logps=True requires both 'ref_chosen_logps' and "
                    "'ref_rejected_logps' in the input batch."
                )
            return inputs["ref_chosen_logps"], inputs["ref_rejected_logps"]

        ref_model = getattr(self, "ref_model", None)
        if ref_model is None and self.require_explicit_ref_model:
            raise ValueError(
                "This trainer is configured to require an explicit ref_model, "
                "but self.ref_model is None."
            )

        if ref_model is None:
            raise ValueError(
                "ref_model is None. For this custom trainer, pass an explicit frozen "
                "reference model or enable precomputed reference log-probs."
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

    def _stable_alpha_scores(
        self,
        logratios: torch.Tensor,
        alpha: float,
        coef: float,
    ) -> torch.Tensor:
        t = (alpha - 1.0) * logratios
        dtype = t.dtype

        clamp_max_map = {
            torch.float16: 11.0,
            torch.bfloat16: 80.0,
            torch.float32: 80.0,
            torch.float64: 80.0,
        }
        clamp_max = clamp_max_map.get(dtype, 80.0)

        t_float = torch.clamp(t.float(), max=clamp_max)
        return torch.exp(t_float).to(dtype) * coef

    def _project_scores(
        self,
        chosen_logratios: torch.Tensor,
        rejected_logratios: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        kind = getattr(self, "f_divergence_type", "reverse_kl")

        if kind == "reverse_kl":
            return chosen_logratios, rejected_logratios

        if kind == "forward_kl":
            return -torch.exp(-chosen_logratios), -torch.exp(-rejected_logratios)

        if kind == "js_divergence":
            return F.logsigmoid(chosen_logratios), F.logsigmoid(rejected_logratios)

        if kind == "alpha_divergence":
            alpha = float(getattr(self, "f_alpha_divergence_coef", 1.0))
            if abs(alpha - 1.0) < 1e-6:
                return chosen_logratios, rejected_logratios

            coef = 1.0 / (alpha - 1.0)
            chosen_scores = self._stable_alpha_scores(chosen_logratios, alpha, coef)
            rejected_scores = self._stable_alpha_scores(rejected_logratios, alpha, coef)
            return chosen_scores, rejected_scores

        raise ValueError(f"Unknown f_divergence_type: {kind}")

    def _maybe_log_margin(self, margin_tensor: torch.Tensor) -> None:
        if not self.model.training:
            return

        log_every = self.margin_log_steps
        step = int(self.state.global_step)

        if log_every <= 0:
            return
        if step % log_every != 0:
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

    def _compute_loss(
        self,
        model,
        inputs: Dict[str, torch.Tensor],
        return_outputs: bool = False,
    ):
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

        margin = (chosen_logps - rejected_logps) - (
            ref_chosen_logps - ref_rejected_logps
        )
        self._maybe_log_margin(margin.detach())

        chosen_logratios = chosen_logps - ref_chosen_logps
        rejected_logratios = rejected_logps - ref_rejected_logps

        chosen_scores, rejected_scores = self._project_scores(
            chosen_logratios=chosen_logratios,
            rejected_logratios=rejected_logratios,
        )

        delta_score = chosen_scores - rejected_scores
        beta = float(getattr(self, "beta", 0.1))

        per_seq_loss = -F.logsigmoid(beta * delta_score)
        loss = per_seq_loss.mean()

        if return_outputs:
            aux = {
                "logits": outputs.logits,
                "margin": margin.detach(),
                "chosen_logps": chosen_logps.detach(),
                "rejected_logps": rejected_logps.detach(),
                "ref_chosen_logps": ref_chosen_logps.detach(),
                "ref_rejected_logps": ref_rejected_logps.detach(),
                "chosen_logratios": chosen_logratios.detach(),
                "rejected_logratios": rejected_logratios.detach(),
                "delta_score": delta_score.detach(),
                "per_seq_loss": per_seq_loss.detach(),
            }
            return loss, aux

        return loss
