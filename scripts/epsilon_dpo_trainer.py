from contextlib import nullcontext
from typing import Any, Dict, Literal, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from tokenized_dpo_trainer import TokenizedDPOTrainer
from trainer_configs import EpsilonDPOConfig


class EpsilonDPOTrainer(TokenizedDPOTrainer):
    _tag_names = ["trl", "epsilon-dpo"]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if not isinstance(self.args, EpsilonDPOConfig):
            raise TypeError("EpsilonDPOTrainer requires args=EpsilonDPOConfig")
        if getattr(self.args, "precompute_ref_log_probs", False):
            raise ValueError(
                "EpsilonDPOTrainer requires precompute_ref_log_probs=False because epsilon-step computation "
                "needs live reference logits, not only cached sequence log-probs."
            )
        if self.ref_model is None:
            raise ValueError("EpsilonDPOTrainer requires an explicit ref_model when precompute_ref_log_probs=False.")

        self.epsilon = float(self.args.epsilon)
        self.aux_loss_enabled = bool(getattr(self.model.config, "output_router_logits", False))
        self.use_weighting = bool(getattr(self.args, "use_weighting", False))
        self._pending_steps = torch.zeros((), device=self.accelerator.device)

    @staticmethod
    def _score_sequence_logits(
        logits: torch.FloatTensor,
        labels: torch.LongTensor,
        label_pad_token_id: int = -100,
        is_encoder_decoder: bool = False,
    ) -> Dict[str, torch.Tensor]:
        if logits.shape[:-1] != labels.shape:
            raise ValueError("Logits (batch and sequence length dim) and labels must have the same shape.")

        if is_encoder_decoder:
            scoring_logits = logits
            shifted_labels = labels.clone()
        else:
            scoring_logits = logits[:, :-1, :]
            shifted_labels = labels[:, 1:].clone()

        loss_mask = shifted_labels != label_pad_token_id
        safe_labels = shifted_labels.masked_fill(~loss_mask, 0)
        per_token_logps = TokenizedDPOTrainer._compute_token_logps(scoring_logits, safe_labels)
        sequence_logps = (per_token_logps * loss_mask).sum(-1)

        return {
            "sequence_logps": sequence_logps,
            "safe_labels": safe_labels,
            "loss_mask": loss_mask,
            "per_token_logps": per_token_logps,
            "scoring_logits": scoring_logits,
        }

    @staticmethod
    def _score_aligned_logits(
        aligned_logits: torch.FloatTensor,
        safe_labels: torch.LongTensor,
        loss_mask: torch.Tensor,
    ) -> torch.Tensor:
        per_token_logps = TokenizedDPOTrainer._compute_token_logps(aligned_logits, safe_labels)
        return (per_token_logps * loss_mask).sum(-1)

    @staticmethod
    def _compute_steps(
        logratios: torch.Tensor,
        p_epsilon_logratios: torch.Tensor,
        n_epsilon_logratios: torch.Tensor,
    ) -> torch.Tensor:
        p_epsilon_steps = (p_epsilon_logratios > logratios) & (logratios > n_epsilon_logratios)
        n_epsilon_steps = (n_epsilon_logratios > logratios) & (logratios > p_epsilon_logratios)
        return p_epsilon_steps.to(torch.long) - n_epsilon_steps.to(torch.long)

    def _mean_loss_position_logits(
        self,
        scoring_logits: torch.Tensor,
        loss_mask: torch.Tensor,
    ) -> torch.Tensor:
        if bool(loss_mask.any()):
            return scoring_logits[loss_mask].mean()
        return scoring_logits.mean()

    def _cross_entropy_from_scored_tokens(
        self,
        scoring_logits: torch.Tensor,
        safe_labels: torch.Tensor,
    ) -> torch.Tensor:
        return F.cross_entropy(
            scoring_logits.reshape(-1, scoring_logits.shape[-1]),
            safe_labels.reshape(-1),
            ignore_index=0,
        )

    def _epsilon_forward(
        self,
        model: nn.Module,
        batch: Dict[str, Union[torch.Tensor, Any]],
        logit_source: str,
        force_logits_to_float32: bool = False,
    ) -> Dict[str, torch.Tensor]:
        model_device = self._get_model_device(model)
        if model_device is None or model_device.type == "cpu":
            model_device = self.accelerator.device

        concatenated_batch = self.concatenated_inputs(
            batch,
            is_encoder_decoder=self.is_encoder_decoder,
            label_pad_token_id=self.label_pad_token_id,
            padding_value=self.padding_value,
            device=model_device,
        )
        len_chosen = batch["chosen_labels"].shape[0]

        model_kwargs = (
            {
                "labels": concatenated_batch["concatenated_labels"],
                "decoder_input_ids": concatenated_batch.pop("concatenated_decoder_input_ids", None),
            }
            if self.is_encoder_decoder
            else {}
        )
        if self.aux_loss_enabled:
            model_kwargs["output_router_logits"] = True

        outputs = model(
            concatenated_batch["concatenated_input_ids"],
            attention_mask=concatenated_batch["concatenated_attention_mask"],
            use_cache=False,
            **model_kwargs,
        )
        all_logits = outputs.logits
        if force_logits_to_float32:
            all_logits = all_logits.float()
        all_logits = self._handle_non_finite_logits(all_logits, logit_source=logit_source)

        scored = self._score_sequence_logits(
            all_logits,
            concatenated_batch["concatenated_labels"],
            label_pad_token_id=self.label_pad_token_id,
            is_encoder_decoder=self.is_encoder_decoder,
        )
        all_logps = scored["sequence_logps"]
        chosen_logps = all_logps[:len_chosen]
        rejected_logps = all_logps[len_chosen:]

        scoring_logits = scored["scoring_logits"]
        loss_mask = scored["loss_mask"]
        safe_labels = scored["safe_labels"]

        output = {
            "chosen_logps": chosen_logps,
            "rejected_logps": rejected_logps,
            "all_logits": all_logits,
            "scoring_logits": scoring_logits,
            "safe_labels": safe_labels,
            "loss_mask": loss_mask,
            "per_token_logps": scored["per_token_logps"],
            "mean_chosen_logits": self._mean_loss_position_logits(scoring_logits[:len_chosen], loss_mask[:len_chosen]),
            "mean_rejected_logits": self._mean_loss_position_logits(scoring_logits[len_chosen:], loss_mask[len_chosen:]),
        }

        if self.args.rpo_alpha is not None:
            output["nll_loss"] = self._cross_entropy_from_scored_tokens(
                scoring_logits[:len_chosen],
                safe_labels[:len_chosen],
            )

        if self.use_weighting:
            logprobs = F.log_softmax(scoring_logits, dim=-1)
            weights_adjustment_factor = torch.logsumexp(2 * logprobs, dim=-1)
            per_token_logps_adjusted = scored["per_token_logps"] - weights_adjustment_factor
            token_counts = loss_mask.sum(-1).clamp_min(1)
            all_weights = (per_token_logps_adjusted * loss_mask).sum(-1) / token_counts
            chosen_weights = all_weights[:len_chosen]
            rejected_weights = all_weights[len_chosen:]
            output["policy_weights"] = torch.clamp(torch.exp(chosen_weights + rejected_weights), max=1.0)

        if self.aux_loss_enabled and hasattr(outputs, "aux_loss"):
            output["aux_loss"] = outputs.aux_loss

        return output

    @torch.no_grad()
    def compute_reference_model_output(
        self,
        padded_batch: Dict[str, Union[torch.Tensor, Any]],
    ) -> Dict[str, torch.Tensor]:
        if self.ref_model is None:
            raise ValueError("EpsilonDPOTrainer requires a live ref_model for epsilon-step computation.")

        compute_ref_context = torch.cuda.amp.autocast if self._peft_has_been_casted_to_bf16 else nullcontext
        with compute_ref_context():
            return self._epsilon_forward(
                self.ref_model,
                padded_batch,
                logit_source="reference",
            )

    @torch.no_grad()
    def compute_reference_log_probs(
        self,
        padded_batch: Dict[str, Union[torch.Tensor, Any]],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        reference_output = self.compute_reference_model_output(padded_batch)
        return reference_output["chosen_logps"], reference_output["rejected_logps"]

    def dpo_loss(
        self,
        chosen_logps: torch.FloatTensor,
        rejected_logps: torch.FloatTensor,
        ref_chosen_logps: torch.FloatTensor,
        ref_rejected_logps: torch.FloatTensor,
        steps: torch.LongTensor,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        logratios = chosen_logps - rejected_logps
        ref_logratios = ref_chosen_logps - ref_rejected_logps
        logits = logratios - ref_logratios

        updated_beta = self.beta / (1 + self.epsilon * steps.to(logits.dtype))
        losses = (
            -F.logsigmoid(updated_beta * logits) * (1 - self.label_smoothing)
            - F.logsigmoid(updated_beta * logits) * self.label_smoothing
        )
        chosen_rewards = updated_beta * (chosen_logps - ref_chosen_logps).detach()
        rejected_rewards = updated_beta * (rejected_logps - ref_rejected_logps).detach()
        return losses, chosen_rewards, rejected_rewards

    def _accumulate_microbatch_steps(self, steps: torch.Tensor) -> None:
        self._pending_steps.add_(steps.float().mean() / float(self.args.gradient_accumulation_steps))

    def _apply_pending_beta_update(self, metrics: Dict[str, torch.Tensor], prefix: str) -> None:
        mean_steps = self.accelerator.gather(self._pending_steps.detach().reshape(1)).mean()
        metrics[f"{prefix}kl/beta"] = float(self.beta)
        metrics[f"{prefix}kl/avg_steps"] = mean_steps.detach().cpu()
        self.beta = self.beta / (1 + float(mean_steps.item()) * self.epsilon)
        self._pending_steps.zero_()

    def get_batch_loss_metrics(
        self,
        model,
        batch: Dict[str, Union[torch.Tensor, Any]],
        train_eval: Literal["train", "eval"] = "train",
    ):
        prefix = "eval_" if train_eval == "eval" else ""
        metrics = {}

        policy_output = self._epsilon_forward(model, batch, logit_source="policy")
        reference_output = self.compute_reference_model_output(batch)

        policy_logratios = policy_output["chosen_logps"] - policy_output["rejected_logps"]
        reference_logratios = reference_output["chosen_logps"] - reference_output["rejected_logps"]

        p_epsilon_logits = self._handle_non_finite_logits(
            ((1 + self.epsilon) * policy_output["scoring_logits"]) - (self.epsilon * reference_output["scoring_logits"]),
            logit_source="policy_plus_epsilon",
        )
        n_epsilon_logits = self._handle_non_finite_logits(
            ((1 - self.epsilon) * policy_output["scoring_logits"]) + (self.epsilon * reference_output["scoring_logits"]),
            logit_source="policy_minus_epsilon",
        )

        p_epsilon_scores = self._score_aligned_logits(
            p_epsilon_logits,
            policy_output["safe_labels"],
            policy_output["loss_mask"],
        )
        n_epsilon_scores = self._score_aligned_logits(
            n_epsilon_logits,
            policy_output["safe_labels"],
            policy_output["loss_mask"],
        )

        len_chosen = policy_output["chosen_logps"].shape[0]
        p_epsilon_logratios = p_epsilon_scores[:len_chosen] - p_epsilon_scores[len_chosen:]
        n_epsilon_logratios = n_epsilon_scores[:len_chosen] - n_epsilon_scores[len_chosen:]
        steps = self._compute_steps(policy_logratios, p_epsilon_logratios, n_epsilon_logratios)

        losses, chosen_rewards, rejected_rewards = self.dpo_loss(
            policy_output["chosen_logps"],
            policy_output["rejected_logps"],
            reference_output["chosen_logps"],
            reference_output["rejected_logps"],
            steps,
        )

        if self.args.rpo_alpha is not None:
            losses = losses + (self.args.rpo_alpha * policy_output["nll_loss"])

        if self.use_weighting and "policy_weights" in policy_output:
            losses = losses * policy_output["policy_weights"]

        loss = losses.mean()
        if self.aux_loss_enabled and "aux_loss" in policy_output:
            aux_loss_coef = float(getattr(model.config, "router_aux_loss_coef", 0.0))
            loss = loss + (aux_loss_coef * policy_output["aux_loss"])
            metrics[f"{prefix}aux_loss"] = self.accelerator.gather_for_metrics(policy_output["aux_loss"].detach()).mean().cpu()

        reward_accuracies = (chosen_rewards > rejected_rewards).float()

        metrics[f"{prefix}rewards/chosen"] = self.accelerator.gather_for_metrics(chosen_rewards).mean().cpu()
        metrics[f"{prefix}rewards/rejected"] = self.accelerator.gather_for_metrics(rejected_rewards).mean().cpu()
        metrics[f"{prefix}rewards/accuracies"] = self.accelerator.gather_for_metrics(reward_accuracies).mean().cpu()
        metrics[f"{prefix}rewards/margins"] = self.accelerator.gather_for_metrics(
            chosen_rewards - rejected_rewards
        ).mean().cpu()
        metrics[f"{prefix}logps/chosen"] = self.accelerator.gather_for_metrics(
            policy_output["chosen_logps"].detach()
        ).mean().cpu()
        metrics[f"{prefix}logps/rejected"] = self.accelerator.gather_for_metrics(
            policy_output["rejected_logps"].detach()
        ).mean().cpu()
        metrics[f"{prefix}logps/ref_chosen"] = self.accelerator.gather_for_metrics(
            reference_output["chosen_logps"].detach()
        ).mean().cpu()
        metrics[f"{prefix}logps/ref_rejected"] = self.accelerator.gather_for_metrics(
            reference_output["rejected_logps"].detach()
        ).mean().cpu()
        metrics[f"{prefix}logits/chosen"] = self.accelerator.gather_for_metrics(
            policy_output["mean_chosen_logits"].detach().reshape(1)
        ).mean().cpu()
        metrics[f"{prefix}logits/rejected"] = self.accelerator.gather_for_metrics(
            policy_output["mean_rejected_logits"].detach().reshape(1)
        ).mean().cpu()
        metrics[f"{prefix}kl/p_epsilon_steps"] = self.accelerator.gather_for_metrics((steps == 1).float()).mean().cpu()
        metrics[f"{prefix}kl/n_epsilon_steps"] = self.accelerator.gather_for_metrics((steps == -1).float()).mean().cpu()

        if self.args.rpo_alpha is not None:
            metrics[f"{prefix}nll_loss"] = self.accelerator.gather_for_metrics(
                policy_output["nll_loss"].detach().reshape(1)
            ).mean().cpu()

        if train_eval == "train":
            self._accumulate_microbatch_steps(steps.detach())
            if self.accelerator.gradient_state.sync_gradients:
                self._apply_pending_beta_update(metrics, prefix=prefix)

        return loss, metrics
