from typing import Any, Dict, Literal, Tuple, Union

import torch

from tokenized_dpo_trainer import TokenizedDPOTrainer
from trainer_configs import SLiCHFConfig


class SLiCHFTrainer(TokenizedDPOTrainer):
    _tag_names = ["trl", "slic-hf"]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, ref_model=None, **kwargs)

        if not isinstance(self.args, SLiCHFConfig):
            raise TypeError("SLiCHFTrainer requires args=SLiCHFConfig")

        self.slic_margin = float(self.args.slic_margin)
        self.slic_lambda = float(self.args.slic_lambda)

    def slic_hf_loss(
        self,
        chosen_logps: torch.FloatTensor,
        rejected_logps: torch.FloatTensor,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        rank_losses = torch.relu(self.slic_margin - (chosen_logps - rejected_logps))
        ce_losses = self.slic_lambda * (-chosen_logps)
        total_losses = rank_losses + ce_losses
        return total_losses, rank_losses, ce_losses

    def get_batch_loss_metrics(
        self,
        model,
        batch: Dict[str, Union[torch.Tensor, Any]],
        train_eval: Literal["train", "eval"] = "train",
    ):
        prefix = "eval_" if train_eval == "eval" else ""
        metrics = {}

        chosen_logps, rejected_logps, chosen_logits, rejected_logits, _ = self.concatenated_forward(
            model,
            batch,
            average_log_prob=False,
            logit_source="policy",
        )

        losses, rank_losses, ce_losses = self.slic_hf_loss(chosen_logps, rejected_logps)
        loss = losses.mean()

        preference_margin = chosen_logps - rejected_logps
        reward_accuracies = (preference_margin > 0).float()

        metrics[f"{prefix}rewards/chosen"] = self.accelerator.gather_for_metrics(chosen_logps.detach()).mean().cpu()
        metrics[f"{prefix}rewards/rejected"] = self.accelerator.gather_for_metrics(rejected_logps.detach()).mean().cpu()
        metrics[f"{prefix}rewards/accuracies"] = self.accelerator.gather_for_metrics(reward_accuracies).mean().cpu()
        metrics[f"{prefix}rewards/margins"] = self.accelerator.gather_for_metrics(preference_margin.detach()).mean().cpu()
        metrics[f"{prefix}logps/chosen"] = self.accelerator.gather_for_metrics(chosen_logps.detach()).mean().cpu()
        metrics[f"{prefix}logps/rejected"] = self.accelerator.gather_for_metrics(rejected_logps.detach()).mean().cpu()
        metrics[f"{prefix}slic/rank_loss"] = self.accelerator.gather_for_metrics(rank_losses.detach()).mean().cpu()
        metrics[f"{prefix}slic/ce_loss"] = self.accelerator.gather_for_metrics(ce_losses.detach()).mean().cpu()
        metrics[f"{prefix}logits/chosen"] = self.accelerator.gather_for_metrics(
            chosen_logits.detach().mean().reshape(1)
        ).mean().cpu()
        metrics[f"{prefix}logits/rejected"] = self.accelerator.gather_for_metrics(
            rejected_logits.detach().mean().reshape(1)
        ).mean().cpu()

        return loss, metrics
