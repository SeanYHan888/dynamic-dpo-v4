#!/usr/bin/env python
import argparse
import json
import logging
import math
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
def _build_examples(raw_json: str) -> List[Dict[str, str]]:
    examples = json.loads(raw_json)
    if not isinstance(examples, list) or not examples:
        raise ValueError("examples_json must decode to a non-empty list.")
    for example in examples:
        for key in ("prompt", "chosen", "rejected"):
            if key not in example:
                raise ValueError(f"Each example must contain '{key}'.")
            if not isinstance(example[key], str):
                raise ValueError(f"Each example field '{key}' must be a string.")
    return examples


def _common_training_kwargs(
    output_dir: str,
    beta: float,
    epsilon: float,
    grad_accum: int,
    max_length: int,
    max_prompt_length: int,
    learning_rate: float,
    train_batch_size: int,
) -> Dict[str, Any]:
    return {
        "output_dir": output_dir,
        "per_device_train_batch_size": train_batch_size,
        "per_device_eval_batch_size": train_batch_size,
        "gradient_accumulation_steps": grad_accum,
        "max_length": max_length,
        "max_prompt_length": max_prompt_length,
        "remove_unused_columns": False,
        "disable_dropout": True,
        "report_to": [],
        "do_train": False,
        "do_eval": False,
        "seed": 0,
        "beta": beta,
        "epsilon": epsilon,
        "learning_rate": learning_rate,
    }


def _load_examples_from_repo_config(
    config_path: str,
    split: str,
    max_examples: int | None,
    model_name_override: str | None,
    tokenizer_name_override: str | None,
) -> Dict[str, Any]:
    sys.path.insert(0, str(REPO_ROOT))
    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    from alignment import DataArguments, H4ArgumentParser, ModelArguments
    from trainer_configs import EpsilonDPOConfig
    from utils.runtime import prepare_preference_datasets

    parser = H4ArgumentParser((ModelArguments, DataArguments, EpsilonDPOConfig))
    model_args, data_args, training_args = parser.parse_yaml_file(str(Path(config_path).resolve()))

    if model_name_override is not None:
        model_args.model_name_or_path = model_name_override
    if tokenizer_name_override is not None:
        model_args.tokenizer_name_or_path = tokenizer_name_override

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(name)s - %(message)s")
    run_logger = logging.getLogger("epsilon_dpo_parity.config")
    raw_datasets, tokenizer = prepare_preference_datasets(model_args, data_args, training_args, run_logger)
    if split not in raw_datasets:
        raise ValueError(f"Split '{split}' not found in config dataset. Available: {list(raw_datasets.keys())}")

    split_dataset = raw_datasets[split]
    if max_examples is None:
        max_examples = len(split_dataset)
    max_examples = min(max_examples, len(split_dataset))
    examples = [
        {
            "prompt": split_dataset[i]["prompt"],
            "chosen": split_dataset[i]["chosen"],
            "rejected": split_dataset[i]["rejected"],
        }
        for i in range(max_examples)
    ]

    return {
        "examples": examples,
        "model_name_or_path": model_args.model_name_or_path,
        "tokenizer_name_or_path": model_args.tokenizer_name_or_path or model_args.model_name_or_path,
        "beta": float(training_args.beta),
        "epsilon": float(training_args.epsilon),
        "gradient_accumulation_steps": int(training_args.gradient_accumulation_steps),
        "per_device_train_batch_size": int(training_args.per_device_train_batch_size),
        "max_length": int(training_args.max_length),
        "max_prompt_length": int(training_args.max_prompt_length),
        "learning_rate": float(training_args.learning_rate),
        "tokenizer_name": getattr(tokenizer, "name_or_path", None),
    }


def _resolve_examples_and_hparams(args: argparse.Namespace) -> Dict[str, Any]:
    if args.examples_json is not None:
        examples = _build_examples(args.examples_json)
        return {
            "examples": examples,
            "model_name_or_path": args.model_name_or_path,
            "tokenizer_name_or_path": args.tokenizer_name_or_path or args.model_name_or_path,
            "beta": 0.1 if args.beta is None else args.beta,
            "epsilon": 0.01 if args.epsilon is None else args.epsilon,
            "gradient_accumulation_steps": 2 if args.gradient_accumulation_steps is None else args.gradient_accumulation_steps,
            "per_device_train_batch_size": 2 if args.train_batch_size is None else args.train_batch_size,
            "max_length": 256 if args.max_length is None else args.max_length,
            "max_prompt_length": 128 if args.max_prompt_length is None else args.max_prompt_length,
            "learning_rate": 1e-6 if args.learning_rate is None else args.learning_rate,
        }

    if args.config_path is None:
        raise ValueError("Provide either --examples-json or --config-path.")

    resolved = _load_examples_from_repo_config(
        config_path=args.config_path,
        split=args.dataset_split,
        max_examples=args.max_examples,
        model_name_override=args.model_name_or_path,
        tokenizer_name_override=args.tokenizer_name_or_path,
    )

    if args.beta is not None:
        resolved["beta"] = args.beta
    if args.epsilon is not None:
        resolved["epsilon"] = args.epsilon
    if args.gradient_accumulation_steps is not None:
        resolved["gradient_accumulation_steps"] = args.gradient_accumulation_steps
    if args.train_batch_size is not None:
        resolved["per_device_train_batch_size"] = args.train_batch_size
    if args.max_length is not None:
        resolved["max_length"] = args.max_length
    if args.max_prompt_length is not None:
        resolved["max_prompt_length"] = args.max_prompt_length
    if args.learning_rate is not None:
        resolved["learning_rate"] = args.learning_rate

    return resolved


def _scalarize(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return float(value.detach().cpu().item())
        return value.detach().cpu().float()
    if isinstance(value, (float, int, str, bool)) or value is None:
        return value
    return str(value)


def _optimizer_kwargs(args) -> Dict[str, Any]:
    return {
        "lr": float(args.learning_rate),
        "betas": (0.9, 0.999),
        "eps": 1e-8,
        "weight_decay": float(getattr(args, "weight_decay", 0.0)),
    }


def _set_sync_gradients(trainer: Any, sync_gradients: bool) -> None:
    current_state = getattr(trainer.accelerator, "gradient_state", None)
    if current_state is not None and hasattr(current_state, "sync_gradients"):
        try:
            current_state.sync_gradients = sync_gradients
            return
        except Exception:
            pass

    trainer.accelerator.gradient_state = SimpleNamespace(sync_gradients=sync_gradients)


def _build_current_trainer(
    model_name_or_path: str,
    tokenizer_name_or_path: str,
    examples: List[Dict[str, str]],
    train_kwargs: Dict[str, Any],
):
    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    from datasets import Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from epsilon_dpo_trainer import EpsilonDPOTrainer
    from trainer_configs import EpsilonDPOConfig

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_name_or_path)
    ref_model = AutoModelForCausalLM.from_pretrained(model_name_or_path)
    dataset = Dataset.from_list(examples)

    trainer = EpsilonDPOTrainer(
        model=model,
        ref_model=ref_model,
        args=EpsilonDPOConfig(**train_kwargs),
        train_dataset=dataset,
        eval_dataset=None,
        tokenizer=tokenizer,
    )
    return trainer


def _build_reference_trainer(
    model_name_or_path: str,
    tokenizer_name_or_path: str,
    examples: List[Dict[str, str]],
    train_kwargs: Dict[str, Any],
):
    reference_root = REPO_ROOT / "reference-codes" / "e-dpo"
    sys.path.insert(0, str(reference_root))
    from datasets import Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from config import EpsilonDPOConfig
    from trainer import EpsilonDPOTrainer

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_name_or_path)
    ref_model = AutoModelForCausalLM.from_pretrained(model_name_or_path)
    dataset = Dataset.from_list(
        [{"chosen": ex["prompt"] + ex["chosen"], "rejected": ex["prompt"] + ex["rejected"]} for ex in examples]
    )

    trainer = EpsilonDPOTrainer(
        model=model,
        ref_model=ref_model,
        args=EpsilonDPOConfig(**train_kwargs),
        train_dataset=dataset,
        eval_dataset=None,
        processing_class=tokenizer,
    )
    return trainer


def _freeze_model_state(model: torch.nn.Module) -> Dict[str, torch.Tensor]:
    return {name: param.detach().cpu().float().clone() for name, param in model.named_parameters()}


def _trim_row(row: torch.Tensor, mask: torch.Tensor | None = None, pad_value: int | None = None) -> List[int]:
    values = row.detach().cpu().tolist()
    if mask is not None:
        mask_values = mask.detach().cpu().tolist()
        return [value for value, keep in zip(values, mask_values) if int(keep) != 0]
    if pad_value is not None:
        return [value for value in values if value != pad_value]
    return values


def _decode_ids(tokenizer: Any, token_ids: List[int]) -> List[str]:
    return [tokenizer.decode([token_id], clean_up_tokenization_spaces=False) for token_id in token_ids]


def _loss_token_rows(
    tokenizer: Any,
    safe_labels: torch.Tensor,
    loss_mask: torch.Tensor,
    per_token_logps: torch.Tensor,
) -> List[List[Dict[str, Any]]]:
    rows: List[List[Dict[str, Any]]] = []
    for row_idx in range(safe_labels.shape[0]):
        row_entries: List[Dict[str, Any]] = []
        for position in range(safe_labels.shape[1]):
            if not bool(loss_mask[row_idx, position].item()):
                continue
            label_id = int(safe_labels[row_idx, position].item())
            row_entries.append(
                {
                    "position": position,
                    "label_id": label_id,
                    "label_text": tokenizer.decode([label_id], clean_up_tokenization_spaces=False),
                    "logp": float(per_token_logps[row_idx, position].detach().cpu().item()),
                }
            )
        rows.append(row_entries)
    return rows


def _reward_decomposition(
    trainer: Any,
    chosen_logps: torch.Tensor,
    rejected_logps: torch.Tensor,
    ref_chosen_logps: torch.Tensor,
    ref_rejected_logps: torch.Tensor,
    steps: torch.Tensor,
) -> Dict[str, Any]:
    losses, chosen_rewards, rejected_rewards = trainer.dpo_loss(
        chosen_logps,
        rejected_logps,
        ref_chosen_logps,
        ref_rejected_logps,
        steps,
    )
    updated_beta = trainer.beta / (1 + trainer.epsilon * steps.to(chosen_logps.dtype))
    reward_margin = chosen_rewards - rejected_rewards
    reward_accuracy = (chosen_rewards > rejected_rewards).float()
    return {
        "chosen_logps": chosen_logps.detach().cpu(),
        "rejected_logps": rejected_logps.detach().cpu(),
        "ref_chosen_logps": ref_chosen_logps.detach().cpu(),
        "ref_rejected_logps": ref_rejected_logps.detach().cpu(),
        "steps": steps.detach().cpu(),
        "updated_beta": updated_beta.detach().cpu(),
        "losses": losses.detach().cpu(),
        "chosen_rewards": chosen_rewards.detach().cpu(),
        "rejected_rewards": rejected_rewards.detach().cpu(),
        "reward_margins": reward_margin.detach().cpu(),
        "reward_accuracies": reward_accuracy.detach().cpu(),
    }


def _current_example_token_view(trainer: Any, batch: Dict[str, torch.Tensor]) -> List[Dict[str, Any]]:
    tokenizer = trainer.tokenizer
    example_views: List[Dict[str, Any]] = []
    batch_size = batch["prompt_input_ids"].shape[0]
    for idx in range(batch_size):
        prompt_mask = batch["prompt_attention_mask"][idx]
        prompt_ids = _trim_row(batch["prompt_input_ids"][idx], mask=prompt_mask)
        chosen_mask = batch["chosen_attention_mask"][idx]
        rejected_mask = batch["rejected_attention_mask"][idx]
        chosen_ids = _trim_row(batch["chosen_input_ids"][idx], mask=chosen_mask)
        rejected_ids = _trim_row(batch["rejected_input_ids"][idx], mask=rejected_mask)

        prompt_len = len(prompt_ids)
        chosen_completion_ids = chosen_ids[prompt_len:]
        rejected_completion_ids = rejected_ids[prompt_len:]

        example_views.append(
            {
                "prompt_ids": prompt_ids,
                "prompt_tokens": _decode_ids(tokenizer, prompt_ids),
                "chosen_full_ids": chosen_ids,
                "chosen_full_tokens": _decode_ids(tokenizer, chosen_ids),
                "chosen_completion_ids": chosen_completion_ids,
                "chosen_completion_tokens": _decode_ids(tokenizer, chosen_completion_ids),
                "rejected_full_ids": rejected_ids,
                "rejected_full_tokens": _decode_ids(tokenizer, rejected_ids),
                "rejected_completion_ids": rejected_completion_ids,
                "rejected_completion_tokens": _decode_ids(tokenizer, rejected_completion_ids),
            }
        )
    return example_views


def _reference_debug_forward(trainer: Any, model: torch.nn.Module, batch: Dict[str, torch.Tensor]) -> Dict[str, Any]:
    concatenated_batch = trainer.concatenated_inputs(batch, padding_value=trainer.padding_value)
    num_examples = batch["prompt_input_ids"].shape[0]
    prompt_input_ids = concatenated_batch["prompt_input_ids"]
    prompt_attention_mask = concatenated_batch["prompt_attention_mask"]
    completion_input_ids = concatenated_batch["completion_input_ids"]
    completion_attention_mask = concatenated_batch["completion_attention_mask"]

    debug: Dict[str, Any] = {
        "concatenated_prompt_input_ids": prompt_input_ids.detach().cpu(),
        "concatenated_prompt_attention_mask": prompt_attention_mask.detach().cpu(),
        "concatenated_completion_input_ids": completion_input_ids.detach().cpu(),
        "concatenated_completion_attention_mask": completion_attention_mask.detach().cpu(),
    }

    if trainer.is_encoder_decoder:
        raise NotImplementedError("Reference scoring debug currently supports decoder-only models only.")

    input_ids = torch.cat((prompt_input_ids, completion_input_ids), dim=1)
    attention_mask = torch.cat((prompt_attention_mask, completion_attention_mask), dim=1)
    loss_mask = torch.cat((torch.zeros_like(prompt_attention_mask), completion_attention_mask), dim=1)

    for i in range(attention_mask.size(0)):
        first_one_idx = torch.nonzero(attention_mask[i])[0].item()
        input_ids[i] = torch.roll(input_ids[i], shifts=-first_one_idx)
        attention_mask[i] = torch.roll(attention_mask[i], shifts=-first_one_idx)
        loss_mask[i] = torch.roll(loss_mask[i], shifts=-first_one_idx)

    empty_cols = torch.sum(attention_mask, dim=0) == 0
    first_empty_col = torch.nonzero(empty_cols)[0].item() if empty_cols.any() else attention_mask.size(1)
    input_ids = input_ids[:, :first_empty_col]
    attention_mask = attention_mask[:, :first_empty_col]
    loss_mask = loss_mask[:, :first_empty_col]

    if trainer.args.max_length is not None:
        input_ids = input_ids[:, : trainer.args.max_length]
        attention_mask = attention_mask[:, : trainer.args.max_length]
        loss_mask = loss_mask[:, : trainer.args.max_length]

    outputs = model(input_ids)
    logits = outputs.logits
    labels = torch.roll(input_ids, shifts=-1, dims=1)
    loss_mask = torch.roll(loss_mask, shifts=-1, dims=1).bool()

    if logits.shape[:2] != labels.shape[:2]:
        seq_len = labels.shape[1]
        logits = logits[:, -seq_len:]

    labels[~loss_mask] = 0
    per_token_logps_unrolled = torch.gather(logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)).squeeze(2)
    per_token_logps_unrolled[~loss_mask] = 0
    per_token_logps = torch.roll(per_token_logps_unrolled, shifts=1, dims=1)
    all_logps = per_token_logps.sum(-1)

    debug.update(
        {
            "input_ids": input_ids.detach().cpu(),
            "attention_mask": attention_mask.detach().cpu(),
            "labels": labels.detach().cpu(),
            "loss_mask": loss_mask.detach().cpu(),
            "per_token_logps_unrolled": per_token_logps_unrolled.detach().cpu(),
            "per_token_logps": per_token_logps.detach().cpu(),
            "sequence_logps": all_logps.detach().cpu(),
        }
    )

    tokenizer = trainer.processing_class
    example_views: List[Dict[str, Any]] = []
    for idx in range(num_examples):
        prompt_ids = _trim_row(prompt_input_ids[idx], mask=prompt_attention_mask[idx])
        chosen_completion_ids = _trim_row(completion_input_ids[idx], mask=completion_attention_mask[idx])
        rejected_completion_ids = _trim_row(
            completion_input_ids[idx + num_examples],
            mask=completion_attention_mask[idx + num_examples],
        )
        example_views.append(
            {
                "prompt_ids": prompt_ids,
                "prompt_tokens": _decode_ids(tokenizer, prompt_ids),
                "chosen_completion_ids": chosen_completion_ids,
                "chosen_completion_tokens": _decode_ids(tokenizer, chosen_completion_ids),
                "rejected_completion_ids": rejected_completion_ids,
                "rejected_completion_tokens": _decode_ids(tokenizer, rejected_completion_ids),
            }
        )
    debug["example_token_view"] = example_views
    debug["loss_token_rows"] = _loss_token_rows(tokenizer, labels.detach().cpu(), loss_mask.detach().cpu(), per_token_logps.detach().cpu())
    return debug


def _current_scoring_debug_payload(trainer: Any) -> Dict[str, Any]:
    batch = trainer.data_collator([trainer.train_dataset[i] for i in range(len(trainer.train_dataset))])
    policy_output = trainer._epsilon_forward(trainer.model, batch, logit_source="policy")
    reference_output = trainer.compute_reference_model_output(batch)
    len_chosen = policy_output["chosen_logps"].shape[0]
    p_epsilon_logits = ((1 + trainer.epsilon) * policy_output["scoring_logits"]) - (
        trainer.epsilon * reference_output["scoring_logits"]
    )
    n_epsilon_logits = ((1 - trainer.epsilon) * policy_output["scoring_logits"]) + (
        trainer.epsilon * reference_output["scoring_logits"]
    )
    p_epsilon_scores = trainer._score_aligned_logits(
        p_epsilon_logits,
        policy_output["safe_labels"],
        policy_output["loss_mask"],
    )
    n_epsilon_scores = trainer._score_aligned_logits(
        n_epsilon_logits,
        policy_output["safe_labels"],
        policy_output["loss_mask"],
    )
    policy_logratios = policy_output["chosen_logps"] - policy_output["rejected_logps"]
    p_epsilon_logratios = p_epsilon_scores[:len_chosen] - p_epsilon_scores[len_chosen:]
    n_epsilon_logratios = n_epsilon_scores[:len_chosen] - n_epsilon_scores[len_chosen:]
    steps = trainer._compute_steps(policy_logratios, p_epsilon_logratios, n_epsilon_logratios)
    concatenated_batch = trainer.concatenated_inputs(
        batch,
        is_encoder_decoder=trainer.is_encoder_decoder,
        label_pad_token_id=trainer.label_pad_token_id,
        padding_value=trainer.padding_value,
        device=torch.device("cpu"),
    )

    payload = {
        "example_token_view": _current_example_token_view(trainer, batch),
        "batch": {key: value.detach().cpu() if isinstance(value, torch.Tensor) else value for key, value in batch.items()},
        "concatenated_batch": {
            key: value.detach().cpu() if isinstance(value, torch.Tensor) else value
            for key, value in concatenated_batch.items()
        },
        "reward_decomposition": _reward_decomposition(
            trainer,
            policy_output["chosen_logps"],
            policy_output["rejected_logps"],
            reference_output["chosen_logps"],
            reference_output["rejected_logps"],
            steps,
        ),
        "policy": {
            "safe_labels": policy_output["safe_labels"].detach().cpu(),
            "loss_mask": policy_output["loss_mask"].detach().cpu(),
            "per_token_logps": policy_output["per_token_logps"].detach().cpu(),
            "loss_token_rows": _loss_token_rows(
                trainer.tokenizer,
                policy_output["safe_labels"].detach().cpu(),
                policy_output["loss_mask"].detach().cpu(),
                policy_output["per_token_logps"].detach().cpu(),
            ),
        },
        "reference": {
            "safe_labels": reference_output["safe_labels"].detach().cpu(),
            "loss_mask": reference_output["loss_mask"].detach().cpu(),
            "per_token_logps": reference_output["per_token_logps"].detach().cpu(),
            "loss_token_rows": _loss_token_rows(
                trainer.tokenizer,
                reference_output["safe_labels"].detach().cpu(),
                reference_output["loss_mask"].detach().cpu(),
                reference_output["per_token_logps"].detach().cpu(),
            ),
        },
    }
    return payload


def _reference_scoring_debug_payload(trainer: Any) -> Dict[str, Any]:
    batch = trainer.data_collator([trainer.train_dataset[i] for i in range(len(trainer.train_dataset))])
    policy_debug = _reference_debug_forward(trainer, trainer.model, batch)
    reference_debug = _reference_debug_forward(trainer, trainer.ref_model, batch)
    ref_model_output = trainer.compute_ref_log_probs(batch)
    model_output = trainer.concatenated_forward(trainer.model, batch, ref_model_output["logits"])
    payload = {
        "batch": {key: value.detach().cpu() if isinstance(value, torch.Tensor) else value for key, value in batch.items()},
        "policy": policy_debug,
        "reference": reference_debug,
        "reward_decomposition": _reward_decomposition(
            trainer,
            model_output["chosen_logps"],
            model_output["rejected_logps"],
            ref_model_output["chosen_logps"],
            ref_model_output["rejected_logps"],
            model_output["steps"],
        ),
    }
    return payload


def _first_loss_token_mismatch(
    reference_rows: List[List[Dict[str, Any]]],
    current_rows: List[List[Dict[str, Any]]],
    atol: float,
    rtol: float,
) -> Dict[str, Any] | None:
    for row_idx, (reference_row, current_row) in enumerate(zip(reference_rows, current_rows)):
        if len(reference_row) != len(current_row):
            return {
                "row_index": row_idx,
                "reason": "loss_token_count_mismatch",
                "reference_count": len(reference_row),
                "current_count": len(current_row),
            }
        for token_idx, (reference_token, current_token) in enumerate(zip(reference_row, current_row)):
            if reference_token["label_id"] != current_token["label_id"]:
                return {
                    "row_index": row_idx,
                    "token_index": token_idx,
                    "reason": "label_id_mismatch",
                    "reference": reference_token,
                    "current": current_token,
                }
            if not math.isclose(reference_token["logp"], current_token["logp"], abs_tol=atol, rel_tol=rtol):
                return {
                    "row_index": row_idx,
                    "token_index": token_idx,
                    "reason": "logp_mismatch",
                    "reference": reference_token,
                    "current": current_token,
                    "abs_diff": abs(reference_token["logp"] - current_token["logp"]),
                }
    return None


def _scoring_debug_summary(reference_payload: Dict[str, Any], current_payload: Dict[str, Any], atol: float, rtol: float) -> Dict[str, Any]:
    reward_order_flip = None
    reference_reward_decomp = reference_payload["reward_decomposition"]
    current_reward_decomp = current_payload["reward_decomposition"]
    reference_accuracies = reference_reward_decomp["reward_accuracies"].tolist()
    current_accuracies = current_reward_decomp["reward_accuracies"].tolist()
    for example_index, (reference_accuracy, current_accuracy) in enumerate(zip(reference_accuracies, current_accuracies)):
        if bool(reference_accuracy) != bool(current_accuracy):
            reward_order_flip = {
                "example_index": example_index,
                "reference": {
                    key: _scalarize(value[example_index]) if isinstance(value, torch.Tensor) and value.ndim > 0 else _scalarize(value)
                    for key, value in reference_reward_decomp.items()
                },
                "current": {
                    key: _scalarize(value[example_index]) if isinstance(value, torch.Tensor) and value.ndim > 0 else _scalarize(value)
                    for key, value in current_reward_decomp.items()
                },
            }
            break

    summary: Dict[str, Any] = {
        "example_token_view_matches": reference_payload["policy"]["example_token_view"] == current_payload["example_token_view"],
        "policy_loss_token_mismatch": _first_loss_token_mismatch(
            reference_payload["policy"]["loss_token_rows"],
            current_payload["policy"]["loss_token_rows"],
            atol=atol,
            rtol=rtol,
        ),
        "reference_loss_token_mismatch": _first_loss_token_mismatch(
            reference_payload["reference"]["loss_token_rows"],
            current_payload["reference"]["loss_token_rows"],
            atol=atol,
            rtol=rtol,
        ),
        "first_reward_order_flip": reward_order_flip,
    }
    return summary


def _model_state_summary(model: torch.nn.Module) -> Dict[str, float]:
    total_sum = 0.0
    total_abs_sum = 0.0
    total_sq_sum = 0.0
    max_abs = 0.0
    for _, param in model.named_parameters():
        tensor = param.detach().cpu().float()
        total_sum += float(tensor.sum().item())
        total_abs_sum += float(tensor.abs().sum().item())
        total_sq_sum += float((tensor * tensor).sum().item())
        if tensor.numel() > 0:
            max_abs = max(max_abs, float(tensor.abs().max().item()))
    return {
        "sum": total_sum,
        "abs_sum": total_abs_sum,
        "sq_sum": total_sq_sum,
        "max_abs": max_abs,
    }


def _manual_training_trajectory(
    trainer: Any,
    optimizer: torch.optim.Optimizer,
    optimizer_steps: int,
    train_batch_size: int,
    grad_accum: int,
    include_model_state: bool,
) -> Dict[str, Any]:
    required_examples = optimizer_steps * train_batch_size * grad_accum
    if len(trainer.train_dataset) < required_examples:
        raise ValueError(
            f"Trajectory parity requires at least {required_examples} examples but dataset only has "
            f"{len(trainer.train_dataset)}. Increase --max-examples or reduce steps/batch size/grad-accum."
        )

    trainer.model.train()
    if trainer.ref_model is not None:
        trainer.ref_model.eval()

    microbatch_records: List[Dict[str, Any]] = []
    optimizer_step_records: List[Dict[str, Any]] = []
    dataset_index = 0

    for optimizer_step_index in range(optimizer_steps):
        optimizer.zero_grad(set_to_none=True)
        optimizer_step_loss = 0.0
        beta_before_step = float(trainer.beta)

        for accumulation_index in range(grad_accum):
            batch_examples = [
                trainer.train_dataset[i]
                for i in range(dataset_index, dataset_index + train_batch_size)
            ]
            dataset_index += train_batch_size
            batch = trainer.data_collator(batch_examples)
            sync_gradients = accumulation_index == (grad_accum - 1)
            _set_sync_gradients(trainer, sync_gradients)

            beta_before_forward = float(trainer.beta)
            loss, metrics = trainer.get_batch_loss_metrics(trainer.model, batch, train_eval="train")
            scaled_loss = loss / float(grad_accum)
            scaled_loss.backward()
            beta_after_forward = float(trainer.beta)
            optimizer_step_loss += float(loss.detach().cpu().item())

            microbatch_records.append(
                {
                    "optimizer_step": optimizer_step_index,
                    "accumulation_index": accumulation_index,
                    "dataset_index_start": dataset_index - train_batch_size,
                    "sync_gradients": sync_gradients,
                    "loss": float(loss.detach().cpu().item()),
                    "beta_before_forward": beta_before_forward,
                    "beta_after_forward": beta_after_forward,
                    "metrics": {key: _scalarize(value) for key, value in metrics.items()},
                }
            )

        optimizer.step()
        optimizer_step_records.append(
            {
                "optimizer_step": optimizer_step_index,
                "beta_before_step": beta_before_step,
                "beta_after_step": float(trainer.beta),
                "optimizer_step_loss_sum": optimizer_step_loss,
                "model_state_summary": _model_state_summary(trainer.model),
            }
        )

    payload: Dict[str, Any] = {
        "microbatch_records": microbatch_records,
        "optimizer_step_records": optimizer_step_records,
        "final_beta": float(trainer.beta),
        "final_model_state_summary": _model_state_summary(trainer.model),
    }
    if include_model_state:
        payload["final_model_state"] = _freeze_model_state(trainer.model)
    return payload


def _dump_current(
    output_path: str,
    model_name_or_path: str,
    tokenizer_name_or_path: str,
    examples_json: str,
    beta: float,
    epsilon: float,
    grad_accum: int,
    max_length: int,
    max_prompt_length: int,
    learning_rate: float,
    train_batch_size: int,
):
    examples = _build_examples(examples_json)
    with tempfile.TemporaryDirectory(prefix="epsilon-dpo-current-") as output_dir:
        train_kwargs = _common_training_kwargs(
            output_dir=output_dir,
            beta=beta,
            epsilon=epsilon,
            grad_accum=grad_accum,
            max_length=max_length,
            max_prompt_length=max_prompt_length,
            learning_rate=learning_rate,
            train_batch_size=train_batch_size,
        )
        trainer = _build_current_trainer(
            model_name_or_path=model_name_or_path,
            tokenizer_name_or_path=tokenizer_name_or_path,
            examples=examples,
            train_kwargs=train_kwargs,
        )

        batch = trainer.data_collator([trainer.train_dataset[i] for i in range(len(trainer.train_dataset))])
        policy_output = trainer._epsilon_forward(trainer.model, batch, logit_source="policy")
        reference_output = trainer.compute_reference_model_output(batch)

        p_epsilon_logits = ((1 + trainer.epsilon) * policy_output["scoring_logits"]) - (
            trainer.epsilon * reference_output["scoring_logits"]
        )
        n_epsilon_logits = ((1 - trainer.epsilon) * policy_output["scoring_logits"]) + (
            trainer.epsilon * reference_output["scoring_logits"]
        )
        p_epsilon_logps = trainer._score_aligned_logits(
            p_epsilon_logits,
            policy_output["safe_labels"],
            policy_output["loss_mask"],
        )
        n_epsilon_logps = trainer._score_aligned_logits(
            n_epsilon_logits,
            policy_output["safe_labels"],
            policy_output["loss_mask"],
        )

        len_chosen = policy_output["chosen_logps"].shape[0]
        logratios = policy_output["chosen_logps"] - policy_output["rejected_logps"]
        p_epsilon_logratios = p_epsilon_logps[:len_chosen] - p_epsilon_logps[len_chosen:]
        n_epsilon_logratios = n_epsilon_logps[:len_chosen] - n_epsilon_logps[len_chosen:]
        steps = trainer._compute_steps(logratios, p_epsilon_logratios, n_epsilon_logratios)

        losses, chosen_rewards, rejected_rewards = trainer.dpo_loss(
            policy_output["chosen_logps"],
            policy_output["rejected_logps"],
            reference_output["chosen_logps"],
            reference_output["rejected_logps"],
            steps,
        )

        payload = {
            "chosen_logps": policy_output["chosen_logps"].cpu(),
            "rejected_logps": policy_output["rejected_logps"].cpu(),
            "ref_chosen_logps": reference_output["chosen_logps"].cpu(),
            "ref_rejected_logps": reference_output["rejected_logps"].cpu(),
            "steps": steps.cpu(),
            "losses": losses.cpu(),
            "chosen_rewards": chosen_rewards.cpu(),
            "rejected_rewards": rejected_rewards.cpu(),
            "beta_before": torch.tensor(float(trainer.beta)),
            "beta_after_boundary": torch.tensor(float(trainer.beta / (1 + steps.float().mean().item() * trainer.epsilon))),
        }
        torch.save(payload, output_path)


def _dump_reference(
    output_path: str,
    model_name_or_path: str,
    tokenizer_name_or_path: str,
    examples_json: str,
    beta: float,
    epsilon: float,
    grad_accum: int,
    max_length: int,
    max_prompt_length: int,
    learning_rate: float,
    train_batch_size: int,
):
    examples = _build_examples(examples_json)
    with tempfile.TemporaryDirectory(prefix="epsilon-dpo-reference-") as output_dir:
        train_kwargs = _common_training_kwargs(
            output_dir=output_dir,
            beta=beta,
            epsilon=epsilon,
            grad_accum=grad_accum,
            max_length=max_length,
            max_prompt_length=max_prompt_length,
            learning_rate=learning_rate,
            train_batch_size=train_batch_size,
        )
        trainer = _build_reference_trainer(
            model_name_or_path=model_name_or_path,
            tokenizer_name_or_path=tokenizer_name_or_path,
            examples=examples,
            train_kwargs=train_kwargs,
        )

        batch = trainer.data_collator([trainer.train_dataset[i] for i in range(len(trainer.train_dataset))])
        reference_output = trainer.compute_ref_log_probs(batch)
        policy_output = trainer.concatenated_forward(trainer.model, batch, reference_output["logits"])
        losses, chosen_rewards, rejected_rewards = trainer.dpo_loss(
            policy_output["chosen_logps"],
            policy_output["rejected_logps"],
            reference_output["chosen_logps"],
            reference_output["rejected_logps"],
            policy_output["steps"],
        )

        payload = {
            "chosen_logps": policy_output["chosen_logps"].cpu(),
            "rejected_logps": policy_output["rejected_logps"].cpu(),
            "ref_chosen_logps": reference_output["chosen_logps"].cpu(),
            "ref_rejected_logps": reference_output["rejected_logps"].cpu(),
            "steps": policy_output["steps"].cpu(),
            "losses": losses.cpu(),
            "chosen_rewards": chosen_rewards.cpu(),
            "rejected_rewards": rejected_rewards.cpu(),
            "beta_before": torch.tensor(float(trainer.beta)),
            "beta_after_boundary": torch.tensor(float(trainer.beta / (1 + policy_output["steps"].float().mean().item() * trainer.epsilon))),
        }
        torch.save(payload, output_path)


def _dump_current_scoring_debug(
    output_path: str,
    model_name_or_path: str,
    tokenizer_name_or_path: str,
    examples_json: str,
    beta: float,
    epsilon: float,
    grad_accum: int,
    max_length: int,
    max_prompt_length: int,
    learning_rate: float,
    train_batch_size: int,
):
    examples = _build_examples(examples_json)
    with tempfile.TemporaryDirectory(prefix="epsilon-dpo-current-debug-") as output_dir:
        train_kwargs = _common_training_kwargs(
            output_dir=output_dir,
            beta=beta,
            epsilon=epsilon,
            grad_accum=grad_accum,
            max_length=max_length,
            max_prompt_length=max_prompt_length,
            learning_rate=learning_rate,
            train_batch_size=train_batch_size,
        )
        trainer = _build_current_trainer(
            model_name_or_path=model_name_or_path,
            tokenizer_name_or_path=tokenizer_name_or_path,
            examples=examples,
            train_kwargs=train_kwargs,
        )
        torch.save(_current_scoring_debug_payload(trainer), output_path)


def _dump_reference_scoring_debug(
    output_path: str,
    model_name_or_path: str,
    tokenizer_name_or_path: str,
    examples_json: str,
    beta: float,
    epsilon: float,
    grad_accum: int,
    max_length: int,
    max_prompt_length: int,
    learning_rate: float,
    train_batch_size: int,
):
    examples = _build_examples(examples_json)
    with tempfile.TemporaryDirectory(prefix="epsilon-dpo-reference-debug-") as output_dir:
        train_kwargs = _common_training_kwargs(
            output_dir=output_dir,
            beta=beta,
            epsilon=epsilon,
            grad_accum=grad_accum,
            max_length=max_length,
            max_prompt_length=max_prompt_length,
            learning_rate=learning_rate,
            train_batch_size=train_batch_size,
        )
        trainer = _build_reference_trainer(
            model_name_or_path=model_name_or_path,
            tokenizer_name_or_path=tokenizer_name_or_path,
            examples=examples,
            train_kwargs=train_kwargs,
        )
        torch.save(_reference_scoring_debug_payload(trainer), output_path)


def _dump_current_trajectory(
    output_path: str,
    model_name_or_path: str,
    tokenizer_name_or_path: str,
    examples_json: str,
    beta: float,
    epsilon: float,
    grad_accum: int,
    max_length: int,
    max_prompt_length: int,
    learning_rate: float,
    train_batch_size: int,
    optimizer_steps: int,
    include_model_state: bool,
):
    examples = _build_examples(examples_json)
    with tempfile.TemporaryDirectory(prefix="epsilon-dpo-current-traj-") as output_dir:
        train_kwargs = _common_training_kwargs(
            output_dir=output_dir,
            beta=beta,
            epsilon=epsilon,
            grad_accum=grad_accum,
            max_length=max_length,
            max_prompt_length=max_prompt_length,
            learning_rate=learning_rate,
            train_batch_size=train_batch_size,
        )
        trainer = _build_current_trainer(
            model_name_or_path=model_name_or_path,
            tokenizer_name_or_path=tokenizer_name_or_path,
            examples=examples,
            train_kwargs=train_kwargs,
        )
        optimizer = torch.optim.AdamW(trainer.model.parameters(), **_optimizer_kwargs(trainer.args))
        payload = _manual_training_trajectory(
            trainer=trainer,
            optimizer=optimizer,
            optimizer_steps=optimizer_steps,
            train_batch_size=train_batch_size,
            grad_accum=grad_accum,
            include_model_state=include_model_state,
        )
        torch.save(payload, output_path)


def _dump_reference_trajectory(
    output_path: str,
    model_name_or_path: str,
    tokenizer_name_or_path: str,
    examples_json: str,
    beta: float,
    epsilon: float,
    grad_accum: int,
    max_length: int,
    max_prompt_length: int,
    learning_rate: float,
    train_batch_size: int,
    optimizer_steps: int,
    include_model_state: bool,
):
    examples = _build_examples(examples_json)
    with tempfile.TemporaryDirectory(prefix="epsilon-dpo-reference-traj-") as output_dir:
        train_kwargs = _common_training_kwargs(
            output_dir=output_dir,
            beta=beta,
            epsilon=epsilon,
            grad_accum=grad_accum,
            max_length=max_length,
            max_prompt_length=max_prompt_length,
            learning_rate=learning_rate,
            train_batch_size=train_batch_size,
        )
        trainer = _build_reference_trainer(
            model_name_or_path=model_name_or_path,
            tokenizer_name_or_path=tokenizer_name_or_path,
            examples=examples,
            train_kwargs=train_kwargs,
        )
        optimizer = torch.optim.AdamW(trainer.model.parameters(), **_optimizer_kwargs(trainer.args))
        payload = _manual_training_trajectory(
            trainer=trainer,
            optimizer=optimizer,
            optimizer_steps=optimizer_steps,
            train_batch_size=train_batch_size,
            grad_accum=grad_accum,
            include_model_state=include_model_state,
        )
        torch.save(payload, output_path)


def _assert_close(reference_value: Any, current_value: Any, path: str, atol: float, rtol: float) -> None:
    if isinstance(reference_value, torch.Tensor):
        if not isinstance(current_value, torch.Tensor):
            raise AssertionError(f"Type mismatch at {path}: expected tensor, got {type(current_value)}")
        if reference_value.dtype == torch.bool and current_value.dtype == torch.bool:
            if not torch.equal(reference_value, current_value):
                raise AssertionError(f"Mismatch at {path}\nreference={reference_value}\ncurrent={current_value}")
            return
        if not torch.allclose(reference_value.float(), current_value.float(), atol=atol, rtol=rtol):
            raise AssertionError(f"Mismatch at {path}\nreference={reference_value}\ncurrent={current_value}")
        return

    if isinstance(reference_value, dict):
        if not isinstance(current_value, dict):
            raise AssertionError(f"Type mismatch at {path}: expected dict, got {type(current_value)}")
        if set(reference_value.keys()) != set(current_value.keys()):
            raise AssertionError(
                f"Key mismatch at {path}\nreference={sorted(reference_value.keys())}\ncurrent={sorted(current_value.keys())}"
            )
        for key in sorted(reference_value.keys()):
            _assert_close(reference_value[key], current_value[key], f"{path}.{key}", atol, rtol)
        return

    if isinstance(reference_value, (list, tuple)):
        if not isinstance(current_value, (list, tuple)):
            raise AssertionError(f"Type mismatch at {path}: expected sequence, got {type(current_value)}")
        if len(reference_value) != len(current_value):
            raise AssertionError(f"Length mismatch at {path}: reference={len(reference_value)} current={len(current_value)}")
        for index, (reference_item, current_item) in enumerate(zip(reference_value, current_value)):
            _assert_close(reference_item, current_item, f"{path}[{index}]", atol, rtol)
        return

    if isinstance(reference_value, (float, int)) and isinstance(current_value, (float, int)):
        if not math.isclose(float(reference_value), float(current_value), abs_tol=atol, rel_tol=rtol):
            raise AssertionError(f"Mismatch at {path}\nreference={reference_value}\ncurrent={current_value}")
        return

    if reference_value != current_value:
        raise AssertionError(f"Mismatch at {path}\nreference={reference_value}\ncurrent={current_value}")


def _compare(reference_dump: str, current_dump: str, atol: float, rtol: float) -> None:
    reference_payload = torch.load(reference_dump, map_location="cpu")
    current_payload = torch.load(current_dump, map_location="cpu")
    _assert_close(reference_payload, current_payload, path="payload", atol=atol, rtol=rtol)


def _run_compare_subprocess(
    args: argparse.Namespace,
    mode: str,
    resolved: Dict[str, Any],
    reference_dump: Path,
    current_dump: Path,
) -> None:
    import subprocess

    common_cli = [
        "--output-path",
        str(reference_dump),
        "--model-name-or-path",
        resolved["model_name_or_path"],
        "--tokenizer-name-or-path",
        resolved["tokenizer_name_or_path"],
        "--examples-json",
        json.dumps(resolved["examples"]),
        "--beta",
        str(resolved["beta"]),
        "--epsilon",
        str(resolved["epsilon"]),
        "--gradient-accumulation-steps",
        str(resolved["gradient_accumulation_steps"]),
        "--max-length",
        str(resolved["max_length"]),
        "--max-prompt-length",
        str(resolved["max_prompt_length"]),
        "--learning-rate",
        str(resolved["learning_rate"]),
        "--train-batch-size",
        str(resolved["per_device_train_batch_size"]),
    ]

    if mode in {"reference_trajectory_dump", "current_trajectory_dump"}:
        common_cli.extend(
            [
                "--trajectory-optimizer-steps",
                str(args.trajectory_optimizer_steps),
                "--trajectory-include-model-state",
                str(args.trajectory_include_model_state).lower(),
            ]
        )

    subprocess.run(
        [args.reference_python, str(Path(__file__).resolve()), "--mode", mode] + common_cli,
        check=True,
    )

    if mode == "reference_dump":
        _dump_current(
            output_path=str(current_dump),
            model_name_or_path=resolved["model_name_or_path"],
            tokenizer_name_or_path=resolved["tokenizer_name_or_path"],
            examples_json=json.dumps(resolved["examples"]),
            beta=resolved["beta"],
            epsilon=resolved["epsilon"],
            grad_accum=resolved["gradient_accumulation_steps"],
            max_length=resolved["max_length"],
            max_prompt_length=resolved["max_prompt_length"],
            learning_rate=resolved["learning_rate"],
            train_batch_size=resolved["per_device_train_batch_size"],
        )
        return

    _dump_current_trajectory(
        output_path=str(current_dump),
        model_name_or_path=resolved["model_name_or_path"],
        tokenizer_name_or_path=resolved["tokenizer_name_or_path"],
        examples_json=json.dumps(resolved["examples"]),
        beta=resolved["beta"],
        epsilon=resolved["epsilon"],
        grad_accum=resolved["gradient_accumulation_steps"],
        max_length=resolved["max_length"],
        max_prompt_length=resolved["max_prompt_length"],
        learning_rate=resolved["learning_rate"],
        train_batch_size=resolved["per_device_train_batch_size"],
        optimizer_steps=args.trajectory_optimizer_steps,
        include_model_state=args.trajectory_include_model_state,
    )


def _run_compare_scoring_debug(
    args: argparse.Namespace,
    resolved: Dict[str, Any],
    reference_dump: Path,
    current_dump: Path,
) -> Dict[str, Any]:
    import subprocess

    common_cli = [
        "--model-name-or-path",
        resolved["model_name_or_path"],
        "--tokenizer-name-or-path",
        resolved["tokenizer_name_or_path"],
        "--examples-json",
        json.dumps(resolved["examples"]),
        "--beta",
        str(resolved["beta"]),
        "--epsilon",
        str(resolved["epsilon"]),
        "--gradient-accumulation-steps",
        str(resolved["gradient_accumulation_steps"]),
        "--max-length",
        str(resolved["max_length"]),
        "--max-prompt-length",
        str(resolved["max_prompt_length"]),
        "--learning-rate",
        str(resolved["learning_rate"]),
        "--train-batch-size",
        str(resolved["per_device_train_batch_size"]),
    ]

    subprocess.run(
        [
            args.reference_python,
            str(Path(__file__).resolve()),
            "--mode",
            "reference_scoring_debug_dump",
            "--output-path",
            str(reference_dump),
        ] + common_cli,
        check=True,
    )

    _dump_current_scoring_debug(
        output_path=str(current_dump),
        model_name_or_path=resolved["model_name_or_path"],
        tokenizer_name_or_path=resolved["tokenizer_name_or_path"],
        examples_json=json.dumps(resolved["examples"]),
        beta=resolved["beta"],
        epsilon=resolved["epsilon"],
        grad_accum=resolved["gradient_accumulation_steps"],
        max_length=resolved["max_length"],
        max_prompt_length=resolved["max_prompt_length"],
        learning_rate=resolved["learning_rate"],
        train_batch_size=resolved["per_device_train_batch_size"],
    )

    reference_payload = torch.load(reference_dump, map_location="cpu")
    current_payload = torch.load(current_dump, map_location="cpu")
    return {
        "reference_dump": str(reference_dump),
        "current_dump": str(current_dump),
        "summary": _scoring_debug_summary(reference_payload, current_payload, atol=args.atol, rtol=args.rtol),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare vendored original e-DPO and the repo-native 0.10.1 port.")
    parser.add_argument(
        "--mode",
        choices=[
            "current_dump",
            "reference_dump",
            "compare",
            "current_scoring_debug_dump",
            "reference_scoring_debug_dump",
            "compare_scoring_debug",
            "current_trajectory_dump",
            "reference_trajectory_dump",
            "compare_trajectory",
        ],
        required=True,
    )
    parser.add_argument("--output-path")
    parser.add_argument("--reference-python")
    parser.add_argument("--model-name-or-path")
    parser.add_argument("--tokenizer-name-or-path")
    parser.add_argument("--examples-json")
    parser.add_argument("--config-path")
    parser.add_argument("--dataset-split", default="train")
    parser.add_argument("--max-examples", type=int)
    parser.add_argument("--beta", type=float)
    parser.add_argument("--epsilon", type=float)
    parser.add_argument("--gradient-accumulation-steps", type=int)
    parser.add_argument("--max-length", type=int)
    parser.add_argument("--max-prompt-length", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--train-batch-size", type=int)
    parser.add_argument("--trajectory-optimizer-steps", type=int, default=2)
    parser.add_argument("--trajectory-include-model-state", type=lambda value: value.lower() == "true", default=True)
    parser.add_argument("--atol", type=float, default=1e-5)
    parser.add_argument("--rtol", type=float, default=1e-5)
    args = parser.parse_args()

    resolved = None
    if args.mode in {"compare", "compare_scoring_debug", "compare_trajectory"}:
        if args.reference_python is None:
            raise ValueError("--reference-python is required in compare modes.")
        resolved = _resolve_examples_and_hparams(args)
        if resolved["model_name_or_path"] is None:
            raise ValueError("A model must be provided either explicitly or through --config-path.")

    if args.mode in {
        "current_dump",
        "reference_dump",
        "current_scoring_debug_dump",
        "reference_scoring_debug_dump",
        "current_trajectory_dump",
        "reference_trajectory_dump",
    }:
        if args.output_path is None:
            raise ValueError("--output-path is required for dump modes.")
        resolved = _resolve_examples_and_hparams(args)
        if resolved["model_name_or_path"] is None:
            raise ValueError("A model must be provided either explicitly or through --config-path.")

    if args.mode == "current_dump":
        _dump_current(
            output_path=args.output_path,
            model_name_or_path=resolved["model_name_or_path"],
            tokenizer_name_or_path=resolved["tokenizer_name_or_path"],
            examples_json=json.dumps(resolved["examples"]),
            beta=resolved["beta"],
            epsilon=resolved["epsilon"],
            grad_accum=resolved["gradient_accumulation_steps"],
            max_length=resolved["max_length"],
            max_prompt_length=resolved["max_prompt_length"],
            learning_rate=resolved["learning_rate"],
            train_batch_size=resolved["per_device_train_batch_size"],
        )
        return

    if args.mode == "reference_dump":
        _dump_reference(
            output_path=args.output_path,
            model_name_or_path=resolved["model_name_or_path"],
            tokenizer_name_or_path=resolved["tokenizer_name_or_path"],
            examples_json=json.dumps(resolved["examples"]),
            beta=resolved["beta"],
            epsilon=resolved["epsilon"],
            grad_accum=resolved["gradient_accumulation_steps"],
            max_length=resolved["max_length"],
            max_prompt_length=resolved["max_prompt_length"],
            learning_rate=resolved["learning_rate"],
            train_batch_size=resolved["per_device_train_batch_size"],
        )
        return

    if args.mode == "current_scoring_debug_dump":
        _dump_current_scoring_debug(
            output_path=args.output_path,
            model_name_or_path=resolved["model_name_or_path"],
            tokenizer_name_or_path=resolved["tokenizer_name_or_path"],
            examples_json=json.dumps(resolved["examples"]),
            beta=resolved["beta"],
            epsilon=resolved["epsilon"],
            grad_accum=resolved["gradient_accumulation_steps"],
            max_length=resolved["max_length"],
            max_prompt_length=resolved["max_prompt_length"],
            learning_rate=resolved["learning_rate"],
            train_batch_size=resolved["per_device_train_batch_size"],
        )
        return

    if args.mode == "reference_scoring_debug_dump":
        _dump_reference_scoring_debug(
            output_path=args.output_path,
            model_name_or_path=resolved["model_name_or_path"],
            tokenizer_name_or_path=resolved["tokenizer_name_or_path"],
            examples_json=json.dumps(resolved["examples"]),
            beta=resolved["beta"],
            epsilon=resolved["epsilon"],
            grad_accum=resolved["gradient_accumulation_steps"],
            max_length=resolved["max_length"],
            max_prompt_length=resolved["max_prompt_length"],
            learning_rate=resolved["learning_rate"],
            train_batch_size=resolved["per_device_train_batch_size"],
        )
        return

    if args.mode == "current_trajectory_dump":
        _dump_current_trajectory(
            output_path=args.output_path,
            model_name_or_path=resolved["model_name_or_path"],
            tokenizer_name_or_path=resolved["tokenizer_name_or_path"],
            examples_json=json.dumps(resolved["examples"]),
            beta=resolved["beta"],
            epsilon=resolved["epsilon"],
            grad_accum=resolved["gradient_accumulation_steps"],
            max_length=resolved["max_length"],
            max_prompt_length=resolved["max_prompt_length"],
            learning_rate=resolved["learning_rate"],
            train_batch_size=resolved["per_device_train_batch_size"],
            optimizer_steps=args.trajectory_optimizer_steps,
            include_model_state=args.trajectory_include_model_state,
        )
        return

    if args.mode == "reference_trajectory_dump":
        _dump_reference_trajectory(
            output_path=args.output_path,
            model_name_or_path=resolved["model_name_or_path"],
            tokenizer_name_or_path=resolved["tokenizer_name_or_path"],
            examples_json=json.dumps(resolved["examples"]),
            beta=resolved["beta"],
            epsilon=resolved["epsilon"],
            grad_accum=resolved["gradient_accumulation_steps"],
            max_length=resolved["max_length"],
            max_prompt_length=resolved["max_prompt_length"],
            learning_rate=resolved["learning_rate"],
            train_batch_size=resolved["per_device_train_batch_size"],
            optimizer_steps=args.trajectory_optimizer_steps,
            include_model_state=args.trajectory_include_model_state,
        )
        return

    with tempfile.TemporaryDirectory(prefix="epsilon-dpo-parity-") as tmpdir:
        tmpdir_path = Path(tmpdir)
        reference_dump = tmpdir_path / "reference.pt"
        current_dump = tmpdir_path / "current.pt"

        if args.mode == "compare":
            _run_compare_subprocess(
                args=args,
                mode="reference_dump",
                resolved=resolved,
                reference_dump=reference_dump,
                current_dump=current_dump,
            )
            _compare(str(reference_dump), str(current_dump), atol=args.atol, rtol=args.rtol)
            return

        if args.mode == "compare_scoring_debug":
            debug_result = _run_compare_scoring_debug(
                args=args,
                resolved=resolved,
                reference_dump=reference_dump,
                current_dump=current_dump,
            )
            print(json.dumps(debug_result, indent=2))
            return

        _run_compare_subprocess(
            args=args,
            mode="reference_trajectory_dump",
            resolved=resolved,
            reference_dump=reference_dump,
            current_dump=current_dump,
        )
        _compare(str(reference_dump), str(current_dump), atol=args.atol, rtol=args.rtol)


if __name__ == "__main__":
    main()
