#!/usr/bin/env python
import argparse
import json
import sys
import tempfile
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]


def _build_examples(raw_json: str):
    examples = json.loads(raw_json)
    if not isinstance(examples, list) or not examples:
        raise ValueError("examples_json must decode to a non-empty list.")
    for example in examples:
        for key in ("prompt", "chosen", "rejected"):
            if key not in example:
                raise ValueError(f"Each example must contain '{key}'.")
    return examples


def _common_training_kwargs(output_dir: str, beta: float, epsilon: float, grad_accum: int):
    return {
        "output_dir": output_dir,
        "per_device_train_batch_size": 2,
        "per_device_eval_batch_size": 2,
        "gradient_accumulation_steps": grad_accum,
        "max_length": 256,
        "max_prompt_length": 128,
        "remove_unused_columns": False,
        "disable_dropout": True,
        "report_to": [],
        "do_train": False,
        "do_eval": False,
        "seed": 0,
        "beta": beta,
        "epsilon": epsilon,
    }


def _dump_current(output_path: str, model_name_or_path: str, examples_json: str, beta: float, epsilon: float, grad_accum: int):
    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    from datasets import Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from epsilon_dpo_trainer import EpsilonDPOTrainer
    from trainer_configs import EpsilonDPOConfig

    examples = _build_examples(examples_json)
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_name_or_path)
    ref_model = AutoModelForCausalLM.from_pretrained(model_name_or_path)
    dataset = Dataset.from_list(examples)

    with tempfile.TemporaryDirectory(prefix="epsilon-dpo-current-") as output_dir:
        args = EpsilonDPOConfig(**_common_training_kwargs(output_dir, beta, epsilon, grad_accum))
        trainer = EpsilonDPOTrainer(
            model=model,
            ref_model=ref_model,
            args=args,
            train_dataset=dataset,
            eval_dataset=None,
            tokenizer=tokenizer,
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


def _dump_reference(output_path: str, model_name_or_path: str, examples_json: str, beta: float, epsilon: float, grad_accum: int):
    reference_root = REPO_ROOT / "reference-codes" / "e-dpo"
    sys.path.insert(0, str(reference_root))
    from datasets import Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from config import EpsilonDPOConfig
    from trainer import EpsilonDPOTrainer

    examples = _build_examples(examples_json)
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_name_or_path)
    ref_model = AutoModelForCausalLM.from_pretrained(model_name_or_path)
    reference_examples = [
        {
            "chosen": example["prompt"] + example["chosen"],
            "rejected": example["prompt"] + example["rejected"],
        }
        for example in examples
    ]
    dataset = Dataset.from_list(reference_examples)

    with tempfile.TemporaryDirectory(prefix="epsilon-dpo-reference-") as output_dir:
        args = EpsilonDPOConfig(**_common_training_kwargs(output_dir, beta, epsilon, grad_accum))
        trainer = EpsilonDPOTrainer(
            model=model,
            ref_model=ref_model,
            args=args,
            train_dataset=dataset,
            eval_dataset=None,
            processing_class=tokenizer,
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


def _compare(reference_dump: str, current_dump: str, atol: float, rtol: float):
    reference_payload = torch.load(reference_dump, map_location="cpu")
    current_payload = torch.load(current_dump, map_location="cpu")

    keys = [
        "chosen_logps",
        "rejected_logps",
        "ref_chosen_logps",
        "ref_rejected_logps",
        "steps",
        "losses",
        "chosen_rewards",
        "rejected_rewards",
        "beta_before",
        "beta_after_boundary",
    ]
    for key in keys:
        reference_value = reference_payload[key]
        current_value = current_payload[key]
        if not torch.allclose(reference_value.float(), current_value.float(), atol=atol, rtol=rtol):
            raise AssertionError(
                f"Mismatch for {key}\nreference={reference_value}\ncurrent={current_value}"
            )


def main():
    parser = argparse.ArgumentParser(description="Compare vendored original e-DPO and the repo-native 0.10.1 port.")
    parser.add_argument("--mode", choices=["current_dump", "reference_dump", "compare"], required=True)
    parser.add_argument("--output-path")
    parser.add_argument("--reference-dump")
    parser.add_argument("--current-dump")
    parser.add_argument("--reference-python")
    parser.add_argument("--model-name-or-path")
    parser.add_argument("--examples-json")
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--epsilon", type=float, default=0.01)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=2)
    parser.add_argument("--atol", type=float, default=1e-5)
    parser.add_argument("--rtol", type=float, default=1e-5)
    args = parser.parse_args()

    if args.mode == "current_dump":
        _dump_current(
            output_path=args.output_path,
            model_name_or_path=args.model_name_or_path,
            examples_json=args.examples_json,
            beta=args.beta,
            epsilon=args.epsilon,
            grad_accum=args.gradient_accumulation_steps,
        )
        return

    if args.mode == "reference_dump":
        _dump_reference(
            output_path=args.output_path,
            model_name_or_path=args.model_name_or_path,
            examples_json=args.examples_json,
            beta=args.beta,
            epsilon=args.epsilon,
            grad_accum=args.gradient_accumulation_steps,
        )
        return

    if args.reference_python is None:
        raise ValueError("--reference-python is required in compare mode.")
    if args.model_name_or_path is None or args.examples_json is None:
        raise ValueError("--model-name-or-path and --examples-json are required in compare mode.")

    with tempfile.TemporaryDirectory(prefix="epsilon-dpo-parity-") as tmpdir:
        tmpdir_path = Path(tmpdir)
        reference_dump = tmpdir_path / "reference.pt"
        current_dump = tmpdir_path / "current.pt"

        import subprocess

        subprocess.run(
            [
                args.reference_python,
                str(Path(__file__).resolve()),
                "--mode",
                "reference_dump",
                "--output-path",
                str(reference_dump),
                "--model-name-or-path",
                args.model_name_or_path,
                "--examples-json",
                args.examples_json,
                "--beta",
                str(args.beta),
                "--epsilon",
                str(args.epsilon),
                "--gradient-accumulation-steps",
                str(args.gradient_accumulation_steps),
            ],
            check=True,
        )
        _dump_current(
            output_path=str(current_dump),
            model_name_or_path=args.model_name_or_path,
            examples_json=args.examples_json,
            beta=args.beta,
            epsilon=args.epsilon,
            grad_accum=args.gradient_accumulation_steps,
        )
        _compare(str(reference_dump), str(current_dump), atol=args.atol, rtol=args.rtol)


if __name__ == "__main__":
    main()
