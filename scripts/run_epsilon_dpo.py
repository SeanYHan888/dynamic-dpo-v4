#!/usr/bin/env python
# coding=utf-8
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from alignment import DataArguments, H4ArgumentParser, ModelArguments, get_peft_config
from epsilon_dpo_trainer import EpsilonDPOTrainer
from trainer_configs import EpsilonDPOConfig
from utils.runtime import finalize_training, prepare_preference_datasets, setup_run

logger = logging.getLogger(__name__)


def main():
    # Reuse the same runner shape as the other trainers so CLI, dataset preparation, and model
    # loading stay uniform across algorithms.
    parser = H4ArgumentParser((ModelArguments, DataArguments, EpsilonDPOConfig))
    model_args, data_args, training_args = parser.parse()

    setup_run(model_args, data_args, training_args, logger)
    logger.info(
        "Epsilon-DPO parameters: "
        f"beta={training_args.beta}, epsilon={training_args.epsilon}, "
        f"gradient_accumulation_steps={training_args.gradient_accumulation_steps}"
    )
    raw_datasets, tokenizer = prepare_preference_datasets(model_args, data_args, training_args, logger)
    eval_dataset = raw_datasets["test"] if training_args.do_eval and "test" in raw_datasets else None

    # Pass both policy and frozen reference models explicitly: ε-DPO needs live reference logits for
    # step estimation, so it cannot rely on reference-free mode or cached ref sequence scores alone.
    trainer = EpsilonDPOTrainer(
        model=model_args.model_name_or_path,
        ref_model=model_args.model_name_or_path,
        args=training_args,
        train_dataset=raw_datasets["train"],
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        peft_config=get_peft_config(model_args),
    )

    finalize_training(
        trainer=trainer,
        training_args=training_args,
        model_args=model_args,
        data_args=data_args,
        raw_datasets=raw_datasets,
        run_logger=logger,
        tags=["alignment-handbook", "epsilon-dpo"],
    )


if __name__ == "__main__":
    main()
