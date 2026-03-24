#!/usr/bin/env python
# coding=utf-8
import logging

from alignment import DataArguments, H4ArgumentParser, ModelArguments, get_peft_config
from alpha_dpo_trainer import AlphaDPOTrainer
from run_preference_utils import finalize_training, prepare_preference_datasets, setup_run
from simpo_trainer import SimPOTrainer
from trainer_configs import SimPOConfig

logger = logging.getLogger(__name__)


def main():
    parser = H4ArgumentParser((ModelArguments, DataArguments, SimPOConfig))
    model_args, data_args, training_args = parser.parse()

    setup_run(model_args, data_args, training_args, logger)
    logger.info(f"Trainer type: {training_args.trainer_type}")
    if training_args.trainer_type == "alpha_dpo":
        logger.info(f"Alpha-DPO parameters: alpha={training_args.alpha}, ln={training_args.ln}")

    raw_datasets, tokenizer = prepare_preference_datasets(model_args, data_args, training_args, logger)

    if training_args.trainer_type == "simpo":
        trainer = SimPOTrainer(
            model=model_args.model_name_or_path,
            args=training_args,
            train_dataset=raw_datasets["train"],
            eval_dataset=raw_datasets["test"],
            tokenizer=tokenizer,
            peft_config=get_peft_config(model_args),
        )
        tags = ["alignment-handbook", "simpo"]
    elif training_args.trainer_type == "alpha_dpo":
        trainer = AlphaDPOTrainer(
            model=model_args.model_name_or_path,
            ref_model=model_args.model_name_or_path,
            args=training_args,
            train_dataset=raw_datasets["train"],
            eval_dataset=raw_datasets["test"],
            tokenizer=tokenizer,
            peft_config=get_peft_config(model_args),
        )
        tags = ["alignment-handbook", "alpha-dpo"]
    else:
        raise ValueError(f"Unknown trainer_type: {training_args.trainer_type}. Should be 'simpo' or 'alpha_dpo'.")

    finalize_training(
        trainer=trainer,
        training_args=training_args,
        model_args=model_args,
        data_args=data_args,
        raw_datasets=raw_datasets,
        run_logger=logger,
        tags=tags,
    )


if __name__ == "__main__":
    main()
