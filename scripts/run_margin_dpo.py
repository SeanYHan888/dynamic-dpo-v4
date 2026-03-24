#!/usr/bin/env python
# coding=utf-8
import logging

from alignment import DataArguments, H4ArgumentParser, ModelArguments, get_peft_config
from margin_dpo_trainer import MarginDPOTrainer
from run_preference_utils import finalize_training, prepare_preference_datasets, setup_run
from trainer_configs import MarginDPOConfig

logger = logging.getLogger(__name__)


def main():
    parser = H4ArgumentParser((ModelArguments, DataArguments, MarginDPOConfig))
    model_args, data_args, training_args = parser.parse()

    setup_run(model_args, data_args, training_args, logger)
    logger.info(
        "Margin-DPO parameters: "
        f"beta={training_args.beta}, f_divergence_type={training_args.f_divergence_type}, "
        f"margin_log_steps={training_args.margin_log_steps}"
    )
    raw_datasets, tokenizer = prepare_preference_datasets(model_args, data_args, training_args, logger)

    trainer = MarginDPOTrainer(
        model=model_args.model_name_or_path,
        ref_model=model_args.model_name_or_path,
        args=training_args,
        train_dataset=raw_datasets["train"],
        eval_dataset=raw_datasets["test"],
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
        tags=["alignment-handbook", "margin-dpo"],
    )


if __name__ == "__main__":
    main()
