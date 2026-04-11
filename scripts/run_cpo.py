#!/usr/bin/env python
# coding=utf-8
import logging
import sys
from pathlib import Path

from trl import CPOTrainer

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from alignment import CPOConfig, DataArguments, H4ArgumentParser, ModelArguments
from utils.runtime import (
    finalize_training,
    prepare_pairwise_preference_datasets,
    prepare_trl_trainer_models,
    setup_run,
)

logger = logging.getLogger(__name__)


def main():
    parser = H4ArgumentParser((ModelArguments, DataArguments, CPOConfig))
    model_args, data_args, training_args = parser.parse()

    setup_run(model_args, data_args, training_args, logger)
    training_args.loss_type = "sigmoid"
    logger.info(
        "CPO parameters: beta=%s, cpo_alpha=%s",
        training_args.beta,
        training_args.cpo_alpha,
    )

    raw_datasets, tokenizer = prepare_pairwise_preference_datasets(model_args, data_args, training_args, logger)
    eval_dataset = raw_datasets["test"] if training_args.do_eval and "test" in raw_datasets else None
    model, _, peft_config = prepare_trl_trainer_models(
        model_args,
        training_args,
        logger,
        require_reference_model=False,
    )

    trainer = CPOTrainer(
        model=model,
        args=training_args,
        train_dataset=raw_datasets["train"],
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        peft_config=peft_config,
    )

    finalize_training(
        trainer=trainer,
        training_args=training_args,
        model_args=model_args,
        data_args=data_args,
        raw_datasets=raw_datasets,
        run_logger=logger,
        tags=["alignment-handbook", "cpo"],
    )


if __name__ == "__main__":
    main()
