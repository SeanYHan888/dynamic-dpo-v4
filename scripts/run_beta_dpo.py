#!/usr/bin/env python
# coding=utf-8
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from alignment import DataArguments, H4ArgumentParser, ModelArguments, get_peft_config
from beta_dpo_trainer import BetaDPOTrainer
from run_preference_utils import finalize_training, prepare_preference_datasets, setup_run
from trainer_configs import BetaDPOConfig

logger = logging.getLogger(__name__)


def main():
    parser = H4ArgumentParser((ModelArguments, DataArguments, BetaDPOConfig))
    model_args, data_args, training_args = parser.parse()

    setup_run(model_args, data_args, training_args, logger)
    logger.info(
        "Beta-DPO parameters: "
        f"beta={training_args.beta}, rho={training_args.rho}, alpha={training_args.alpha}, "
        f"ema_momentum={training_args.ema_momentum}"
    )
    raw_datasets, tokenizer = prepare_preference_datasets(model_args, data_args, training_args, logger)
    eval_dataset = raw_datasets["test"] if training_args.do_eval and "test" in raw_datasets else None

    trainer = BetaDPOTrainer(
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
        tags=["alignment-handbook", "beta-dpo"],
    )


if __name__ == "__main__":
    main()
