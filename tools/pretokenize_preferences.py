#!/usr/bin/env python
# coding=utf-8
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Type

import yaml
from transformers import AutoConfig

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from alignment import DataArguments, H4ArgumentParser, ModelArguments
from tokenized_dpo_trainer import PreferenceTokenizationProcessor
from trainer_configs import BetaDPOConfig, MarginDPOConfig, SimPOConfig
from utils.preprocessing_cache import maybe_prepare_tokenized_datasets
from utils.runtime import prepare_preference_datasets, setup_run

logger = logging.getLogger(__name__)


@dataclass
class OfflineTokenizationAdapter:
    tokenizer: Any
    is_encoder_decoder: bool
    max_length: int
    max_prompt_length: int
    max_target_length: Optional[int]
    truncation_mode: str
    label_pad_token_id: int
    padding_value: int
    trainer_type: str
    tokenization_processor: PreferenceTokenizationProcessor

    def _tokenize_without_max_length_warning(self, text: str, **kwargs) -> Dict[str, Any]:
        return self.tokenization_processor._tokenize_without_max_length_warning(text, **kwargs)

    def build_tokenized_answer(self, prompt, answer):
        return self.tokenization_processor.build_tokenized_answer(prompt, answer)

    def tokenize_row(self, feature, model=None):
        return self.tokenization_processor.tokenize_row(feature, model=model)

    def tokenize_batch(self, features, model=None):
        return self.tokenization_processor.tokenize_batch(features, model=model)

    def get_tokenization_code_hashes(self) -> Dict[str, str]:
        return self.tokenization_processor.get_code_sources()


def _infer_training_config_class(config_path: str) -> Type:
    config_data = yaml.safe_load(Path(config_path).read_text()) or {}
    trainer_type = config_data.get("trainer_type")

    if trainer_type in {"simpo", "alpha_dpo"}:
        return SimPOConfig
    if trainer_type == "margin_dpo" or "margin_log_path" in config_data:
        return MarginDPOConfig
    return BetaDPOConfig


def _build_offline_adapter(model_args, training_args, tokenizer) -> OfflineTokenizationAdapter:
    is_encoder_decoder = training_args.is_encoder_decoder
    if is_encoder_decoder is None:
        model_config = AutoConfig.from_pretrained(model_args.model_name_or_path)
        is_encoder_decoder = model_config.is_encoder_decoder

    max_length = 512 if training_args.max_length is None else training_args.max_length
    max_prompt_length = 128 if training_args.max_prompt_length is None else training_args.max_prompt_length
    max_target_length = 128 if training_args.max_target_length is None and is_encoder_decoder else training_args.max_target_length
    padding_value = training_args.padding_value if training_args.padding_value is not None else tokenizer.pad_token_id

    processor = PreferenceTokenizationProcessor(
        tokenizer=tokenizer,
        is_encoder_decoder=is_encoder_decoder,
        max_length=max_length,
        max_prompt_length=max_prompt_length,
        max_target_length=max_target_length,
        truncation_mode=training_args.truncation_mode,
        label_pad_token_id=training_args.label_pad_token_id,
        long_sequence_warning_key="sequence-length-is-longer-than-the-specified-maximum",
    )
    trainer_type = getattr(training_args, "trainer_type", "beta_dpo")
    return OfflineTokenizationAdapter(
        tokenizer=tokenizer,
        is_encoder_decoder=is_encoder_decoder,
        max_length=max_length,
        max_prompt_length=max_prompt_length,
        max_target_length=max_target_length,
        truncation_mode=training_args.truncation_mode,
        label_pad_token_id=training_args.label_pad_token_id,
        padding_value=padding_value,
        trainer_type=trainer_type,
        tokenization_processor=processor,
    )


def main():
    if len(sys.argv) < 2:
        raise ValueError("Usage: pretokenize_preferences.py <config.yaml> [overrides...]")

    config_class = _infer_training_config_class(sys.argv[1])
    parser = H4ArgumentParser((ModelArguments, DataArguments, config_class))
    model_args, data_args, training_args = parser.parse()

    training_args.tokenization_mode = "offline_only"
    setup_run(model_args, data_args, training_args, logger)
    raw_datasets, tokenizer = prepare_preference_datasets(model_args, data_args, training_args, logger)

    adapter = _build_offline_adapter(model_args, training_args, tokenizer)
    train_dataset = raw_datasets["train"]
    eval_dataset = raw_datasets["test"] if "test" in raw_datasets else None

    maybe_prepare_tokenized_datasets(adapter, training_args, train_dataset, eval_dataset)
    logger.info("Offline tokenization complete.")


if __name__ == "__main__":
    main()
