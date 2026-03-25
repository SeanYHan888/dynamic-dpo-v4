from __future__ import annotations

from dataclasses import dataclass
import inspect
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
import torch
import torch.nn as nn


@dataclass
class PreferenceTokenizationProcessor:
    tokenizer: Any
    is_encoder_decoder: bool
    max_length: int
    max_prompt_length: int
    max_target_length: Optional[int]
    truncation_mode: str
    label_pad_token_id: int
    long_sequence_warning_key: str

    def _tokenize_without_max_length_warning(self, texts: Union[str, List[str]], **kwargs) -> Dict[str, Any]:
        previous = self.tokenizer.deprecation_warnings.get(self.long_sequence_warning_key, False)
        self.tokenizer.deprecation_warnings[self.long_sequence_warning_key] = True
        try:
            return self.tokenizer(texts, **kwargs)
        finally:
            self.tokenizer.deprecation_warnings[self.long_sequence_warning_key] = previous

    def build_tokenized_answer(
        self,
        prompt: str,
        answer: str,
        prompt_input_ids: Optional[List[int]] = None,
        prompt_attention_mask: Optional[List[int]] = None,
    ) -> Dict[str, List[int]]:
        if prompt_input_ids is None or prompt_attention_mask is None:
            prompt_tokenized = self._tokenize_without_max_length_warning(prompt, add_special_tokens=False)
            prompt_input_ids = prompt_tokenized["input_ids"]
            prompt_attention_mask = prompt_tokenized["attention_mask"]

        full_tokenized = self._tokenize_without_max_length_warning(prompt + answer, add_special_tokens=False)
        return self._build_answer_tokens_from_full(
            prompt_input_ids=prompt_input_ids,
            prompt_attention_mask=prompt_attention_mask,
            full_input_ids=full_tokenized["input_ids"],
            full_attention_mask=full_tokenized["attention_mask"],
        )

    def tokenize_row(self, feature: Dict[str, Any], model: Optional[Union[nn.Module, Any]] = None) -> Dict[str, Any]:
        tokenized = self.tokenize_batch(
            {
                "prompt": [feature["prompt"]],
                "chosen": [feature["chosen"]],
                "rejected": [feature["rejected"]],
            },
            model=model,
        )
        return {key: value[0] for key, value in tokenized.items()}

    def tokenize_batch(
        self,
        features: Dict[str, Sequence[Any]],
        model: Optional[Union[nn.Module, Any]] = None,
    ) -> Dict[str, List[Any]]:
        prompts = list(features["prompt"])
        chosen = list(features["chosen"])
        rejected = list(features["rejected"])

        if not self.is_encoder_decoder:
            return self._tokenize_decoder_only_batch(prompts, chosen, rejected)

        return self._tokenize_encoder_decoder_batch(prompts, chosen, rejected, model=model)

    def _tokenize_decoder_only_batch(
        self,
        prompts: List[str],
        chosen: List[str],
        rejected: List[str],
    ) -> Dict[str, List[Any]]:
        for name, values in (("prompt", prompts), ("chosen", chosen), ("rejected", rejected)):
            if not all(isinstance(value, str) for value in values):
                bad_type = next(type(value) for value in values if not isinstance(value, str))
                raise ValueError(f"{name} should be an str but got {bad_type}")

        prompt_batch = self._tokenize_without_max_length_warning(prompts, add_special_tokens=False)
        chosen_full_batch = self._tokenize_without_max_length_warning(
            [prompt + answer for prompt, answer in zip(prompts, chosen)],
            add_special_tokens=False,
        )
        rejected_full_batch = self._tokenize_without_max_length_warning(
            [prompt + answer for prompt, answer in zip(prompts, rejected)],
            add_special_tokens=False,
        )

        output: Dict[str, List[Any]] = {
            "chosen_input_ids": [],
            "chosen_attention_mask": [],
            "chosen_labels": [],
            "rejected_input_ids": [],
            "rejected_attention_mask": [],
            "rejected_labels": [],
            "prompt_input_ids": [],
            "prompt_attention_mask": [],
        }

        for idx in range(len(prompts)):
            prompt_tokens = {
                "prompt_input_ids": list(prompt_batch["input_ids"][idx]),
                "prompt_attention_mask": list(prompt_batch["attention_mask"][idx]),
            }
            chosen_tokens = self._build_answer_tokens_from_full(
                prompt_input_ids=prompt_tokens["prompt_input_ids"],
                prompt_attention_mask=prompt_tokens["prompt_attention_mask"],
                full_input_ids=list(chosen_full_batch["input_ids"][idx]),
                full_attention_mask=list(chosen_full_batch["attention_mask"][idx]),
            )
            rejected_tokens = self._build_answer_tokens_from_full(
                prompt_input_ids=prompt_tokens["prompt_input_ids"],
                prompt_attention_mask=prompt_tokens["prompt_attention_mask"],
                full_input_ids=list(rejected_full_batch["input_ids"][idx]),
                full_attention_mask=list(rejected_full_batch["attention_mask"][idx]),
            )

            example = self._finalize_decoder_only_example(prompt_tokens, chosen_tokens, rejected_tokens)
            for key, value in example.items():
                output[key].append(value)

        return output

    def _tokenize_encoder_decoder_batch(
        self,
        prompts: List[str],
        chosen: List[str],
        rejected: List[str],
        model: Optional[Union[nn.Module, Any]] = None,
    ) -> Dict[str, List[Any]]:
        chosen_tokens = self.tokenizer(
            chosen,
            truncation=True,
            max_length=self.max_target_length,
            add_special_tokens=True,
        )
        rejected_tokens = self.tokenizer(
            rejected,
            truncation=True,
            max_length=self.max_target_length,
            add_special_tokens=True,
        )
        prompt_tokens = self.tokenizer(
            prompts,
            truncation=True,
            max_length=self.max_prompt_length,
            add_special_tokens=True,
        )

        batch: Dict[str, List[Any]] = {
            "chosen_labels": [list(tokens) for tokens in chosen_tokens["input_ids"]],
            "rejected_labels": [list(tokens) for tokens in rejected_tokens["input_ids"]],
            "prompt_input_ids": [list(tokens) for tokens in prompt_tokens["input_ids"]],
            "prompt_attention_mask": [list(tokens) for tokens in prompt_tokens["attention_mask"]],
        }

        if model is not None and hasattr(model, "prepare_decoder_input_ids_from_labels"):
            batch["rejected_decoder_input_ids"] = [
                model.prepare_decoder_input_ids_from_labels(labels=torch.tensor(labels)).tolist()
                for labels in batch["rejected_labels"]
            ]
            batch["chosen_decoder_input_ids"] = [
                model.prepare_decoder_input_ids_from_labels(labels=torch.tensor(labels)).tolist()
                for labels in batch["chosen_labels"]
            ]

        return batch

    def get_code_sources(self) -> Dict[str, str]:
        return {
            "tokenize_row": inspect.getsource(self.tokenize_row),
            "tokenize_batch": inspect.getsource(self.tokenize_batch),
            "build_tokenized_answer": inspect.getsource(self.build_tokenized_answer),
        }

    def _build_answer_tokens_from_full(
        self,
        prompt_input_ids: List[int],
        prompt_attention_mask: List[int],
        full_input_ids: List[int],
        full_attention_mask: List[int],
    ) -> Dict[str, List[int]]:
        answer_input_ids = full_input_ids[len(prompt_input_ids) :]
        answer_attention_mask = full_attention_mask[len(prompt_input_ids) :]
        full_concat_input_ids = np.concatenate([prompt_input_ids, answer_input_ids])
        full_input_ids_array = np.array(full_input_ids)

        if len(full_input_ids_array) != len(full_concat_input_ids):
            raise ValueError("Prompt input ids and answer input ids should have the same length.")

        response_token_ids_start_idx = len(prompt_input_ids)
        if prompt_input_ids != full_input_ids[:response_token_ids_start_idx]:
            response_token_ids_start_idx -= 1

        corrected_prompt_input_ids = full_input_ids[:response_token_ids_start_idx]
        corrected_prompt_attention_mask = full_attention_mask[:response_token_ids_start_idx]

        if len(corrected_prompt_input_ids) != len(corrected_prompt_attention_mask):
            raise ValueError("Prompt input ids and attention mask should have the same length.")

        corrected_answer_input_ids = full_input_ids[response_token_ids_start_idx:]
        corrected_answer_attention_mask = full_attention_mask[response_token_ids_start_idx:]

        return {
            "prompt_input_ids": corrected_prompt_input_ids,
            "prompt_attention_mask": corrected_prompt_attention_mask,
            "input_ids": corrected_answer_input_ids,
            "attention_mask": corrected_answer_attention_mask,
        }

    def _finalize_decoder_only_example(
        self,
        prompt_tokens: Dict[str, List[int]],
        chosen_tokens: Dict[str, List[int]],
        rejected_tokens: Dict[str, List[int]],
    ) -> Dict[str, List[int]]:
        prompt_len_input_ids = len(prompt_tokens["prompt_input_ids"])
        chosen_prompt_len_input_ids = len(chosen_tokens["prompt_input_ids"])
        rejected_prompt_len_input_ids = len(rejected_tokens["prompt_input_ids"])
        prompt_len_input_ids = min(chosen_prompt_len_input_ids, rejected_prompt_len_input_ids)

        prompt_tokens = {
            key: value[:prompt_len_input_ids]
            for key, value in prompt_tokens.items()
        }
        chosen_tokens = {
            key: list(value)
            for key, value in chosen_tokens.items()
        }
        rejected_tokens = {
            key: list(value)
            for key, value in rejected_tokens.items()
        }

        num_diff_tokens = sum(
            a != b for a, b in zip(chosen_tokens["prompt_input_ids"], rejected_tokens["prompt_input_ids"])
        )
        num_diff_len = abs(chosen_prompt_len_input_ids - rejected_prompt_len_input_ids)
        if num_diff_tokens > 1 or num_diff_len > 1:
            raise ValueError(
                "Chosen and rejected prompt_input_ids might only differ on the last token due to tokenizer merge ops."
            )

        bos_token_id = self.tokenizer.bos_token_id
        if prompt_len_input_ids == 0 or bos_token_id != prompt_tokens["prompt_input_ids"][0]:
            prompt_tokens["prompt_input_ids"] = [bos_token_id] + prompt_tokens["prompt_input_ids"]
            prompt_tokens["prompt_attention_mask"] = [1] + prompt_tokens["prompt_attention_mask"]
        if chosen_prompt_len_input_ids == 0 or bos_token_id != chosen_tokens["prompt_input_ids"][0]:
            chosen_tokens["prompt_input_ids"] = [bos_token_id] + chosen_tokens["prompt_input_ids"]
            chosen_tokens["prompt_attention_mask"] = [1] + chosen_tokens["prompt_attention_mask"]
        if rejected_prompt_len_input_ids == 0 or bos_token_id != rejected_tokens["prompt_input_ids"][0]:
            rejected_tokens["prompt_input_ids"] = [bos_token_id] + rejected_tokens["prompt_input_ids"]
            rejected_tokens["prompt_attention_mask"] = [1] + rejected_tokens["prompt_attention_mask"]

        eos_token_id = self.tokenizer.eos_token_id
        if len(chosen_tokens["input_ids"]) == 0 or eos_token_id != chosen_tokens["input_ids"][-1]:
            chosen_tokens["input_ids"].append(eos_token_id)
            chosen_tokens["attention_mask"].append(1)
        if len(rejected_tokens["input_ids"]) == 0 or eos_token_id != rejected_tokens["input_ids"][-1]:
            rejected_tokens["input_ids"].append(eos_token_id)
            rejected_tokens["attention_mask"].append(1)

        longer_response_length = max(len(chosen_tokens["input_ids"]), len(rejected_tokens["input_ids"]))

        for answer_tokens in [chosen_tokens, rejected_tokens, prompt_tokens]:
            if len(answer_tokens["prompt_input_ids"]) + longer_response_length > self.max_length:
                if self.truncation_mode == "keep_start":
                    for key in ["prompt_input_ids", "prompt_attention_mask"]:
                        answer_tokens[key] = answer_tokens[key][: self.max_prompt_length]
                elif self.truncation_mode == "keep_end":
                    for key in ["prompt_input_ids", "prompt_attention_mask"]:
                        answer_tokens[key] = answer_tokens[key][-self.max_prompt_length :]
                else:
                    raise ValueError(f"Unknown truncation mode: {self.truncation_mode}")

        for answer_tokens in [chosen_tokens, rejected_tokens]:
            if len(answer_tokens["prompt_input_ids"]) + longer_response_length > self.max_length:
                for key in ["input_ids", "attention_mask"]:
                    answer_tokens[key] = answer_tokens[key][: self.max_length - self.max_prompt_length]

        chosen_sequence_tokens = {
            key: chosen_tokens[f"prompt_{key}"] + chosen_tokens[key]
            for key in ["input_ids", "attention_mask"]
        }
        rejected_sequence_tokens = {
            key: rejected_tokens[f"prompt_{key}"] + rejected_tokens[key]
            for key in ["input_ids", "attention_mask"]
        }
        chosen_sequence_tokens["labels"] = chosen_sequence_tokens["input_ids"][:]
        chosen_sequence_tokens["labels"][: len(chosen_tokens["prompt_input_ids"])] = [
            self.label_pad_token_id
        ] * len(chosen_tokens["prompt_input_ids"])
        rejected_sequence_tokens["labels"] = rejected_sequence_tokens["input_ids"][:]
        rejected_sequence_tokens["labels"][: len(rejected_tokens["prompt_input_ids"])] = [
            self.label_pad_token_id
        ] * len(rejected_tokens["prompt_input_ids"])

        batch: Dict[str, List[int]] = {}
        for prefix, toks in {
            "chosen_": chosen_sequence_tokens,
            "rejected_": rejected_sequence_tokens,
            "": prompt_tokens,
        }.items():
            for type_key, tokens in toks.items():
                if type_key == "token_type_ids":
                    continue
                batch[f"{prefix}{type_key}"] = tokens
        return batch
