import logging
import sys
from pathlib import Path
from types import SimpleNamespace

from datasets import Dataset, DatasetDict

from alignment import DataArguments, H4ArgumentParser, ModelArguments, SFTConfig, get_tokenizer
from alignment.model_utils import tokenizer_needs_chat_format_setup
from utils import runtime as run_preference_utils

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from tokenized_dpo_trainer import PreferenceTokenizationProcessor
from trainer_configs import BetaDPOConfig, MarginDPOConfig


class BoslessTokenizer:
    def __init__(self):
        self.deprecation_warnings = {}
        self.bos_token_id = None
        self.eos_token_id = 999

    def _encode(self, text):
        return [ord(char) for char in text]

    def __call__(self, texts, add_special_tokens=False, truncation=False, max_length=None):
        del add_special_tokens
        single_input = isinstance(texts, str)
        if single_input:
            texts = [texts]

        input_ids = []
        attention_mask = []
        for text in texts:
            ids = self._encode(text)
            if truncation and max_length is not None:
                ids = ids[:max_length]
            input_ids.append(ids)
            attention_mask.append([1] * len(ids))

        if single_input:
            return {"input_ids": input_ids[0], "attention_mask": attention_mask[0]}
        return {"input_ids": input_ids, "attention_mask": attention_mask}


class NativeQwenLikeTokenizer:
    def __init__(self):
        self.chat_template = "<|im_start|>{% for message in messages %}{{ message['role'] }}{% endfor %}<|im_end|>"
        self.default_chat_template = None
        self.bos_token = None
        self.bos_token_id = None
        self.eos_token = "<|im_end|>"
        self.eos_token_id = 2
        self.pad_token_id = 2
        self.unk_token_id = 0
        self.name_or_path = "Qwen/Qwen3-8B-Base"

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        del tokenize, add_generation_prompt
        chunks = [f"<|im_start|>{message['role']}\n{message['content']}<|im_end|>" for message in messages]
        return "".join(chunks)

    def convert_tokens_to_ids(self, token):
        return {
            "<|im_start|>": 10,
            "<|im_end|>": 11,
        }.get(token, self.unk_token_id)


class MissingChatMlSetupTokenizer(NativeQwenLikeTokenizer):
    def __init__(self):
        super().__init__()
        self.pad_token_id = None

    def convert_tokens_to_ids(self, token):
        return self.unk_token_id


class ExistingTemplateTokenizer:
    def __init__(self):
        self.chat_template = "native-qwen-template"
        self.default_chat_template = None
        self.pad_token_id = None
        self.eos_token_id = 42
        self.model_max_length = 1_000_001


class PlainTokenizer:
    def __init__(self):
        self.chat_template = None
        self.default_chat_template = None
        self.pad_token_id = None
        self.eos_token_id = 7
        self.model_max_length = 4096


class RuntimeDummyTokenizer:
    def __init__(self):
        self.chat_template = "system"
        self.default_chat_template = None
        self.bos_token = "<s>"

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        del tokenize, add_generation_prompt
        rendered = " || ".join(f"{message['role']}:{message['content']}" for message in messages)
        return f"{self.bos_token}{rendered}"


def _raw_preference_dataset():
    return Dataset.from_list(
        [
            {
                "prompt": [{"role": "user", "content": "Question"}],
                "chosen": [{"role": "assistant", "content": "Answer"}],
                "rejected": [{"role": "assistant", "content": "Bad"}],
            }
        ]
    )


def test_decoder_only_tokenization_supports_bosless_tokenizers():
    processor = PreferenceTokenizationProcessor(
        tokenizer=BoslessTokenizer(),
        is_encoder_decoder=False,
        max_length=32,
        max_prompt_length=16,
        max_target_length=None,
        truncation_mode="keep_end",
        label_pad_token_id=-100,
        long_sequence_warning_key="sequence-length-is-longer-than-the-specified-maximum",
    )

    tokenized = processor.tokenize_batch(
        {
            "prompt": ["Prompt"],
            "chosen": [" chosen"],
            "rejected": [" rejected"],
        }
    )

    for key in ("prompt_input_ids", "chosen_input_ids", "rejected_input_ids"):
        assert None not in tokenized[key][0]
    assert tokenized["prompt_input_ids"][0][0] == ord("P")
    assert tokenized["chosen_input_ids"][0][-1] == 999
    assert tokenized["rejected_input_ids"][0][-1] == 999


def test_apply_preference_chat_template_preserves_native_qwen_template():
    tokenizer = NativeQwenLikeTokenizer()
    formatted = run_preference_utils.apply_preference_chat_template(
        {
            "prompt": [{"role": "user", "content": "Hi"}],
            "chosen": [{"role": "assistant", "content": "Hello"}],
            "rejected": [{"role": "assistant", "content": "No"}],
        },
        tokenizer=tokenizer,
        auto_insert_empty_system_msg=False,
    )

    assert formatted["text_prompt"] == "<|im_start|>user\nHi<|im_end|>"
    assert formatted["text_chosen"] == "<|im_start|>assistant\nHello<|im_end|>"
    assert formatted["text_rejected"] == "<|im_start|>assistant\nNo<|im_end|>"


def test_get_tokenizer_preserves_existing_native_chat_template(monkeypatch):
    tokenizer = ExistingTemplateTokenizer()
    monkeypatch.setattr("alignment.model_utils.AutoTokenizer.from_pretrained", lambda *args, **kwargs: tokenizer)

    model_args = ModelArguments(model_name_or_path="Qwen/Qwen2.5-7B")
    data_args = DataArguments()
    resolved = get_tokenizer(model_args, data_args)

    assert resolved.chat_template == "native-qwen-template"
    assert resolved.pad_token_id == resolved.eos_token_id
    assert resolved.model_max_length == 2048


def test_get_tokenizer_falls_back_to_repo_default_template(monkeypatch):
    tokenizer = PlainTokenizer()
    monkeypatch.setattr("alignment.model_utils.AutoTokenizer.from_pretrained", lambda *args, **kwargs: tokenizer)

    model_args = ModelArguments(model_name_or_path="plain/model")
    data_args = DataArguments()
    resolved = get_tokenizer(model_args, data_args)

    assert resolved.chat_template is not None
    assert "<|user|>" in resolved.chat_template


def test_qwen_like_chatml_tokenizer_does_not_need_setup():
    assert tokenizer_needs_chat_format_setup(NativeQwenLikeTokenizer()) is False


def test_tokenizer_without_chatml_tokens_still_needs_setup():
    assert tokenizer_needs_chat_format_setup(MissingChatMlSetupTokenizer()) is True


def test_prepare_preference_datasets_does_not_override_mistral_template(monkeypatch, tmp_path):
    tokenizer = RuntimeDummyTokenizer()

    monkeypatch.setattr(
        run_preference_utils,
        "get_datasets",
        lambda *args, **kwargs: DatasetDict({"train": _raw_preference_dataset()}),
    )
    monkeypatch.setattr(run_preference_utils, "get_tokenizer", lambda *args, **kwargs: tokenizer)
    monkeypatch.setattr(run_preference_utils, "build_model_init_kwargs", lambda *args, **kwargs: {"ok": True})
    monkeypatch.setattr(run_preference_utils, "configure_persistent_hf_cache", lambda *args, **kwargs: None)

    model_args = SimpleNamespace(
        model_name_or_path="mistralai/Mistral-7B-v0.1",
        torch_dtype=None,
        model_revision="main",
        trust_remote_code=False,
        attn_implementation=None,
    )
    data_args = DataArguments(preprocessing_log_samples=0)
    training_args = SimpleNamespace(
        output_dir=str(tmp_path / "output"),
        seed=17,
        process_index=0,
        gradient_checkpointing=False,
    )

    run_preference_utils.prepare_preference_datasets(
        model_args=model_args,
        data_args=data_args,
        training_args=training_args,
        run_logger=logging.getLogger("test_mistral_override"),
    )

    assert tokenizer.chat_template == "system"


def test_qwen25_sft_yaml_parses_without_custom_chat_template():
    parser = H4ArgumentParser((ModelArguments, DataArguments, SFTConfig))
    config_path = REPO_ROOT / "training_configs" / "qwen2.5-7b-base" / "sft" / "qwen2.5-7b-base-ultrachat-sft.yaml"

    model_args, data_args, training_args = parser.parse_yaml_file(str(config_path))

    assert model_args.model_name_or_path == "Qwen/Qwen2.5-7B"
    assert data_args.chat_template is None
    assert training_args.output_dir.endswith("qwen2.5-7b-base-ultrachat-sft-4xh100")


def test_qwen3_margin_dpo_yaml_parses_without_custom_chat_template():
    parser = H4ArgumentParser((ModelArguments, DataArguments, MarginDPOConfig))
    config_path = REPO_ROOT / "training_configs" / "qwen3-8b-base" / "dpo" / "qwen3-8b-base-margin-dpo.yaml"

    model_args, data_args, training_args = parser.parse_yaml_file(str(config_path))

    assert model_args.model_name_or_path == "outputs/qwen3-8b-base-ultrachat-sft-4xh100"
    assert data_args.chat_template is None
    assert model_args.attn_implementation == "flash_attention_2"


def test_qwen3_beta_dpo_smoke_yaml_parses_without_custom_chat_template():
    parser = H4ArgumentParser((ModelArguments, DataArguments, BetaDPOConfig))
    config_path = REPO_ROOT / "training_configs" / "qwen3-8b-base" / "dpo" / "qwen3-8b-base-beta-dpo-smoke.yaml"

    model_args, data_args, training_args = parser.parse_yaml_file(str(config_path))

    assert model_args.model_name_or_path == "outputs/qwen3-8b-base-ultrachat-sft-smoke"
    assert data_args.chat_template is None
    assert model_args.attn_implementation == "sdpa"
