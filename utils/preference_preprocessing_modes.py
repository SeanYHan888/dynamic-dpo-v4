import hashlib
import inspect
from typing import Dict, Tuple

from alignment.data import is_openai_format, maybe_insert_system_message

MISTRAL_CHAT_TEMPLATE = "{% if messages[0]['role'] == 'system' %}{% set loop_messages = messages[1:] %}{% set system_message = messages[0]['content'].strip() + '\n\n' %}{% else %}{% set loop_messages = messages %}{% set system_message = '' %}{% endif %}{% for message in loop_messages %}{% if loop.index0 == 0 %}{% set content = system_message + message['content'] %}{% else %}{% set content = message['content'] %}{% endif %}{% if message['role'] == 'user' %}{{ '[INST] ' + content.strip() + ' [/INST]' }}{% elif message['role'] == 'assistant' %}{{ ' '  + content.strip() + ' ' + eos_token }}{% endif %}{% endfor %}"
ORIGINAL_BETA_DPO_STYLE_IDENTIFIER = "original_beta_dpo_plaintext_v1"
NEW_STYLE_IDENTIFIER = "chat_template_v1"


def prepare_preference_dataset_texts(raw_datasets, tokenizer, data_args, model_args, column_names, run_logger):
    preprocessing_mode = data_args.preprocessing_mode
    formatter, style_identifier = _resolve_preprocessing_mode(preprocessing_mode)
    change_template = "mistral" if "mistral" in model_args.model_name_or_path.lower() else None

    if preprocessing_mode == "new" and change_template == "mistral":
        tokenizer.chat_template = MISTRAL_CHAT_TEMPLATE

    raw_datasets = raw_datasets.map(
        formatter,
        fn_kwargs={
            "tokenizer": tokenizer,
            "auto_insert_empty_system_msg": data_args.auto_insert_empty_system_msg,
            "change_template": change_template,
        },
        num_proc=data_args.preprocessing_num_workers,
        remove_columns=column_names,
        desc=f"Formatting comparisons with preprocessing_mode={preprocessing_mode}",
    )

    for split in raw_datasets.keys():
        raw_datasets[split] = raw_datasets[split].rename_columns(
            {"text_prompt": "prompt", "text_chosen": "chosen", "text_rejected": "rejected"}
        )

    _log_preprocessing_diagnostics(raw_datasets, run_logger, preprocessing_mode, style_identifier)
    return raw_datasets


def build_preprocessing_metadata(tokenizer, data_args, model_args) -> Dict[str, str]:
    preprocessing_mode = data_args.preprocessing_mode
    formatter, style_identifier = _resolve_preprocessing_mode(preprocessing_mode)
    change_template = "mistral" if "mistral" in model_args.model_name_or_path.lower() else None

    active_chat_template = None
    if preprocessing_mode == "new":
        if change_template == "mistral":
            tokenizer.chat_template = MISTRAL_CHAT_TEMPLATE
        active_chat_template = tokenizer.chat_template
        if active_chat_template is None:
            active_chat_template = getattr(tokenizer, "default_chat_template", None)

    return {
        "preprocessing_mode": preprocessing_mode,
        "tokenizer_name_or_path": getattr(tokenizer, "name_or_path", None),
        "tokenizer_class": tokenizer.__class__.__name__,
        "active_chat_template": active_chat_template,
        "auto_insert_empty_system_msg": getattr(data_args, "auto_insert_empty_system_msg", True),
        "prompt_format_function_hash": _callable_source_hash(formatter),
        "preprocessing_style_identifier": style_identifier,
    }


def _resolve_preprocessing_mode(preprocessing_mode: str) -> Tuple:
    if preprocessing_mode == "new":
        return _apply_new_preference_chat_template, NEW_STYLE_IDENTIFIER
    if preprocessing_mode == "original_beta_dpo":
        return _apply_original_beta_dpo_preference_format, ORIGINAL_BETA_DPO_STYLE_IDENTIFIER
    raise ValueError(f"Unsupported preprocessing_mode: {preprocessing_mode}")


def _extract_preference_messages(example):
    if not all(key in example.keys() for key in ("chosen", "rejected")):
        raise ValueError(
            "Could not format example as dialogue for preference training; expected either "
            "`[chosen, rejected]` or `[prompt, chosen, rejected]`."
        )
    if not is_openai_format(example["chosen"]) or not is_openai_format(example["rejected"]):
        raise ValueError("Preference training expects OpenAI-style message dictionaries.")

    if "prompt" in example and is_openai_format(example["prompt"]):
        prompt_messages = list(example["prompt"])
        chosen_messages = list(example["chosen"])
        rejected_messages = list(example["rejected"])
    else:
        prompt_messages = list(example["chosen"][:-1])
        chosen_messages = list(example["chosen"][-1:])
        rejected_messages = list(example["rejected"][-1:])

    return prompt_messages, chosen_messages, rejected_messages


def _apply_new_preference_chat_template(
    example,
    tokenizer,
    auto_insert_empty_system_msg: bool = True,
    change_template=None,
):
    if change_template == "mistral":
        tokenizer.chat_template = MISTRAL_CHAT_TEMPLATE

    prompt_messages, chosen_messages, rejected_messages = _extract_preference_messages(example)

    if auto_insert_empty_system_msg:
        maybe_insert_system_message(prompt_messages, tokenizer)

    example["text_prompt"] = tokenizer.apply_chat_template(prompt_messages, tokenize=False)
    example["text_chosen"] = tokenizer.apply_chat_template(chosen_messages, tokenize=False)
    if tokenizer.bos_token and example["text_chosen"].startswith(tokenizer.bos_token):
        example["text_chosen"] = example["text_chosen"][len(tokenizer.bos_token) :]
    example["text_rejected"] = tokenizer.apply_chat_template(rejected_messages, tokenize=False)
    if tokenizer.bos_token and example["text_rejected"].startswith(tokenizer.bos_token):
        example["text_rejected"] = example["text_rejected"][len(tokenizer.bos_token) :]
    return example


def _apply_original_beta_dpo_preference_format(
    example,
    tokenizer,
    auto_insert_empty_system_msg: bool = True,
    change_template=None,
):
    del tokenizer, auto_insert_empty_system_msg, change_template
    prompt_messages, chosen_messages, rejected_messages = _extract_preference_messages(example)

    example["text_prompt"] = _render_original_beta_dpo_prompt(prompt_messages)
    example["text_chosen"] = _render_original_beta_dpo_response(chosen_messages)
    example["text_rejected"] = _render_original_beta_dpo_response(rejected_messages)
    return example


def _render_original_beta_dpo_prompt(messages) -> str:
    rendered = "".join(_render_role_message(message) for message in messages)
    return rendered + "\n\nAssistant:"


def _render_original_beta_dpo_response(messages) -> str:
    if len(messages) == 0:
        return ""

    rendered_parts = []
    for idx, message in enumerate(messages):
        role = _rendered_role_name(message["role"])
        content = message["content"].strip()
        if idx == 0 and role == "Assistant":
            rendered_parts.append(f" {content}")
        else:
            rendered_parts.append(f"\n\n{role}: {content}")
    return "".join(rendered_parts)


def _render_role_message(message) -> str:
    role = _rendered_role_name(message["role"])
    content = message["content"].strip()
    return f"\n\n{role}: {content}"


def _rendered_role_name(role: str) -> str:
    if role == "user":
        return "Human"
    if role == "assistant":
        return "Assistant"
    if role == "system":
        return "System"
    return role.capitalize()


def _log_preprocessing_diagnostics(raw_datasets, run_logger, preprocessing_mode: str, style_identifier: str) -> None:
    if "train" not in raw_datasets or len(raw_datasets["train"]) == 0:
        return

    sample_size = min(1024, len(raw_datasets["train"]))
    sample = raw_datasets["train"].select(range(sample_size))
    avg_prompt_chars = sum(len(text) for text in sample["prompt"]) / sample_size
    avg_chosen_chars = sum(len(text) for text in sample["chosen"]) / sample_size
    avg_rejected_chars = sum(len(text) for text in sample["rejected"]) / sample_size
    run_logger.info(
        "Preprocessing diagnostics: "
        f"mode={preprocessing_mode}, style={style_identifier}, sample_size={sample_size}, "
        f"avg_prompt_chars={avg_prompt_chars:.1f}, avg_chosen_chars={avg_chosen_chars:.1f}, "
        f"avg_rejected_chars={avg_rejected_chars:.1f}"
    )


def _callable_source_hash(fn) -> str:
    target = getattr(fn, "__func__", fn)
    try:
        source = inspect.getsource(target)
    except (OSError, TypeError):
        source = repr(target)
    return hashlib.sha256(source.encode("utf-8")).hexdigest()
