import inspect
import random
import warnings
from collections import defaultdict
from contextlib import nullcontext
from functools import wraps
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union

import torch
import torch.distributed as dist
import torch.nn as nn
from accelerate import PartialState
from accelerate.utils import is_deepspeed_available
from datasets import Dataset
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from utils.preference_tokenization import PreferenceTokenizationProcessor
from utils.preprocessing_cache import maybe_prepare_tokenized_datasets
from transformers import AutoModelForCausalLM, DataCollator, PreTrainedModel, PreTrainedTokenizerBase, Trainer
from transformers.trainer_callback import TrainerCallback
from transformers.trainer_utils import EvalLoopOutput

from trainer_configs import TokenizedPreferenceConfig
from torch_dtype_utils import normalize_torch_dtype
from trl.import_utils import is_peft_available, is_wandb_available
from trl.trainer.utils import (
    DPODataCollatorWithPadding,
    disable_dropout_in_model,
    pad_to_length,
    peft_module_casting_to_bf16,
    trl_sanitze_kwargs_for_tagging,
)

if is_peft_available():
    from peft import PeftModel, get_peft_model, prepare_model_for_kbit_training

if is_wandb_available():
    import wandb

if is_deepspeed_available():
    import deepspeed


class TokenizedDPOTrainer(Trainer):
    _tag_names = ["trl"]
    _LONG_SEQUENCE_WARNING_KEY = "sequence-length-is-longer-than-the-specified-maximum"

    def __init__(
        self,
        model: Optional[Union[PreTrainedModel, nn.Module, str]] = None,
        ref_model: Optional[Union[PreTrainedModel, nn.Module, str]] = None,
        args: Optional[TokenizedPreferenceConfig] = None,
        data_collator: Optional[DataCollator] = None,
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Union[Dataset, Dict[str, Dataset]]] = None,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        model_init: Optional[Callable[[], PreTrainedModel]] = None,
        callbacks: Optional[List[TrainerCallback]] = None,
        optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),
        preprocess_logits_for_metrics: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
        peft_config: Optional[Dict] = None,
        compute_metrics: Optional[Callable[[EvalLoopOutput], Dict]] = None,
    ):
        if args.model_init_kwargs is None:
            model_init_kwargs = {}
        elif not isinstance(model, str):
            raise ValueError("You passed model_kwargs to the trainer, but your model is already instantiated.")
        else:
            model_init_kwargs = dict(args.model_init_kwargs)
            model_init_kwargs["torch_dtype"] = normalize_torch_dtype(model_init_kwargs.get("torch_dtype"))

        self.precompute_ref_logps = bool(
            getattr(args, "precompute_ref_log_probs", False) or getattr(args, "precompute_ref_logps", False)
        )
        self.non_finite_logits_handling = getattr(args, "non_finite_logits_handling", "sanitize")
        self._precompute_ref_model_path = ref_model if isinstance(ref_model, str) and self.precompute_ref_logps else None
        self._precompute_ref_model = None
        self._precompute_ref_model_init_kwargs = self._build_precompute_ref_model_init_kwargs(model_init_kwargs)

        if isinstance(model, str):
            warnings.warn(
                "You passed a model_id to the trainer. This will automatically create an "
                "`AutoModelForCausalLM` or a `PeftModel` (if you passed a `peft_config`) for you."
            )
            model = AutoModelForCausalLM.from_pretrained(model, **model_init_kwargs)

        if isinstance(ref_model, str):
            self.ref_model = None if self.precompute_ref_logps else AutoModelForCausalLM.from_pretrained(
                ref_model, **model_init_kwargs
            )
        else:
            self.ref_model = ref_model

        self._peft_has_been_casted_to_bf16 = False

        if not is_peft_available() and peft_config is not None:
            raise ValueError("PEFT is not installed but `peft_config` was provided.")
        elif is_peft_available() and peft_config is not None:
            if isinstance(model, PeftModel):
                model = model.merge_and_unload()

            if getattr(model, "is_loaded_in_8bit", False) or getattr(model, "is_loaded_in_4bit", False):
                support_gc_kwargs = hasattr(args, "gradient_checkpointing_kwargs") and "gradient_checkpointing_kwargs" in list(
                    inspect.signature(prepare_model_for_kbit_training).parameters
                )
                prepare_model_kwargs = {"use_gradient_checkpointing": args.gradient_checkpointing}
                if support_gc_kwargs:
                    prepare_model_kwargs["gradient_checkpointing_kwargs"] = args.gradient_checkpointing_kwargs
                model = prepare_model_for_kbit_training(model, **prepare_model_kwargs)
            elif getattr(args, "gradient_checkpointing", False):
                if hasattr(model, "enable_input_require_grads"):
                    model.enable_input_require_grads()
                else:

                    def make_inputs_require_grad(module, input, output):
                        output.requires_grad_(True)

                    model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

            model = get_peft_model(model, peft_config)
            if args.bf16 and getattr(model, "is_loaded_in_4bit", False):
                peft_module_casting_to_bf16(model)
                self._peft_has_been_casted_to_bf16 = True
        elif getattr(args, "gradient_checkpointing", False):
            if hasattr(model, "enable_input_require_grads"):
                model.enable_input_require_grads()
            else:

                def make_inputs_require_grad(module, input, output):
                    output.requires_grad_(True)

                model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

        if args.generate_during_eval and not is_wandb_available():
            raise ValueError("`generate_during_eval=True` requires `wandb` to be installed.")

        if model is not None:
            self.is_encoder_decoder = model.config.is_encoder_decoder
        elif args.is_encoder_decoder is None:
            raise ValueError("When no model is provided, you need to pass `is_encoder_decoder`.")
        else:
            self.is_encoder_decoder = args.is_encoder_decoder

        if self.is_encoder_decoder:
            self.decoder_start_token_id = model.config.decoder_start_token_id
            self.pad_token_id = model.config.pad_token_id

        if tokenizer is None:
            raise ValueError("tokenizer must be specified to tokenize a preference dataset.")

        max_length = 512 if args.max_length is None else args.max_length
        max_prompt_length = 128 if args.max_prompt_length is None else args.max_prompt_length
        max_target_length = 128 if args.max_target_length is None and self.is_encoder_decoder else args.max_target_length

        if data_collator is None:
            data_collator = DPODataCollatorWithPadding(
                pad_token_id=tokenizer.pad_token_id,
                label_pad_token_id=args.label_pad_token_id,
                is_encoder_decoder=self.is_encoder_decoder,
            )

            if args.remove_unused_columns:
                args.remove_unused_columns = False
                warnings.warn(
                    "When using DPODataCollatorWithPadding, `remove_unused_columns=False` is required. "
                    "It has been set automatically for this run.",
                    UserWarning,
                )

            self.use_dpo_data_collator = True
        else:
            self.use_dpo_data_collator = False

        if args.disable_dropout:
            disable_dropout_in_model(model)
            if self.ref_model is not None:
                disable_dropout_in_model(self.ref_model)

        self.max_length = max_length
        self.generate_during_eval = args.generate_during_eval
        self.label_pad_token_id = args.label_pad_token_id
        self.padding_value = args.padding_value if args.padding_value is not None else tokenizer.pad_token_id
        self.max_prompt_length = max_prompt_length
        self.truncation_mode = args.truncation_mode
        self.max_target_length = max_target_length
        self.tokenizer = tokenizer
        self.tokenization_processor = PreferenceTokenizationProcessor(
            tokenizer=tokenizer,
            is_encoder_decoder=self.is_encoder_decoder,
            max_length=self.max_length,
            max_prompt_length=self.max_prompt_length,
            max_target_length=self.max_target_length,
            truncation_mode=self.truncation_mode,
            label_pad_token_id=self.label_pad_token_id,
            long_sequence_warning_key=self._LONG_SEQUENCE_WARNING_KEY,
        )
        self.beta = getattr(args, "beta", 0.1)
        self.sft_weight = getattr(args, "sft_weight", 0.0)
        self.label_smoothing = getattr(args, "label_smoothing", 0.0)
        self.loss_type = getattr(args, "loss_type", "sigmoid")
        self._precomputed_train_ref_logps = False
        self._precomputed_eval_ref_logps = False
        self._stored_metrics = defaultdict(lambda: defaultdict(list))

        with PartialState().local_main_process_first():
            train_dataset, eval_dataset = maybe_prepare_tokenized_datasets(self, args, train_dataset, eval_dataset)

        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            model_init=model_init,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            optimizers=optimizers,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        )

        if hasattr(self.model, "add_model_tags"):
            self.model.add_model_tags(self._tag_names)

        if not hasattr(self, "accelerator"):
            raise AttributeError("Your `Trainer` does not have an `accelerator` object. Consider upgrading `transformers`.")

        if self.ref_model is not None and not self.precompute_ref_logps:
            if self.is_deepspeed_enabled:
                self.ref_model = self._prepare_deepspeed(self.ref_model)
            else:
                # Under FSDP, `evaluation_mode=True` skips distributed wrapping and the default
                # device placement is disabled, so the reference model would otherwise remain on CPU.
                self.ref_model = self.accelerator.prepare_model(
                    self.ref_model,
                    device_placement=True,
                    evaluation_mode=True,
                )
                self._sync_unwrapped_model_from_rank0(self.ref_model, model_name="reference")

    def _prepare_deepspeed(self, model: PreTrainedModel):
        deepspeed_plugin = self.accelerator.state.deepspeed_plugin
        config_kwargs = deepspeed_plugin.deepspeed_config
        if model is not None and hasattr(model, "config"):
            hidden_size = (
                max(model.config.hidden_sizes)
                if getattr(model.config, "hidden_sizes", None)
                else getattr(model.config, "hidden_size", None)
            )
            if hidden_size is not None and config_kwargs["zero_optimization"]["stage"] == 3:
                config_kwargs.update(
                    {
                        "zero_optimization.reduce_bucket_size": hidden_size * hidden_size,
                        "zero_optimization.stage3_param_persistence_threshold": 10 * hidden_size,
                        "zero_optimization.stage3_prefetch_bucket_size": 0.9 * hidden_size * hidden_size,
                    }
                )
        if config_kwargs["zero_optimization"]["stage"] != 3:
            config_kwargs["zero_optimization"]["stage"] = 0
        model, *_ = deepspeed.initialize(model=model, config=config_kwargs)
        model.eval()
        return model

    def _is_distributed_run(self) -> bool:
        return dist.is_available() and dist.is_initialized() and self.accelerator.num_processes > 1

    def _sync_unwrapped_model_from_rank0(self, model: nn.Module, model_name: str) -> None:
        if not self._is_distributed_run():
            return

        # Manually loaded auxiliary models are not FSDP-wrapped, so they do not benefit from
        # rank-0 state syncing when Accelerate's FSDP CPU-RAM-efficient loading is enabled.
        for _, tensor in model.named_parameters():
            dist.broadcast(tensor.data, src=0)
        for _, tensor in model.named_buffers():
            dist.broadcast(tensor.data, src=0)
        dist.barrier()

    @staticmethod
    def _build_precompute_ref_model_init_kwargs(model_init_kwargs: Dict[str, Any]) -> Dict[str, Any]:
        precompute_kwargs = dict(model_init_kwargs)
        precompute_kwargs["torch_dtype"] = torch.float32
        precompute_kwargs["use_cache"] = False
        precompute_kwargs["device_map"] = None
        precompute_kwargs["quantization_config"] = None
        return precompute_kwargs

    def _load_precompute_ref_model(self) -> nn.Module:
        if self._precompute_ref_model is not None:
            return self._precompute_ref_model

        if self._precompute_ref_model_path is None:
            raise ValueError(
                "precompute_ref_log_probs=True requires ref_model to be provided as a model identifier or path "
                "so a dedicated fp32 precompute model can be loaded."
            )

        ref_model = AutoModelForCausalLM.from_pretrained(
            self._precompute_ref_model_path,
            **self._precompute_ref_model_init_kwargs,
        )
        if getattr(self.args, "disable_dropout", False):
            disable_dropout_in_model(ref_model)
        ref_model.to(device=self.accelerator.device)
        self._sync_unwrapped_model_from_rank0(ref_model, model_name="reference_precompute")
        ref_model.eval()
        self._precompute_ref_model = ref_model
        return self._precompute_ref_model

    def _release_precompute_ref_model(self) -> None:
        if self._precompute_ref_model is None:
            return

        del self._precompute_ref_model
        self._precompute_ref_model = None
        torch.cuda.empty_cache()
        self.accelerator.free_memory()

    @torch.no_grad()
    def compute_reference_log_probs(
        self,
        padded_batch: Dict[str, Union[torch.Tensor, Any]],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        ref_model = self._load_precompute_ref_model() if self.precompute_ref_logps else self.ref_model
        if ref_model is None:
            raise ValueError("Reference model is required to precompute reference log-probs.")

        autocast_context = (
            torch.autocast(device_type=self.accelerator.device.type, enabled=False)
            if self.precompute_ref_logps and self.accelerator.device.type != "cpu"
            else nullcontext()
        )
        inference_context = torch.inference_mode() if self.precompute_ref_logps else nullcontext()
        with inference_context, autocast_context:
            ref_chosen_logps, ref_rejected_logps, _, _, _ = self.concatenated_forward(
                ref_model,
                padded_batch,
                average_log_prob=False,
                force_logits_to_float32=self.precompute_ref_logps,
                logit_source="reference_precompute" if self.precompute_ref_logps else "reference",
            )
        return ref_chosen_logps, ref_rejected_logps

    def _precompute_dataset_reference_logps(
        self,
        dataset: Dataset,
        batch_size: int,
        description: str,
    ) -> Dataset:
        dataloader_params = {
            "batch_size": batch_size,
            "collate_fn": self.data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "shuffle": False,
        }
        data_loader = self.accelerator.prepare(DataLoader(dataset, **dataloader_params))

        reference_chosen_logps = []
        reference_rejected_logps = []
        try:
            for padded_batch in tqdm(iterable=data_loader, desc=description):
                ref_chosen_logp, ref_rejected_logp = self.compute_reference_log_probs(padded_batch)
                if not torch.isfinite(ref_chosen_logp).all() or not torch.isfinite(ref_rejected_logp).all():
                    raise ValueError(
                        f"Non-finite reference log-probs encountered during {description}. "
                        "This usually indicates numerical instability in the reference forward pass."
                    )
                ref_chosen_logp, ref_rejected_logp = self.accelerator.gather_for_metrics(
                    (ref_chosen_logp, ref_rejected_logp)
                )
                reference_chosen_logps.append(ref_chosen_logp.cpu())
                reference_rejected_logps.append(ref_rejected_logp.cpu())

                # Reduce allocator pressure before the actual training loop starts.
                torch.cuda.empty_cache()
                self.accelerator.free_memory()

            all_reference_chosen_logps = torch.cat(reference_chosen_logps).float().numpy()[: len(dataset)]
            all_reference_rejected_logps = torch.cat(reference_rejected_logps).float().numpy()[: len(dataset)]

            dataset = dataset.add_column(name="ref_chosen_logps", column=all_reference_chosen_logps)
            dataset = dataset.add_column(name="ref_rejected_logps", column=all_reference_rejected_logps)
            return dataset
        finally:
            self._release_precompute_ref_model()

    def _maybe_release_ref_model(self) -> None:
        if self.ref_model is None or not self.precompute_ref_logps:
            return

        if not self._precomputed_train_ref_logps:
            return

        if self.args.do_eval and not self._precomputed_eval_ref_logps:
            return

        self.ref_model = None
        torch.cuda.empty_cache()
        self.accelerator.free_memory()

    def get_train_dataloader(self) -> DataLoader:
        if self.precompute_ref_logps and not self._precomputed_train_ref_logps:
            precompute_batch_size = getattr(self.args, "precompute_ref_batch_size", None)
            if precompute_batch_size is None:
                precompute_batch_size = self.args.per_device_train_batch_size
            self.train_dataset = self._precompute_dataset_reference_logps(
                self.train_dataset,
                batch_size=precompute_batch_size,
                description="Train dataset reference log probs",
            )
            self._precomputed_train_ref_logps = True
            self._maybe_release_ref_model()

        return super().get_train_dataloader()

    def get_eval_dataloader(self, eval_dataset: Optional[Dataset] = None) -> DataLoader:
        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        if eval_dataset is None:
            raise ValueError("Trainer: evaluation requires an eval_dataset.")

        if self.precompute_ref_logps and not self._precomputed_eval_ref_logps:
            precompute_eval_batch_size = getattr(self.args, "precompute_ref_eval_batch_size", None)
            if precompute_eval_batch_size is None:
                precompute_eval_batch_size = self.args.per_device_eval_batch_size
            eval_dataset = self._precompute_dataset_reference_logps(
                eval_dataset,
                batch_size=precompute_eval_batch_size,
                description="Eval dataset reference log probs",
            )
            if self.eval_dataset is not None:
                self.eval_dataset = eval_dataset
            self._precomputed_eval_ref_logps = True
            self._maybe_release_ref_model()

        return super().get_eval_dataloader(eval_dataset=eval_dataset)

    def _tokenize_without_max_length_warning(self, text: str, **kwargs) -> Dict[str, List[int]]:
        return self.tokenization_processor._tokenize_without_max_length_warning(text, **kwargs)

    def build_tokenized_answer(self, prompt, answer):
        return self.tokenization_processor.build_tokenized_answer(prompt, answer)

    def tokenize_row(self, feature, model: Optional[Union[PreTrainedModel, nn.Module]] = None) -> Dict:
        return self.tokenization_processor.tokenize_row(feature, model=model)

    def tokenize_batch(self, features, model: Optional[Union[PreTrainedModel, nn.Module]] = None) -> Dict[str, List[Any]]:
        return self.tokenization_processor.tokenize_batch(features, model=model)

    def get_tokenization_code_hashes(self) -> Dict[str, str]:
        return self.tokenization_processor.get_code_sources()

    @staticmethod
    def concatenated_inputs(
        batch: Dict[str, Union[List, torch.LongTensor]],
        is_encoder_decoder: bool = False,
        label_pad_token_id: int = -100,
        padding_value: int = 0,
        device: Optional[torch.device] = None,
    ) -> Dict[str, torch.LongTensor]:
        concatenated_batch = {}

        if is_encoder_decoder:
            max_length = max(batch["chosen_labels"].shape[1], batch["rejected_labels"].shape[1])
        else:
            max_length = max(batch["chosen_input_ids"].shape[1], batch["rejected_input_ids"].shape[1])

        for key in batch:
            if key.startswith("chosen") and isinstance(batch[key], torch.Tensor):
                if "labels" in key or is_encoder_decoder:
                    pad_value = label_pad_token_id
                elif key.endswith("_input_ids"):
                    pad_value = padding_value
                elif key.endswith("_attention_mask"):
                    pad_value = 0
                else:
                    continue
                concatenated_key = key.replace("chosen", "concatenated")
                concatenated_batch[concatenated_key] = pad_to_length(
                    batch[key],
                    max_length,
                    pad_value=pad_value,
                ).to(device=device)

        for key in batch:
            if key.startswith("rejected") and isinstance(batch[key], torch.Tensor):
                if "labels" in key or is_encoder_decoder:
                    pad_value = label_pad_token_id
                elif key.endswith("_input_ids"):
                    pad_value = padding_value
                elif key.endswith("_attention_mask"):
                    pad_value = 0
                else:
                    continue
                concatenated_key = key.replace("rejected", "concatenated")
                concatenated_batch[concatenated_key] = torch.cat(
                    (
                        concatenated_batch[concatenated_key],
                        pad_to_length(batch[key], max_length, pad_value=pad_value),
                    ),
                    dim=0,
                ).to(device=device)

        if is_encoder_decoder:
            concatenated_batch["concatenated_input_ids"] = batch["prompt_input_ids"].repeat(2, 1).to(device=device)
            concatenated_batch["concatenated_attention_mask"] = batch["prompt_attention_mask"].repeat(2, 1).to(device=device)
        else:
            for key, value in concatenated_batch.items():
                concatenated_batch[key] = value.to(device=device)

        return concatenated_batch

    def concatenated_forward(
        self,
        model: nn.Module,
        batch: Dict[str, Union[List, torch.LongTensor]],
        average_log_prob: bool = False,
        force_logits_to_float32: bool = False,
        logit_source: str = "unknown",
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.LongTensor]:
        model_device = self._get_model_device(model)
        if model_device is None or model_device.type == "cpu":
            model_device = self.accelerator.device
        concatenated_batch = self.concatenated_inputs(
            batch,
            is_encoder_decoder=self.is_encoder_decoder,
            label_pad_token_id=self.label_pad_token_id,
            padding_value=self.padding_value,
            device=model_device,
        )
        len_chosen = batch["chosen_labels"].shape[0]

        model_kwargs = (
            {
                "labels": concatenated_batch["concatenated_labels"],
                "decoder_input_ids": concatenated_batch.pop("concatenated_decoder_input_ids", None),
            }
            if self.is_encoder_decoder
            else {}
        )

        all_logits = model(
            concatenated_batch["concatenated_input_ids"],
            attention_mask=concatenated_batch["concatenated_attention_mask"],
            use_cache=False,
            **model_kwargs,
        ).logits
        if force_logits_to_float32:
            all_logits = all_logits.float()
        all_logits = self._handle_non_finite_logits(all_logits, logit_source=logit_source)

        all_logps = self.get_batch_logps(
            all_logits,
            concatenated_batch["concatenated_labels"],
            average_log_prob=average_log_prob,
            is_encoder_decoder=self.is_encoder_decoder,
            label_pad_token_id=self.label_pad_token_id,
        )

        chosen_logps = all_logps[:len_chosen]
        rejected_logps = all_logps[len_chosen:]
        chosen_logits = all_logits[:len_chosen]
        rejected_logits = all_logits[len_chosen:]
        chosen_labels = concatenated_batch["concatenated_labels"][:len_chosen]

        return chosen_logps, rejected_logps, chosen_logits, rejected_logits, chosen_labels

    @staticmethod
    def _get_model_device(model: nn.Module) -> Optional[torch.device]:
        for tensor in model.parameters():
            return tensor.device
        for tensor in model.buffers():
            return tensor.device
        return None

    @staticmethod
    def get_batch_logps(
        logits: torch.FloatTensor,
        labels: torch.LongTensor,
        average_log_prob: bool = False,
        label_pad_token_id: int = -100,
        is_encoder_decoder: bool = False,
    ) -> torch.FloatTensor:
        if logits.shape[:-1] != labels.shape:
            raise ValueError("Logits (batch and sequence length dim) and labels must have the same shape.")

        if not is_encoder_decoder:
            labels = labels[:, 1:]
            logits = logits[:, :-1, :]
        loss_mask = labels != label_pad_token_id
        safe_labels = labels.masked_fill(~loss_mask, 0)
        per_token_logps = TokenizedDPOTrainer._compute_token_logps(logits, safe_labels)

        if average_log_prob:
            return (per_token_logps * loss_mask).sum(-1) / loss_mask.sum(-1).clamp_min(1)
        return (per_token_logps * loss_mask).sum(-1)

    @staticmethod
    def _compute_token_logps(logits: torch.FloatTensor, labels: torch.LongTensor) -> torch.FloatTensor:
        if not torch.isfinite(logits).all():
            warnings.warn(
                "Detected non-finite logits after the forward-pass safety check; sanitizing logits as a fallback.",
                RuntimeWarning,
            )
            logits = TokenizedDPOTrainer._sanitize_logits_for_logps(logits)
            selected_logits = torch.gather(logits, dim=2, index=labels.unsqueeze(2)).squeeze(2)
            log_normalizers = TokenizedDPOTrainer._chunked_logsumexp_fp32(logits)
            return selected_logits - log_normalizers

        selected_logits = torch.gather(logits, dim=2, index=labels.unsqueeze(2)).squeeze(2)
        log_normalizers = torch.logsumexp(logits, dim=-1)
        per_token_logps = selected_logits - log_normalizers

        if torch.isfinite(per_token_logps).all():
            return per_token_logps

        if logits.dtype not in (torch.float16, torch.bfloat16):
            return per_token_logps

        selected_logits_fp32 = selected_logits.float()
        log_normalizers_fp32 = TokenizedDPOTrainer._chunked_logsumexp_fp32(logits)
        return selected_logits_fp32 - log_normalizers_fp32

    @staticmethod
    def _sanitize_logits_for_logps(logits: torch.FloatTensor) -> torch.FloatTensor:
        sanitized_logits = torch.nan_to_num(logits.float(), nan=0.0, posinf=1e4, neginf=-1e4)
        return sanitized_logits.clamp_(min=-1e4, max=1e4)

    @staticmethod
    def _chunked_logsumexp_fp32(logits: torch.FloatTensor, chunk_size: int = 2048) -> torch.FloatTensor:
        log_normalizers = None
        vocab_size = logits.shape[-1]
        for start in range(0, vocab_size, chunk_size):
            chunk = logits[..., start : start + chunk_size].float()
            chunk_logsumexp = torch.logsumexp(chunk, dim=-1)
            log_normalizers = (
                chunk_logsumexp if log_normalizers is None else torch.logaddexp(log_normalizers, chunk_logsumexp)
            )
        return log_normalizers

    def _format_non_finite_logits_message(self, logits: torch.FloatTensor, logit_source: str) -> str:
        non_finite_count = int((~torch.isfinite(logits)).sum().item())
        nan_count = int(torch.isnan(logits).sum().item())
        posinf_count = int(torch.isposinf(logits).sum().item())
        neginf_count = int(torch.isneginf(logits).sum().item())
        return (
            f"Detected {non_finite_count} non-finite values in {logit_source} logits before token log-prob computation "
            f"(global_step={int(self.state.global_step)}, epoch={self.state.epoch}, process_index={self.accelerator.process_index}, "
            f"shape={tuple(logits.shape)}, dtype={logits.dtype}, device={logits.device}, nan={nan_count}, "
            f"posinf={posinf_count}, neginf={neginf_count})."
        )

    def _handle_non_finite_logits(self, logits: torch.FloatTensor, logit_source: str) -> torch.FloatTensor:
        if torch.isfinite(logits).all():
            return logits

        message = self._format_non_finite_logits_message(logits, logit_source)
        if self.non_finite_logits_handling == "error":
            raise ValueError(message)

        warnings.warn(
            message + " Sanitizing logits to keep reference/preference scoring numerically stable.",
            RuntimeWarning,
        )
        return TokenizedDPOTrainer._sanitize_logits_for_logps(logits)

    def compute_loss(
        self,
        model: Union[PreTrainedModel, nn.Module],
        inputs: Dict[str, Union[torch.Tensor, Any]],
        return_outputs: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        if not self.use_dpo_data_collator:
            warnings.warn(
                "compute_loss is implemented for DPODataCollatorWithPadding; a custom data collator may require a custom prediction_step."
            )

        compute_loss_context_manager = torch.cuda.amp.autocast if self._peft_has_been_casted_to_bf16 else nullcontext

        with compute_loss_context_manager():
            loss, metrics = self.get_batch_loss_metrics(model, inputs, train_eval="train")

        self.store_metrics(metrics, train_eval="train")

        if return_outputs:
            return loss, metrics
        return loss

    def get_batch_samples(self, model, batch: Dict[str, torch.LongTensor]) -> Tuple[str, str]:
        generate_context_manager = nullcontext if not self._peft_has_been_casted_to_bf16 else torch.cuda.amp.autocast

        with generate_context_manager():
            policy_output = model.generate(
                input_ids=batch["prompt_input_ids"],
                attention_mask=batch["prompt_attention_mask"],
                max_length=self.max_length,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        policy_output = pad_to_length(policy_output, self.max_length, self.tokenizer.pad_token_id)
        policy_output_decoded = self.tokenizer.batch_decode(policy_output, skip_special_tokens=True)
        return policy_output_decoded

    def prediction_step(
        self,
        model: Union[PreTrainedModel, nn.Module],
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ):
        if not self.use_dpo_data_collator:
            warnings.warn(
                "prediction_step is implemented for DPODataCollatorWithPadding; a custom data collator may require a custom prediction_step."
            )
        if ignore_keys is None:
            if hasattr(model, "config"):
                ignore_keys = getattr(model.config, "keys_to_ignore_at_inference", [])
            else:
                ignore_keys = []

        prediction_context_manager = torch.cuda.amp.autocast if self._peft_has_been_casted_to_bf16 else nullcontext

        with torch.no_grad(), prediction_context_manager():
            loss, metrics = self.get_batch_loss_metrics(model, inputs, train_eval="eval")

        self.store_metrics(metrics, train_eval="eval")

        if prediction_loss_only:
            return loss.detach(), None, None

        logits_dict = {
            "eval_logits/chosen": metrics["eval_logits/chosen"],
            "eval_logits/rejected": metrics["eval_logits/rejected"],
        }
        logits = tuple(value.unsqueeze(dim=0) for key, value in logits_dict.items() if key not in ignore_keys)
        logits = torch.stack(logits).mean(axis=1).to(self.accelerator.device)
        labels = torch.zeros(logits.shape[0], device=self.accelerator.device)
        return loss.detach(), logits, labels

    def store_metrics(self, metrics: Dict[str, float], train_eval: Literal["train", "eval"] = "train") -> None:
        for key, value in metrics.items():
            self._stored_metrics[train_eval][key].append(value)

    def evaluation_loop(
        self,
        dataloader: DataLoader,
        description: str,
        prediction_loss_only: Optional[bool] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> EvalLoopOutput:
        if self.generate_during_eval:
            num_samples = len(dataloader.dataset)
            sample_count = min(self.args.eval_batch_size, num_samples)
            random_indices = random.sample(range(num_samples), k=sample_count)
            random_batch_dataset = dataloader.dataset.select(random_indices)
            random_batch = self.data_collator(random_batch_dataset)
            random_batch = self._prepare_inputs(random_batch)
            policy_output_decoded = self.get_batch_samples(self.model, random_batch)

            self.log(
                {
                    "game_log": wandb.Table(
                        columns=["Prompt", "Policy"],
                        rows=[[prompt, pol[len(prompt) :]] for prompt, pol in zip(random_batch["prompt"], policy_output_decoded)],
                    )
                }
            )
            self.state.log_history.pop()

        return super().evaluation_loop(dataloader, description, prediction_loss_only, ignore_keys, metric_key_prefix)

    def log(self, logs: Dict[str, float]) -> None:
        train_eval = "train" if "loss" in logs else "eval"
        for key, metrics in self._stored_metrics[train_eval].items():
            logs[key] = torch.tensor(metrics).mean().item()
        del self._stored_metrics[train_eval]
        super().log(logs)

    @wraps(Trainer.push_to_hub)
    def push_to_hub(self, commit_message: Optional[str] = "End of training", blocking: bool = True, **kwargs) -> str:
        kwargs = trl_sanitze_kwargs_for_tagging(model=self.model, tag_names=self._tag_names, kwargs=kwargs)
        return super().push_to_hub(commit_message=commit_message, blocking=blocking, **kwargs)
