import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, Literal, Optional

from alignment import DPOConfig


@dataclass
class TokenizedPreferenceConfig(DPOConfig):
    model_init_kwargs: Optional[Dict[str, Any]] = field(default=None, repr=False)
    dataset_num_proc: Optional[int] = field(default=None)
    tokenization_mode: Literal["online", "offline_only", "reuse_only"] = field(default="online")
    tokenization_batch_size: int = field(default=64)
    post_tokenization_log_samples: int = field(default=0)
    post_tokenization_log_dir: Optional[str] = field(default=None)
    precompute_ref_batch_size: Optional[int] = field(default=None)
    precompute_ref_eval_batch_size: Optional[int] = field(default=None)
    reuse_tokenized_dataset: bool = field(default=False)
    tokenized_dataset_cache_dir: Optional[str] = field(default=None)
    disable_dropout: bool = field(default=True)
    generate_during_eval: bool = field(default=False)
    sft_weight: float = field(default=0.0)
    truncation_mode: str = field(default="keep_end")
    max_length: Optional[int] = field(default=512)
    max_prompt_length: Optional[int] = field(default=128)
    max_target_length: Optional[int] = field(default=None)
    beta: float = field(default=0.1)
    label_smoothing: float = field(default=0.0)
    loss_type: str = field(default="sigmoid")
    non_finite_logits_handling: Literal["sanitize", "error"] = field(default="sanitize")
    label_pad_token_id: int = field(default=-100)
    padding_value: Optional[int] = field(default=None)
    is_encoder_decoder: Optional[bool] = field(default=None)

    def __post_init__(self):
        if self.tokenization_mode not in {"online", "offline_only", "reuse_only"}:
            raise ValueError("tokenization_mode must be one of {'online', 'offline_only', 'reuse_only'}.")
        if self.tokenization_batch_size <= 0:
            raise ValueError("tokenization_batch_size must be > 0.")
        if self.post_tokenization_log_samples < 0:
            raise ValueError("post_tokenization_log_samples must be >= 0.")
        if self.precompute_ref_batch_size is not None and self.precompute_ref_batch_size <= 0:
            raise ValueError("precompute_ref_batch_size must be > 0 when provided.")
        if self.precompute_ref_eval_batch_size is not None and self.precompute_ref_eval_batch_size <= 0:
            raise ValueError("precompute_ref_eval_batch_size must be > 0 when provided.")
        if self.non_finite_logits_handling not in {"sanitize", "error"}:
            raise ValueError("non_finite_logits_handling must be one of {'sanitize', 'error'}.")
        super().__post_init__()


@dataclass
class SimPOConfig(TokenizedPreferenceConfig):
    trainer_type: str = field(default="simpo")
    gamma_beta_ratio: float = field(default=0.5)
    alpha: float = field(default=0.0)
    ln: bool = field(default=False)


@dataclass
class BetaDPOConfig(TokenizedPreferenceConfig):
    trainer_type: str = field(default="beta_dpo")
    rho: float = field(default=0.8)
    alpha: float = field(default=1.0)
    ema_momentum: float = field(default=0.9)
    beta_min: float = field(default=1e-3)
    sync_global_mask: bool = field(default=True)
    deterministic_eval: bool = field(default=False)
    require_equal_local_batch_size: bool = field(default=True)

    def __post_init__(self):
        if not (0.0 < self.rho <= 1.0):
            raise ValueError("rho must be in (0, 1].")
        if self.alpha < 0:
            raise ValueError("alpha must be >= 0.")
        if not (0.0 <= self.ema_momentum < 1.0):
            raise ValueError("ema_momentum must be in [0, 1).")
        if self.beta_min <= 0.0:
            raise ValueError("beta_min must be > 0.")
        super().__post_init__()


@dataclass
class MarginDPOConfig(TokenizedPreferenceConfig):
    trainer_type: str = field(default="margin_dpo")
    margin_log_path: str = field(default="./margin_logs")
    margin_log_steps: int = field(default=50)
    margin_save_full: bool = field(default=False)
    push_margin_dataset: bool = field(default=True)
    hub_margin_dataset_id: Optional[str] = field(default=None)
    margin_dataset_private: Optional[bool] = field(default=None)
    margin_dataset_split: str = field(default="train")
    require_explicit_ref_model: bool = field(default=True)
    f_divergence_type: str = field(default="reverse_kl")
    f_alpha_divergence_coef: float = field(default=1.0)


@dataclass
class EpsilonDPOConfig(TokenizedPreferenceConfig):
    # Keep the repo-wide tokenization / training argument surface, but add the one ε-DPO-specific
    # hyperparameter needed by the original method.
    trainer_type: str = field(default="epsilon_dpo")
    epsilon: float = field(default=0.01)

    def __post_init__(self):
        if self.epsilon < 0.0:
            raise ValueError("epsilon must be >= 0.")

        # These three settings are hard requirements from the original ε-DPO implementation. We
        # override them here so the trainer cannot silently drift away from the intended math.
        if self.reference_free:
            warnings.warn(
                "When using `EpsilonDPOTrainer`, `reference_free=False` is required. Overriding to False.",
                UserWarning,
            )
            self.reference_free = False

        if self.precompute_ref_log_probs:
            warnings.warn(
                "When using `EpsilonDPOTrainer`, `precompute_ref_log_probs=False` is required. Overriding to False.",
                UserWarning,
            )
            self.precompute_ref_log_probs = False

        if self.loss_type != "sigmoid":
            warnings.warn(
                "When using `EpsilonDPOTrainer`, `loss_type='sigmoid'` is required. Overriding to 'sigmoid'.",
                UserWarning,
            )
            self.loss_type = "sigmoid"

        super().__post_init__()
