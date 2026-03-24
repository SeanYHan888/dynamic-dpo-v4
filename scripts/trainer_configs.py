from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from alignment import DPOConfig


@dataclass
class TokenizedPreferenceConfig(DPOConfig):
    model_init_kwargs: Optional[Dict[str, Any]] = field(default=None, repr=False)
    dataset_num_proc: Optional[int] = field(default=None)
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
    label_pad_token_id: int = field(default=-100)
    padding_value: Optional[int] = field(default=None)
    is_encoder_decoder: Optional[bool] = field(default=None)


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
    deterministic_eval: bool = field(default=True)
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
    require_explicit_ref_model: bool = field(default=True)
    f_divergence_type: str = field(default="reverse_kl")
    f_alpha_divergence_coef: float = field(default=1.0)
