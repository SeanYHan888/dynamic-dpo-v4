#!/usr/bin/env python
# coding=utf-8
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path

from accelerate import Accelerator
from accelerate.utils import DistributedType
from accelerate.utils.fsdp_utils import load_fsdp_model
from transformers import AutoModelForCausalLM

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from alignment import DataArguments, H4ArgumentParser, ModelArguments, get_tokenizer
from trainer_configs import BetaDPOConfig
from utils.checkpoint_io import save_hf_compatible_model_artifacts
from utils.runtime import build_model_init_kwargs, setup_run

logger = logging.getLogger(__name__)


@dataclass
class ExportArguments:
    checkpoint_dir: str = field(metadata={"help": "Path to the FSDP checkpoint directory to export from."})
    export_dir: str | None = field(default=None, metadata={"help": "Optional output directory for exported HF files."})


def main():
    parser = H4ArgumentParser((ModelArguments, DataArguments, BetaDPOConfig, ExportArguments))
    model_args, data_args, training_args, export_args = parser.parse()

    setup_run(model_args, data_args, training_args, logger)
    accelerator = Accelerator()
    if accelerator.distributed_type != DistributedType.FSDP:
        raise ValueError("export_fsdp_checkpoint.py must be launched under accelerate FSDP.")

    tokenizer = get_tokenizer(model_args, data_args)
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        **build_model_init_kwargs(model_args, training_args),
    )
    model = accelerator.prepare_model(model)

    checkpoint_dir = Path(export_args.checkpoint_dir).expanduser().resolve()
    logger.info(f"Loading FSDP model weights from {checkpoint_dir}")
    load_fsdp_model(
        accelerator.state.fsdp_plugin,
        accelerator,
        model,
        str(checkpoint_dir),
        adapter_only=False,
    )

    export_dir = export_args.export_dir or training_args.output_dir
    logger.info(f"Exporting validated HF artifacts to {export_dir}")
    save_hf_compatible_model_artifacts(
        accelerator,
        model,
        export_dir,
        logger,
        processing_artifact=tokenizer,
        safe_serialization=bool(getattr(training_args, "save_safetensors", True)),
    )


if __name__ == "__main__":
    main()
