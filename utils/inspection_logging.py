import json
import random
from pathlib import Path
from typing import Iterable, Mapping, Optional


DEFAULT_PREPROCESSING_LOG_DIRNAME = "preprocessing_logs"


def is_main_process(process_index: Optional[int]) -> bool:
    return (0 if process_index is None else process_index) == 0


def resolve_inspection_log_dir(explicit_dir: Optional[str], output_dir: Optional[str]) -> Path:
    if explicit_dir:
        return Path(explicit_dir).expanduser()
    if output_dir is None:
        raise ValueError("output_dir is required when no explicit inspection log dir is provided.")
    return Path(output_dir) / DEFAULT_PREPROCESSING_LOG_DIRNAME


def select_sample_indices(dataset_size: int, sample_count: int, seed: int) -> list[int]:
    if dataset_size <= 0 or sample_count <= 0:
        return []
    effective_count = min(dataset_size, sample_count)
    return random.Random(seed).sample(range(dataset_size), k=effective_count)


def write_jsonl_records(log_dir: Path, filename: str, records: Iterable[Mapping]) -> Path:
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / filename
    with log_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return log_path
