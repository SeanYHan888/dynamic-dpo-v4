import argparse
import re
import shlex
import subprocess
import tempfile
from pathlib import Path


DEFAULT_BENCHMARKS = ["alpacaeval2", "arenahard", "mt-bench"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run benchmark commands for AlpacaEval2, Arena-Hard, and MT-Bench."
    )
    parser.add_argument(
        "--benchmarks",
        type=str,
        default="alpacaeval2,arenahard,mt-bench",
        help="Comma-separated benchmarks to run.",
    )
    parser.add_argument(
        "--assets_dir",
        type=str,
        default="evaluation/benchmarks",
        help="Directory containing benchmark assets copied from eval/.",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually execute commands. By default commands are printed only.",
    )

    parser.add_argument(
        "--alpacaeval_config",
        type=str,
        default="evaluation/benchmarks/alpacaeval2/configs/Llama-3-Instruct-8B-SimPO.yaml",
        help="AlpacaEval2 model config YAML.",
    )
    parser.add_argument(
        "--alpacaeval_output_dir",
        type=str,
        default="evaluation/results/alpacaeval2",
        help="Output directory for AlpacaEval2 run.",
    )
    parser.add_argument(
        "--alpacaeval_extra_args",
        type=str,
        default="",
        help="Extra args appended to the alpaca_eval command.",
    )

    parser.add_argument(
        "--arenahard_repo",
        type=str,
        default="",
        help="Path to arena-hard-auto repo root.",
    )
    parser.add_argument(
        "--arenahard_model_config",
        type=str,
        default="Llama-3-Instruct-8B-SimPO",
        help="Model config folder name under arenahard/configs/.",
    )

    parser.add_argument(
        "--fastchat_repo",
        type=str,
        default="",
        help="Path to FastChat repo root.",
    )
    parser.add_argument(
        "--mtbench_model_path",
        type=str,
        default="",
        help="Model path for FastChat MT-Bench generation.",
    )
    parser.add_argument(
        "--mtbench_model_id",
        type=str,
        default="simpo-model",
        help="Model id for FastChat MT-Bench generation/judging.",
    )
    parser.add_argument(
        "--mtbench_judge_model",
        type=str,
        default="gpt-4-1106-preview",
        help="Judge model passed to FastChat judgment.",
    )
    return parser


def _selected_benchmarks(benchmarks: str):
    selected = [item.strip() for item in benchmarks.split(",") if item.strip()]
    if not selected:
        raise ValueError("At least one benchmark must be selected.")
    unknown = [item for item in selected if item not in DEFAULT_BENCHMARKS]
    if unknown:
        raise ValueError(
            f"Unknown benchmarks: {unknown}. Supported: {DEFAULT_BENCHMARKS}"
        )
    return selected


def _run_or_print(cmd: str, execute: bool, cwd: str | None = None):
    if cwd:
        print(f"[cwd={cwd}] {cmd}")
    else:
        print(cmd)
    if execute:
        subprocess.run(cmd, shell=True, check=True, cwd=cwd)


def _prepare_alpacaeval_config(config_path: Path) -> Path:
    config_dir = config_path.parent
    templates_root = config_dir.parent
    changed = False
    updated_lines = []

    with open(config_path, "r", encoding="utf-8") as f:
        for line in f:
            match = re.match(r"^(\s*prompt_template:\s*)(\S+)(\s*)$", line)
            if not match:
                updated_lines.append(line)
                continue

            prefix, value, suffix = match.groups()
            quote = ""
            prompt_template = value
            if (value.startswith('"') and value.endswith('"')) or (
                value.startswith("'") and value.endswith("'")
            ):
                quote = value[0]
                prompt_template = value[1:-1]

            template_path = Path(prompt_template)
            if template_path.is_absolute():
                updated_lines.append(line)
                continue

            candidates = [
                (config_dir / template_path),
                (templates_root / template_path),
            ]
            resolved = next(
                (candidate.resolve() for candidate in candidates if candidate.exists()),
                None,
            )
            if resolved is None:
                raise FileNotFoundError(
                    f"prompt_template not found: {prompt_template} "
                    f"(checked: {candidates[0]}, {candidates[1]})"
                )

            resolved_value = f"{quote}{resolved}{quote}" if quote else str(resolved)
            updated_lines.append(f"{prefix}{resolved_value}{suffix}\n")
            changed = True

    if not changed:
        return config_path

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", prefix="alpacaeval_", delete=False, encoding="utf-8"
    ) as tmp:
        tmp.writelines(updated_lines)
        return Path(tmp.name)


def _run_alpacaeval2(args: argparse.Namespace):
    Path(args.alpacaeval_output_dir).mkdir(parents=True, exist_ok=True)
    config_path = Path(args.alpacaeval_config).expanduser().resolve()
    if not config_path.exists():
        raise FileNotFoundError(
            f"AlpacaEval config not found: {config_path}. "
            "Check the path (for example, use 'Llama-3-...' not 'LLlama-3-...')."
        )
    runtime_config_path = config_path
    if args.execute:
        runtime_config_path = _prepare_alpacaeval_config(config_path)
    cmd = (
        f"alpaca_eval evaluate_from_model "
        f"--model_configs {shlex.quote(str(runtime_config_path))} "
        f"--output_path {shlex.quote(args.alpacaeval_output_dir)}"
    )
    if args.alpacaeval_extra_args:
        cmd = f"{cmd} {args.alpacaeval_extra_args}"
    try:
        _run_or_print(cmd, execute=args.execute)
    finally:
        if args.execute and runtime_config_path != config_path:
            runtime_config_path.unlink(missing_ok=True)


def _run_arenahard(args: argparse.Namespace):
    if not args.arenahard_repo and args.execute:
        raise ValueError("--arenahard_repo is required for arenahard when --execute is set.")

    base = (
        Path(args.assets_dir)
        / "arenahard"
        / "configs"
        / args.arenahard_model_config
    )
    api_config = base / "api_config.yaml"
    gen_answer_config = base / "gen_answer_config.yaml"
    judge_config = base / "judge_config.yaml"

    commands = [
        f"python3 gen_answer.py --api-config {shlex.quote(str(api_config))} --config {shlex.quote(str(gen_answer_config))}",
        f"python3 gen_judgment.py --api-config {shlex.quote(str(api_config))} --config {shlex.quote(str(judge_config))}",
        f"python3 show_result.py --config {shlex.quote(str(judge_config))}",
    ]
    cwd = args.arenahard_repo if args.arenahard_repo else "/path/to/arena-hard-auto"
    for command in commands:
        _run_or_print(command, execute=args.execute, cwd=cwd)


def _run_mtbench(args: argparse.Namespace):
    if (not args.fastchat_repo or not args.mtbench_model_path) and args.execute:
        raise ValueError(
            "--fastchat_repo and --mtbench_model_path are required for mt-bench when --execute is set."
        )

    assets_dir = Path(args.assets_dir)
    src_ref = assets_dir / "mt-bench" / "gpt-4-1106-preview.jsonl"
    fastchat_repo = args.fastchat_repo if args.fastchat_repo else "/path/to/FastChat"
    mtbench_model_path = (
        args.mtbench_model_path if args.mtbench_model_path else "/path/to/your/model"
    )

    dst_ref = (
        Path(fastchat_repo)
        / "fastchat"
        / "llm_judge"
        / "data"
        / "mt_bench"
        / "reference_answer"
        / "gpt-4.jsonl"
    )
    if args.execute:
        dst_ref.parent.mkdir(parents=True, exist_ok=True)
    copy_cmd = f"cp {shlex.quote(str(src_ref))} {shlex.quote(str(dst_ref))}"
    _run_or_print(copy_cmd, execute=args.execute)

    commands = [
        (
            "python3 -m fastchat.llm_judge.gen_model_answer "
            f"--model-path {shlex.quote(mtbench_model_path)} "
            f"--model-id {shlex.quote(args.mtbench_model_id)} "
            "--bench-name mt_bench"
        ),
        (
            "python3 -m fastchat.llm_judge.gen_judgment "
            "--bench-name mt_bench "
            f"--judge-model {shlex.quote(args.mtbench_judge_model)} "
            f"--model-list {shlex.quote(args.mtbench_model_id)}"
        ),
        (
            "python3 -m fastchat.llm_judge.show_result "
            "--bench-name mt_bench "
            f"--model-list {shlex.quote(args.mtbench_model_id)}"
        ),
    ]
    for command in commands:
        _run_or_print(command, execute=args.execute, cwd=fastchat_repo)


def run(args: argparse.Namespace):
    selected = _selected_benchmarks(args.benchmarks)
    for benchmark in selected:
        print(f"\n=== {benchmark} ===")
        if benchmark == "alpacaeval2":
            _run_alpacaeval2(args)
        elif benchmark == "arenahard":
            _run_arenahard(args)
        elif benchmark == "mt-bench":
            _run_mtbench(args)


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    print(args)
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
