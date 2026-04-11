import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation import decode, post_process, reward_model_annotate, run_benchmarks


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run evaluation pipeline.")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="HuggingFaceH4/ultrafeedback_binarized",
        help="Directory containing prompt data",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="google/gemma-2-9b-it",
        help="Path to the LLM model",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="Temperature for sampling",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.95,
        help="Top-p probability for sampling",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=4096,
        help="Maximum number of tokens to generate",
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default="13,21,42,79,100",
        help="Comma-separated random seeds for decoding",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="datasets/gemma2_ultrafeedback",
        help="Output directory",
    )
    parser.add_argument(
        "--reward_model",
        type=str,
        default="RLHFlow/ArmoRM-Llama3-8B-v0.1",
        help="Path to reward model",
    )
    parser.add_argument(
        "--run_benchmarks",
        action="store_true",
        help="Run benchmark commands after data generation pipeline.",
    )
    parser.add_argument(
        "--benchmarks",
        type=str,
        default="alpacaeval2,arenahard,mt-bench",
        help="Comma-separated benchmarks to run.",
    )
    parser.add_argument(
        "--benchmark_assets_dir",
        type=str,
        default="evaluation/benchmarks",
        help="Benchmark assets directory.",
    )
    parser.add_argument(
        "--execute_benchmarks",
        action="store_true",
        help="Actually execute benchmark commands. Otherwise print commands only.",
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
        help="AlpacaEval2 output directory.",
    )
    parser.add_argument(
        "--alpacaeval_extra_args",
        type=str,
        default="",
        help="Extra args for alpaca_eval command.",
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
        help="Model config folder name under arenahard/configs.",
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
        help="Model path for MT-Bench generation.",
    )
    parser.add_argument(
        "--mtbench_model_id",
        type=str,
        default="simpo-model",
        help="Model id for MT-Bench generation/judging.",
    )
    parser.add_argument(
        "--mtbench_judge_model",
        type=str,
        default="gpt-4-1106-preview",
        help="Judge model for MT-Bench.",
    )
    return parser


def run(args: argparse.Namespace):
    seed_list = [int(seed.strip()) for seed in args.seeds.split(",") if seed.strip()]

    for seed in seed_list:
        decode.run(
            argparse.Namespace(
                data_dir=args.data_dir,
                model=args.model,
                temperature=args.temperature,
                top_p=args.top_p,
                max_tokens=args.max_tokens,
                seed=seed,
                output_dir=args.output_dir,
            )
        )

    generation_file = post_process.run(
        argparse.Namespace(generation_file_dir=args.output_dir)
    )

    reward_model_annotate.run(
        argparse.Namespace(
            generation_file=generation_file,
            reward_model=args.reward_model,
            output_dir=args.output_dir,
        )
    )

    if args.run_benchmarks:
        run_benchmarks.run(
            argparse.Namespace(
                benchmarks=args.benchmarks,
                assets_dir=args.benchmark_assets_dir,
                execute=args.execute_benchmarks,
                alpacaeval_config=args.alpacaeval_config,
                alpacaeval_output_dir=args.alpacaeval_output_dir,
                alpacaeval_extra_args=args.alpacaeval_extra_args,
                arenahard_repo=args.arenahard_repo,
                arenahard_model_config=args.arenahard_model_config,
                fastchat_repo=args.fastchat_repo,
                mtbench_model_path=args.mtbench_model_path,
                mtbench_model_id=args.mtbench_model_id,
                mtbench_judge_model=args.mtbench_judge_model,
            )
        )


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    print(args)
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
