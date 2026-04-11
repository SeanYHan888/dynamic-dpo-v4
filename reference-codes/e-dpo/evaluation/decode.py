import argparse
import importlib.util
import json
import os


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Decode with vllm")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="HuggingFaceH4/ultrafeedback_binarized",
        help="Directory containing the data",
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
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="datasets/gemma2_ultrafeedback",
        help="output_dir",
    )
    parser.add_argument(
        "--attention_backend",
        type=str,
        choices=["auto", "flashinfer", "default"],
        default="auto",
        help=(
            "Attention backend policy for vLLM. "
            "`auto` uses FlashInfer if available, "
            "`flashinfer` requires it, and `default` leaves vLLM backend unset."
        ),
    )
    return parser


def configure_attention_backend(attention_backend: str):
    if attention_backend == "default":
        os.environ.pop("VLLM_ATTENTION_BACKEND", None)
        return

    has_flashinfer = importlib.util.find_spec("flashinfer") is not None

    if attention_backend == "flashinfer":
        if not has_flashinfer:
            raise RuntimeError(
                "Requested --attention_backend flashinfer, but `flashinfer` is not installed."
            )
        os.environ["VLLM_ATTENTION_BACKEND"] = "FLASHINFER"
        return

    if has_flashinfer:
        os.environ["VLLM_ATTENTION_BACKEND"] = "FLASHINFER"
    else:
        os.environ.pop("VLLM_ATTENTION_BACKEND", None)
        print(
            "Warning: `flashinfer` is not installed. "
            "Proceeding with vLLM default attention backend."
        )


def run(args: argparse.Namespace) -> str:
    configure_attention_backend(getattr(args, "attention_backend", "auto"))

    from datasets import load_dataset
    from vllm import LLM, SamplingParams

    llm = LLM(model=args.model)
    tokenizer = llm.get_tokenizer()

    train_dataset = load_dataset(args.data_dir, split="train_prefs")
    prompts = sorted(list(set(train_dataset["prompt"])))
    conversations = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for prompt in prompts
    ]

    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        seed=args.seed,
    )
    outputs = llm.generate(conversations, sampling_params)

    output_data = []
    for i, output in enumerate(outputs):
        output_data.append(
            {
                "prompt": prompts[i],
                "format_prompt": output.prompt,
                "generated_text": output.outputs[0].text,
            }
        )

    output_file = f"output_{args.seed}.json"
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    output_path = os.path.join(args.output_dir, output_file)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=4)

    print(f"Outputs saved to {output_path}")
    return output_path


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    print(args)
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
