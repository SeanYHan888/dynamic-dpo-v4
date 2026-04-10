import argparse
import json
import os


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--generation_file_dir",
        type=str,
        help="Diretory containing the generation files",
        default="datasets/gemma2_ultrafeedback",
    )
    return parser


def run(args: argparse.Namespace) -> str:
    all_data = []
    for file_name in os.listdir(args.generation_file_dir):
        if file_name.startswith("output") and file_name.endswith(".json"):
            generation_file = os.path.join(args.generation_file_dir, file_name)
            with open(generation_file, "r", encoding="utf-8") as f:
                output_data = json.load(f)
                all_data.append(output_data)

    num_samples = len(all_data[0])
    all_res = []
    num_identical = 0
    for i in range(num_samples):
        prompt = all_data[0][i]["prompt"]
        gen_text = []
        for data in all_data:
            gen_text.append(data[i]["generated_text"])

        if len(set(gen_text)) == 1:
            num_identical += 1
            continue

        all_res.append(
            {
                "prompt": prompt,
                "all_generated_responses": gen_text,
            }
        )

    print(f"Filtered out {num_identical} samples with identical generated responses")

    output_path = os.path.join(args.generation_file_dir, "all_outputs.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_res, f, indent=4)

    print(f"Processed outputs saved to {output_path}")
    return output_path


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    print(args)
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

