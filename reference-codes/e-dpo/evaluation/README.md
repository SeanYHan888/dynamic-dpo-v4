# Evaluation Pipeline

This folder contains the evaluation data generation pipeline extracted from
`on_policy_data_gen`.

It also contains benchmark assets and command wrappers for:
- AlpacaEval 2
- Arena-Hard
- MT-Bench

## Environment

Use Python 3.10 for compatibility with evaluation and acceleration packages.

Install dependencies from the repository root:

```bash
pip install -r requirements.txt
```

## Steps

1. Decode model outputs under one or more seeds:

```bash
python evaluation/decode.py \
  --data_dir $DATASET_DIR \
  --seed $SEED \
  --attention_backend auto
```

`--attention_backend` options:
- `auto` (default): use FlashInfer when available, else fall back to vLLM default backend.
- `flashinfer`: require FlashInfer and fail if unavailable.
- `default`: always use vLLM default backend.

2. Merge outputs and remove identical generations:

```bash
python evaluation/post_process.py --generation_file_dir $OUTPUT_DIR
```

3. Score with reward model and binarize:

```bash
python evaluation/reward_model_annotate.py \
  --generation_file $OUTPUT_DIR/all_outputs.json \
  --reward_model $MODEL \
  --output_dir $OUTPUT_DIR
```

4. Run the full pipeline in one command:

```bash
python evaluation/pipeline.py \
  --data_dir $DATASET_DIR \
  --model $MODEL \
  --reward_model $REWARD_MODEL \
  --output_dir $OUTPUT_DIR
```

## Benchmarks

Benchmark configs/templates are under:
- `evaluation/benchmarks/alpacaeval2`
- `evaluation/benchmarks/arenahard`
- `evaluation/benchmarks/mt-bench`

Print benchmark commands (no execution):

```bash
python evaluation/run_benchmarks.py \
  --benchmarks alpacaeval2,arenahard,mt-bench
```

Execute commands (requires external benchmark repos/tools installed):

```bash
python evaluation/run_benchmarks.py \
  --benchmarks alpacaeval2,arenahard,mt-bench \
  --execute \
  --arenahard_repo /path/to/arena-hard-auto \
  --fastchat_repo /path/to/FastChat \
  --mtbench_model_path /path/to/your/model
```

When using `--execute`, ensure:
- `--arenahard_repo` points to a valid `arena-hard-auto` checkout.
- `--fastchat_repo` points to a valid `FastChat` checkout.
- `--mtbench_model_path` points to the local model to evaluate.
- Required API credentials are set in your environment (for example `OPENAI_API_KEY`).

You can also trigger benchmark commands at the end of `evaluation/pipeline.py`
with `--run_benchmarks` (and `--execute_benchmarks` to actually run them).
