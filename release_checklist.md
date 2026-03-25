# Release Checklist

Use this checklist whenever you train, export, and publish a model checkpoint.

## 1. Before Training

- Use a unique `output_dir` and `run_name` for every run.
- Do not reuse an older run directory.
- Do not make Hugging Face the only copy you plan to keep.
- For important runs, avoid `save_strategy: "no"` unless you have another durable checkpoint copy.

## 2. After Training Finishes

- Confirm the final local checkpoint folder exists and is complete.
- Keep that folder intact until all validation and publishing steps succeed.
- Do not delete the training machine copy yet.

Expected files in the final checkpoint folder:

- `config.json`
- `generation_config.json` if present
- `model.safetensors.index.json`
- `model-*.safetensors`
- `tokenizer.json`
- `tokenizer_config.json`
- `special_tokens_map.json`

## 3. Validate the Local Checkpoint

Run the validation script from this repo:

```bash
uv run python scripts/validate_and_upload.py \
  --checkpoint-dir /ABS/PATH/TO/CHECKPOINT \
  --skip-upload
```

This checks:

- required export files exist
- tensor headers match the model config
- tokenizer and config load cleanly

Optional heavier smoke test:

```bash
uv run python scripts/validate_and_upload.py \
  --checkpoint-dir /ABS/PATH/TO/CHECKPOINT \
  --skip-upload \
  --full-model-load
```

Use `--full-model-load` only on a machine with enough RAM/VRAM.

## 4. Validate Inference Before Publishing

Do not publish a checkpoint that has never been loaded for inference.

Minimum:

- `transformers` config load succeeds
- tokenizer load succeeds
- vLLM or inference backend load succeeds on the target machine class
- one short generation succeeds

For this repo, the eval pipeline now has a preflight check and will fail early on broken exports.

## 5. Upload the Validated Folder

Upload from the already-validated local folder, not by re-saving through `Trainer.push_to_hub()`.

```bash
uv run python scripts/validate_and_upload.py \
  --checkpoint-dir /ABS/PATH/TO/CHECKPOINT \
  --repo-id W-61/YOUR-MODEL-NAME \
  --commit-message "Upload validated checkpoint"
```

If you are replacing an existing repo and want to remove stale old shard files that are not present locally:

```bash
uv run python scripts/validate_and_upload.py \
  --checkpoint-dir /ABS/PATH/TO/CHECKPOINT \
  --repo-id W-61/YOUR-MODEL-NAME \
  --commit-message "Replace checkpoint with validated export" \
  --delete-stale-remote-artifacts
```

Only use `--delete-stale-remote-artifacts` when the local folder is the complete desired repo state for model artifacts.

## 6. Verify the Published Repo

After upload:

- download or load the model fresh from Hugging Face
- run the same validation again against the published artifact
- run one inference smoke test
- run the exact benchmark command you plan to use

For AlpacaEval in this repo:

```bash
uv run alpacaeval-infer --config alpacaeval/config_alpacaeval_llama3_8b_base_beta_dpo_benchmark.yaml
```

## 7. Only Then Clean Up

Do not delete local copies until:

- local validation passed
- upload succeeded
- fresh Hub validation passed
- benchmark startup succeeded

Recommended retention:

- one training-machine copy
- one second backup copy
- one published Hugging Face repo

## 8. Failure Rules

If any of these happen, stop and fix before publishing:

- embedding or `lm_head` shape mismatch
- tokenizer/config mismatch
- vLLM load failure
- missing shard files
- partial upload

Never try to repair a corrupted checkpoint by editing only `config.json` or tokenizer metadata unless you have independently verified the weights are valid.
