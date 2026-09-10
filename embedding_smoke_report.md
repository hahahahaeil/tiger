# Beauty embedding smoke report

Status: **failed before embedding inference** on 2026-09-10. Per the smoke
stop condition, RQ-VAE training, Semantic-ID export, and TIGER forward/beam
search were not started.

## Executed command

```bash
/usr/bin/time -v python -m src.inference \
  experiment=beauty_smoke_sem_embeds \
  data_dir=/data2/hnlcm/dataset/amazon_data/amazon_data/beauty \
  hydra.run.dir=logs/audit/beauty_smoke_embedding
```

Resolved configuration evidence was printed and identifies
`google/flan-t5-xl`, one GPU, one item shard, batch size 4, seed 42, and the
intended output directory
`/data2/hnlcm/lab12/tiger/logs/audit/beauty_smoke_embedding`.

## Root cause

The initial Hugging Face metadata request to `https://huggingface.co` timed out
after 10 seconds. Transformers then confirmed that `google/flan-t5-xl` was not
available in the server's local Hugging Face cache. Failure happened while
Hydra instantiated `AutoTokenizer.from_pretrained`; no item TFRecord was read,
no FLAN-T5-XL model was loaded, and no embedding tensor was created.

The process exited with code 1 after 37.22 seconds. Maximum resident memory was
1,072,316 KiB. No per-process GPU-memory value can be attributed to the failed
run because inference never initialized CUDA model weights. The expected file
`pickle/merged_predictions_tensor.pt` does not exist; the subsequent audit
script's `FileNotFoundError` is a correct downstream consequence, not a second
root cause.

## Separate cleanup defect observed

After the tokenizer exception, `pipeline_launcher` raised a secondary
`UnboundLocalError` because its `finally` block evaluates
`pipeline_modules.trainer` even if `initialize_pipeline_modules` has failed.
This does not cause the embedding failure, but it obscures clean error
reporting. The minimal non-model-code repair is to initialize
`pipeline_modules = None` before the `try` and call `finalize_loggers` only
when initialization succeeded. This repair has **not** been applied under the
smoke stop condition.

## Minimal retry prerequisite

Keep `google/flan-t5-xl`; do not substitute a smaller encoder. First make the
model available in the server cache or restore stable Hugging Face connectivity.
When retrying the same stage, use a longer Hub metadata/download timeout:

```bash
export HF_HUB_ETAG_TIMEOUT=60
export HF_HUB_DOWNLOAD_TIMEOUT=600
```

Before retrying, record `du -sh "${HF_HOME:-$HOME/.cache/huggingface}"` and
check whether `models--google--flan-t5-xl` exists. The successful retry must
run `tools/audit_smoke_tensor.py embedding` only after the inference command
returns exit code 0.

## Retry 1 result

Retry 1 used `HF_HUB_ETAG_TIMEOUT=60` and `HF_HUB_DOWNLOAD_TIMEOUT=600`, but
failed after 74.62 seconds with `ConnectionResetError: [Errno 104] Connection
reset by peer` during the TLS handshake to `huggingface.co`. Increasing the
timeout did not change the outcome, so this is now classified as an upstream
network-connectivity block rather than an application timeout. No model files
were cached and no embedding output exists.

The minimal recovery path is to obtain the unchanged official
`google/flan-t5-xl` snapshot on a host with stable Hugging Face access, transfer
it to the server, and pass that local directory through `embedding_model`.
This preserves the configured encoder and does not substitute a model.
