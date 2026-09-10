# TIGER Beauty environment report

Status: pre-run audit. Runtime evidence was collected from the server `grid`
environment on 2026-09-10; the final GPU model and memory record must be
captured by `tools/audit_environment.sh` in the same shell used for training.

| Field | Recorded value |
| --- | --- |
| Repository commit | `b98f06ad3ad851edddedfb6b64b08dcf02a605d0` |
| Remote | `https://github.com/hahahahaeil/tiger.git` |
| Root marker | `.project-root` exists at repository root |
| Server Python | 3.10.20 (`grid`) |
| PyTorch | 2.6.0+cu124 |
| CUDA runtime reported by PyTorch | 12.4 |
| Transformers / Lightning / Hydra | 4.47.0 / 2.5.0 / 1.3.2 |
| GPUs visible | 8; CUDA available |
| GPU model and total memory | pending `nvidia-smi` capture |
| Seed | 42 |

The local desktop Python is 3.12.14 and has no PyTorch installed. It is only
used for static auditing; every smoke and formal run must use server `grid`.

Run and retain the resulting terminal log before the smoke test:

```bash
cd /data2/hnlcm/lab12/tiger
conda activate grid
bash tools/audit_environment.sh | tee logs/audit/environment_seed1.txt
```

The resolved Hydra configuration is part of the run evidence. Each command
must retain `<run_dir>/.hydra/config.yaml`, `<run_dir>/.hydra/overrides.yaml`,
and the command log. Do not overwrite a previous run directory.
