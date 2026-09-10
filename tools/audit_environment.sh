#!/usr/bin/env bash
set -euo pipefail
python --version
python - <<'PY'
import platform
import torch
print('platform=', platform.platform())
print('torch=', torch.__version__)
print('cuda_available=', torch.cuda.is_available())
print('torch_cuda=', torch.version.cuda)
print('gpu_count=', torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    print(f'gpu_{i}=name:{p.name}; total_memory_bytes:{p.total_memory}')
PY
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader
git rev-parse HEAD
git status --short
