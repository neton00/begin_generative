"""로컬 CUDA 설정과 동기화된 벤치마크. import 시 연산을 실행하지 않는다.

GEN_DEVICE=cuda (기본): CUDA가 없으면 명확히 실패한다.
GEN_DEVICE=cpu: CPU 실행을 명시적으로 허용한다.
타이밍 근거: https://docs.pytorch.org/docs/stable/notes/cuda.html
"""
from __future__ import annotations

import os
import random
import time


def configure(seed=42):
    import numpy as np
    import torch

    requested = os.environ.get('GEN_DEVICE', 'cuda')
    if requested.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError(
            'CUDA를 사용할 수 없습니다. setup_gpu.ps1로 설치한 .venv 커널과 '
            'NVIDIA 드라이버를 확인하세요. CPU 실행은 GEN_DEVICE=cpu로 명시하세요.'
        )
    device = torch.device(requested)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(seed)
    # 수치 검증에 FP32 기본 정밀도를 유지. AMP는 GPT 학습에서만 별도 사용.
    torch.set_float32_matmul_precision('highest')
    print(f'PyTorch {torch.__version__} | CUDA runtime {torch.version.cuda} | device {device}')
    if device.type == 'cuda':
        props = torch.cuda.get_device_properties(device)
        print(f'{props.name} | VRAM {props.total_memory / 1024**3:.1f} GiB')
    return torch, device


def benchmark(fn, device, warmup=3, repeat=10):
    import torch

    def sync():
        if device.type == 'cuda':
            torch.cuda.synchronize(device)

    with torch.inference_mode():
        for _ in range(warmup):
            fn()
        sync()
        samples = []
        for _ in range(repeat):
            sync()
            start = time.perf_counter()
            fn()
            sync()
            samples.append(time.perf_counter() - start)
    return min(samples)


def matmul_benchmark(device):
    import pandas as pd
    import torch

    generator = torch.Generator(device=device).manual_seed(0)
    rows = []
    dtype = torch.float16 if device.type == 'cuda' else torch.float32
    for n in [256, 512, 1024, 2048]:
        a = torch.randn(n, n, device=device, dtype=dtype, generator=generator)
        b = torch.randn(n, n, device=device, dtype=dtype, generator=generator)
        v = torch.randn(n, 1, device=device, dtype=dtype, generator=generator)
        mm = torch.empty_like(a)
        mv = torch.empty_like(v)
        t_mm = benchmark(lambda: torch.mm(a, b, out=mm), device)
        t_mv = benchmark(lambda: torch.mm(a, v, out=mv), device)
        rows.append(dict(n=n, dtype=str(dtype), matmul_ms=1000*t_mm,
                         matvec_ms=1000*t_mv, matmul_GFLOPS=2*n**3/t_mm/1e9,
                         matvec_GFLOPS=2*n**2/t_mv/1e9))
    return pd.DataFrame(rows).set_index('n')

