#!/usr/bin/env bash
#SBATCH --job-name=mw-h100-memcheck
#SBATCH --account=gpu
#SBATCH --partition=gpu-4farm-bf
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h100:4
#SBATCH --cpus-per-task=112
#SBATCH --mem=896G
#SBATCH --time=01:00:00
set -euo pipefail
cd /home/psk6950/miniworld-engine
export CONDA_PREFIX="$PWD/.pixi/envs/default"
export PATH="$CONDA_PREFIX/bin:$PATH"
export CUDA_HOME="$CONDA_PREFIX"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export PYTHONNOUSERSITE=1 PYTHONPATH="$PWD/src"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MAX_JOBS=28
export CUDA_LAUNCH_BLOCKING=1
export TRITON_CACHE_DIR="$PWD/.scratch/h100-build/triton-cache"
export XDG_CACHE_HOME="$PWD/.scratch/h100-build/cache"
export MINIWORLD_PLAN_CACHE_DIR="$PWD/.scratch/h100-build/plans"
export TORCH_EXTENSIONS_DIR="$PWD/.scratch/h100-build/torch-extensions"
export QUACK_CACHE_DIR="$PWD/.scratch/h100-build/quack-cache"
nvidia-smi
python - <<'PY'
import concurrent.futures
import os
from pathlib import Path
import subprocess
import sys

root = Path.cwd()
out = root / '.scratch/h100-memory-debug' / os.environ['SLURM_JOB_ID']
out.mkdir(parents=True, exist_ok=True)
devices = os.environ['CUDA_VISIBLE_DEVICES'].split(',')
assert len(devices) == 4, devices

def run(index):
    attention = index < 2
    length = (128, 256, 128, 5120)[index]
    name = f'{"attention" if attention else "adaln"}-L{length}'
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = devices[index]
    family = 'attention' if attention else 'adaln'
    warps = '4' if index % 2 == 0 else '2'
    # Isolate each candidate in its own CUDA context. Memcheck supplies the exact
    # instruction and address space; a later candidate never inherits a device fault.
    for stages in ('1', '2'):
        probe = f'{family}-w{warps}-s{stages}'
        command = ['compute-sanitizer', '--tool', 'memcheck', '--error-exitcode', '99',
                   sys.executable, 'scripts/repro-h100-memory.py', family,
                   '--warps', warps, '--stages', stages]
        with (out / f'{probe}.log').open('w') as log:
            try:
                rc = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT,
                                    timeout=300).returncode
            except subprocess.TimeoutExpired:
                rc = 124
        print(probe, 'rc', rc, flush=True)
    args = [sys.executable, '-m', 'miniworld_engine.autotune.builder',
            '--shard', str(out / f'{name}.json'), '--length', str(length),
            '--compile-jobs', '28', '--dtype', 'bfloat16', '--rebuild-cached',
            '--bench-clear-mb', '16', '--bench-rep-ms', '10',
            '--config-dir', str(root / 'src/miniworld_engine/autotune/configs/grid')]
    if attention:
        args += ['--case', 'augmented_attention', '--dims', '2', '--mode', 'train',
                 '--compute-dtype', 'bfloat16']
    else:
        args += ['--op', 'adaln_gemm_gate_triton', '--width', '768', '--heads', '384',
                 '--side', 'token']
        env.update(MINIWORLD_DRIVER_LENGTH=str(length), MINIWORLD_DRIVER_WIDTH='768',
                   MINIWORLD_DRIVER_HEADS='384', MINIWORLD_DRIVER_SIDE='token',
                   MINIWORLD_DRIVER_DTYPE='bf16')
    with (out / f'{name}.log').open('w') as log:
        print('COMMAND', args, file=log, flush=True)
        rc = subprocess.run(args, env=env, stdout=log, stderr=subprocess.STDOUT,
                            timeout=3300).returncode
    print(name, 'rc', rc, 'log', out / f'{name}.log', flush=True)
    return rc

with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
    results = list(pool.map(run, range(4)))
sys.exit(int(any(results)))
PY
