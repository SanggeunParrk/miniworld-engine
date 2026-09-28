"""Shared setup for the short-large-D optimization experiments."""
import hashlib
import os
from pathlib import Path
import statistics
import sys

ROOT = Path('/workspace/experiments/trimul-large-d/runs')
STAGE = ROOT / 'trimul_d256_bwd_sol90_stage2_20260923'
PRE = ROOT / 'trimul_d256_bwd_sol90_20260923'
sys.path.insert(0, str(PRE))
ns = {'__file__': str(PRE / 'check.py')}
exec(compile((PRE / 'check.py').read_text().split('ap=argparse.ArgumentParser()')[0],
             str(PRE / 'check.py'), 'exec'), ns)
sys.path.insert(0, str(STAGE))
from runtime import pin_lt_library
pin_lt_library()
setup, T, error = ns['setup'], ns['T'], ns['error']
import torch
from validate_engine import capture

OUT = Path(os.environ.get('TRIMUL_SHORT_RESULTS','/workspace/vast-results/trimul-short-large-d'))
OUT.mkdir(parents=True,exist_ok=True)
os.environ.update(PREFIX_IMPL='blas', PREFIX_COPY='tma', CHECKPOINT_LN_THREADS='0')

def training(width):
    if width == 256:
        from d256_pool_checkpoint import Training, configure
        configure()
    elif width == 384:
        from wide_checkpoint23 import Training
    elif width == 512:
        from wide_checkpoint24 import Training
    else:
        raise ValueError(f'No wide checkpoint for D={width}')
    return Training

def paired(graphs, count=51):
    for g in graphs.values():
        for _ in range(5): g.replay()
    vals = {k: [] for k in graphs}
    for i in range(count):
        for k in (list(graphs) if i % 2 else list(graphs)[::-1]):
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record(); graphs[k].replay(); end.record(); end.synchronize()
            vals[k].append(start.elapsed_time(end) * 1000)
    return {k: {'median_us': statistics.median(v), 'samples_us': v} for k, v in vals.items()}

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def strict(errors):
    return all(v < (2e-5 if n == 'dx' else 5e-6 if n.startswith(('dgamma', 'dbeta')) else 5e-4)
               for n, v in errors.items())
