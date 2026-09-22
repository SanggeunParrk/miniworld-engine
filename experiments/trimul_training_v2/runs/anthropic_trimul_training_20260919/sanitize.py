import importlib.util
from pathlib import Path
import torch
p=Path('/home/psk6950/MiniWorld/runs/trimul_sm90_parity_20260917/engine/tests/numerics/test_trimul_anthropic_training_gpu.py')
s=importlib.util.spec_from_file_location('checks',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
for cfg in ((2,64,4,1,232,1),(1,64,4,1,232,1),(1,128,4,2,240,0)):
 m.test_padded_ragged_tma(cfg)
 torch.cuda.synchronize()
 print('PASS',cfg,flush=True)
