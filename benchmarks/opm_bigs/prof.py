"""Profile driver: one OPM case (row, L, S, backend) run N times inside cudaProfilerStart/Stop (for nsys / ncu --profile-from-start off).
   python prof.py ROW L S [backend] [iters]"""
import copy, dataclasses, sys, torch
from foldjet_engine.measure import config, speed
from foldjet_engine.modules.primitives import unpadded_masks

row, L, S = map(int, sys.argv[1:4]); backend = sys.argv[4] if len(sys.argv) > 4 else "cuda"; iters = int(sys.argv[5]) if len(sys.argv) > 5 else 3
dev = torch.device("cuda")
cfg = config.Config(modules=("OuterProductMean",), rows=(row,), lengths=(L,), depths="min", modes=("eval",), samples="min")
case = dataclasses.replace(next(iter(config.cases(cfg))), depth=S)
print("case", case.row, case.kwargs, case.length, case.depth, flush=True)
base = config.build(case, device=dev); module = copy.deepcopy(base)
with unpadded_masks():
    inputs = config.inputs(case, module, device=dev)
step = speed.make_step(module, inputs, backend=backend, train=False, autocast=config.autocast(case), seed=config.seed(case), compile_mode=None)
for _ in range(3): step()
torch.cuda.synchronize()
torch.cuda.cudart().cudaProfilerStart()
for _ in range(iters): step()
torch.cuda.synchronize()
torch.cuda.cudart().cudaProfilerStop()
print("done", flush=True)
