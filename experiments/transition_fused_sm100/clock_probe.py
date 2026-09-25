import sys, time, subprocess, threading, torch
from common import make_inputs
from fwd_op import FusedFwd
from bwd_op import FusedTrain
what = sys.argv[1]
x, wa, wb, ws, g, b = make_inputs(768)
f = FusedFwd(); f.set_weights(wa, wb, ws)
if what == "fwd":
    run, *_ = f.bind(x, g, b, save=False)
elif what == "train":
    run = FusedTrain(f).bind(x, g, b, torch.randn_like(x) * 0.1)
else:
    A = torch.randn(8192, 8192, device="cuda", dtype=torch.bfloat16); B_ = torch.randn_like(A)
    run = lambda: torch.mm(A, B_)
samples = []
def sample():
    for _ in range(8):
        time.sleep(0.5)
        out = subprocess.run(["nvidia-smi", "--query-gpu=clocks.sm,power.draw,temperature.gpu,clocks_throttle_reasons.active", "--format=csv,noheader", "-i", "7"], capture_output=True, text=True).stdout.strip()
        samples.append(out)
th = threading.Thread(target=sample); th.start()
t0 = time.time()
while th.is_alive():
    for _ in range(50): run()
    torch.cuda.synchronize()
th.join()
print(what, samples[2:])
