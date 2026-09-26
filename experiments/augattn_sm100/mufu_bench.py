import sys, torch, drv
from common import graph_time
k = drv.Kernel(sys.argv[1], "mufu_bench", 0)
out = torch.zeros(1, device="cuda"); n = 2000; nch = int(sys.argv[2])
nsm = torch.cuda.get_device_properties(0).multi_processor_count
for th in (128, 256, 384, 512, 1024):
    t = graph_time(lambda: k((nsm, 1, 1), (th, 1, 1), out, n), reps=5)
    ops = nsm * th * n * nch
    print(f"threads {th:5d}: {t:8.1f} us  {ops / t / 1e6:8.2f} Tex2/s  per SM per ns {ops / t / 1e3 / nsm:6.2f}", flush=True)
