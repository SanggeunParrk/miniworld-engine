"""SM clock / power while (gemm, epilogue) or epilogue alone loop for a few seconds: nvidia-smi sampled alongside."""
import os, subprocess, sys, time, pathlib, torch
sys.argv = [sys.argv[0]]
exec(open(pathlib.Path(__file__).parent / "t_epibits.py").read().split("A2 = torch.randn")[0].split("for name, a in")[0])
A2 = torch.randn(N * CH, S, device="cuda", dtype=bf); BT = torch.randn(N * CH, S, device="cuda", dtype=bf); Ob = torch.empty_like(O)
gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0]
for name, fn in (("epilogue alone", lambda: ext.opm_epilogue(O, bits, wo, bias, N, N, res)),
                 ("gemm + epilogue", lambda: (torch.matmul(A2, BT.t(), out=Ob), ext.opm_epilogue(Ob, bits, wo, bias, N, N, res)))):
    p = subprocess.Popen(["nvidia-smi", "-i", gpu, "--query-gpu=clocks.sm,clocks.mem,power.draw,clocks_throttle_reasons.active", "--format=csv,noheader", "-lms", "200"], stdout=subprocess.PIPE, text=True)
    t0 = time.time()
    while time.time() - t0 < 4:
        for _ in range(50): fn()
        torch.cuda.synchronize()
    p.terminate(); out = p.communicate()[0].strip().splitlines()
    print(name, "|", " ; ".join(out[len(out)//2:len(out)//2+4]))
