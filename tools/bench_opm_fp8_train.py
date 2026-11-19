"""OPM FP8 training step-time benchmark (H100, sm90). Run under Slurm on one GPU, e.g.
    python tools/bench_opm_fp8_train.py cases.txt out.jsonl     (cases: "L S d_msa d_pair after" per line, after 1 = AF3 order)
Same process, CUDA graph of forward + backward (autograd.grad wrt msa and all weights), routes interleaved:
  def = default route, ctl = default again (same-cubin control), e4m3 / e5m2 = MINIWORLD_OPM_FP8_TRAIN forced on (no shape gate).
cool = 1.5 s idle then a ~40 ms burst (median of 3); sustained = after 1 s back-to-back, 3 blocks of ~60 ms (mean). Resumable (rows already
in the output file are skipped); stops cleanly when $DEADLINE (epoch seconds) is within 90 s."""
import copy
import json
import os
import sys
import time

import torch

from miniworld_engine.kernels.integrations import opm_fp8_train as F8
from miniworld_engine.measure import config as C
from miniworld_engine.modules import OuterProductMean


def case(L, S, cm, cz, after, seed=0):
    gen = torch.Generator(device="cuda").manual_seed(seed * 7919 + L + S)
    base = OuterProductMean(cm, cz, 32, normalize_before_proj=not after).to("cuda")
    C.randomize(base, gen)
    base = C.cast(base, "master")
    inp = base.make_inputs(length=L, samples=1, depth=S, dtype=torch.bfloat16, device="cuda", generator=gen)
    msa, mask = inp["msa"], inp["mask"]
    mask[:, :] = True
    mask[:, S - S // 8 :, L // 2 :] = False  # ragged MSA: the last S/8 rows cover half the tokens
    gz = torch.randn((1, L, L, cz), device="cuda", generator=gen, dtype=torch.float32).to(torch.bfloat16)
    return base, msa, mask, gz


def make_graph(mod, msa, mask, gz, mode):
    mode, _, ostore = (mode or "").partition(":")  # route "e4m3:fp8" = e4m3 with O kept in fp8 ("e4m3:recompute": recomputed)
    mode = mode or None
    F8.OSTORE = ostore or "bf16"
    F8.MODE = mode
    F8.MIN_S = 0
    F8.MIN_L = 0
    F8.MAX_CZ = 0
    params = list(mod.parameters())
    x = msa.detach().clone().requires_grad_(True)

    def step():
        with torch.autocast("cuda", dtype=torch.bfloat16):
            y = mod(x, mask)
        return torch.autograd.grad(y, [x, *params], gz)

    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            step()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        out = step()
    torch.cuda.synchronize()
    F8.MODE = None
    F8.OSTORE = "bf16"
    return g, out


def burst(g, n):
    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    e0.record()
    for _ in range(n):
        g.replay()
    e1.record()
    e1.synchronize()
    return e0.elapsed_time(e1) / n


def run(L, S, cm, cz, after, routes=("def", "ctl", "e4m3", "e4m3:fp8", "e4m3:recompute")):
    base, msa, mask, gz = case(L, S, cm, cz, after)
    mod = copy.deepcopy(base).set_backend("cuda")
    gs = {r: make_graph(mod, msa, mask, gz, None if r in ("def", "ctl") else r)[0] for r in routes}
    t1 = {r: burst(g, 1) for r, g in gs.items()}
    nb = {r: max(1, min(200, int(40.0 / max(t, 1e-3)))) for r, t in t1.items()}
    cool = {r: [] for r in gs}
    for _ in range(3):
        for r, g in gs.items():
            time.sleep(1.5)
            cool[r].append(burst(g, nb[r]))
    t_end = time.time() + 1.0
    while time.time() < t_end:
        for g in gs.values():
            g.replay()
    sus = {r: [] for r in gs}
    for _ in range(3):
        for r, g in gs.items():
            sus[r].append(burst(g, max(1, int(nb[r] * 1.5))))
    med = lambda v: sorted(v)[len(v) // 2]  # noqa: E731
    row = {"L": L, "S": S, "cm": cm, "cz": cz, "after": after,
           "cool": {r: round(med(v), 4) for r, v in cool.items()}, "sus": {r: round(sum(v) / len(v), 4) for r, v in sus.items()}}
    row["ratio_cool"] = {r: round(row["cool"][r] / row["cool"]["def"], 3) for r in gs}
    row["ratio_sus"] = {r: round(row["sus"][r] / row["sus"]["def"], 3) for r in gs}
    print(json.dumps(row), flush=True)
    return row


if __name__ == "__main__":
    out = sys.argv[2]
    deadline = float(os.environ.get("DEADLINE", "1e12"))
    done = set()
    if os.path.exists(out):
        for ln in open(out):
            if ln.startswith("{"):
                d = json.loads(ln)
                done.add((d["L"], d["S"], d["cm"], d["cz"], d["after"]))
    for ln in open(sys.argv[1]):
        if not ln.strip() or ln.startswith("#"):
            continue
        L, S, cm, cz, after = (int(t) for t in ln.split())
        if (L, S, cm, cz, bool(after)) in done:
            print("SKIP", ln.strip(), flush=True)
            continue
        if deadline - time.time() < 90:
            print("DEADLINE", flush=True)
            break
        try:
            row = run(L, S, cm, cz, bool(after))
            with open(out, "a") as f:
                f.write(json.dumps(row) + "\n")
        except Exception as e:  # noqa: BLE001
            import traceback

            traceback.print_exc()
            print("ERR", ln.strip(), e, flush=True)
        torch.cuda.empty_cache()
