import sys, time, torch
from common import make_inputs
from fwd_op import FusedFwd2
from bwd_op import FusedTrain
L = int(sys.argv[1]); R = int(sys.argv[2])
x, wa, wb, ws, g, b = make_inputs(L); dy = torch.randn_like(x) * 0.1
f = FusedFwd2(); f.set_weights(wa, wb, ws)
tr = FusedTrain(f, repl=R, x=True, cubin=(sys.argv[3] if len(sys.argv) > 3 else None)); st = tr.bind(x, g, b, dy)
run_f, run_b = st.keep
run_f(); torch.cuda.synchronize(); print("fwd ok", flush=True)
for k in range(3):
    run_b(); ev = torch.cuda.Event(); ev.record(); t0 = time.time()
    while not ev.query():
        if time.time() - t0 > 8: print("HANG at launch", k); dab, dflags, epoch = run_b.keep[8:]; raise SystemExit
        time.sleep(0.05)
    dab, dflags, epoch = run_b.keep[8:]
    print("launch", k, "ok; flags min/max", dflags.min().item(), dflags.max().item(), "epoch", epoch.item(), flush=True)
