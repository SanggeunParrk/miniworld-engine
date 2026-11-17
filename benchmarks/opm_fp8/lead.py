"""Lead vs Anthropic (opm_win fair protocol): lead = (PyTorch_engine / CUDA) / (PyTorch_kit / Anthropic) (> 1: the engine is faster), cool (median) and sustained.
Default CUDA = opm_win v3_after* rows (2.5.11 = 2.5.12 OPM), FP8 = res/fair_fp8.jsonl (bound scales, lazy guard)."""
import glob, json
R = "/home/psk6950/miniworld-engine-scratch/wt-fp8x-opm/benchmarks/opm_win/raw/"
def load(paths):
    d = {}
    for p in paths:
        for l in open(p):
            r = json.loads(l)
            if r.get("status") == "ok":
                d[(r["row"], r["length"], r["depth"], r["backend"])] = r
    return d
kit = load(glob.glob(R + "v3_kit*.jsonl"))
dflt = load(sorted(glob.glob(R + "v3_after*.jsonl")))
f8 = load(["/home/psk6950/miniworld-engine-scratch/fp8x-opm/res/fair_fp8.jsonl"])
print("| row | L | S | Anthropic ms cool | default ms cool | FP8 ms cool | lead default cool / sus | lead FP8 cool / sus | FP8 vs default (PyTorch-normalised) cool / sus |")
print("|---|---|---|---|---|---|---|---|---|")
for (row, L, S, b) in sorted(f8):
    if b != "cuda":
        continue
    try:
        kp, ka = kit[(row, L, S, "pytorch")], kit[(row, L, S, "anthropic")]
        ep, ec = dflt[(row, L, S, "pytorch")], dflt[(row, L, S, "cuda")]
        fp, fc = f8[(row, L, S, "pytorch")], f8[(row, L, S, "cuda")]
    except KeyError as e:
        print("missing", e); continue
    out = []
    for k in ("cool_ms", "sus_ms"):
        la = kp[k] / ka[k]
        out.append(((ep[k] / ec[k]) / la, (fp[k] / fc[k]) / la, (ep[k] / ec[k]) / (fp[k] / fc[k])))
    print(f"| {row} | {L} | {S} | {ka['cool_ms']:.2f} | {ec['cool_ms']:.2f} | {fc['cool_ms']:.2f} | {out[0][0]:.3f} / {out[1][0]:.3f} | {out[0][1]:.3f} / {out[1][1]:.3f} | {out[0][2]:.3f} / {out[1][2]:.3f} |")
