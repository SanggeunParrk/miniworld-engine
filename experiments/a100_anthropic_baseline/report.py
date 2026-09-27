"""Aggregate results/*.json into results.csv and the markdown tables pasted into README.md."""
import csv
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
RES = HERE / (sys.argv[1] if len(sys.argv) > 1 else "results")
MODULES = [("trimul", "양방향 TriMul"), ("single", "단방향 TriMul (outgoing)"), ("transition", "Transition"),
           ("block", "MiniPairformer 1블록"), ("opm", "OPM"), ("pwa", "PWA")]

rows = []
for f in sorted(RES.glob("*.json")):
    r = json.loads(f.read_text())
    for mode, v in r.get("modes", {}).items():
        rows.append(dict(module=r["module"], L=r["L"], arm=r["arm"], mode=mode, ms=v.get("ms"),
                         rel_rms=v.get("rel_rms_vs_fp32"), finite=v.get("finite"),
                         selection=r.get("selection") if r["arm"].startswith("anth:") else None,
                         error=(v.get("error") or "").strip().splitlines()[-1][:160] if v.get("error") else None,
                         gpu=r.get("gpu"), host=r.get("host")))
with open(RES.parent / f"{RES.name}.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)


def get(m, L, arm, mode):
    for r in rows:
        if (r["module"], r["L"], r["arm"], r["mode"]) == (m, L, arm, mode):
            return r
    return None


def best_anth(m, L):
    ok = [r for r in rows if r["module"] == m and r["L"] == L and r["arm"].startswith("anth:") and r["ms"] is not None]
    return min(ok, key=lambda r: r["ms"]) if ok else None


def fmt(r):
    return "—" if r is None else ("오류" if r["ms"] is None else f"{r['ms']:.3f}")


out = []
out.append("## 추론 forward (ms)\n")
out.append("| 모듈 | L | PyTorch | cuEquivariance | Engine(main, A100) | **Anthropic 최선** | 최선 row | Anthropic rel-RMS |")
out.append("|---|---:|---:|---:|---:|---:|---|---:|")
for m, name in MODULES:
    for L in (384, 768):
        b = best_anth(m, L)
        err = f"{b['rel_rms']:.2e}" if b else "—"
        out.append(f"| {name} | {L} | {fmt(get(m, L, 'pytorch', 'inference'))} | {fmt(get(m, L, 'cuequiv', 'inference'))} | "
                   f"{fmt(get(m, L, 'engine', 'inference'))} | **{fmt(b)}** | {b['arm'][5:] if b else '—'} | {err} |")
out.append("\n## 학습 forward + backward (ms) — Anthropic 공개 커널에는 backward가 없음\n")
out.append("| 모듈 | L | PyTorch | cuEquivariance | Engine(main, A100) |")
out.append("|---|---:|---:|---:|---:|")
for m, name in MODULES:
    for L in (384, 768):
        out.append(f"| {name} | {L} | {fmt(get(m, L, 'pytorch', 'training'))} | {fmt(get(m, L, 'cuequiv', 'training'))} | "
                   f"{fmt(get(m, L, 'engine', 'training'))} |")
out.append("\n## Anthropic row/config 전체 (추론 ms, rel-RMS vs fp32)\n")
for m, name in MODULES:
    out.append(f"**{name}**\n")
    out.append("| row | L384 | L768 |")
    out.append("|---|---|---|")
    arms = sorted({r["arm"] for r in rows if r["module"] == m and r["arm"].startswith("anth:")})
    for a in arms:
        cells = []
        for L in (384, 768):
            r = get(m, L, a, "inference")
            if r is None:
                cells.append("—")
            elif r["ms"] is None:
                cells.append("실패: " + (r["error"] or "?").replace("|", "/")[:90])
            else:
                cells.append(f"{r['ms']:.3f} ({r['rel_rms']:.2e})")
        out.append(f"| {a[5:]} | {cells[0]} | {cells[1]} |")
    out.append("")
text = "\n".join(out)
(RES.parent / f"{RES.name}-tables.md").write_text(text + "\n")
print(text)
