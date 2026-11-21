"""E5_<card>.csv (collect.py) + the accuracy records -> E5_results.md: one table per operator class per card, the cells where a candidate beats the engine, the accuracy
against fp64 per candidate, and the candidates table.    python report.py   (reads E5_{h100,a100,rtx5090,b200}.csv and records/ next to this file; writes E5_results.md)"""

from __future__ import annotations

import csv
import re
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
import externals
from collect import class_of
from paper_common import csvio

CARDS = [("h100", "H100 (sm_90)"), ("a100", "A100 (sm_80)"), ("rtx5090", "RTX 5090 (sm_120)"), ("b200", "B200 (sm_100)")]
CLASSES = ["TriAttn", "TriMul", "AttentionPairBias-384", "AttentionPairBias-768", "OPM", "PWA", "Transition-d128", "Transition-d64"]
TITLE = {
    "TriAttn": "TriAttn (TriangleAttention, starting, d_pair 128, 4 heads of 32)",
    "TriMul": "TriMul (TriangleMultiplication, outgoing, d_pair 128)",
    "AttentionPairBias-384": "AttentionPairBias d_single 384, 16 heads (confidence / Pairformer single attention; master dtype)",
    "AttentionPairBias-768": "AttentionPairBias d_single 768, 16 heads x 48 channels (AF3 token DiT attention; master dtype)",
    "OPM": "OuterProductMean (d_msa 64, d_pair 128, d_hidden 32, depth 1024)",
    "PWA": "MSAPairWeightedAveraging (d_msa 64, d_pair 128, 8 heads, depth 1024)",
    "Transition-d128": "Transition d_hidden 128, n 4 (pair transition)",
    "Transition-d64": "Transition d_hidden 64, n 4 (MSA transition, depth 384)",
}
CONTEXT = ["pytorch", "cuda", "cueq"]
FASTER, EQUAL = 0.97, 1.03  # ratio to the engine's cuda: <= FASTER: candidate faster; < EQUAL: equal within 3 %


def code_of(status: str, reason: str) -> str:
    r = reason or ""
    if "OutOfMemory" in r or "out of memory" in r:
        return "OOM"
    if status == "failed":
        return "FAIL"
    if re.search(r"no backward|forward-only|inference only|no backward kernel", r):
        return "NOBWD"
    if re.search(r"Hopper|GLIBC|glibc|INVALID_IMAGE|cannot be loaded|not compatible", r):
        return "HW"
    return "SHAPE"


CODE_TEXT = {
    "OOM": "out of memory on this card (skipped; the case is not re-routed)",
    "FAIL": "the library failed on this card (error recorded)",
    "NOBWD": "no backward kernel: training not served (forward-only or inference-only library)",
    "HW": "hardware / software not supported on this card (see the footnotes)",
    "SHAPE": "the library (or, for the engine, its kernel table) does not serve this shape / dtype / layout",
}


def load(card: str) -> list[dict]:
    with open(HERE / f"E5_{card}.csv") as fh:
        rows = list(csv.DictReader(fh))
    for r in rows:
        r["L"] = int(r["L"])
        r["ms"] = float(r["ms_median"]) if r["ms_median"] != "" else None
        r["ms_e"] = float(r["ms_eager"]) if r.get("ms_eager", "") != "" else None
        r["inv"] = r["status"] == "invalid_graph"
        if r["inv"] and r["ms_e"] is not None:
            r["ms"] = r["ms_e"]  # the graph time is not a measurement; the corrected (eager) time stands in, marked
        r["iterations"] = int(r["iterations"]) if r["iterations"] else None
    return rows


def eng_ms(eng: dict, r: dict) -> float:
    """The engine time a candidate cell is compared with: eager for an eager (invalid-graph) cell, else the graph time."""
    return (eng["ms_e"] or eng["ms"]) if r["inv"] and eng["ms_e"] else eng["ms"]


def fmt(ms: float) -> str:
    return f"{ms:.3f}" if ms < 100 else f"{ms:.1f}"


def main() -> None:
    data = {c: load(c) for c, _ in CARDS}
    order = list(externals.AVAILABLE)
    notes: dict[tuple, set] = defaultdict(set)  # (backend, code, reason) -> cards
    out: list[str] = []
    w = out.append
    w("# E5: other public kernels (all operator classes, all candidates)\n")
    w(
        "Source: `E5_<card>.csv` (collect.py) from the existing records only: the first E5 run (`ext`: TriAttn + ds4sci / trifast), the third-party-trimul run (`ext_trimul`), the third-party-attn run"
        " (`ext_attn`) and a few small runs of this integration (`e5`: cueq for TriMul on every card, all TriMul backends on the 5090 at 20 iterations). AF3 setup, L = 512 / 1024 / 2048, eval and train, 3 repeats"
        " (median of the repeats' medians), 20 timed iterations (warmup 5, L2 flushed, CUDA-graph timing) unless the `iterations` column of the CSV says otherwise. Median ms per call of the whole module"
        " (LayerNorm, projections, gate, out-projection and residual are the module's own; only the core differs for the attention libraries, the whole update for the TriMul libraries). Cells: `ms (xR)`"
        " with R = candidate / engine cuda (R < 1: faster than the engine). `e` after a number = eager timing (CUDA events) of a cell whose graph timing failed the CUDA-graph validity check (section 0), compared with the engine's eager time. Codes of cells without a number: "
        + "; ".join(f"`{k}` {v}" for k, v in CODE_TEXT.items())
        + ".\n"
    )

    # ---------------------------------------------------------------- CUDA-graph validity
    w("## 0. CUDA-graph validity of every candidate (graph_check.py)\n")
    w(
        "The measurer times `graph.replay()` of a step captured on a side stream. A library that launches its kernels on the legacy default stream (DeepSpeed `evoformer_attn`: "
        "attention_cu.cu / attention_back.cu launch without a stream argument) leaves the capture: the replay then times an empty graph. `graph_check.py` runs each (backend, case) with the same step "
        "and checks (a) two replays give the eager output, (b) train: parameter gradients non-zero and of the eager norm, (c) replay time not below half of the eager GPU kernel time, "
        "(d) the largest eager kernels appear in the profile of the replay. A cell that fails is `invalid_graph` in the CSV (excluded from the tables' graph columns); its corrected eager "
        "time (CUDA events, same warmup / iterations / L2 flush) is shown with an `e` and compared with the engine's eager time. `pytorch` is not checked (plain torch ops).\n"
    )
    w("| backend | " + " | ".join(label for _c, label in CARDS) + " |\n|---|" + "---|" * len(CARDS))
    allb = ["cuda", "cueq", *externals.AVAILABLE]
    for b in allb:
        cells = []
        any_row = False
        for card, _ in CARDS:
            rs = [r for r in data[card] if r["backend"] == b]
            if not rs:
                cells.append("n/a")
                continue
            any_row = True
            val = [r for r in rs if r.get("graph_valid") == "valid"]
            inv = [r for r in rs if r.get("graph_valid") == "invalid"]
            unc = [r for r in rs if r.get("graph_valid") == "not checked" and r["status"] == "ok"]
            cerr = [r for r in rs if r.get("graph_valid") == "check error" and r["status"] == "ok"]
            if inv:
                why = min({re.sub(r"\s+", " ", r["reason"]).split(";")[0][:110] for r in inv})
                cells.append(f"INVALID {len(inv)} of {len(inv) + len(val)} cells ({why}); valid {len(val)}")
            elif val:
                cells.append(f"valid ({len(val)} cells)" + (f"; {len(unc)} ok cells unchecked" if unc else ""))
            elif unc or cerr:
                cells.append(f"not checked ({len(unc) + len(cerr)} ok cells)")
            else:
                cells.append("no ok cell (unsupported / failed / skipped)")
            if cerr and (inv or val):
                cells[-1] += f"; {len(cerr)} ok cells: check could not run (out of memory / crash / library error)"
        if any_row:
            w(f"| `{b}` | " + " | ".join(cells) + " |")
    w("")
    # ---------------------------------------------------------------- faster cells
    faster, equal, noengine = [], [], []
    best_rows = defaultdict(list)
    for card, _ in CARDS:
        cells = defaultdict(dict)
        for r in data[card]:
            if r["class"] in CLASSES and r["dtype"] != "float32":
                cells[(r["class"], r["mode"], r["L"])][r["backend"]] = r
        for (cls, mode, L), bs in cells.items():
            eng = bs.get("cuda")
            cands = {b: r for b, r in bs.items() if b not in ("pytorch", "cuda") and r["ms"] is not None}
            if not cands:
                continue
            b, rb = min(cands.items(), key=lambda kv: kv[1]["ms"])
            if eng is None or eng["ms"] is None:
                noengine.append((card, cls, mode, L, eng["status"] if eng else "no record", b, rb["ms"]))
                best_rows[(card, cls)].append((mode, L, None, b, rb["ms"]))
                continue
            best_rows[(card, cls)].append((mode, L, eng_ms(eng, rb), b, rb["ms"]))
            for b2, r2 in cands.items():
                em = eng_ms(eng, r2)
                q = r2["ms"] / em
                tag = f"{b2}" + (" (eager)" if r2["inv"] else "")
                if q <= FASTER:
                    faster.append((card, cls, mode, L, tag, r2["ms"], em, q))
                elif q < EQUAL:
                    equal.append((card, cls, mode, L, tag, r2["ms"], em, q))
    w("## 1. Cells where a candidate is faster than the engine\n")
    w(
        f"Engine = the engine's `cuda` backend. Faster: candidate <= {FASTER:.2f} x engine; equal: within +-3 %. Every class on every card was checked (all `ok` cells with an engine `cuda` number).\n"
    )
    if faster:
        w("| card | class | mode | L | candidate | candidate ms | engine ms | candidate / engine |\n|---|---|---|---|---|---|---|---|")
        for card, cls, mode, L, b, ms, e, q in sorted(faster, key=lambda t: (t[0], CLASSES.index(t[1]), t[2], t[3], t[4])):
            w(f"| {card} | {cls} | {mode} | {L} | `{b}` | {fmt(ms)} | {fmt(e)} | x{q:.2f} |")
    else:
        w(
            "No cell. (An earlier reading that `ds4sci` beats the engine at TriAttn train L = 1024 / 2048 and eval L = 2048 on H100 and A100 is withdrawn: those graph timings were of an empty graph, see section 0; the eager timings of `ds4sci` are slower than the engine in every TriAttn cell.)"
        )
    w("")
    if equal:
        w("Equal within 3 % (candidate / engine in 0.97 .. 1.03):\n")
        w("| card | class | mode | L | candidate | candidate ms | engine ms | candidate / engine |\n|---|---|---|---|---|---|---|---|")
        for card, cls, mode, L, b, ms, e, q in sorted(equal, key=lambda t: (t[0], CLASSES.index(t[1]), t[2], t[3], t[4])):
            w(f"| {card} | {cls} | {mode} | {L} | `{b}` | {fmt(ms)} | {fmt(e)} | x{q:.2f} |")
        w("")
    if noengine:
        w("Cells where the engine's `cuda` backend has no number (no kernel for the shape: any candidate that runs is ahead by default):\n")
        w("| card | class | mode | L | engine cuda | best candidate | its ms |\n|---|---|---|---|---|---|---|")
        for card, cls, mode, L, st, b, ms in sorted(noengine, key=lambda t: (t[0], CLASSES.index(t[1]), t[2], t[3])):
            w(f"| {card} | {cls} | {mode} | {L} | {st} | `{b}` | {fmt(ms)} |")
        w("")

    # ---------------------------------------------------------------- engine vs best candidate
    w("## 2. Engine against the best candidate (per class, per card)\n")
    w(
        "Best candidate = the fastest `ok` backend other than `pytorch` and the engine's `cuda` (cuEquivariance counts). Ratio = best candidate / engine (> 1: the engine is ahead).\n"
    )
    w("| card | class | " + " | ".join(f"{m} {L}" for m in ("eval", "train") for L in (512, 1024, 2048)) + " |\n|---|---|" + "---|" * 6)
    for card, _ in CARDS:
        for cls in CLASSES:
            rows = {(m, L): (e, b, ms) for m, L, e, b, ms in best_rows.get((card, cls), [])}
            cells = []
            for m in ("eval", "train"):
                for L in (512, 1024, 2048):
                    t = rows.get((m, L))
                    cells.append(
                        "-" if not t else (f"no engine number; `{t[1]}` {fmt(t[2])}" if t[0] is None else f"`{t[1]}` x{t[2] / t[0]:.2f}")
                    )
            w(f"| {card} | {cls} | " + " | ".join(cells) + " |")
    w("")

    # ---------------------------------------------------------------- per class tables
    w("## 3. Tables per operator class and card\n")
    for card, label in CARDS:
        w(f"### {label}\n")
        rows = data[card]
        for cls in CLASSES:
            crs = [r for r in rows if r["class"] == cls and r["dtype"] != "float32"]
            if not crs:
                continue
            backs = sorted(
                {r["backend"] for r in crs}, key=lambda b: CONTEXT.index(b) if b in CONTEXT else 3 + (order.index(b) if b in order else 99)
            )
            by = {(r["mode"], r["L"], r["backend"]): r for r in crs}
            w(f"**{TITLE[cls]}**\n")
            w("| mode | L | " + " | ".join(f"`{b}`" for b in backs) + " |\n|---|---|" + "---|" * len(backs))
            for mode in ("eval", "train"):
                for L in (512, 1024, 2048):
                    eng = by.get((mode, L, "cuda"))
                    cells = []
                    for b in backs:
                        r = by.get((mode, L, b))
                        if r is None:
                            cells.append("n/a")
                        elif r["ms"] is None:
                            c = "INV" if r["inv"] else code_of(r["status"], r["reason"])
                            cells.append(c)
                            notes[(b, c, re.sub(r"\s+", " ", r["reason"])[:170])].add(card)
                        elif b in ("pytorch", "cuda") or eng is None or eng["ms"] is None:
                            cells.append(
                                fmt(r["ms"]) + ("e" if r["inv"] else "") + ("" if r["iterations"] == 20 else f" ({r['iterations']} it)")
                            )
                        else:
                            cells.append(
                                f"{fmt(r['ms'])}{'e' if r['inv'] else ''} (x{r['ms'] / eng_ms(eng, r):.2f})"
                                + (" INVALID-GRAPH" if r["inv"] else "")
                                + ("" if r["iterations"] == 20 else f" ({r['iterations']} it)")
                            )
                    w(f"| {mode} | {L} | " + " | ".join(cells) + " |")
            w("")
            fl = sorted([t for t in faster if t[0] == card and t[1] == cls], key=lambda t: (t[2], t[3], t[4]))
            eq = sorted([t for t in equal if t[0] == card and t[1] == cls], key=lambda t: (t[2], t[3], t[4]))
            ne = sorted([t for t in noengine if t[0] == card and t[1] == cls], key=lambda t: (t[2], t[3]))
            line = "Faster than the engine: " + (
                "; ".join(f"{m} L={L} `{b}` {fmt(ms)} ms vs engine {fmt(e)} ms (x{q:.2f})" for _c, _k, m, L, b, ms, e, q in fl)
                if fl
                else "none"
            )
            if eq:
                line += ". Equal within 3 %: " + "; ".join(f"{m} L={L} `{b}` {fmt(ms)} vs {fmt(e)} ms" for _c, _k, m, L, b, ms, e, q in eq)
            if ne:
                line += ". Engine cuda without a number: " + "; ".join(f"{m} L={L} ({st})" for _c, _k, m, L, st, _b, _ms in ne)
            w(line + ".\n")
    w("Reasons behind the codes (backend, code: recorded reason, cards):\n")
    for (b, c, why), cards in sorted(notes.items(), key=lambda kv: (kv[0][0], kv[0][1], kv[0][2])):
        if c == "OOM":
            why = "CUDA out of memory (allocation larger than the free memory of the card at that size)"
        w(f"- `{b}` {c}: {why} ({', '.join(sorted(cards))})")
    w("")

    # ---------------------------------------------------------------- accuracy
    w("## 4. Accuracy against the float64 gold, per candidate\n")
    w(
        "Two harnesses, both against the module's own float64 PyTorch result with the same weights and inputs. Part A (`miniworld_engine.measure.accuracy`, the engine's accuracy standard; smallest length L = 512,"
        " master dtype): relative error of the checked tensor (`out` for eval, the worst gradient tensor for train) and the PyTorch backend's own error on the same case; the light compares with the tolerance"
        " (GREEN within, YELLOW slightly above, RED far above; backends that are external baselines are informational). The accuracy runs are eager (not graph) runs: the CUDA-graph problem of `ds4sci` does not touch them. Part B (TriMul, `acc_trimul.py`): relative rms of the output and of the gradients"
        " (`dx`, `dWo`) at L = 512, d_pair 128, no padding.\n"
    )
    acc_files = {"ext_attn_acc": "third-party-attn run", "e5acc": "this integration (ds4sci / trifast added)"}
    for card, label in CARDS:
        recs = {}
        for stem in acc_files:
            for f in sorted((HERE / "records" / card).glob(f"{stem}.r*.jsonl")):
                for r in csvio.read_jsonl(str(f)):
                    if r.get("dtype") != "master" or r["length"] != 512:
                        continue
                    cls = class_of(r["module"], r["kwargs"])
                    recs[(cls, r["backend"], r["mode"])] = (r, stem)
        if not recs:
            w(f"### {label}\n")
            w(
                f"No E5 accuracy record exists for {label}: the B200 campaign ran the speed stages and the graph check; the E5 accuracy stages (`e5acc`, `acc_trimul.py`) were not run there. "
                "The engine's own accuracy against float64 on this card (cuda, triton, cueq, pytorch; E1 cases) is in `E3_accuracy/E3_b200.csv`; the candidates' accuracy is not measured on B200.\n"
            )
            continue
        w(f"### {label}: part A (L = 512)\n")
        w("| class | backend | eval error (light) | train error (light, tensor) | note |\n|---|---|---|---|---|")
        backs = sorted(
            {(c, b) for c, b, _m in recs},
            key=lambda t: (
                CLASSES.index(t[0]) if t[0] in CLASSES else 99,
                CONTEXT.index(t[1]) if t[1] in CONTEXT else 3 + (order.index(t[1]) if t[1] in order else 99),
            ),
        )
        for cls, b in backs:
            cells = []
            note = ""
            for mode in ("eval", "train"):
                t = recs.get((cls, b, mode))
                if not t:
                    cells.append("-")
                    continue
                r, stem = t
                if r["status"] == "measured":
                    base = f"{r['err_backend']:.2e} ({r['light']}" + (f", {r['tensor']})" if mode == "train" else ")")
                    cells.append(base + (f" [pytorch {r['err_pytorch']:.2e}]" if b != "pytorch" else ""))
                else:
                    cells.append(f"{r['status']}: {code_of(r['status'], r.get('reason', ''))}")
                if stem == "e5acc" and b in ("ds4sci", "trifast"):
                    note = "new in this integration"
            w(f"| {cls} | `{b}` | {cells[0]} | {cells[1]} | {note} |")
        w("")
        f = HERE / "records" / "acc" / f"ext_trimul_acc.{card}.jsonl"
        if not recs:
            continue
        w(f"{label}: part B (TriMul, L = 512, d_pair 128)\n")
        w("| backend | output rel rms | output rel max | update rel rms | dx rel rms | dWo rel rms |\n|---|---|---|---|---|---|")
        out_rows, gr = {}, {}
        for r in csvio.read_jsonl(str(f)):
            if r.get("case") != "512:128" or r.get("padded_mask"):
                continue
            (gr if r.get("kind") == "grad" else out_rows)[r["backend"]] = r
        for b in ["pytorch", "cuda", "fast_trimul", "gpumode", "geonn"]:
            r = out_rows.get(b)
            if not r:
                continue
            g = gr.get(b)
            if r["status"] != "ok":
                w(f"| `{b}` | {r['status']}: {code_of(r['status'], r.get('reason', ''))} | | | | |")
                continue
            gx = f"{g['dx']['rel_rms']:.2e}" if g and g.get("status") == "ok" else "-"
            gw = f"{g['dWo']['rel_rms']:.2e}" if g and g.get("status") == "ok" else "-"
            w(f"| `{b}` | {r['rel_rms']:.2e} | {r['rel_max']:.2e} | {r['upd_rel_rms']:.2e} | {gx} | {gw} |")
        w("")

    # ---------------------------------------------------------------- candidates
    w("## 5. Candidates\n")
    lic = {
        "ds4sci": ("DeepSpeed4Science `DS4Sci_EvoformerAttention` (deepspeed 0.20.0)", "Apache-2.0"),
        "trifast": ("trifast 0.1.13 (Triton)", "MIT"),
        "fast_trimul": ("fast_trimul 3.0.6 (CuTe DSL, fp16)", "Apache-2.0"),
        "gpumode": ("GPU MODE TriMul competition kernels (arseniivanov/trimul, Triton)", "MIT"),
        "geonn": ("AMD geometric-nn-ops `triangle_multiplicative_update` (Triton)", "Apache-2.0"),
        "gnops": ("AMD geometric-nn-ops `triangle_attention` (Triton)", "Apache-2.0"),
        "of3tri": ("OpenFold3 Triton Evoformer attention", "Apache-2.0"),
        "megafold": ("MegaFold EvoAttention (Triton)", "MIT"),
        "bionemo_ir": ("NVIDIA BioNeMo Inference Runtime (CuTeDSL CUBINs)", "Apache-2.0"),
        "turbopf": ("TurboPairFormer Hopper triangle attention (CUDA)", "Apache-2.0"),
        "ligo_pwa": ("Ligo AlphaFold3 Triton PWA kernel (patched)", "Apache-2.0"),
        "megafold_tr": ("MegaFold fused Transition (Triton)", "MIT"),
        "cueq": ("NVIDIA cuEquivariance (engine's external baseline)", "Apache-2.0"),
    }
    why_not = {
        "bionemo_ir": "H100: the host glibc 2.28 < 2.34 required by its CuTeDSL library; A100: CUDA_ERROR_INVALID_IMAGE (its CUBINs do not load on driver 570); B200: runs (inference only); inference only (no backward); Transition: no shipped silu_gate variant for width 128 / hidden 512 or 64 / 256",
        "turbopf": "Hopper (sm_90) only: A100, RTX 5090 and B200 refuse; TriAttn only; d <= 64",
        "ds4sci": "RTX 5090: the CUTLASS extension does not JIT-load on sm_120 (`evoformer_attn op ... not compatible`); needs CUTLASS headers + nvcc; B200: loads, but the graph timing is invalid (eager timings used, section 0)",
        "gpumode": "forward only (no backward): training cells unsupported; its output-gate placement is remapped by the adapter",
        "geonn": "forward only (autograd path is its PyTorch reference): training cells unsupported",
        "fast_trimul": "fp16 kernels (bf16 cast), L % 8 == 0, no mask; training = kernel forward + torch-recompute backward",
        "ligo_pwa": "PWA only",
        "megafold_tr": "Transition only (bf16)",
        "gnops": "none recorded (TriAttn + AttentionPairBias; its `attention_pair_bias` op is pure torch and not used)",
        "of3tri": "none recorded",
        "megafold": "none recorded",
        "trifast": "none recorded",
        "cueq": "TriAttn and TriMul only",
    }
    w(
        "| candidate (`backend`) | library | license | classes it serves | cards with a measured cell (ok / cells) | why not elsewhere |\n|---|---|---|---|---|---|"
    )
    for b in [
        "ds4sci",
        "trifast",
        "cueq",
        "fast_trimul",
        "gpumode",
        "geonn",
        "gnops",
        "of3tri",
        "megafold",
        "bionemo_ir",
        "turbopf",
        "ligo_pwa",
        "megafold_tr",
    ]:
        cl = sorted(
            {r["class"] for c in data for r in data[c] if r["backend"] == b and r["status"] in ("ok", "invalid_graph")},
            key=lambda c: CLASSES.index(c) if c in CLASSES else 99,
        )
        cards = []
        for c, _ in CARDS:
            n = [r for r in data[c] if r["backend"] == b]
            ok = [r for r in n if r["status"] == "ok"]
            inv = [r for r in n if r["status"] == "invalid_graph"]
            cards.append(f"{c} {len(ok)}/{len(n)}" + (f" (+{len(inv)} eager-timing only, graph invalid)" if inv else ""))
        name, lc = lic[b]
        w(f"| `{b}` | {name} | {lc} | {', '.join(cl) or '-'} | {'; '.join(cards)} | {why_not[b]} |")
    w(
        "| FlashBias | low-rank-bias attention | MIT | none | not connected | NOT APPLICABLE: the pair bias of AF3 attention is dense (not low-rank) |"
    )
    w("")
    w(
        "Checked and not connected (documented in README_trimul.md / ext_attn_results.md): Hugging Face `trimul_gpumode` `kernel_global` (derived from a repository without a license), the TTT-Discover H100 TriMul kernel (printed in a paper appendix only), Anthropic kits (E1 / E2 baseline, outside E5).\n"
    )
    Path(HERE / "E5_results.md").write_text("\n".join(out) + "\n")
    print(f"[report] {len(out)} lines -> {HERE / 'E5_results.md'}")


if __name__ == "__main__":
    main()
