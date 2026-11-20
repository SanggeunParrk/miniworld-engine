"""Converters from the harness CSVs to the table shapes the paper scripts read (card names: h100 -> H100 / h100_sm90, ...).

    python paper_common/tables.py --card h100 [--card a100 ...] --out DIR      # needs E1_<card>.csv / E2 / E3 of the card where present

Writes (only the ones whose inputs exist), in the layout of the paper's tables/:
  operator_paired_ratios.csv        card,cls,mode,vs,cases,geomean_ratio,min,max,cases_cuda_faster      (E1: ratio = other / engine cuda, same case)
  supp_operator_speed_3cards.csv    card,cls,mode,backend,speedup,n                                      (E1: geomean of pytorch / backend per class)
  sweeps.csv                        card,module,mode,sweep,d,L,backend,ms                                (E2: L sweep d=128, D sweep L=768)
  accuracy_lights.csv               card,backend,measured,GREEN,YELLOW,RED                               (E3)
Pairing is per case (same requirements row, variant, L, depth, samples, dtype, mode); the Anthropic ratio pairs the engine-stack cuda with the
kit-stack anthropic, the PyTorch / Triton / cuEquivariance ratios use the engine stack. Only ``ok`` rows enter.
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
KEYS = {"h100": ("H100", "h100_sm90"), "a100": ("A100", "a100_sm80"), "rtx5090": ("RTX5090", "5090_sm120")}
OTHERS = {"pytorch": "pytorch", "triton": "triton", "anthropic": "anthropic", "cueq": "cueq"}


def _read(exp: str, eid: str, card: str) -> list[dict]:
    p = HERE / exp / f"{eid}_{card}.csv"
    if not p.exists():
        return []
    with open(p, newline="") as fh:
        return list(csv.DictReader(fh))


def _case(r: dict) -> tuple:
    return (r["module"], r["mode"], r["row"], r["variant"], r["L"], r["depth"], r["samples"], r["dtype"])


def _ms(rows: list[dict]) -> dict[tuple, dict[str, float]]:
    """case -> {backend: ms} over ok rows; a backend measured on both stacks keeps the engine stack's (anthropic: the kit's)."""
    out: dict[tuple, dict[str, float]] = defaultdict(dict)
    for r in rows:
        if r["status"] != "ok" or not r["ms_median"]:
            continue
        stack_ok = (r["backend"] == "anthropic") == (r["stack"] == "kit")
        if r["backend"] == "pytorch" and r["stack"] != "engine":
            continue
        if stack_ok or r["backend"] == "pytorch":
            out[_case(r)][r["backend"]] = float(r["ms_median"])
    return out


def _gm(xs: list[float]) -> float:
    return math.exp(sum(math.log(x) for x in xs) / len(xs))


def operator_tables(cards: list[str], out: Path) -> None:
    paired, supp = [], []
    for card in cards:
        upper, key = KEYS[card]
        rows = _read("E1_operator_speed", "E1", card)
        if not rows:
            continue
        ms = _ms(rows)
        for mode in ("eval", "train"):
            by_cls: dict[str, list[dict]] = defaultdict(list)
            for case, b in ms.items():
                if case[1] == mode:
                    by_cls[case[0]].append(b)
            for cls, cases in sorted(by_cls.items()):
                for vs in OTHERS:
                    r = [b[vs] / b["cuda"] for b in cases if "cuda" in b and vs in b]
                    if r:
                        paired.append(dict(card=upper, cls=cls, mode=mode, vs=vs, cases=len(r), geomean_ratio=_gm(r), min=min(r), max=max(r),
                                           cases_cuda_faster=sum(x > 1 for x in r)))  # fmt: skip
                for be, name in (("cuda", "engine"), ("anthropic", "anthropic"), ("cueq", "cueq"), ("triton", "triton")):
                    r = [b["pytorch"] / b[be] for b in cases if "pytorch" in b and be in b]
                    if r:
                        supp.append(dict(card=key, cls=cls, mode=mode, backend=name, speedup=round(_gm(r), 2), n=len(r)))
    if paired:
        _write(
            out / "operator_paired_ratios.csv",
            paired,
            ["card", "cls", "mode", "vs", "cases", "geomean_ratio", "min", "max", "cases_cuda_faster"],
        )
        _write(out / "supp_operator_speed_3cards.csv", supp, ["card", "cls", "mode", "backend", "speedup", "n"])


def sweeps(cards: list[str], out: Path) -> None:
    res = []
    for card in cards:
        upper = KEYS[card][0]
        ms = _read("E2_sweeps", "E2", card)
        for r in ms:
            if r["status"] != "ok" or r["backend"] not in ("cuda", "cueq", "anthropic"):
                continue
            if (r["backend"] == "anthropic") != (r["stack"] == "kit"):
                continue
            d, length = int(r["d"]), int(r["L"])
            for sweep, hit in (("L", d == 128), ("D", length == 768)):
                if hit:
                    res.append(dict(card=upper, module=r["module"].replace("TriangleMultiplication", "TriangleMultiplication"), mode=r["mode"],
                                    sweep=sweep, d=d, L=length, backend=r["backend"], ms=r["ms_median"]))  # fmt: skip
    if res:
        _write(out / "sweeps.csv", res, ["card", "module", "mode", "sweep", "d", "L", "backend", "ms"])


def accuracy(cards: list[str], out: Path) -> None:
    res = []
    for card in cards:
        rows = _read("E3_accuracy", "E3", card)
        c: dict[str, Counter] = defaultdict(Counter)
        for r in rows:
            if r["status"] == "measured" and r["backend"] != "pytorch":
                c[r["backend"]][r["light"]] += 1
        for be, n in sorted(c.items()):
            res.append(dict(card=KEYS[card][0], backend=be, measured=sum(n.values()), GREEN=n["GREEN"], YELLOW=n["YELLOW"], RED=n["RED"]))
    if res:
        _write(out / "accuracy_lights.csv", res, ["card", "backend", "measured", "GREEN", "YELLOW", "RED"])


def _write(path: Path, rows: list[dict], cols: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    print(f"[tables] {len(rows)} rows -> {path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--card", action="append", required=True, choices=list(KEYS))
    p.add_argument("--out", required=True)
    a = p.parse_args()
    o = Path(a.out)
    operator_tables(a.card, o)
    sweeps(a.card, o)
    accuracy(a.card, o)
