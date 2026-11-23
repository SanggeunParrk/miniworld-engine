"""E4 raw records (records/<card>/fp8_<gates>.r<k>.jsonl) -> E4_<card>.csv: two rows per case and gate setting (backend ``cuda`` = the default
path, ``engine-fp8``), ms_median / ms_min / ms_max over the alternating rounds of the repeats, the ratio default / FP8 on the FP8 row,
whether an FP8 route fired and whether the saturation check tripped.   python collect.py --card h100"""

import argparse
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from paper_common import csvio

EXTRA = ["gates", "model", "ratio_default_over_fp8", "fp8_route_fired", "saturated", "stack", "row", "variant", "samples", "repeats"]


def rows(card: str) -> list[dict]:
    from paper_common.run_measure import case_key

    groups: dict[tuple, list[tuple[dict, dict]]] = {}
    for f in csvio.record_files(HERE, card):
        meta = csvio.meta_of(f)
        for rec in csvio.read_jsonl(f):
            groups.setdefault((rec["gates"], *case_key(rec)), []).append((rec, meta))
    out = []
    for items in groups.values():
        ok = [(r, m) for r, m in items if r["status"] == "ok"]
        rec, meta = (ok or items)[-1]
        d, h = csvio.dims_of(rec["kwargs"])
        no_route = rec["status"] == "ok" and meta.get("card") not in ("h100", "b200") and not rec["fp8_route_fired"]
        for backend in ("cuda", "engine-fp8"):
            row = {
                "id": "E4", "card": card, "gpu": meta.get("gpu", ""), "engine_version": meta.get("engine_version", ""),
                "commit": meta.get("commit", ""), "module": rec["module"], "mode": rec["mode"], "backend": backend, "L": rec["length"],
                "d": d, "n_head": h, "dtype": rec["dtype"], "status": "ok" if ok else rec["status"],
                "reason": rec.get("reason", "") if not ok else ("the FP8 route did not fire (no route for this case on this card, or below its gate): engine-fp8 ran the default kernels" if no_route else ""),
                "gates": rec["gates"], "model": rec["model"], "stack": "engine", "row": rec["row"], "variant": csvio.variant_of(rec["kwargs"]),
                "samples": rec.get("samples", ""), "repeats": len(ok),
            }  # fmt: skip
            if ok:
                per = [statistics.median(r["ms"][backend]) for r, _ in ok]
                row.update(ms_median=statistics.median(per), ms_min=min(per), ms_max=max(per))
                if backend == "engine-fp8":
                    row.update(
                        ratio_default_over_fp8=statistics.median(r["ratio_default_over_fp8"] for r, _ in ok),
                        fp8_route_fired=rec["fp8_route_fired"],
                        saturated=any(r["saturated_calibration"] or r["saturated_replay"] for r, _ in ok),
                    )
            out.append(row)
    return out


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--card", required=True, choices=["h100", "a100", "rtx5090", "b200"])
    card = p.parse_args().card
    csvio.write(csvio.csv_path(HERE, f"E4_{card}.csv"), rows(card), EXTRA)
