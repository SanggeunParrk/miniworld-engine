"""Union-merge per-target runtime autotune caches into the shipped package data.

Each capture job wrote to an ISOLATED cache dir ($HOME/.cache/mwk_cap/<target>/autotune) to avoid
concurrent read-modify-write races on shared op files (e.g. layernorm_linear is written by
transition + triangle_attention + bias_only). This merges them: for each (op, gpu) it UNIONS the
bucket entries across all targets; a bucket seen in multiple targets keeps the union of its
configs re-ranked by ms, top-5. All sources for an op must share one config_space_hash (same
kernel grid) — mismatches are reported and skipped (stale). Existing shipped buckets NOT
re-measured this run are preserved (union with the shipped file)."""
import glob
import json
import os
import sys

ROOT = os.path.expanduser("~/.cache/mwk_cap")
DST = "src/miniworld_kernels/autotune/data"
TOPK = 5


def _sig(e):
    kw = tuple(sorted((k, tuple(v) if isinstance(v, list) else v) for k, v in e["kwargs"].items()))
    return (kw, e.get("num_warps", 0), e.get("num_stages", 0))


def main():
    # collect: op -> gpu_file -> {hash, entries{bucket: [cfg,...]}} merged
    per_target = sorted(glob.glob(f"{ROOT}/*/autotune"))
    if not per_target:
        print(f"no per-target caches under {ROOT}"); return
    merged = {}   # (op, gpufile) -> {"hash":h, "entries":{bucket:{sig:cfg}}}
    for root in per_target:
        for opdir in sorted(p for p in glob.glob(f"{root}/*") if os.path.isdir(p)):
            op = os.path.basename(opdir)
            for fp in glob.glob(f"{opdir}/*.json"):
                gpu = os.path.basename(fp)
                d = json.load(open(fp))
                key = (op, gpu)
                slot = merged.setdefault(key, {"hash": d.get("config_space_hash"), "meta": d, "entries": {}})
                if d.get("config_space_hash") != slot["hash"]:
                    print(f"  !! {op}/{gpu}: config_space_hash mismatch across targets "
                          f"({d.get('config_space_hash')} vs {slot['hash']}) — skipping this source")
                    continue
                for bucket, cfgs in d.get("entries", {}).items():
                    bslot = slot["entries"].setdefault(bucket, {})
                    for c in cfgs:
                        s = _sig(c)
                        if s not in bslot or c.get("ms", 9e9) < bslot[s].get("ms", 9e9):
                            bslot[s] = c

    for (op, gpu), slot in sorted(merged.items()):
        dstfp = f"{DST}/{op}/{gpu}"
        os.makedirs(os.path.dirname(dstfp), exist_ok=True)
        # start from shipped (preserve buckets not re-measured), if same hash
        out = slot["meta"].copy()
        entries = {}
        if os.path.exists(dstfp):
            old = json.load(open(dstfp))
            if old.get("config_space_hash") == slot["hash"]:
                entries = old.get("entries", {})  # preserve old buckets under same grid
        for bucket, bslot in slot["entries"].items():
            ranked = sorted(bslot.values(), key=lambda c: c.get("ms", 9e9))[:TOPK]
            entries[bucket] = ranked
        out["entries"] = entries
        json.dump(out, open(dstfp, "w"), indent=2, sort_keys=True)
        print(f"  {op}/{gpu}: {len(entries)} buckets (hash {slot['hash']})")
    print("MERGE DONE")


if __name__ == "__main__":
    main()
