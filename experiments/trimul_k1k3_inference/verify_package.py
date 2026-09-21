"""CPU-only integrity checks: overlay hashes, agreement with the sibling vendored upstream subset, patch coverage, portable Python, records."""
import ast
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parent
ov = json.loads((root / "OVERLAY.json").read_text())
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
for f, h in ov["overlay_files"].items():
    assert sha(root / "overlay" / f) == h, f
    assert ov["base_files"][f] != h, "overlay identical to base: " + f
vendor = root.parent / "trimul_b7b12" / "vendor" / "anthropic_v5"
if (vendor / "UPSTREAM.json").is_file():
    up = json.loads((vendor / "UPSTREAM.json").read_text())
    assert up["revision"] == ov["upstream"]["revision"]
    for f, h in up["files"].items():
        if f in ov["base_files"]:
            assert ov["base_files"][f] == h, "base hash disagrees with vendored upstream subset: " + f
patch = (root / "k1k3-inference.patch").read_text()
for f in ov["overlay_files"]:
    assert ("+++ b/" + f) in patch, "patch lacks " + f
for path in root.glob("*.py"):
    text = path.read_text()
    ast.parse(text, filename=str(path))
    if path.name != "verify_package.py":
        assert "/home/" not in text, path
for name in ("mask", "pdl"):
    pass
for L in (384, 768):
    for tag in ("orig", "final2", "pdl"):
        for r in (1, 2):
            rec = json.loads((root / "records" / "engine-bench" / f"{tag}-L{L}-r{r}.json").read_text())
            assert json.dumps(rec).count('"ms"') >= 1, (tag, L, r)
    for v in ("final2", "mt", "mt-bool", "mt-fp32", "pdl-on", "pdl-off"):
        rec = json.loads((root / "records" / "kernel-times" / f"{v}-L{L}.json").read_text())["results"][0]
        assert {"k1", "k3", "cublas"} <= set(rec["kernels"]) and rec["error"]["finite"], (v, L)
        assert abs(rec["error"]["rel_rms"] - 0.002587) < 2e-5, (v, L, rec["error"])
    pdl = json.loads((root / "records" / "probes" / f"pdl-alternating-L{L}.json").read_text())["res"]
    assert pdl["chain"]["err_on"]["finite"] and pdl["single"]["err_on"]["finite"]
    ts = json.loads((root / "records" / "probes" / f"cta-timeline-L{L}.json").read_text())["res"]
    assert ts["tmn.k1"]["ctas"] == 132 and ts["tmn.k3"]["ctas"] == 132
man = json.loads((root / "records" / "payload-manifest.json").read_text())["units"]["tmn90_z128_h128/sm_90a"]
for k in ("tmn_k1_z128_h128_b_t6x32_s8k2_m1_l2_v0", "tmn_k1_z128_h128_b_t6x32_s8k2_m2_l2_v0", "tmn_k1_z128_h128_b_t6x32_s8k2_m3_l2_v0"):
    assert man["kernels"][k]["spill_stores"] == 0 and man["kernels"][k]["spill_loads"] == 0, k
for d in ov["defines"]:
    assert any(fl == "-D%s=%s" % (d, ov["defines"][d]) for fl in man["flags"]), d
# the bidirectional round: the c_hidden 256 unit, its interleaved rows and the tile the table now defaults to
bd = root / "records" / "bidirectional"
for L in (384, 768):
    best = {}
    for tag in ("base", "tab", "t3x64", "t6x32"):
        for r in (1, 2, 3):
            res = json.loads((bd / "interleaved" / ("%s-L%d-r%d.json" % (tag, L, r))).read_text())
            assert res["c_hidden"] == 256, tag
            one = res["results"][0]
            assert one["error"]["finite"] and abs(one["error"]["rel_rms"] - 0.002586) < 2e-5, (tag, L, one["error"])
            best.setdefault(tag, []).append(one["op_us"]["median"])
    assert min(best["t3x64"]) < min(best["base"]), (L, best)          # the tabled tile is the measured winner, not an assumption
    eng = json.loads((bd / "interleaved" / ("engine-cute-L%d.json" % L)).read_text())
    assert eng["backend"].endswith("CUTE") and eng["op_us"]["median"] > 2 * min(best["t3x64"]), (L, eng["op_us"])
h256 = json.loads((bd / "payload-manifest-h256.json").read_text())["units"]["tmn90_z128_h256/sm_90a"]
assert any("_t3x64_" in k and k.startswith("tmn_k1_z128_h256") for k in h256["kernels"]), "the tabled K1 tile is not in the unit"
for d in json.loads((root / "OVERLAY.json").read_text())["defines"]:
    assert any(fl.startswith("-D%s=" % d) for fl in h256["flags"]), d
print("PASS: overlay hashes, vendored-upstream agreement, patch coverage, portable Python, records, the recorded payload build and the bidirectional round")
