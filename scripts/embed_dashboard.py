"""Inline the benchmark sweep PNGs into the dashboard as base64 data URIs.

The Artifact CSP blocks file/CDN image refs, so the dashboard must carry its
images inline. This reads ``benchmarks/reports/deck/dashboard.html`` (which uses
``__SWEEP_*__`` tokens as <img> sources), replaces each token with a ``data:image/png;base64,…``
URI built from the matplotlib speedup bar charts we generated, and writes
``benchmarks/reports/deck/dashboard.built.html`` — the file that gets published.

Pure stdlib; CPU only. Run via srun, never the login node.
"""

import base64
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DECK = ROOT / "benchmarks" / "reports" / "deck"
# (source -> built) files that use the __SWEEP_*__ image tokens.
PAGES = [
    (DECK / "dashboard.html", DECK / "dashboard.built.html"),
    (DECK / "slides.html", DECK / "slides.built.html"),
]

IMAGES = {
    "__SWEEP_TRIMUL__": ROOT / "benchmarks/artifacts/deck_inputs/trimul_forward_speedup.png",
    "__SWEEP_TRANSITION__": ROOT / "benchmarks/artifacts/deck_inputs/transition_forward_speedup.png",
    "__SWEEP_LNL__": ROOT / "benchmarks/artifacts/deck_inputs/layernorm_linear_fwd_speedup.png",
}


def data_uri(path: Path) -> str:
    b64 = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{b64}"


def main() -> None:
    uris = {tok: data_uri(img) for tok, img in IMAGES.items() if img.exists()}
    missing = [str(img) for tok, img in IMAGES.items() if not img.exists()]
    if missing:
        raise SystemExit("missing images: " + ", ".join(missing))
    for src, out in PAGES:
        if not src.exists():
            print(f"skip (no source): {src.name}")
            continue
        html = src.read_text()
        for tok, uri in uris.items():
            html = html.replace(tok, uri)
        out.write_text(html)
        print(f"wrote {out} ({len(html) // 1024} KB)")


if __name__ == "__main__":
    main()
