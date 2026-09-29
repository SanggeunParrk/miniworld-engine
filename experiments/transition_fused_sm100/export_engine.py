"""Write the engine copies of this capsule's kernels (miniworld_engine/kernels/transition/cuda/sm100/) from src/.

  python export_engine.py <engine repo root>

Every conditional on an experiment switch (OLD_SYNC, TRACE, the ablations and rejected variants: any macro not in BUILD_MACROS) is
resolved as "not defined", i.e. to the default build, so the engine carries only the adopted code; conditionals on the build
macros (DIM, SAVE_AB, ITEM_SCHED, ...) are kept, the engine builds its variants with them. The first comment line names the research
source; each kernel exports its dynamic shared-memory size (``transition_smem_bytes``) for the host. A default build of an exported
file has the SASS of the same build of its source (checked on the B200 box with ``cuobjdump -sass``)."""
import re
import sys
from pathlib import Path

BUILD_MACROS = {"DIM", "KDIM", "LPR", "LNB_MINB", "NST_", "NSTG", "SAVE_AB", "ITEM_SCHED", "NO_H", "EPI_PLAIN", "DEVI"}
SRC = Path(__file__).parent / "src"
# engine file name <- capsule file, and the smem symbol the D128 binding already reads
FILES = {
    "sm100.cuh": ("sm100.cuh", None),
    "tr_fwd_sm100.cu": ("tfwd2.cu", "transition_sm100_fwd_smem_bytes"),
    "tr_bwd_sm100.cu": ("tbwd.cu", "transition_sm100_bwd_smem_bytes"),
    "widths/tfwd_d64.cu": ("tfwd_d64.cu", "transition_smem_bytes"),
    "widths/tbwd_d64.cu": ("tbwd_d64.cu", "transition_smem_bytes"),
    "widths/tfwd_d256.cu": ("tfwd_d256.cu", "transition_smem_bytes"),
    "widths/tgate_w.cu": ("tgate_w.cu", "transition_smem_bytes"),
    "widths/tgate_ab.cu": ("tgate_ab.cu", "transition_smem_bytes"),
    "widths/tswiglu_w.cu": ("tswiglu_w.cu", "transition_smem_bytes"),
    "widths/tgemm_nd.cu": ("tgemm_nd.cu", "transition_smem_bytes"),
    "widths/tln_w.cu": ("tln_w.cu", None),
    "widths/tlnbwd_w.cu": ("tlnbwd_w.cu", None),
}
DIRECTIVE = re.compile(r"^\s*#\s*(ifdef|ifndef|if|elif|else|endif)\b(.*)$")


def macros_in(expr):
    return set(re.findall(r"defined\s*\(?\s*([A-Za-z_]\w*)", expr)) | ({expr.split()[0]} if expr.split() else set())


def evaluate(expr, defined):
    """Value of a condition whose macros are all experiment switches: defined only if the file itself defines them (in code
    already kept), e.g. tbwd.cu's ``#ifndef GATE1 / #define GATE2``."""
    e = re.sub(r"defined\s*\(\s*(\w+)\s*\)|defined\s+(\w+)", lambda m: "1" if (m.group(1) or m.group(2)) in defined else "0",
               expr.split("//")[0])
    e = e.replace("&&", " and ").replace("||", " or ").replace("!", " not ")
    return bool(eval(e, {}, {}))  # noqa: S307 -- the expression is 0 / 1 / and / or / not only


def resolve(text):
    out, stack, defined = [], [], set()                       # frame: [keep, active, taken]
    for line in text.splitlines():
        m = DIRECTIVE.match(line)
        live = all(f[1] for f in stack)
        if not m:
            if live:
                out.append(line)
                d = re.match(r"\s*#\s*define\s+([A-Za-z_]\w*)", line)
                if d:
                    defined.add(d.group(1))
            continue
        kind, rest = m.group(1), m.group(2).split("//")[0].strip()
        if kind in ("ifdef", "ifndef", "if"):
            names = {rest.split()[0]} if kind != "if" else set(re.findall(r"[A-Za-z_]\w*", rest)) - {"defined"}
            if names & BUILD_MACROS or kind == "if" and not re.fullmatch(r"[!\s()&|]*(defined\s*\(\s*\w+\s*\)[!\s()&|]*)+", rest):
                stack.append([True, True, True])
                if live:
                    out.append(line)
            else:
                v = {"ifdef": rest.split()[0] in defined, "ifndef": rest.split()[0] not in defined}.get(kind) if kind != "if" else None
                v = evaluate(rest, defined) if v is None else v
                stack.append([False, v, v])
        elif kind == "elif":
            f = stack[-1]
            if f[0]:
                if all(g[1] for g in stack[:-1]):
                    out.append(line)
            else:
                v = not f[2] and evaluate(rest, defined)
                f[1], f[2] = v, f[2] or v
        elif kind == "else":
            f = stack[-1]
            if f[0]:
                if all(g[1] for g in stack[:-1]):
                    out.append(line)
            else:
                f[1], f[2] = not f[2], True
        else:
            f = stack.pop()
            if f[0] and all(g[1] for g in stack):
                out.append(line)
    assert not stack, "unbalanced conditionals"
    return out


def strip_trace(lines):
    """Drop the clock64 trace stamps (TR(k)) the TRACE build used."""
    keep = []
    for ln in lines:
        if re.fullmatch(r"\s*#define TR\(k\) do \{ \} while \(0\)\s*", ln):
            continue
        if re.fullmatch(r"\s*(if \(.*\) )?TR\([^;]*\);\s*", ln):
            continue
        keep.append(re.sub(r"\s*TR\([^;]*\);", "", ln))
    return keep


def main(root):
    dst_dir = Path(root) / "src/miniworld_engine/kernels/transition/cuda/sm100"
    for dst, (src, smem_sym) in FILES.items():
        lines = strip_trace(resolve((SRC / src).read_text()))
        if lines and lines[0].startswith("// " + src):
            lines[0] = (f"// {Path(dst).name} (research source: experiments/transition_fused_sm100/src/{src}, branch "
                        f"perf/transition-sm100-b200; generated by export_engine.py)\n//" + lines[0][len("// " + src):])
        if smem_sym:
            lines += ["", "// ================================================================================== wiring surface",
                      "// Built into a cubin by the newest nvcc that knows sm_100a and launched through the driver API from `transition_sm100.cu`;",
                      "// the dynamic shared-memory size the launch must request is exported here so the host never restates it.",
                      f'extern "C" __device__ int {smem_sym} = SMEM_BYTES;']
        path = dst_dir / dst
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(lines) + "\n")
        print(f"{dst:24s} <- {src:14s} {len(lines):5d} lines")


if __name__ == "__main__":
    main(sys.argv[1])
