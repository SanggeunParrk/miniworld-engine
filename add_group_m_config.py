"""Add the GROUP_M ladder to a kernel's config sets. Idempotent."""
import sys
from pathlib import Path

CFG = Path("src/miniworld_engine/autotune/configs")
LADDER, KEEP = "4 16 65536", "65536"


def add(kernel: str) -> tuple[int, int]:
    a = m = 0
    for d in sorted(x for x in CFG.iterdir() if x.is_dir()):
        f = d / f"{kernel}.csv"
        if not f.is_file():
            continue
        lines = f.read_text().splitlines()
        if lines[0].startswith("axis,"):
            if any(l.startswith("GROUP_M,") for l in lines):
                continue
            blocks = [j for j, l in enumerate(lines) if l.startswith("BLOCK_")]
            lines.insert(max(blocks) + 1 if blocks else 1, f"GROUP_M,{LADDER}")
            f.write_text("\n".join(lines) + "\n")
            a += 1
        else:
            hdr = lines[0].split(",")
            if "GROUP_M" in hdr:
                continue
            i = hdr.index("num_warps")
            hdr.insert(i, "GROUP_M")
            out = [",".join(hdr)]
            for row in lines[1:]:
                if row.strip():
                    c = row.split(",")
                    c.insert(i, KEEP)
                    out.append(",".join(c))
            f.write_text("\n".join(out) + "\n")
            m += 1
    return a, m


if __name__ == "__main__":
    ta = tm = 0
    for k in sys.argv[1:]:
        a, m = add(k)
        print(f"  {k:46s} 축 {a} · 구체화 {m}")
        ta, tm = ta + a, tm + m
    print(f"  합계: 축 {ta} · 구체화 {tm}")
