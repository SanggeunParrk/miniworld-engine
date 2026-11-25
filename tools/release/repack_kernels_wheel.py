"""Repack a released kernels wheel under a new distribution name / version, byte for byte (used for foldjet-engine 1.0.0 from miniworld-engine 3.0.12).

usage: python tools/release/repack_kernels_wheel.py OLD_WHEEL OUT_DIR   (on a compute node; any python with the standard library)

Every payload file keeps its bytes; only its top directory moves (miniworld_engine/_prebuilt/... -> foldjet_engine/_prebuilt/...).
The licenses are copied from the old wheel's dist-info; METADATA / WHEEL / RECORD are written for the new name and version.
"""

import base64
import hashlib
import sys
import zipfile
from pathlib import Path

old, out = Path(sys.argv[1]), Path(sys.argv[2])
OLD_TOP, NEW_TOP = "miniworld_engine/", "foldjet_engine/"
VERSION = "1.0.0"

z_old = zipfile.ZipFile(old)
old_info = next(n.split("/")[0] for n in z_old.namelist() if n.split("/")[0].endswith(".dist-info"))
old_meta = z_old.read(f"{old_info}/METADATA").decode()
old_wheel = z_old.read(f"{old_info}/WHEEL").decode()
old_name = next(line.split(": ", 1)[1] for line in old_meta.splitlines() if line.startswith("Name: "))
assert old_name.startswith("miniworld-engine-kernels-"), old_name
dist = "foldjet-engine-kernels-" + old_name[len("miniworld-engine-kernels-") :]
norm = dist.replace("-", "_")
info = f"{norm}-{VERSION}.dist-info"
tag = next(line.split(": ", 1)[1] for line in old_wheel.splitlines() if line.startswith("Tag: "))

meta_lines = []
for line in old_meta.splitlines():
    if line.startswith("Name: "):
        line = f"Name: {dist}"
    elif line.startswith("Version: "):
        line = f"Version: {VERSION}"
    meta_lines.append(line)
meta = "\n".join(meta_lines) + "\n"
meta = meta.replace("miniworld-engine==3.0.12", f"foldjet-engine=={VERSION}").replace(" 3.0.12 ", f" {VERSION} ")
meta = meta.replace("[miniworld-engine](https://pypi.org/project/miniworld-engine/)", "[foldjet-engine](https://pypi.org/project/foldjet-engine/)")
meta = meta.replace("`pip install foldjet-engine==", "`pip install foldjet-engine==").replace("miniworld-engine prebuilt kernels", "foldjet-engine prebuilt kernels")
assert "miniworld-engine" not in meta.split("\n\n", 1)[0], meta  # the header part names only the new dist
meta += "\nRenamed from miniworld-engine-kernels (3.0.12); the kernel files are the 3.0.12 files, unchanged.\n"


def h(data: bytes) -> str:
    return "sha256=" + base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode()


path = out / f"{norm}-{VERSION}-{tag}.whl"
record, n, moved_bytes = [], 0, 0
with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as z:
    for zi_old in z_old.infolist():
        name = zi_old.filename
        if name.startswith(old_info + "/"):
            continue
        assert name.startswith(OLD_TOP), name
        data = z_old.read(name)
        new_name = NEW_TOP + name[len(OLD_TOP) :]
        zi = zipfile.ZipInfo(new_name, date_time=zi_old.date_time)
        zi.external_attr = zi_old.external_attr
        zi.compress_type = zipfile.ZIP_DEFLATED
        z.writestr(zi, data, compresslevel=9)
        record.append(f"{new_name},{h(data)},{len(data)}")
        n += 1
        moved_bytes += len(data)
    for name in z_old.namelist():
        if name.startswith(f"{old_info}/licenses/"):
            data = z_old.read(name)
            new_name = f"{info}/licenses/" + name[len(f"{old_info}/licenses/") :]
            z.writestr(new_name, data)
            record.append(f"{new_name},{h(data)},{len(data)}")
    for fname, text in (("METADATA", meta), ("WHEEL", old_wheel)):
        z.writestr(f"{info}/{fname}", text.encode())
        record.append(f"{info}/{fname},{h(text.encode())},{len(text.encode())}")
    record.append(f"{info}/RECORD,,")
    z.writestr(f"{info}/RECORD", "\n".join(record) + "\n")
print(f"{old.name} -> {path.name}: {n} payload files ({moved_bytes} bytes), {path.stat().st_size} bytes")
