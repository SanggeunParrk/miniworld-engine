"""Recreate the frozen research snapshot locally; verify every recorded SHA256."""
import argparse, hashlib, json, pathlib, shutil
p=argparse.ArgumentParser();p.add_argument('--destination',type=pathlib.Path,default=pathlib.Path('.bench/trimul-large-d-source/runs'));a=p.parse_args()
m=json.loads(pathlib.Path(__file__).with_name('source-manifest.json').read_text());root=pathlib.Path(m['origin'])
for name,digest in m['files'].items():
 rel=pathlib.Path(name)
 if rel.is_absolute() or '..' in rel.parts:raise ValueError(name)
 src=root/rel;raw=src.read_bytes()
 if hashlib.sha256(raw).hexdigest()!=digest:raise RuntimeError(f'Historical source changed: {src}; inspect before making a new snapshot')
 dst=a.destination/rel;dst.parent.mkdir(parents=True,exist_ok=True);dst.write_bytes(raw)
print(f'Verified and staged {len(m["files"])} source/config files in {a.destination}')
