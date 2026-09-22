import hashlib,json
from pathlib import Path
E=Path('/home/psk6950/MiniWorld/runs/trimul_sm90_parity_20260917/engine')
R=Path(__file__).resolve().parent
T=E/'third_party/anthropic';V=T/'upstream'
m=json.loads((T/'UPSTREAM.json').read_text());changes=[];missing=[];binaries=[];known={}
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
for f in m['files']:
 p=V/f['path'];known[f['path']]=True
 if not p.exists():missing.append(f['path']);continue
 h=sha(p)
 if h!=f['sha256']:changes.append(dict(path=f['path'],original_sha256=f['sha256'],local_sha256=h))
for p in V.rglob('*'):
 if not p.is_file() or '__pycache__' in str(p):continue
 if p.suffix in ('.so','.cubin','.a','.o','.ptx') or ('manifest' in p.name and str(p.relative_to(V)) not in known):
  binaries.append(dict(path=str(p.relative_to(V)),sha256=sha(p),bytes=p.stat().st_size))
report=dict(upstream_revision=m['revision'],original_files=len(m['files']),missing=missing,changed_original_files=changes,local_binaries_and_manifests=binaries,
 policy='No imported kernel source edits; only runtime binaries and build manifests/checksums may differ. Local rebuilt binaries have no inherited upstream certification.')
(T/'LOCAL_BUILD.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:v for k,v in report.items() if k!='local_binaries_and_manifests'},indent=2))
assert not missing
assert all(f['path'].endswith(('SHA256SUMS','manifest.json','.cubin')) for f in changes),changes
