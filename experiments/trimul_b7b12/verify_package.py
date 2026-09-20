"""CPU-only integrity checks for the reproducible checkpoint and its evidence."""
from pathlib import Path
import ast
import hashlib
import json
import statistics

root = Path(__file__).resolve().parent
vendor = root/'vendor/anthropic_v5'
manifest = json.loads((vendor/'UPSTREAM.json').read_text())
assert manifest['revision'] == 'f4f62fa6592ae4938d49b1757bea0cfeff9f468e'
for name, digest in manifest['files'].items():
    assert hashlib.sha256((vendor/name).read_bytes()).hexdigest() == digest, name
assert (vendor/'LICENSE').is_file() and (vendor/'NOTICE').is_file()
original = json.loads((root/'records/source-provenance.json').read_text())['raw_input_sha256']
for name in ('front_prefetch_lnpair_storepipe.cu', 'front_ring96_cache3.cu',
             'front_primitives.cuh', 'front_mn_primitives.cuh', 'warp_primitives.cuh',
             'front_prefetch_lnpair_storepipe.launch.json', 'front_ring96_cache3.launch.json'):
    assert hashlib.sha256((root/name).read_bytes()).hexdigest() == original[name], name
for path in root.glob('*.py'):
    text = path.read_text()
    ast.parse(text, filename=str(path))
    if path.name != 'verify_package.py':
        assert '/home/' not in text and 'anthropic_ln_equal_saves_20260919' not in text, path
for length in (384,768):
    record=json.loads((root/f'records/train-total-qualified-L{length}.json').read_text())
    for time in record['times'].values():
        assert len(time['samples_us']) == 600
        assert statistics.median(time['samples_us']) == time['median_us']
    for error in record['replay_errors'].values():
        assert error['finite'] and error['relative_l2'] <= error['limit']
    ratio=record['times']['baseline']['median_us']/record['times']['b7b12_cuda']['median_us']
    assert abs(ratio-record['speedup']) < 1e-12
print('PASS: pinned upstream, unchanged selected CUDA/configs, portable Python and 600-sample training records')
