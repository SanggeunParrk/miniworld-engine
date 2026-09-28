"""Publish a validation manifest only after the selected runs are complete."""
import argparse
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

p = argparse.ArgumentParser()
p.add_argument('results', type=Path)
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
records = {'benchmarks': [], 'sanitizers': [], 'pytest': [], 'before_after': []}
repo = Path(__file__).resolve().parents[2]
changed_d128 = {'src/miniworld_engine/kernels/transition/cuda/fused_sm90a.py',
                'src/miniworld_engine/kernels/transition/cuda/transition_fused_bwd_sm90a_kernel.cu'}

def load(path):
    r = json.loads(path.read_text())
    assert r['complete'], path
    return r

def evidence(path):
    return {'path': str(path.relative_to(a.results)), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}

for d in (128, 256, 384, 512):
    gpu = 0 if d in (128, 384) else 1
    version = 2 if d == 128 else 1
    for length in (384, 768):
        path = a.results / f'transition-shapes-v{version}/gpu{gpu}/D{d}-L{length}.json'
        r = load(path)
        assert r['times'] and r['native_cuda_kernel_names']
        for source, digest in r['source_sha256'].items():
            if d != 128 and source in changed_d128:
                continue  # Only the rejected-by-width D128 loader/kernel changed.
            assert hashlib.sha256((repo / source).read_bytes()).hexdigest() == digest, source
        records['benchmarks'].append(evidence(path))
    san = a.results / ('transition-sanitize-fixed/gpu0' if gpu == 0 else 'transition-sanitize/gpu1')
    for tool, length in [('memcheck', 384), ('memcheck', 768), ('racecheck', 768), ('synccheck', 768)]:
        stem = ('race-isolated-v2' if d >= 384 else 'race-once') if tool == 'racecheck' and d != 128 else tool
        path = san / stem / f'D{d}-L{length}.json'
        r = load(path)
        log = san / f'{stem}-D{d}-L{length}.log'
        text = log.read_text()
        marker = 'RACECHECK SUMMARY: 0 hazards displayed (0 errors, 0 warnings)' if tool == 'racecheck' else 'ERROR SUMMARY: 0 errors'
        assert marker in text, log
        records['sanitizers'].append({**evidence(path), 'tool': tool, 'scope': r.get('scope', r.get('protocol', 'full F+B and graph')), 'log': evidence(log)})
for name in ('transition-tests-gpu0.xml', 'transition-tests-gpu1.xml', 'transition-tests-d128-fixed.xml'):
    path = a.results / name
    suites = ET.parse(path).getroot().findall('testsuite')
    assert suites
    for suite in suites:
        assert suite.attrib['failures'] == suite.attrib['errors'] == '0'
    records['pytest'].append({**evidence(path), 'tests': sum(int(s.attrib['tests']) for s in suites)})
for length in (384, 768):
    path = a.results / f'transition-d128-fix-paired/D128-L{length}.json'
    r = load(path)
    assert max(r['errors'].values()) == 0
    records['before_after'].append(evidence(path))
records['complete'] = True
records['limits'] = ['D384/512 racecheck is isolated handwritten CUDA kernels at L768. Full-module filtered racecheck hit Internal Sanitizer Error followed by CUBLAS_STATUS_EXECUTION_FAILED; it is not qualified. Unfiltered memcheck/synccheck and ordinary full-gradient/graph checks passed.']
records['retained_failures'] = [
    evidence(a.results / 'transition-sanitize/gpu0/racecheck-D128-L768.log'),
    evidence(a.results / 'transition-sanitize-fixed/gpu0/race-native-D384-L768.log'),
    evidence(a.results / 'transition-sanitize-fixed/gpu0/race-old-D384-L768.log'),
    evidence(a.results / 'transition-sanitize/gpu1/race-native-D512-L768.log'),
]
a.output.write_text(json.dumps(records, indent=2) + '\n')
print('PASS: 8 benchmark cells, 16 sanitizer runs, 3 pytest records, 2 bitwise paired fix comparisons')
