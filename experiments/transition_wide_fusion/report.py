"""Audit final qualification, then write small durable experiment records."""
import hashlib
import json
from pathlib import Path
import statistics

root=Path(__file__).resolve().parents[2]
here=Path(__file__).resolve().parent
raw=root/'.bench/transition-wide-local'
rows=[]
qualification=[]
for d in (128,256,384,512):
    for length in (384,768):
        path=raw/f'qualification/D{d}-L{length}.json'
        data=json.loads(path.read_text())
        assert data['complete'],path
        assert data['identity']['qos']=='normal_h100',path
        assert max(data['candidate_errors'].values())<1e-4,path
        assert max(data['changed_candidate_errors'].values())<1e-4,path
        for group in ('compiled_graph_errors','changed_graph_errors'):
            assert all(max(v.values())<1e-5 for v in data[group].values()),path
        for name in ('ln_residual.py','selected.py'):
            rel='experiments/transition_wide_fusion/'+name
            assert data['identity']['sources'][rel]==hashlib.sha256((here/name).read_bytes()).hexdigest(),name
        times={k:v['median_ms'] for k,v in data['times'].items()}
        ratios=sorted(a/b for a,b in zip(data['times']['baseline']['samples_ms'],data['times']['candidate']['samples_ms']))
        rows.append(dict(D=d,L=length,**times,
                         speedup_native=times['baseline']/times['candidate'],
                         speedup_pytorch=times['pytorch']/times['candidate'],
                         median_paired_ratio=statistics.median(ratios),
                         paired_wins=sum(r>1 for r in ratios),paired_samples=len(ratios),
                         paired_ratio_p10=ratios[len(ratios)//10-1],paired_ratio_p90=ratios[9*len(ratios)//10-1]))
        qualification.append(dict(path=str(path.relative_to(root)),job=data['identity']['job'],
                                  candidate_errors=data['candidate_errors'],changed_candidate_errors=data['changed_candidate_errors']))
sanitizers=[]
for d in (384,512):
    for mode in ('memcheck','racecheck','synccheck'):
        stem=f'D{d}-L768-{mode}'
        path=raw/'sanitize'/(stem+'.json')
        data=json.loads(path.read_text())
        assert data['complete'],path
        log=(raw/'sanitize'/(stem+'.txt')).read_text()
        expected='RACECHECK SUMMARY: 0 hazards' if mode=='racecheck' else 'ERROR SUMMARY: 0 errors'
        assert expected in log,(path,log[-1000:])
        sanitizers.append(dict(D=d,L=768,tool=mode,scope='full module F+B' if data['full'] else 'changed epilogue kernels',compiled=data.get('compiled',False),path=str(path.relative_to(root)),job=data['identity']['job']))
validation=dict(status='qualified explicit experiment; production unchanged',rows=rows,
                qualification=qualification,sanitizers=sanitizers,
                runtime_sources={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest()
                                 for p in (here/'selected.py',here/'ln_residual.py',here/'common.py')},
                retained_harness_failure_logs=[str(p.relative_to(root)) for p in sorted((raw/'sanitize').glob('*failure.txt'))],
                limitations=['D128 and D256 unchanged; D128 added as a comparison row, new D256 fusion candidates rejected',
                             'No full-module racecheck claim for unchanged kernels',
                             'No hardware-counter or SOL claim'])
(here/'VALIDATION.json').write_text(json.dumps(validation,indent=2)+'\n')
lines=['# Local H100 wide Transition results','',
       'One allocated H100 SXM, `normal_h100` QoS; BF16 pair tensors `[1,L,L,D]`, expansion 4.',
       'Actual compiled modules, fresh full F+B, input and all five parameter gradients, 100 alternating CUDA Graph event samples.',
       'Candidate is explicit only. D128/D256 use unchanged native kernels. Timings are milliseconds.',
       'D128 was added in a separate local job with the same harness/protocol; wide rows retain the preceding qualified measurements.','',
       '| L | D | PyTorch compiled | Native before | Experiment | vs native | vs PyTorch |',
       '| ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
for r in sorted(rows,key=lambda r:(r['L'],r['D'])):
    lines.append(f"| {r['L']} | {r['D']} | {r['pytorch']:.3f} | {r['baseline']:.3f} | {r['candidate']:.3f} | {r['speedup_native']:.3f}x | {r['speedup_pytorch']:.3f}x |")
lines+=['','All eight shapes passed full-gradient, compiled graph and changed-state replay checks.',
        'Both changed widths passed full-L768 module memcheck and isolated changed-kernel racecheck/synccheck.',
        'See [VALIDATION.json](VALIDATION.json) for exact gates, job IDs and source hashes, and [README.md](README.md) for rejected fusion designs.',
        'D384/D512 retain cuBLAS GEMMs; the gain comes from the persistent LN backward + residual epilogue. This does not solve full gate/weight-gradient fusion.',
        'Raw records and unsuccessful attempts remain under `.bench/transition-wide-local/`.']
(here/'RESULTS.md').write_text('\n'.join(lines)+'\n')
print('\n'.join(lines))
