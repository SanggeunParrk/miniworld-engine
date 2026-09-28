"""Explicit K1 schedule audit of installed wide training; never publishes defaults."""
import argparse
import collections
import hashlib
import json
import pathlib
import statistics
import sys
import time

import torch

from miniworld_engine.kernels.trimul_inproj.cuda import _h100_runtime as T
from miniworld_engine.kernels.trimul_inproj.cuda import h100_native as N
from miniworld_engine.kernels.trimul_inproj.cuda.h100_width import Training

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'experiments/trimul_training_v2/runs/trimul_cuda_widths_opt_20260923'))
from fixture import setup

p = argparse.ArgumentParser()
p.add_argument('--width', type=int, default=256)
p.add_argument('--length', type=int, required=True)
a = p.parse_args()
out = pathlib.Path('/workspace/vast-results/trimul-config-audit-20260927')
out.mkdir(parents=True, exist_ok=True)
dest = out / f'K1-D{a.width}-L{a.length}.json'
record = dict(width=a.width, length=a.length, complete=False, rows=[],
              scope='Explicit current-engine reusable Training full F+B; K1 only',
              script_sha256=hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),
              source_sha256={str(f.relative_to(ROOT)): hashlib.sha256(f.read_bytes()).hexdigest()
                             for f in (ROOT/'src/miniworld_engine/kernels/trimul_inproj/cuda').rglob('*') if f.is_file() and f.suffix in ('.py','.cu','.cuh','.inc','.json')})
def save():
    dest.write_text(json.dumps(record, indent=2)+'\n')
def capture(fn):
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3): fn()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g, stream=s): result = fn()
    g.replay()
    torch.cuda.synchronize()
    return g, result

def flatten(result): return (result[0], *result[1])
def check(actual, expected):
    errs = {}
    for name, v, r in zip(names, flatten(actual), expected):
        err = float((v.float()-r.float()).norm()/r.float().norm().clamp_min(1e-20))
        limit = 2e-5 if name == 'dx' else 5e-6 if name.startswith(('dgamma','dbeta')) else 5e-4
        errs[name] = dict(relative_l2=err, passed=bool(torch.isfinite(v).all()) and err < limit)
    return errs

def paired(base, candidate):
    values = {'baseline':[], 'candidate':[]}
    for _ in range(4): base.replay(); candidate.replay()
    for rep in range(3):
        for i in range(15):
            order = [('baseline',base),('candidate',candidate)]
            for name,g in order if (i+rep)%2 else reversed(order):
                start,end = torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                start.record();g.replay();end.record();end.synchronize()
                values[name].append(start.elapsed_time(end)*1000)
    return {name:dict(median_us=statistics.median(v), samples_us=v) for name,v in values.items()}

leaves, dy, mask, ds, ref, _, names = setup(a.width, a.length)
with torch.no_grad(), T.native_context(leaves[0].device):
    plan = Training(*leaves, mask, ds, dy)
    original = plan.front
    baseline = [x.clone() for x in flatten(plan())]
    saved = [original.ab.clone(), original.xn.clone()]
    base_graph, _ = capture(plan)
    record['baseline_config'] = list(original.cfg)
    record['device'] = torch.cuda.get_device_name()
    record['torch'] = torch.__version__
    configs = [tuple(original.cfg)] + [tuple(c) for c in N.configs(a.width) if tuple(c) != tuple(original.cfg)]
    old = set(configs)
    rejected = []
    for bi,bj in ((1,64),(2,64),(1,128)):
        for sk in (1,2,3,4,6,8):
            for slots in (3,5,7):
                for mb in ((2,1) if bi*bj==64 else (1,)):
                    cfg = (bi,bj,slots,sk,mb)
                    try: N.k1_smem(a.width,cfg)
                    except ValueError as e:
                        rejected.append(dict(config=cfg, reason=str(e)));continue
                    if cfg not in configs: configs.append(cfg)
    record.update(declared_existing_count=len(old), expanded_count=len(configs), host_rejections=rejected)
    save()
    for cfg in configs:
        row = dict(config=cfg, origin='existing' if cfg in old else 'odd_slots_expansion')
        record['rows'].append(row)
        try:
            candidate = N.Front(leaves[0][0], plan.w1, plan.mask, plan.gi, plan.bi, cfg,
                                saved=(original.ab,plan.tri,original.xn))
            row.update(smem=candidate.smem, threads=candidate.threads,
                       artifact_sha256=hashlib.sha256(pathlib.Path(candidate.path).read_bytes()).hexdigest())
        except (RuntimeError,ValueError,AssertionError) as e:
            row.update(status='build_failed',error=str(e)[-2500:]);save();print('BUILD_REJECT',cfg,flush=True);continue
        plan.front = candidate
        actual = plan();torch.cuda.synchronize()
        row['saved_exact'] = [torch.equal(v,r) for v,r in zip((candidate.ab,candidate.xn),saved)]
        row['errors'] = check(actual,baseline)
        if not all(row['saved_exact']) or not all(v['passed'] for v in row['errors'].values()):
            row['status']='numerical_reject';plan.front=original;save();continue
        graph, graph_outputs = capture(plan)
        originals = [t.clone() for t in (leaves[0],leaves[1],dy,mask,ds)]
        leaves[0].mul_(.97);leaves[1].mul_(1.01);dy.mul_(.93);mask.copy_(mask.roll(1,1));ds.copy_(ds.roll(1,2))
        # plan.mask owns a conversion; change that buffer as well.
        plan.mask.copy_(mask.reshape_as(plan.mask))
        plan.front=original
        changed=[v.clone() for v in flatten(plan())]
        plan.front=candidate;graph.replay();torch.cuda.synchronize()
        row['changed_input_graph']=check(graph_outputs,changed)
        for t,r in zip((leaves[0],leaves[1],dy,mask,ds),originals):t.copy_(r)
        plan.mask.copy_(mask.reshape_as(plan.mask))
        if not all(v['passed'] for v in row['changed_input_graph'].values()):
            row['status']='graph_reject'
        else:
            row['times']=paired(base_graph,graph)
            row['speedup']=row['times']['baseline']['median_us']/row['times']['candidate']['median_us']
            row['status']='measured'
        print('RESULT',cfg,row['status'],row.get('speedup'),flush=True)
        plan.front=original
        del graph,candidate,originals,changed
        save()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
        base_graph.replay();torch.cuda.synchronize()
    trace=out/f'trace-D{a.width}-L{a.length}.json'
    prof.export_chrome_trace(str(trace))
    events=json.loads(trace.read_text())['traceEvents']
    record['baseline_kernels']=dict(collections.Counter(e['name'] for e in events if e.get('cat')=='kernel'))
    record['complete']=True
    record['qualification']='Baseline-relative strict checks and changed-input graph only; no promotion or new independent-reference/sanitizer qualification'
    save()
