"""Exercise the exact measured combined graph with changed inputs and weights."""
import inspect
import json
import sys
import torch
import compare_all_training as bench

original_capture = bench.capture_outputs


class Checked(Exception):
    pass


def capture_and_validate(fn, stream=None):
    graph, outputs = original_capture(fn, stream)
    if fn.__name__ != 'current_train':
        return graph, outputs
    closure = inspect.getclosurevars(fn).nonlocals
    d, a = closure['d'], closure['a']
    current_forward = closure['current_forward']
    p1, p7 = closure['p1'], closure['p7']
    bench.settings.configure(trimul_sm90_kernels=())
    with torch.no_grad():
        graph.replay()
        torch.cuda.synchronize()
        before = outputs[0].clone()
        x0 = d['x'].clone()
        w0 = d['leaves'][1].clone()
        dy0 = a['dy'].clone()
        d['x'][..., 0].add_(.03)
        d['leaves'][1][0, 0].mul_(.9)
        a['dy'][..., 0].mul_(.9)
        y, saves = current_forward()
        reference = y, bench.C.backward(d, saves, a['dy'])
        graph.replay()
        torch.cuda.synchronize()
        changed = bench.check_combined(outputs, reference)
        assert not torch.equal(before, outputs[0]), 'Graph ignored changed inputs'
        assert bool((p1.workspace[-1] == 0).all()) and bool((p7.counts[:2] == 0).all())
        d['x'].copy_(x0)
        d['leaves'][1].copy_(w0)
        a['dy'].copy_(dy0)
        y, saves = current_forward()
        reference = y, bench.C.backward(d, saves, a['dy'])
        graph.replay()
        torch.cuda.synchronize()
        restored = bench.check_combined(outputs, reference)
        assert torch.equal(before, outputs[0]), 'Restored forward did not reproduce'
        assert bool((p1.workspace[-1] == 0).all()) and bool((p7.counts[:2] == 0).all())
    report = dict(L=d['n'], changed_input_weight_upstream=True, restored_forward_exact=True,
                  counters_zero=True, changed=changed, restored=restored,
                  benchmark='compare_all_training.py; same captured current_train callable')
    path = bench.R / ('all-training-replay-L%d.json' % d['n'])
    path.write_text(json.dumps(report, indent=2))
    print('REPLAY_VALIDATION_PASS', path, flush=True)
    raise Checked()


bench.capture_outputs = capture_and_validate
try:
    bench.main()
except Checked:
    pass
