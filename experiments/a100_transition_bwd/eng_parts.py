"""Engine sm80 Transition: per-kernel time of a training step (profiler) and step time (graph), for nrep values (env).  python eng_parts.py L..."""
import collections, copy, os, sys
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
from bench_common import graph_ms  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "a100_transition_fwd"))
import transition_a100 as TA  # noqa: E402
from miniworld_engine.modules.exceptions import ImplementationType as I  # noqa: E402
from miniworld_engine.modules.dispatch import resolve_transition  # noqa: E402
for L in map(int, sys.argv[1:]):
    ref, x = TA.fixture(L)
    mod = copy.deepcopy(ref); mod.implementation = I.MINIWORLD; mod._backend = resolve_transition(I.MINIWORLD); mod.train()
    xg = x.view(1, L, L, 128).clone().requires_grad_(True)
    dy = torch.randn_like(xg)
    params = [xg] + list(mod.parameters())
    step = lambda: torch.autograd.grad(mod(xg), params, dy)  # noqa: E731
    def infer():
        with torch.no_grad():
            return mod(xg)
    for nrep in os.environ.get("NREPS", "13").split():
        os.environ["MINIWORLD_TRANSITION_SM80_NREP"] = nrep
        t_step, t_inf = graph_ms(step)[0] * 1e3, graph_ms(infer)[0] * 1e3
        step(); torch.cuda.synchronize()
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
            for _ in range(10):
                step()
            torch.cuda.synchronize()
        kt = collections.defaultdict(float)
        for e in prof.events():
            if e.device_type == torch.autograd.DeviceType.CUDA:
                kt[e.name[:34]] += e.device_time_total / 10
        print(f"L{L} nrep {nrep}: step {t_step:.1f} us  infer {t_inf:.1f} us | " + "  ".join(f"{k} {v:.1f}" for k, v in sorted(kt.items(), key=lambda kv: -kv[1])[:9]), flush=True)
