"""Two questions in one job: does the seeded checker reproduce, and is the OOR the narrowed set?"""
import torch, triton
from miniworld_engine.autotune.cache import heuristic_subset
from miniworld_engine.autotune.configs import configs_for

print("=== how much of each mpnn ladder fits on this card ===")
import miniworld_engine.kernels.mpnn_edge_tail.triton.main as m
for name in ("_edge_tail_replay_kernel", "_edge_tail_dx_kernel"):
    k = getattr(m, name)
    full = list(k.configs)
    sub = heuristic_subset(full, 24)
    print(f"  {name}: grid {len(full)}, heuristic subset {len(sub)}")
    print("    subset:", sorted({(c.kwargs.get('BLOCK_M'), c.kwargs.get('TILES'),
                                  c.num_warps, c.num_stages) for c in sub})[:8])

print("\n=== the seeded checkers, twice, in one process ===")
from miniworld_engine.kernels.checks import mpnn_edge_tail as et, mpnn_node_message as nm
for run in (1, 2):
    for label, fn in (("edge_tail_bwd_dx_saveact", et.mpnn_edge_tail_bwd_dx_saveact_triton),
                      ("node_message_bwd_dx", nm.mpnn_node_message_bwd_dx_triton)):
        try:
            got = fn()
            worst = max((a.float()-e.float()).norm().item()/max(e.float().norm().item(), 1e-30)
                        for a, e in got.values())
            print(f"  run{run} {label:28s} worst rel {worst:.3e}")
        except Exception as exc:
            print(f"  run{run} {label:28s} RAISED {type(exc).__name__}: {str(exc)[:90]}")

print("\n=== the two OOR kernels, through their drivers ===")
from miniworld_engine.kernels.drivers import mpnn_edge_tail as d
for label, fn in (("bwd_recompute", d.mpnn_edge_tail_bwd_recompute_triton),
                  ("bwd_dx_recompute", d.mpnn_edge_tail_bwd_dx_recompute_triton)):
    try:
        fn(); print(f"  {label}: ok")
    except Exception as exc:
        print(f"  {label}: {type(exc).__name__}: {str(exc)[:110]}")
