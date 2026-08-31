"""Two answers: what band do the seeded checkers actually need, and why is the OOR still there."""
import torch, traceback
from miniworld_engine.autotune import cache
from miniworld_engine.kernels.checks import mpnn_edge_tail as et, mpnn_node_message as nm

print("=== worst relative error per checker, across five seeds ===")
CHECKS = [("edge_tail_bwd_layernorm_saveact", et.mpnn_edge_tail_bwd_layernorm_saveact_triton),
          ("edge_tail_bwd_dx_saveact",        et.mpnn_edge_tail_bwd_dx_saveact_triton),
          ("edge_tail_bwd_dx_gather_saveact", et.mpnn_edge_tail_bwd_dx_gather_saveact_triton),
          ("edge_tail_fwd_gemm_gather_saveact", et.mpnn_edge_tail_fwd_gemm_gather_saveact_triton),
          ("node_message_bwd_dx",             nm.mpnn_node_message_bwd_dx_triton),
          ("node_message_fwd_gemm",           nm.mpnn_node_message_fwd_gemm_triton)]
for label, fn in CHECKS:
    worst = {}
    for seed in range(5):
        torch.manual_seed(seed)
        got = fn()
        pairs = got if isinstance(got, dict) else {"out": got}
        for k, (a, e) in pairs.items():
            r = (a.float()-e.float()).norm().item()/max(e.float().norm().item(), 1e-30)
            worst[k] = max(worst.get(k, 0.0), r)
    print(f"  {label:34s} " + "  ".join(f"{k}={v:.2e}" for k, v in sorted(worst.items(), key=lambda x: -x[1])[:4]))

print("\n=== the OOR: what does the reader actually hand back now ===")
import miniworld_engine.kernels.mpnn_edge_tail.triton.main as m
k = m._edge_tail_replay_kernel
full = list(k.configs)
sub = cache.heuristic_subset(full, 24)
print(f"  grid {len(full)}, subset {len(sub)}, stages in subset {sorted({c.num_stages for c in sub})}")
print(f"  smallest 4: {sorted({(c.kwargs.get('BLOCK_M'), c.kwargs.get('TILES'), c.num_warps, c.num_stages) for c in sub})[:4]}")
print(f"  cache reader installed: {cache._reader_installed}")
print(f"  autotuner prune hook: {getattr(k, 'early_config_prune', None) is not None}")
from miniworld_engine.kernels.drivers import mpnn_edge_tail as d
try:
    d.mpnn_edge_tail_bwd_recompute_triton(); print("  driver: ok")
except Exception as exc:
    print(f"  driver: {type(exc).__name__}: {str(exc)[:120]}")
    traceback.print_exc()
