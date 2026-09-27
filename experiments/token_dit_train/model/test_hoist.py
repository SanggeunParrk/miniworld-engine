"""pair_bias_all (hoist_pair_bias=True) against the per-block path, both measured against the truth.

Truth: the same weights through the engine's PYTORCH implementation (einsum attention) at full fp32 matmul precision.
The engine path's Triton attention computes in TF32 whatever the matmul setting, and the two engine paths may pick
different autotune configs (different rounding), so they are not compared with each other but each with the truth:
the hoist passes if it is exact where it differs (the bias, to fp32 rounding) and its error against the truth is no
worse than the per-block path's. Every parameter is randomised (ln_pair.weight is 1 and to_bias.weight 0 at init);
each path is a fresh model instance, so parameter gradients never accumulate across runs.
"""
import torch
from team_gm.modules import DiffusionTransformer
from team_gm.modules.exceptions import ImplementationType
from team_gm.modules.blocks.pair_bias_hoist import pair_bias_all

dev = "cuda"
bad = 0
for prec in ("highest", "medium"):
    for L, A, NB, ckpt, masked in ((64, 4, 3, True, False), (128, 3, 4, True, True), (96, 2, 2, False, False), (256, 4, 2, True, False)):
        torch.manual_seed(0)
        cfg = dict(d_single=768, d_cond=384, d_pair=128, n_head=16, n_block=NB, use_qk_norm=True,
                   n_checkpoint_segments=NB if ckpt else None)
        truth = DiffusionTransformer(DiffusionTransformer.Config(**cfg, implementation=ImplementationType.PYTORCH)).to(dev)
        with torch.no_grad():
            for prm in truth.parameters():
                if prm.ndim >= 2: prm.normal_(std=prm.shape[-1] ** -0.5)
                else: prm.normal_(mean=1.0, std=0.3)
        sd = truth.state_dict()
        single = torch.randn(A, 1, L, 768, device=dev)
        cond = torch.randn(A, 1, L, 384, device=dev)
        pair = torch.randn(1, L, L, 128, device=dev) * 2 + 0.5
        mask = (torch.rand(1, L, device=dev) > 0.2) if masked else None
        w = torch.randn(A, 1, L, 768, device=dev)

        from miniworld_engine.kernels.adaln.triton import training as _adaln_tr

        def run(impl, hoist, p, adaln="auto", core="model"):
            torch.set_float32_matmul_precision(p)
            _adaln_tr.set_forward_mode(adaln)
            m = DiffusionTransformer(DiffusionTransformer.Config(**cfg, implementation=impl, hoist_pair_bias=hoist,
                                                                 attention_core_dtype=core)).to(dev)
            m.load_state_dict(sd)
            ins = [t.clone().requires_grad_(True) for t in (single, cond, pair)]
            out = m(*ins, mask)
            (out * w).sum().backward()
            return m, out.detach(), [t.grad for t in ins], {n: q.grad for n, q in m.named_parameters()}

        _, o_t, gi_t, gp_t = run(ImplementationType.PYTORCH, False, "highest")
        m_r, o_r, gi_r, gp_r = run(ImplementationType.MINIWORLD_ENGINE, False, prec)
        m_h, o_h, gi_h, gp_h = run(ImplementationType.MINIWORLD_ENGINE, True, prec)
        _, o_c, gi_c, gp_c = run(ImplementationType.MINIWORLD_ENGINE, True, prec, "cublas")
        _, o_b, gi_b, gp_b = run(ImplementationType.MINIWORLD_ENGINE, True, prec, "cublas", "bf16")
        tdt = L % 128 == 0                      # the sm_90 training kernels tile L by 128
        if tdt:
            _, o_d, gi_d, gp_d = run(ImplementationType.MINIWORLD_ENGINE, True, prec, "cublas", "tdt")
        _adaln_tr.set_forward_mode("auto")
        with torch.no_grad():
            bh = pair_bias_all([b.attention_pair_bias for b in m_h.blocks], pair)
            br = [b.attention_pair_bias.to_bias(b.attention_pair_bias.ln_pair(pair)) for b in m_r.blocks]
            bias_err = max(float((x - y).norm() / y.norm()) for x, y in zip(bh, br))
        rel = lambda a, b: float((a - b).norm() / b.norm().clamp_min(1e-30))

        def errs(o, gi, gp):
            e = {"out": rel(o, o_t)}
            for n, a, b in zip(("dsingle", "dcond", "dpair"), gi, gi_t):
                e[n] = rel(a, b)
            e["dpair-bias params"] = max(rel(gp[n], gp_t[n]) for n in gp_t if ("ln_pair" in n or "to_bias" in n))
            e["dparams"] = max(rel(gp[n], gp_t[n]) for n in gp_t if gp_t[n] is not None)
            return e

        er, eh, ec = errs(o_r, gi_r, gp_r), errs(o_h, gi_h, gp_h), errs(o_c, gi_c, gp_c)
        ok = bias_err < (1e-6 if prec == "highest" else 2e-3) and all(eh[k] <= 1.5 * er[k] + 1e-6 for k in er) \
            and all(ec[k] <= 1.5 * er[k] + 1e-6 for k in er)
        bad += not ok
        print(f"prec={prec:7s} L{L} A{A} NB{NB} ckpt={int(ckpt)} mask={int(masked)}  bias hoist-vs-block {bias_err:.1e}  "
              f"{'ok' if ok else 'FAIL'}", flush=True)
        print("    vs truth, per-block: " + "  ".join(f"{k} {v:.1e}" for k, v in er.items()), flush=True)
        print("    vs truth, hoisted:   " + "  ".join(f"{k} {v:.1e}" for k, v in eh.items()), flush=True)
        print("    vs truth, +cublas:   " + "  ".join(f"{k} {v:.1e}" for k, v in ec.items()), flush=True)
        eb = errs(o_b, gi_b, gp_b)
        print("    vs truth, +bf16 core:" + "  ".join(f"{k} {v:.1e}" for k, v in eb.items()), flush=True)
        if tdt:
            ed = errs(o_d, gi_d, gp_d)
            print("    vs truth, +tdt core: " + "  ".join(f"{k} {v:.1e}" for k, v in ed.items()), flush=True)
print("ALL OK" if bad == 0 else f"{bad} FAILED", flush=True)
