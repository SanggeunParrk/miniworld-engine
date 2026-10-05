"""Fullgraph SWA compilation must execute upstream kernels with live operands."""

import os

import pytest
import torch
from benchmarks.runners.measurement import CompileProbe, observe_execution

from miniworld_engine.modules.swa_atom_attention import build_attention_params
from miniworld_engine.modules.swa_dit import SWADiTBlock

pytestmark = [pytest.mark.gpu, pytest.mark.skipif(
    not os.environ.get("MINIWORLD_ANTHROPIC_ROOT"), reason="needs pinned Anthropic payload")]


@pytest.mark.parametrize(("dtype", "length"), [(torch.bfloat16, 128), (torch.float16, 384)])
def test_fullgraph_live_inputs_weights_and_masks(dtype, length):
    if torch.cuda.get_device_capability() != (8, 0):
        pytest.skip("requires A100")
    torch.manual_seed(211)
    mod = SWADiTBlock(implementation="anthropic").cuda().to(dtype).eval()
    ref = SWADiTBlock(implementation="pytorch").cuda().float().eval()
    x = torch.randn(3, length, 128, device="cuda", dtype=dtype)
    cond = torch.randn_like(x)
    angles = torch.randn(1, length, 16, device="cuda")
    valid = torch.arange(length, device="cuda")[None] < torch.tensor(
        [length, length - 17, 0], device="cuda")[:, None]
    ap = build_attention_params(angles.cos(), angles.sin(), valid, 3)
    graphs = []

    def backend(gm, example_inputs, **kwargs):
        from torch._dynamo.backends.registry import lookup_backend
        graphs.append(gm)
        return lookup_backend("inductor")(gm, example_inputs, **kwargs)

    probe = CompileProbe("swa", fullgraph=True, backend=backend)
    compiled = torch.compile(mod, fullgraph=True, dynamic=False, backend=probe,
                             options={"triton.cudagraphs": False})
    old_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        with torch.no_grad():
            # Nonzero gates: the default identity block cannot test attention/FFN.
            for p in mod.parameters():
                p.normal_(std=p.shape[-1] ** -.5)
            with observe_execution() as evidence:
                got = compiled(x, cond, ap)  # compile before any eager warmup
            assert evidence.compiled
            assert evidence.scopes == {"swa:fullgraph"}
            targets = [str(n.target) for n in graphs[0].graph.nodes]
            for op in ("swa_rms", "swa_gate", "swa_swiglu", "swa_gather"):
                assert any(op in target for target in targets), targets
            assert any("linear" in target for target in targets), targets
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(3):
                    compiled(x, cond, ap)
            torch.cuda.current_stream().wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                output = compiled(x, cond, ap)
            for update in range(2):
                if update:
                    x.mul_(.7)
                    cond.add_(.2)
                    mod.adaln_modulation[1].weight.mul_(.8)
                    mod.attn.Wqkv.weight.mul_(.9)
                    ap[0].mul_(.95)
                    # Rank-space reference also exercises interior gaps and an
                    # empty row becoming populated without recompiling/capturing.
                    valid[0, ::7] = False
                    valid[2, :length // 2] = True
                ref.load_state_dict(mod.state_dict())
                want = ref(x.float(), cond.float(), ap)
                eager = mod(x, cond, ap)
                got = compiled(x, cond, ap)
                graph.replay()
                torch.testing.assert_close(output, got, rtol=0, atol=0)
                assert torch.isfinite(got).all()
                assert (got.float() - eager.float()).norm() / eager.float().norm() < .01
                delta = want - x.float()
                assert delta.norm() > .1 * x.float().norm()
                assert (got.float() - want).norm() / want.norm() < .04
                assert (got.float() - want).norm() / delta.norm() < .06
            assert probe.graphs_created == 1
        with pytest.raises(RuntimeError, match="inference-only"):
            mod(x, cond, ap)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = old_tf32


def test_fullgraph_preserves_upstream_dtype_refusal():
    from miniworld_engine.integrations import anthropic as upstream
    from miniworld_engine.integrations.anthropic_swa_ops import gather

    q = torch.zeros(1, 16, 128, device="cuda", dtype=torch.float32)
    bias = torch.zeros((), device="cuda").expand(1, 16, 16, 4)
    indices = torch.zeros(1, 16, 1, device="cuda", dtype=torch.int32)
    refusal = upstream.carried_kernel("gather_attn").Refusal

    def call(q, bias, indices):
        return gather(q, q, q, bias, indices, 4, 32 ** -.5)

    compiled = torch.compile(call, fullgraph=True)
    with torch.no_grad():
        for fn in (call, compiled):
            with pytest.raises(refusal, match="v-dtype-float32"):
                fn(q, bias, indices)
