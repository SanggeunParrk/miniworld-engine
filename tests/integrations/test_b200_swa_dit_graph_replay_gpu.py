"""CUDA-graph replays of the SWA atom DiT forward chain on B200 must not hang.

The A = 1 - 3 builds of ``qkvg_fwd2`` / ``ffn_fwd2`` (ATM = 32, one FFN stage) with the early ``griddepcontrol.launch_dependents`` hung the
four-kernel PDL chain (modulation -> qkvg -> attention -> ffn) after a few dozen replays when the rows were ragged (``seqused`` below S):
the GPU spun at full power with the host in ``cudaStreamSynchronize``. Those two cubins are built with ``-DPDL_EARLY=0`` now. A hang
cannot be interrupted in-process, so each case runs in a subprocess with a timeout."""

from __future__ import annotations

import subprocess
import sys
import textwrap

import pytest
import torch

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 0), reason="B200 (sm_100)"),
]

_SCRIPT = textwrap.dedent("""
    import sys, torch
    from miniworld_engine.modules.exceptions import ImplementationType
    from miniworld_engine.modules.swa_atom_attention.module import build_attention_params
    from miniworld_engine.modules.swa_dit import SWADiTBlock
    n, s, shared, replays = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3] == "shared", int(sys.argv[4])
    torch.manual_seed(0)
    block = SWADiTBlock(128, 128, 4, implementation=ImplementationType.MINIWORLD)
    with torch.no_grad():
        for p in block.parameters():
            if p.ndim == 2:
                p.normal_(std=p.shape[1] ** -0.5)
    block = block.to("cuda", torch.bfloat16)
    cos, sin = torch.randn(1, s, 16, device="cuda"), torch.randn(1, s, 16, device="cuda")
    lengths = (torch.rand(n, s, device="cuda") > 0.2).sum(-1)                              # ragged, front-packed
    valid = torch.arange(s, device="cuda")[None, :] < lengths[:, None]
    ap = build_attention_params(cos, sin, valid, num_aug=n)
    x = torch.randn(n, s, 128, device="cuda", dtype=torch.bfloat16)
    cond = torch.randn(1 if shared else n, s, 128, device="cuda", dtype=torch.bfloat16)
    def step():
        with torch.no_grad():
            return block.forward_hoisted(x, cond, ap) if shared else block(x, cond, ap)
    st = torch.cuda.Stream(); st.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(st):
        for _ in range(3):
            step()
    torch.cuda.current_stream().wait_stream(st); torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = step()
    for k in range(replays):
        graph.replay()
        if k % 100 == 99:
            torch.cuda.synchronize()
    torch.cuda.synchronize()
    print("replayed", replays)
""")


# (samples, S, conditioning): one conditioning row per sample (B = N, A = 1: the ATM = 32 / single-stage builds), shared by
# the samples (A = 1, 3 use the same builds; A = 5, 48 the default ones).
@pytest.mark.parametrize("n,s,cond", [(48, 4096, "per_row"), (1, 4096, "shared"), (3, 2048, "shared"), (5, 4096, "shared"),
                                      (48, 4096, "shared")])
def test_forward_chain_survives_graph_replays(n, s, cond):
    try:
        done = subprocess.run([sys.executable, "-c", _SCRIPT, str(n), str(s), cond, "1000"], capture_output=True, text=True,
                              timeout=180, check=False)
    except subprocess.TimeoutExpired:
        pytest.fail(f"N={n} S={s} {cond} conditioning: the graph replays hung (the forward PDL chain)")
    assert done.returncode == 0, done.stderr[-2000:]
    assert "replayed 1000" in done.stdout
