"""Why capture_cudagraph fails for every autograd-backward bench.

Every `_bwd_autograd_result` target (adaln_bwd, transition_b2b_bwd, gemm_epil_bwd,
mpnn_edge_tail) fails under cudagraph with the same pair of errors, while every
`_pure_bwd_result` target captures fine. The difference is that the former captures
`torch.autograd.grad` and the latter captures a plain function.

A failed capture leaves the default generator registered to the dead graph -- capture
begins by registering it and only `capture_epilogue` unregisters it -- so every later
RNG call in the process raises "Offset increment outside graph capture". That poison
does not respect case boundaries, so EACH CASE RUNS IN ITS OWN PROCESS. Sharing one
process is what made the first version of this script unreadable: the harness case
poisoned the generator and the next case reported the poison instead of its own result.

  python submits/_cudagraph_autograd_repro.py          # driver: every case
  python submits/_cudagraph_autograd_repro.py <case>   # one case, fresh process
"""

from __future__ import annotations

import subprocess
import sys

CASES = {
    "pure": "pure function capture (what the passing targets do)",
    "harness": "autograd.grad, forward on the default stream (what the harness does)",
    "fwd_on_side": "autograd.grad, forward built on a non-default stream",
    "single_thread": "autograd.grad, autograd multithreading DISABLED",
    "single_thread_side": "autograd.grad, single-threaded AND forward on side stream",
}

D, N = 512, 4096


def build(on_side_stream: bool):
    import torch

    w = torch.randn(D, D, device="cuda")
    x = torch.randn(N, D, device="cuda", requires_grad=True)
    dy = torch.randn(N, D, device="cuda")
    if on_side_stream:
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            out = torch.relu(x @ w)
        torch.cuda.current_stream().wait_stream(stream)
    else:
        out = torch.relu(x @ w)
    return out, [x], dy


def capture(step, is_train: bool):
    """A faithful copy of benchmarks/runners/bench.py::capture_cudagraph."""
    import torch

    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(8):
            if is_train:
                step()
            else:
                with torch.no_grad():
                    step()
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    if is_train:
        with torch.cuda.graph(graph):
            step()
    else:
        with torch.cuda.graph(graph), torch.no_grad():
            step()
    return graph


def run_case(case: str) -> None:
    import torch

    if case in ("single_thread", "single_thread_side"):
        torch.autograd.set_multithreading_enabled(False)
    if case == "pure":
        w = torch.randn(D, D, device="cuda")
        x = torch.randn(N, D, device="cuda")
        graph = capture(lambda: torch.relu(x @ w), is_train=False)
    else:
        out, leaves, dy = build(on_side_stream=case.endswith("side"))
        graph = capture(
            lambda: torch.autograd.grad(out, leaves, dy, retain_graph=True),
            is_train=True,
        )
    graph.replay()
    torch.cuda.synchronize()
    print("CAPTURE OK", flush=True)
    torch.randn(4, device="cuda")
    torch.cuda.synchronize()
    print("generator still usable", flush=True)


def main() -> int:
    if len(sys.argv) > 1:
        run_case(sys.argv[1])
        return 0

    import torch

    print(torch.cuda.get_device_name(0), torch.__version__, flush=True)
    width = max(len(c) for c in CASES)
    for case, blurb in CASES.items():
        proc = subprocess.run(  # noqa: S603
            [sys.executable, __file__, case],
            capture_output=True, text=True, check=False,
        )
        body = proc.stdout + proc.stderr
        if "CAPTURE OK" in body:
            note = "" if "generator still usable" in body else "   (generator poisoned)"
            verdict = f"OK{note}"
        else:
            first = next(
                (ln.strip() for ln in body.splitlines() if "rror" in ln),
                "unknown failure",
            )
            verdict = f"FAILED   {first[:92]}"
        print(f"  {case:<{width}}  {verdict}", flush=True)
        print(f"  {'':<{width}}  ({blurb})", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
