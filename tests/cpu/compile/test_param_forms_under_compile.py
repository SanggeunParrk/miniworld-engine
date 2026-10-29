"""``_capture.param_forms`` inside a ``torch.compile`` trace: the forms are computed in the graph. Traced, the identity-keyed
cache used to kill Dynamo on its weak references (FoldForge's compiled Protenix-v2 run, H100 AdaLN inference ``packed``);
eager calls still build once and reuse."""
import torch

from miniworld_engine.kernels import _capture


def test_param_forms_traces_as_its_build_and_still_caches_eagerly():
    w = torch.nn.Parameter(torch.randn(8, 4))
    builds = []

    def forms():
        builds.append(1)
        return w.t().contiguous() * 2

    def step(x):
        return x @ _capture.param_forms("test_param_forms_compile", (w,), forms)

    x = torch.randn(3, 4)
    with torch.no_grad():
        eager = step(x)
        assert torch.equal(step(x), eager) and len(builds) == 1          # the second eager call is served the entry
        compiled = torch.compile(step, backend="eager", fullgraph=True)(x)
    torch.testing.assert_close(compiled, eager)
