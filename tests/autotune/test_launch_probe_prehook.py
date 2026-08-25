"""The pre_hook timer has to be installed where triton actually keeps the hook.

``_install_launch_probes`` used to do::

    orig_pre = Autotuner.pre_hook if hasattr(Autotuner, "pre_hook") else None
    if callable(orig_pre):
        ...
        Autotuner.pre_hook = pre_hook

Dead twice over. ``Autotuner.__init__`` assigns ``self.pre_hook`` on every path and the class
carries no default, so ``hasattr(Autotuner, "pre_hook")`` is False and the block never opened --
and had it opened, every instance would have shadowed the class attribute before its first
launch. The build report printed ``pre_hook 0 x -> 0s`` for the whole life of the feature.

The probe now wraps the hook per instance on the first ``run``, which also catches autotuners
built before the installer ran -- which is most of them, since the ``@triton.autotune``
decorators fire at kernel import.
"""
from __future__ import annotations

import pytest

from miniworld_engine.autotune import capture


@pytest.fixture
def probes_installed(monkeypatch):
    """Install the probes onto a fresh copy of the counters, and undo the patch afterwards."""
    from triton.compiler.compiler import CompiledKernel
    from triton.runtime.autotuner import Autotuner
    from triton.runtime.jit import JITFunction

    monkeypatch.setattr(capture, "_PROBES_INSTALLED", False, raising=False)
    monkeypatch.setattr(capture, "_LAUNCH_T", dict.fromkeys(capture._LAUNCH_T, 0))
    # The installer patches four class attributes; put all of them back, or the rest of the
    # session runs against a probe whose counters this fixture has thrown away.
    saved = [(Autotuner, "run", Autotuner.run),
             (JITFunction, "run", JITFunction.run),
             (CompiledKernel, "_init_handles", CompiledKernel._init_handles),
             (CompiledKernel, "__init__", CompiledKernel.__init__)]
    capture._install_launch_probes()
    yield Autotuner
    for cls, name, fn in saved:
        setattr(cls, name, fn)


def test_the_class_has_no_prehook_to_patch():
    """The premise of the old code. If triton ever adds a class default, revisit the probe."""
    from triton.runtime.autotuner import Autotuner

    assert not hasattr(Autotuner, "pre_hook")


def test_an_instance_always_has_one():
    from triton.runtime.autotuner import Autotuner

    at = _make_autotuner(Autotuner)
    assert callable(at.pre_hook)


def test_the_probe_wraps_the_instance_hook_and_counts_it(probes_installed):
    Autotuner = probes_installed
    at = _make_autotuner(Autotuner)
    before = at.pre_hook

    # `run` is what installs the per-instance wrapper. Its body would need a device, so stop at
    # the wrapping: call the patched `run` with the original chained away.
    calls = []
    Autotuner.run = _swap_tail(Autotuner.run, lambda self, *a, **k: calls.append(1))
    at.run()
    assert at.pre_hook is not before, "run() did not wrap the instance's pre_hook"

    assert capture._LAUNCH_T["prehook_calls"] == 0
    at.pre_hook({})
    assert capture._LAUNCH_T["prehook_calls"] == 1, "the wrapper did not count a hook call"
    assert capture._LAUNCH_T["prehook_s"] >= 0.0


def test_wrapping_happens_once_per_instance(probes_installed):
    Autotuner = probes_installed
    at = _make_autotuner(Autotuner)
    Autotuner.run = _swap_tail(Autotuner.run, lambda self, *a, **k: None)
    at.run()
    wrapped = at.pre_hook
    at.run()
    assert at.pre_hook is wrapped, "a second run() re-wrapped, so each call would be counted twice"


def test_probes_install_once(probes_installed):
    Autotuner = probes_installed
    first = Autotuner.run
    capture._install_launch_probes()
    assert Autotuner.run is first, "a second install chained another layer onto run"


# --------------------------------------------------------------------------------------------- #


def _kernel_stub():   # `Autotuner.__init__` walks `fn.fn` until `inspect.isfunction`
    return None


def _make_autotuner(cls):
    """An Autotuner with a stub behind it -- enough to own a `pre_hook`."""
    import triton

    return cls(fn=_kernel_stub, arg_names=[], configs=[triton.Config({})], key=[],
               reset_to_zero=None, restore_value=None)


def _swap_tail(patched, tail):
    """Replace the `orig_at_run(self, ...)` the probe chains to, keeping the probe's own body."""
    cell = next(c for c in patched.__closure__ if callable(c.cell_contents)
                and getattr(c.cell_contents, "__name__", "") != "at_run")
    cell.cell_contents = tail
    return patched
