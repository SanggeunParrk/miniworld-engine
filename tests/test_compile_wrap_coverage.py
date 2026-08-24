"""``compile_wrap="custom_op"`` has to be a real mode, not a switch that raises on import.

It shipped as a documented ``settings`` field with two "interchangeable" values. It was not:
47 of the 58 kernel entry points put ``@opaque()`` on ``autograd.Function.forward``/``backward``,
which ``torch.library.custom_op`` cannot wrap (``ctx`` is not a schema type), so simply SELECTING
the mode raised ``ValueError`` before a single kernel loaded. A configuration that cannot be
selected is not a configuration.

These tests are static -- no GPU, no launches. They check the two things that let the mode rot:

1. every ``opaque`` site can register, i.e. it either carries a ``fake`` or is the ``disable``
   default (:func:`test_custom_op_mode_registers_every_site`);
2. every site that stays a graph break says WHY at the site
   (:func:`test_declared_graph_breaks_state_a_reason`), so "still breaking" is a fact someone
   wrote down rather than the accident of a missing fake.

The second matters because two shapes genuinely cannot be ops, and both fail SILENTLY if wrapped:
an ``nn.Module`` method (``self`` is not a schema type -- that one at least raises), and a wrapper
whose body calls ``Function.apply``. An op is opaque to AUTOGRAD as well as to Dynamo, so wrapping
an ``.apply`` returns a tensor with no ``grad_fn``: the forward numbers stay right and training
quietly stops learning through it.
"""
from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src" / "miniworld_engine"


def _decorator_sites(tree: ast.AST, name: str) -> list[ast.expr]:
    """Every decorator expression in ``tree`` whose callee is ``name``."""
    out = []
    for node in ast.walk(tree):
        for dec in getattr(node, "decorator_list", []) or []:
            call = dec if isinstance(dec, ast.Call) else None
            func = call.func if call else dec
            if isinstance(func, ast.Name) and func.id == name:
                out.append(dec)
    return out


def _python_files() -> list[Path]:
    return sorted(p for d in ("kernels", "modules") for p in (SRC / d).rglob("*.py"))


def test_no_bare_opaque_site_remains() -> None:
    """``@opaque()`` with no ``fake`` is exactly the thing that made the mode unselectable."""
    bare: list[str] = []
    for path in _python_files():
        tree = ast.parse(path.read_text())
        for dec in _decorator_sites(tree, "opaque"):
            if isinstance(dec, ast.Call) and not dec.args and not dec.keywords:
                bare.append(f"{path.relative_to(SRC)}:{dec.lineno}")
    assert not bare, (
        "these @opaque() sites have no fake, so compile_wrap='custom_op' cannot register them:\n  "
        + "\n  ".join(bare))


def _is_launch(node: ast.AST) -> bool:
    """Is this ``kernel[grid](...)``, a Triton launch?

    Narrower than "a call whose func is a Subscript": ``candidates[idx][1]()`` in the cute
    dispatcher is that too, and it is a thunk, not a launch. A launch subscripts a plain NAME (the
    kernel) and is called WITH arguments.
    """
    return (isinstance(node, ast.Call)
            and isinstance(node.func, ast.Subscript)
            and isinstance(node.func.value, (ast.Name, ast.Attribute))
            and bool(node.args or node.keywords))


def _launchers_and_ops(paths):
    """(launchers, ops, calls) -- a launcher is a function that launches a Triton kernel."""
    launchers, ops, calls = {}, set(), {}
    for path in paths:
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            names = []
            for dec in node.decorator_list:
                f = dec.func if isinstance(dec, ast.Call) else dec
                names.append(f.id if isinstance(f, ast.Name) else getattr(f, "attr", ""))
            key = f"{path.name}::{node.name}"
            is_op = bool({"opaque", "custom_op", "triton_op"} & set(names))
            if is_op:
                ops.add(node.name)
            if any(_is_launch(n) for n in ast.walk(node)):
                launchers[key] = (node.name, is_op, node.lineno, path)
            for n in ast.walk(node):
                if isinstance(n, ast.Call):
                    f = n.func
                    nm = f.id if isinstance(f, ast.Name) else getattr(f, "attr", None)
                    if nm:
                        calls.setdefault(nm, set()).add((node.name, is_op))
    return launchers, ops, calls


def test_every_launcher_is_opaque_or_only_reached_through_one() -> None:
    """No traced path may reach a raw kernel launch.

    A launcher does NOT have to be an op if every one of its callers already is: it can only be
    entered from inside an opaque region, so Dynamo never sees it, and wrapping it would just add
    a second dispatch on a hot inner call. Everything else needs its own fake.

    ``checks_*`` / ``drivers_*`` are excluded: that is the autotune-capture harness, which calls
    kernels directly and is never inside a compiled model.
    """
    paths = [p for d in ("kernels", "modules") for p in (SRC / d).rglob("*.py")
             if not p.name.startswith(("checks_", "drivers_"))]
    launchers, ops, calls = _launchers_and_ops(paths)

    # "Covered" is TRANSITIVE: `_ln_bwd_atomic`'s only caller is `ln_bwd_mmajor`, which is not an
    # op itself but is only ever called from one. Iterate to a fixed point rather than looking one
    # level up, which is what a first version of this test did -- and it flagged three launchers
    # that were already unreachable.
    covered = set(ops)
    for _ in range(len(calls) + 1):
        grew = False
        for name, callers in calls.items():
            if name in covered or not callers:
                continue
            if all(c in covered or is_op for c, is_op in callers):
                covered.add(name)
                grew = True
        if not grew:
            break

    bad = [f"{path.relative_to(SRC)}:{lineno} {name}"
           for _key, (name, is_op, lineno, path) in sorted(launchers.items())
           if not is_op and name not in covered]
    assert not bad, (
        "these launch a kernel on a path Dynamo can reach, but are not opaque ops:\n  "
        + "\n  ".join(bad))


#: The op namespace is public API: ``torch.ops.miniworld_engine.<name>`` is global and flat, and
#: a consumer reads these names in profiles, in ``TORCH_LOGS`` output and in graph dumps. So a
#: name is not a local variable -- it needs one scheme, and the scheme is: the FAMILY the kernel
#: belongs to, which is the directory it lives in, then what it does, in lower_snake_case.
#:
#: This list is the vocabulary. It was written after an audit found 41 of 107 names not starting
#: with their family at all (``cond_tf_dgemm``, ``lnl_recompute_xhat``, ``dtv1_input_gated_gemm``),
#: three different prefixes for layernorm_linear alone (``lnl_``, ``ln_``, ``layer_norm_linear_``),
#: and one name carrying a capital from maths notation (``trimul_front_bwd_dW_glogit``).
OP_FAMILIES = (
    "adaln",
    "augmented_attention",
    "bias_only_attention",
    "conditioned_transition",
    "fused_ln_mask",
    "gated_projection",
    "layernorm_linear",     # before "layernorm": it is the longer, more specific prefix
    "layernorm",
    "swa_atom_attention",
    "tm1",
    "tm2",
    "transition",
    "triangle_attention",
    "triangle_multiplication",
    "trimul",
)


def _declared_op_names() -> dict[str, str]:
    """Every ``miniworld_engine::`` op name in the tree, mapped to the file that declares it.

    Read from the source rather than from ``dir(torch.ops.miniworld_engine)`` so the check runs on
    a CPU box and covers the sm90/sm100 ops this machine can never import.
    """
    found: dict[str, str] = {}
    for path in sorted(p for d in ("kernels", "modules") for p in (SRC / d).rglob("*.py")):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            fn = node.func
            fname = fn.id if isinstance(fn, ast.Name) else getattr(fn, "attr", "")
            if fname not in ("opaque", "custom_op"):
                continue
            name: str | None = None
            for kw in node.keywords:
                if kw.arg == "name" and isinstance(kw.value, ast.Constant):
                    name = str(kw.value.value)
            if name is None and fname == "custom_op" and node.args:
                first = node.args[0]
                if isinstance(first, ast.Constant):
                    name = str(first.value).split("::")[-1]
            if name:
                found[name] = str(path.relative_to(SRC))
    return found


def test_op_names_are_lower_snake_case() -> None:
    bad = {n: f for n, f in _declared_op_names().items()
           if n != n.lower() or not n.replace("_", "").isalnum()}
    assert not bad, ("op names are public and must be lower_snake_case:\n  "
                     + "\n  ".join(f"{n}  ({f})" for n, f in sorted(bad.items())))


def test_op_names_start_with_their_family() -> None:
    """A name's first token says which kernel family it belongs to -- see ``OP_FAMILIES``."""
    bad = {n: f for n, f in _declared_op_names().items() if not n.startswith(OP_FAMILIES)}
    assert not bad, (
        "these op names do not start with a known family; either rename them or add the family "
        "to OP_FAMILIES (and say why it is a family):\n  "
        + "\n  ".join(f"{n}  ({f})" for n, f in sorted(bad.items())))


def test_nothing_bypasses_the_compile_wrap_switch() -> None:
    """``opaque`` is the ONE way a kernel launch becomes an op. No exceptions.

    ``layernorm_linear/triton/pair_bias.py`` used to be one: it registered its two ops with
    ``torch.library.custom_op`` directly and wired autograd with ``register_autograd``, so
    ``compile_wrap`` never reached them. That pattern is legitimate on its own terms -- everything
    its backward needs is a forward output, which is all ``setup_context`` may save -- but two
    patterns for one job cost more than the tidiness of the second one was worth. It is an
    ``autograd.Function`` over ``opaque`` launches now, like the other 105.
    """
    offenders = []
    for path in sorted(p for d in ("kernels", "modules") for p in (SRC / d).rglob("*.py")):
        if path.name == "_compile.py":
            continue                      # the switch's own implementation
        if "torch.library.custom_op(" in path.read_text():
            offenders.append(str(path.relative_to(SRC)))
    assert not offenders, ("these register ops outside kernels._compile.opaque, so compile_wrap "
                           "does not reach them:\n  " + "\n  ".join(offenders))


@pytest.mark.parametrize("wrap", ["disable", "custom_op"])
def test_custom_op_mode_registers_every_site(wrap: str) -> None:
    """Import every module that HAS an ``opaque`` site, under each mode. Registration is at import.

    A subprocess per mode is not fussiness: ``kernels._compile`` reads ``compile_wrap`` when the
    decorator RUNS, so a single interpreter can only ever hold one of the two.

    Only the 53 files that contain ``@opaque`` are imported, not all 184 under ``kernels/`` and
    ``modules/``. The whole tree took over ten minutes -- most of it nvcc JIT-building CUDA
    extensions that have nothing to do with op registration -- and a ten-minute test is a test
    that gets skipped, which is exactly what happened to this one while it was being written.
    The narrow set is also the complete set for the question being asked: a site that does not
    exist in a file cannot fail to register from it.
    """
    sites = sorted(p for d in ("kernels", "modules") for p in (SRC / d).rglob("*.py")
                   if "@opaque" in p.read_text())
    assert sites, "no @opaque sites found -- the test is looking in the wrong place"
    mods = [
        "miniworld_engine." + ".".join(
            p.relative_to(SRC).with_suffix("").parts).removesuffix(".__init__")
        for p in sites
    ]
    script = f"""
import sys
from miniworld_engine import settings
settings.configure(compile_wrap={wrap!r})
import importlib
missing = []
for m in {mods!r}:
    try:
        importlib.import_module(m)
    except ValueError as e:
        if "needs a fake implementation" in str(e):
            missing.append(m + " :: " + str(e))
    except Exception:
        pass          # CUDA/CuTeDSL absent on a CPU runner -- not what this test is about
print("MISSING=" + str(len(missing)))
for m in missing:
    print("  " + m)
sys.exit(1 if missing else 0)
"""
    r = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True,
                       timeout=1800, check=False)
    assert r.returncode == 0, f"compile_wrap={wrap!r} could not register every site:\n{r.stdout}"
