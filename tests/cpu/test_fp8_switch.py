"""The FP8 backend ``engine-fp8`` (name resolution, its scope per module call, outside the judge) and the lower-level switch
(``miniworld_engine.fp8``): ``MINIWORLD_FP8=<preset>`` / ``fp8.enable(preset)`` map to the per-route switches;
an explicitly set per-route variable wins; unset = every route off (the default route). TriAttn FP8 only for a model in
``fp8.TRIATTN_MODELS`` (``MINIWORLD_FP8_MODEL`` / ``enable(model=)``); FP8 is outside the judge (no registry dtype, no judge / measure path)."""

from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from miniworld_engine import fp8

_STATUS = "import json; from miniworld_engine import fp8; print(json.dumps(fp8.status()))"
_ROUTE_VARS = ("MINIWORLD_FP8", *fp8.PRESETS["safe"], "MINIWORLD_ATOMDIT_FP8_B200", "MINIWORLD_TOKEN_DIT_FP8_B200")  # "MINIWORLD_FP8" also covers MINIWORLD_FP8_MODEL
_PKG = Path(fp8.__file__).resolve().parent


def _clean_env() -> dict:
    return {k: v for k, v in os.environ.items() if not k.startswith(_ROUTE_VARS)}


def _status_in_process(**env) -> dict:
    out = subprocess.run([sys.executable, "-c", _STATUS], env={**_clean_env(), **env}, capture_output=True, text=True, check=True)
    return json.loads(out.stdout.strip().splitlines()[-1])


NO_TRIATTN = {"pv": [], "qk": False}
OFF = {
    "triattn": NO_TRIATTN,
    "trimul": None,
    "opm": None,
    "pwa": False,
    "transition": None,
    "token_dit": [],
    "atom_dit": [],
}
#: ``safe`` without a model: every route but TriAttn
SAFE = {
    "triattn": NO_TRIATTN,
    "trimul": "calib",
    "opm": "bound",
    "pwa": True,
    "transition": "calib",
    "token_dit": ["expand"],
    "atom_dit": [] if fp8.arch() == "sm100" else ["post", "pre"],  # B200: status lists the b200_atom.f8 stages (opt-in, not in the preset)
}
#: the TriAttn part per validated model (af3 / openfold3: PV fused + unfused and q / k; protenix1: PV fused + q / k; the others: PV only, on their checked kinds)
TRIATTN_OF = {
    "af3": {"pv": ["fused", "unfused"], "qk": True},
    "protenix1": {"pv": ["fused"], "qk": True},
    "opendde": {"pv": ["unfused"], "qk": False},
    "protenix2": {"pv": ["unfused"], "qk": False},
    "openfold3": {"pv": ["fused", "unfused"], "qk": True},
    "rosettafold3": {"pv": ["fused"], "qk": True},
    "chai1": {"pv": ["d64", "unfused"], "qk": False},
}
# (the B200 TriAttn route builds and serves e4m3 q / k since docs/b200/fp8.md "TriAttn e4m3 q / k": status reports the whitelist's qk on both archs; the gate is S >= 1024)


def _routes(status: dict) -> dict:
    return {k: v for k, v in status.items() if k != "guard"}


def test_env_preset(monkeypatch):
    for k in (*_ROUTE_VARS, "MINIWORLD_FP8_MODEL"):
        monkeypatch.delenv(k, raising=False)
    assert fp8.preset() is None and fp8.env("MINIWORLD_TRIMUL_FP8") is None and fp8.model() is None
    monkeypatch.setenv("MINIWORLD_FP8", "safe")
    assert fp8.preset() == "safe" and fp8.env("MINIWORLD_TRIMUL_FP8") == "calib"
    monkeypatch.setenv("MINIWORLD_TRIMUL_FP8", "")  # explicitly off wins over the preset
    assert fp8.env("MINIWORLD_TRIMUL_FP8") == ""
    monkeypatch.setenv("MINIWORLD_FP8_MODEL", " AF3 ")
    assert fp8.model() == "af3"
    monkeypatch.setenv("MINIWORLD_FP8", "fastest")
    with pytest.raises(ValueError):
        fp8.preset()


def test_triattn_whitelist():
    # extend only after an end-level check
    assert set(fp8.TRIATTN_MODELS) == {"af3", "protenix1", "opendde", "protenix2", "openfold3", "rosettafold3", "chai1"}
    assert "boltz2" not in fp8.TRIATTN_MODELS  # 9c90 bistable
    assert "intellifold2" not in fp8.TRIATTN_MODELS  # end-level 10 / 12 (two large targets flip), trunk heads are 64-channel
    # e4m3 q / k: the fused d_pair-128 route only, validated end to end on AF3 and OpenFold3
    assert [m for m, (_k, qk) in fp8.TRIATTN_MODELS.items() if qk] == ["af3", "protenix1", "openfold3", "rosettafold3"]
    assert fp8.triattn_for(None) == (frozenset(), False) and fp8.triattn_for("boltz2") == (frozenset(), False)
    assert fp8.triattn_for("AF3") == fp8.TRIATTN_MODELS["af3"]


def test_default_is_off():
    assert _routes(_status_in_process()) == OFF
    assert _routes(_status_in_process(MINIWORLD_FP8_MODEL="af3")) == OFF  # the model alone switches nothing on


def test_env_safe_without_model_keeps_triattn_off():
    assert _routes(_status_in_process(MINIWORLD_FP8="safe")) == SAFE
    st = _status_in_process(MINIWORLD_FP8="safe", MINIWORLD_TOKEN_DIT_FP8="", MINIWORLD_PWA_FP8_GUARD="lazy")
    assert _routes(st) == {**SAFE, "token_dit": []} and st["guard"]["pwa"] == "lazy"


@pytest.mark.parametrize("model", ["af3", "protenix1", "opendde", "protenix2", "openfold3", "rosettafold3", "chai1", "intellifold2", "boltz2"])
def test_env_safe_with_model(model):
    st = _status_in_process(MINIWORLD_FP8="safe", MINIWORLD_FP8_MODEL=model)
    assert _routes(st) == {**SAFE, "triattn": TRIATTN_OF.get(model, NO_TRIATTN)}


def test_enable_disable_api():
    code = (
        "import json; from miniworld_engine import fp8; out = []\n"
        "fp8.enable('safe', guard='lazy'); out.append(fp8.status())\n"
        "for m in ('af3', 'protenix1', 'opendde', 'protenix2', 'openfold3', 'rosettafold3', 'chai1', 'intellifold2', 'boltz2'):\n"
        "    fp8.enable('safe', model=m); out.append(fp8.status())\n"
        "fp8.enable('safe'); out.append(fp8.status())\n"
        "fp8.disable(); out.append(fp8.status()); print(json.dumps(out))"
    )
    # MINIWORLD_FP8_MODEL is not read by enable(): set it to show that
    out = subprocess.run(
        [sys.executable, "-c", code], env={**_clean_env(), "MINIWORLD_FP8_MODEL": "af3"}, capture_output=True, text=True, check=True
    )
    a, *per_model, again, off = json.loads(out.stdout.strip().splitlines()[-1])
    assert _routes(a) == SAFE and set(a["guard"].values()) == {"lazy"}
    for m, st in zip(("af3", "protenix1", "opendde", "protenix2", "openfold3", "rosettafold3", "chai1", "intellifold2", "boltz2"), per_model, strict=True):
        assert _routes(st) == {**SAFE, "triattn": TRIATTN_OF.get(m, NO_TRIATTN)}, m
    assert _routes(again) == SAFE  # enable() without model= turns the TriAttn part off again
    assert _routes(off) == OFF


def test_pv_whitelist_is_derived():
    from miniworld_engine.kernels.triangle_attention.cuda import h100_sm90 as ta

    derived = {m: kinds for m, (kinds, _qk) in fp8.TRIATTN_MODELS.items()}
    assert derived == ta.FP8_PV_WHITELIST


def test_fp8_is_outside_the_judge():
    """No registry dtype / row for FP8, and no judge / measure / requirements code path that switches or names an FP8 route."""
    for name in ("requirements_module.csv",):
        with (_PKG / "requirements" / name).open() as fh:
            dtypes = {d for row in csv.DictReader(fh) for d in row["dtypes"].split("|") if d}
        assert dtypes <= {"float32", "master"}, dtypes
    for sub in ("judge", "measure", "requirements"):
        for f in (_PKG / sub).rglob("*"):
            if f.suffix in (".py", ".csv", ".json", ".yaml", ".yml", ".toml"):
                text = f.read_text(errors="replace").lower()
                assert "fp8" not in text and "e4m3" not in text and "float8" not in text, f


# ------------------------------------------------------------------------------------------------------------- the engine-fp8 backend
def _backend_classes() -> list[type]:
    import miniworld_engine.modules as mods
    from miniworld_engine.modules.primitives import BackendModule

    for info in __import__("pkgutil").walk_packages(mods.__path__, "miniworld_engine.modules."):
        __import__("importlib").import_module(info.name)
    out, todo = [], [BackendModule]
    while todo:
        cls = todo.pop()
        out.append(cls)
        todo.extend(cls.__subclasses__())
    return out


def test_engine_fp8_name_resolution():
    from miniworld_engine.modules.primitives import BackendModule
    from miniworld_engine.modules.triangle import TriangleMultiplication

    assert (fp8.BACKEND, fp8.BACKEND_ID, fp8.BASE_BACKEND) == ("engine-fp8", "engine_fp8", "cuda")
    assert fp8.resolve_backend("engine-fp8") == ("cuda", True) == fp8.resolve_backend("engine_fp8")
    for name in ("pytorch", "cuda", "triton", "anthropic", "cueq"):
        assert fp8.resolve_backend(name) == (name, False)
    m = TriangleMultiplication(32, backend="engine-fp8")
    assert (m.backend, m.engine_fp8, m.backend_name) == ("cuda", True, "engine-fp8") and "backend=engine-fp8" in repr(m)
    m.set_backend("cuda")
    assert (m.backend, m.engine_fp8, m.backend_name) == ("cuda", False, "cuda")
    m.set_backend("engine_fp8")
    assert m.engine_fp8 and m.backend == "cuda"
    m.set_backend("pytorch")
    assert not m.engine_fp8 and TriangleMultiplication(32).engine_fp8 is False  # the default: no FP8
    assert TriangleMultiplication.has_backend("engine-fp8") and TriangleMultiplication.has_backend("cuda")
    no_cuda = [c for c in _backend_classes() if "cuda" not in c.backends]
    assert BackendModule in no_cuda and not BackendModule.has_backend("engine-fp8")
    with pytest.raises(ValueError, match="engine-fp8"):
        BackendModule().set_backend("engine-fp8")


def test_engine_fp8_reaches_a_composite_s_parts():
    from miniworld_engine.modules.pairformer import PairformerBlock, PairformerConfig

    block = PairformerBlock(
        PairformerConfig(d_pair=32, d_hidden_tri_multi=32, d_hidden_tri_attention=8, n_head_tri_attention=4), backend="engine-fp8"
    )
    parts = block._backend_children()
    assert parts and all(child.engine_fp8 and child.backend == "cuda" for _, child in parts if "cuda" in child.backends)
    block.set_backend("cuda")
    assert not any(child.engine_fp8 for _, child in parts)


def test_engine_fp8_is_not_a_judged_backend():
    """``engine-fp8`` is in no class's ``backends`` (the measurers and the judge enumerate those), and nothing the judge reads names it."""
    for cls in _backend_classes():
        assert fp8.BACKEND not in cls.backends and fp8.BACKEND_ID not in cls.backends, cls
        assert not any(k.split(":")[0] in (fp8.BACKEND, fp8.BACKEND_ID) for k in cls.backend_versions), cls
    for sub in ("judge", "measure", "requirements"):
        for f in (_PKG / sub).rglob("*"):
            if f.suffix in (".py", ".csv", ".json", ".jsonl", ".yaml", ".yml", ".toml"):
                text = f.read_text(errors="replace")
                assert "engine-fp8" not in text and "engine_fp8" not in text, f


_SCOPE = """
import json, torch
from miniworld_engine import fp8
from miniworld_engine.modules.primitives import BackendModule, BackendUnsupported
from miniworld_engine.modules.triangle import TriangleMultiplication

seen = []
def routes():
    return {k: v for k, v in fp8.status().items() if k != "guard"}

class Leaf(BackendModule):
    backends = ("pytorch", "cuda")
    def __init__(self, backend="pytorch"):
        super().__init__(); self.set_backend(backend)
    def _forward_pytorch(self, x):
        seen.append(("pytorch", routes())); return x
    def _forward_cuda(self, x):
        seen.append(("cuda", routes())); return x

class Outer(BackendModule):
    backends = ("pytorch", "cuda")
    def __init__(self, backend="pytorch"):
        super().__init__(); self.a = Leaf(); self.b = Leaf(); self.c = Leaf(); self.set_backend(backend)
    def set_backend(self, backend="pytorch"):
        super().set_backend(backend)
        for leaf in (getattr(self, "a", None), getattr(self, "b", None)):
            if leaf is not None:
                leaf.set_backend(backend)
        return self
    def _forward_pytorch(self, x):
        return x
    def _forward_cuda(self, x):
        seen.append(("outer", routes())); x = self.a(x); self.b.set_backend("cuda"); x = self.b(x); x = self.c(x)
        self.b.set_backend("engine-fp8"); return self.b(x)

x = torch.zeros(1)
out = {"before": routes()}
fp8.set_model(MODEL)
# training: train mode, or autograd on -> the default path (no scope), exactly as cuda
Outer(backend="engine-fp8")(x)
with torch.no_grad():
    Outer(backend="engine-fp8").train()(x)
Outer(backend="engine-fp8").eval()(x)
out["train"] = seen[:]
seen.clear()
torch.set_grad_enabled(False)
Leaf(backend="cuda").eval()(x); Leaf().eval()(x)
Outer(backend="engine-fp8").eval()(x)
out["seen"] = seen[:]
seen.clear()
import copy
leaf = Leaf(backend="engine-fp8").eval()
leaf._forward_cuda(x)  # a runtime that calls _forward_<backend> directly (FoldForge's guard)
c = copy.deepcopy(leaf)
out["copy_ok"] = c._forward_cuda.module is c and leaf._forward_cuda.module is leaf
c.set_backend("cuda")
out["plain_again"] = "_forward_cuda" not in c.__dict__ and not c.engine_fp8
out["direct"] = seen[:]
out["after"] = routes()
out["scopes"] = list(fp8.SCOPES)
try:
    TriangleMultiplication(32, backend="engine-fp8")(torch.zeros(1, 8, 8, 32), torch.ones(1, 8, dtype=torch.bool))
except BackendUnsupported:
    out["refused"] = True
out["after_refusal"] = routes()
out["scopes2"] = list(fp8.SCOPES)
print(json.dumps(out))
"""


@pytest.mark.parametrize(
    ("env", "model"), [({}, "af3"), ({}, None), ({"MINIWORLD_FP8_MODEL": "protenix1"}, None), ({"MINIWORLD_FP8": "safe"}, "boltz2")]
)
def test_engine_fp8_scope(env, model):
    """An engine-fp8 module's call runs with the ``safe`` routes (+ the TriAttn part of its model); a module on another backend called inside
    it runs with the process's own switches; everything is restored after the call (also after a refusal); the default path never opens it."""
    code = _SCOPE.replace("MODEL", repr(model))
    proc = subprocess.run([sys.executable, "-c", code], env={**_clean_env(), **env}, capture_output=True, text=True, check=False)
    assert proc.returncode == 0, proc.stderr[-3000:]
    out = json.loads(proc.stdout.strip().splitlines()[-1])
    base = SAFE if env.get("MINIWORLD_FP8") else OFF  # the process preset without MINIWORLD_FP8_MODEL: no TriAttn part
    on = {**SAFE, "triattn": TRIATTN_OF.get(model or env.get("MINIWORLD_FP8_MODEL"), NO_TRIATTN)}
    assert out["before"] == base and out["after"] == base and out["after_refusal"] == base
    assert out["scopes"] == [] and out["scopes2"] == [] and out.get("refused")
    kinds = [k for k, _ in out["seen"]]
    assert kinds == ["cuda", "pytorch", "outer", "cuda", "cuda", "pytorch", "cuda"]
    routes = [r for _, r in out["seen"]]
    # default Leaf / Leaf: the process's switches; outer + a (engine-fp8): on; b on plain cuda: the process's; c (pytorch): untouched = on;
    # b again engine-fp8: on
    assert routes == [base, base, on, on, base, on, on]
    assert (
        out["direct"] == [["cuda", on]] and out["copy_ok"] and out["plain_again"]
    )  # direct _forward_cuda call; copies follow their module
    # training calls (train mode, or grad on): every part with the process's switches, as on cuda
    assert [r for _, r in out["train"]] == [base] * 15 and len(out["train"]) == 15


# ------------------------------------------------------------------------------------------------------------------ B200 (sm_100a) routes
def test_b200_route_table():
    assert fp8.SLOTS == ("triattn", "trimul", "opm", "pwa", "transition", "token_dit", "atom_dit")
    assert set(fp8.B200_ROUTES) <= set(fp8.SLOTS)
    assert fp8.serves_sm100("transition")
    assert fp8.arch() in ("sm90", "sm100", "sm120")


_B200_POLICY = (
    "from miniworld_engine.kernels.transition.cuda import fp8_sm100a as m; print(repr(m.POLICY))"
)


def test_b200_transition_route_follows_the_preset():
    def policy(**env):
        out = subprocess.run([sys.executable, "-c", _B200_POLICY], env={**_clean_env(), **env}, capture_output=True, text=True, check=True)
        return out.stdout.strip().splitlines()[-1]

    assert policy() == "None"  # default: off
    assert policy(MINIWORLD_FP8="safe") == "'calib'"
    assert policy(MINIWORLD_FP8="safe", MINIWORLD_TRANSITION_FP8="") == "None"  # an explicitly set per-route variable wins
    assert policy(MINIWORLD_TRANSITION_FP8="calib") == "'calib'"


# ---- the final B200 route table (n14 integration: TriMul / TriAttn / Transition replace the slot module, the atom DiT rows route is opt-in)
def test_b200_final_route_table():
    assert set(fp8.B200_ROUTES) == {"triattn", "trimul", "transition"}
    assert set(fp8.B200_OPTIN) == {"atom_dit"} and not set(fp8.B200_OPTIN) & set(fp8.B200_ROUTES)
    for slot in ("triattn", "trimul", "transition", "atom_dit"):
        assert fp8.serves_sm100(slot), slot
    # OPM / PWA / token DiT keep their H100-named module in the slot: their B200 forms live behind the same switches (opm_sm100 / pwa_sm100 / token_dit_fp8)
    for slot in ("opm", "pwa", "token_dit"):
        assert not fp8.serves_sm100(slot), slot
    # the atom DiT rows route is NOT in the preset (measured effect <= 0.3 percent, outside the judge): no variable of the preset names it
    assert "MINIWORLD_ATOMDIT_FP8_B200" not in fp8.PRESETS["safe"]


_B200_SWITCHES = (
    "import json; from miniworld_engine.kernels.trimul_inproj.cuda import b200_fp8 as tm; from miniworld_engine.kernels.transition.cuda import fp8_sm100a as tr;"
    "from miniworld_engine.kernels.conditioned_transition.cuda import token_dit_fp8 as td; from miniworld_engine.kernels.integrations import opm_fp8 as opm, pwa_fp8 as pwa;"
    "from miniworld_engine.kernels.augmented_attention.cuda.b200_atom import f8;"
    "print(json.dumps({'trimul': tm.POLICY, 'transition': tr.POLICY, 'widths': sorted(tr.WIDTHS), 'opm': opm.POLICY, 'pwa': pwa.ENABLED, 'td': sorted(td.PLACES),"
    " 'td_b200': sorted(td.B200_PLACES) if hasattr(td, 'B200_PLACES') else None, 'up': td.B200_UP, 'rf3': td.B200_RF3, 'atom_b200': sorted(f8.ENABLED)}))"
)


def _b200_switches(**env) -> dict:
    out = subprocess.run([sys.executable, "-c", _B200_SWITCHES], env={**_clean_env(), **env}, capture_output=True, text=True, check=True)
    return json.loads(out.stdout.strip().splitlines()[-1])


def test_b200_routes_follow_the_preset_with_per_route_precedence():
    off = _b200_switches()
    assert off["trimul"] is None and off["transition"] is None and off["opm"] is None and off["pwa"] is False and off["td"] == [] and off["atom_b200"] == []
    assert off["widths"] == [128, 256, 512, 768]  # agent F: the Transition WIDTHS default
    on = _b200_switches(MINIWORLD_FP8="safe")
    assert (on["trimul"], on["transition"], on["opm"], on["pwa"], on["td"]) == ("calib", "calib", "bound", True, ["expand"])
    assert on["atom_b200"] == []  # not in the preset
    assert on["up"] is True and on["rf3"] is True  # token DiT B200_UP / B200_RF3 ride on the `expand` placement
    # an explicitly set per-route variable wins over the preset (also set to "")
    p = _b200_switches(MINIWORLD_FP8="safe", MINIWORLD_TRIMUL_FP8="", MINIWORLD_OPM_FP8="", MINIWORLD_PWA_FP8="0", MINIWORLD_TRANSITION_FP8="", MINIWORLD_TOKEN_DIT_FP8="")
    assert (p["trimul"], p["transition"], p["opm"], p["pwa"], p["td"]) == (None, None, None, False, [])
    # the atom DiT rows route is switched by its own variable only (with or without the preset)
    assert _b200_switches(MINIWORLD_ATOMDIT_FP8_B200="1")["atom_b200"] == ["post", "pre"]
    assert _b200_switches(MINIWORLD_FP8="safe", MINIWORLD_ATOMDIT_FP8_B200="pre")["atom_b200"] == ["pre"]
    assert _b200_switches(MINIWORLD_FP8="safe", MINIWORLD_ATOMDIT_FP8="1")["atom_b200"] == []  # the H100 variable does not reach the B200 route
    assert _b200_switches(MINIWORLD_FP8="safe", MINIWORLD_TOKEN_DIT_FP8_B200_UP="0")["up"] is False


# ---- RTX 5090 (sm_120a): TriAttn / TriMul / Transition replace the slot module; the Transition follows the preset exactly as on the H100 (2026-11-23)
def test_sm120_route_table():
    assert set(fp8.SM120_ROUTES) == {"triattn", "trimul", "transition"}
    assert set(fp8.SM120_ROUTES) <= set(fp8.SLOTS)
    assert fp8.SM120_ROUTES["transition"] == "miniworld_engine.kernels.transition.cuda.fp8_sm120"
    assert not fp8.SM120_OPTIN  # nothing outside the preset on the 5090 any more
    # the 5090 slots are switched by the H100 preset variables: every H100 preset slot with a 5090 module is on under `safe`
    assert fp8.PRESETS["safe"]["MINIWORLD_TRANSITION_FP8"] == "calib"


_SM120_TR = (
    "import json; from miniworld_engine.kernels.transition.cuda import fp8_sm120 as m;"
    "print(json.dumps({'policy': m.POLICY, 'guard': m.GUARD, 'minrows': m.MIN_ROWS, 'widths': list(m.WIDTHS), 'headroom': m.HEADROOM}))"
)


def _sm120_tr(**env) -> dict:
    out = subprocess.run([sys.executable, "-c", _SM120_TR], env={**_clean_env(), **env}, capture_output=True, text=True, check=True)
    return json.loads(out.stdout.strip().splitlines()[-1])


def test_sm120_transition_route_follows_the_preset():
    off = _sm120_tr()
    assert off == {"policy": None, "guard": "sync", "minrows": None, "widths": [128, 256], "headroom": 16.0}
    assert _sm120_tr(MINIWORLD_FP8="safe")["policy"] == "calib"  # as the H100 route (fp8_sm90a)
    assert _sm120_tr(MINIWORLD_FP8="safe", MINIWORLD_TRANSITION_FP8="")["policy"] is None  # an explicitly set per-route variable wins
    assert _sm120_tr(MINIWORLD_TRANSITION_FP8="calib")["policy"] == "calib"
    # the guard contract variables of the H100 route
    g = _sm120_tr(MINIWORLD_FP8="safe", MINIWORLD_TRANSITION_FP8_GUARD="lazy", MINIWORLD_TRANSITION_FP8_MINROWS="0")
    assert (g["guard"], g["minrows"]) == ("lazy", 0)
    # the former opt-in variables stay as aliases and win over the generic names when set
    assert _sm120_tr(MINIWORLD_TRANSITION_FP8_SM120="calib")["policy"] == "calib"
    assert _sm120_tr(MINIWORLD_FP8="safe", MINIWORLD_TRANSITION_FP8_SM120="")["policy"] is None
    a = _sm120_tr(MINIWORLD_TRANSITION_FP8_GUARD="lazy", MINIWORLD_TRANSITION_FP8_SM120_GUARD="off", MINIWORLD_TRANSITION_FP8_SM120_MINROWS="4096")
    assert (a["guard"], a["minrows"]) == ("off", 4096)
    assert _sm120_tr(MINIWORLD_TRANSITION_FP8_SM120_WIDTHS="128")["widths"] == [128]
