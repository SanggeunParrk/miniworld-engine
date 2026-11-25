"""Pin `foldjet_engine` (namespace package) to the wt-fp8t3-tokendit tree, before anything imports it."""

import importlib.machinery
import os
import sys
import types
from pathlib import Path

TREE = Path(os.environ.get("PIN_TREE", "."))
pkg = TREE / "src/foldjet_engine"
spec = importlib.machinery.ModuleSpec("foldjet_engine", None, is_package=True)
spec.submodule_search_locations = [str(pkg)]
mod = types.ModuleType("foldjet_engine")
mod.__path__ = [str(pkg)]
mod.__spec__ = spec
mod.__package__ = "foldjet_engine"
sys.modules["foldjet_engine"] = mod
sys.path.insert(0, str(TREE / "src"))
