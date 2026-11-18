"""Pin `miniworld_engine` (namespace package) to the wt-fp8t3-tokendit tree, before anything imports it."""
import importlib.machinery, sys, types
from pathlib import Path

import os
TREE = Path(os.environ.get("PIN_TREE", "."))
pkg = TREE / "src/miniworld_engine"
spec = importlib.machinery.ModuleSpec("miniworld_engine", None, is_package=True)
spec.submodule_search_locations = [str(pkg)]
mod = types.ModuleType("miniworld_engine")
mod.__path__ = [str(pkg)]
mod.__spec__ = spec
mod.__package__ = "miniworld_engine"
sys.modules["miniworld_engine"] = mod
sys.path.insert(0, str(TREE / "src"))
