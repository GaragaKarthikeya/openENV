from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _bootstrap_repo_package() -> None:
    if "linux_sre_gym" in sys.modules:
        return

    repo_root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "linux_sre_gym",
        repo_root / "__init__.py",
        submodule_search_locations=[str(repo_root)],
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("Unable to bootstrap linux_sre_gym package from repository root.")

    module = importlib.util.module_from_spec(spec)
    sys.modules["linux_sre_gym"] = module
    spec.loader.exec_module(module)


_bootstrap_repo_package()
