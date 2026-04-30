from __future__ import annotations

import importlib
from collections.abc import Iterable


def run_test_functions(modules: Iterable[str]) -> list[str]:
    passed: list[str] = []
    for module_name in modules:
        module = importlib.import_module(module_name)
        for name in dir(module):
            if not name.startswith("test_"):
                continue
            candidate = getattr(module, name)
            if callable(candidate):
                candidate()
                passed.append(f"{module_name}.{name}")
    return passed
