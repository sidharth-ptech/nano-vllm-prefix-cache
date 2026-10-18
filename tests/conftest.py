"""Load CPU components without importing the CUDA-backed public LLM entry point."""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]

# nanovllm/__init__.py eagerly imports the GPU engine. Namespace packages let
# these tests import the unchanged engine modules with their real dependencies.
for name, directory in (
    ("nanovllm", ROOT / "nanovllm"),
    ("nanovllm.engine", ROOT / "nanovllm" / "engine"),
):
    package = ModuleType(name)
    package.__path__ = [str(directory)]
    sys.modules[name] = package


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(autouse=True)
def small_blocks(monkeypatch):
    from nanovllm.engine.sequence import Sequence

    monkeypatch.setattr(Sequence, "block_size", 4)


@pytest.fixture(scope="session")
def upstream_manager_type():
    return load_module(
        "upstream_block_manager", ROOT / "tests/fixtures/block_manager_upstream.py"
    ).BlockManager


@pytest.fixture
def scheduler_type(monkeypatch):
    # Scheduler uses Config only as a type annotation. Replacing this import
    # avoids loading transformers or downloading model configuration; scheduling
    # logic, Sequence, and BlockManager are the real production implementations.
    config_module = ModuleType("nanovllm.config")
    config_module.Config = SimpleNamespace
    with monkeypatch.context() as scoped:
        scoped.setitem(sys.modules, "nanovllm.config", config_module)
        module = load_module("cpu_scheduler", ROOT / "nanovllm/engine/scheduler.py")
    return module.Scheduler
