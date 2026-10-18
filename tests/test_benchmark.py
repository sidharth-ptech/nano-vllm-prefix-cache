import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from nanovllm.engine.block_manager import BlockManager
from nanovllm.engine.sequence import Sequence


@pytest.fixture(scope="module")
def benchmark():
    path = Path(__file__).resolve().parents[1] / "benchmarks/allocator.py"
    spec = importlib.util.spec_from_file_location("allocator_benchmark_tests", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_summary_uses_individual_cycles_and_separate_wall_time(benchmark):
    summary = benchmark.summarize(
        [
            {"cycle_latency_ns": [1000, 2000], "loop_wall_ns": 8000},
            {"cycle_latency_ns": [3000, 4000], "loop_wall_ns": 16000},
        ]
    )
    assert summary["cycles"] == 4
    assert summary["cycle_latency_us"] == pytest.approx(
        {
            "p50": 2.5,
            "p95": 3.85,
            "p99": 3.97,
        }
    )
    assert summary["full_cycles_per_second"] == pytest.approx(400_000)
    assert summary["loop_cycles_per_second"] == pytest.approx(4e9 / 24_000)


def test_reachable_memory_counts_aliases_once_and_handles_cycles(benchmark):
    payload = bytearray(32)
    aliased = [payload, payload]
    distinct = [payload, bytearray(payload)]
    assert benchmark.deep_size(aliased) == sys.getsizeof(aliased) + sys.getsizeof(
        payload
    )
    assert benchmark.deep_size(distinct) == sys.getsizeof(distinct) + 2 * sys.getsizeof(
        payload
    )
    cyclic = []
    cyclic.append(cyclic)
    assert benchmark.deep_size(cyclic) == sys.getsizeof(cyclic)


@pytest.mark.parametrize("scenario", ["cold_miss", "reclaimed_prefix", "shared_prefix"])
@pytest.mark.parametrize("implementation", ["candidate", "baseline"])
def test_workload_cycles_preserve_cache_and_block_ownership(
    benchmark, upstream_manager_type, scenario, implementation
):
    manager_type = (
        BlockManager if implementation == "candidate" else upstream_manager_type
    )
    manager, query, seed, expected = benchmark.prepare(
        manager_type, Sequence, 8, 2, scenario
    )
    assert expected == (0 if scenario == "cold_miss" else 2)
    assert manager.can_allocate(query) == expected
    assert len(manager.hash_to_block_id) == expected
    for _ in range(3):
        observation = benchmark.trial(manager, query, expected, 4)
        assert len(observation["cycle_latency_ns"]) == 4
        assert observation["measured_cycle_ns"] == sum(observation["cycle_latency_ns"])
        assert observation["loop_wall_ns"] >= observation["measured_cycle_ns"] > 0
        assert query.block_table == []
        assert query.num_cached_tokens == 0
        assert manager.can_allocate(query) == expected
        pinned = set(seed.block_table) if seed is not None else set()
        assert manager.used_block_ids == pinned
        assert set(manager.free_block_ids) == set(range(8)) - pinned
        assert len(manager.free_block_ids) == 8 - len(pinned)
        assert all(
            block.ref_count == int(block.block_id in pinned) for block in manager.blocks
        )
    if seed is not None and seed.block_table:
        manager.deallocate(seed)
    assert len(manager.free_block_ids) == 8
    assert not manager.used_block_ids


@pytest.mark.parametrize("scenario", ["cold_miss", "reclaimed_prefix", "shared_prefix"])
def test_tiny_case_releases_every_block_and_records_alternating_order(
    benchmark, upstream_manager_type, scenario
):
    managers = []

    def track(manager_type):
        def create(*args):
            manager = manager_type(*args)
            managers.append(manager)
            return manager

        return create

    result = benchmark.benchmark_case(
        SimpleNamespace(repeats=2, warmup=1, iterations=3),
        Sequence,
        {"baseline": track(upstream_manager_type), "candidate": track(BlockManager)},
        8,
        2,
        scenario,
    )
    for manager in managers:
        assert set(manager.free_block_ids) == set(range(8))
        assert len(manager.free_block_ids) == 8
        assert not manager.used_block_ids
        assert all(block.ref_count == 0 for block in manager.blocks)
    for label, entry in result["implementations"].items():
        assert entry["summary"]["cycles"] == 6
        assert [trial["order_position"] for trial in entry["trials"]] == (
            [0, 1] if label == "baseline" else [1, 0]
        )
        assert all(value > 0 for value in entry["memory_bytes"].values())
