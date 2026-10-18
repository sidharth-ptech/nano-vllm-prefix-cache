#!/usr/bin/env python3
"""Compare CPU allocator cycles against a pinned upstream implementation.

No model, CUDA, torch, or inference server is loaded. See benchmarks/README.md
for the measured boundary, workloads, memory accounting, and interpretation.
"""

import argparse
import gc
import hashlib
import importlib
import json
import os
import platform
import subprocess
import sys
from collections import deque
from importlib.machinery import ModuleSpec
from importlib.metadata import version
from pathlib import Path
from time import perf_counter_ns
from types import ModuleType

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SOURCE_PATH = "nanovllm/engine/block_manager.py"
UPSTREAM_REF = "bb823b3e06983d71485a8e1f23715ebd87d98ef8"
UPSTREAM_SHA256 = "d0240374f589eaa85c4c7923b8970f2635ab7bc73adc5044fa2d273955b54242"
SCENARIOS = ("cold_miss", "reclaimed_prefix", "shared_prefix")


def git(*args):
    return subprocess.check_output(
        ["git", "-C", str(ROOT), *args], stderr=subprocess.DEVNULL
    ).decode().strip()


def load_allocators(baseline_ref):
    # The public package initializer eagerly imports the GPU engine. Namespace
    # packages let this standalone process import the real CPU modules directly.
    for name, path in (("nanovllm", ROOT / "nanovllm"),
                       ("nanovllm.engine", ROOT / "nanovllm/engine")):
        package = ModuleType(name)
        package.__path__ = [str(path)]
        package.__package__ = name
        package.__spec__ = ModuleSpec(name, loader=None, is_package=True)
        sys.modules[name] = package
    sequence = importlib.import_module("nanovllm.engine.sequence").Sequence
    try:
        commit = git("rev-parse", "--verify", f"{baseline_ref}^{{commit}}")
        baseline = subprocess.check_output(
            ["git", "-C", str(ROOT), "show", f"{commit}:{SOURCE_PATH}"],
            stderr=subprocess.DEVNULL,
        )
        origin = f"git:{commit}:{SOURCE_PATH}"
    except (subprocess.CalledProcessError, FileNotFoundError):
        # Source archives and shallow forks can use the checked-in, byte-exact
        # upstream fixture. Never silently substitute for a different revision.
        if baseline_ref != UPSTREAM_REF:
            raise RuntimeError(f"Cannot read baseline revision {baseline_ref!r}") from None
        fixture = ROOT / "tests/fixtures/block_manager_upstream.py"
        baseline = fixture.read_bytes()
        if hashlib.sha256(baseline).hexdigest() != UPSTREAM_SHA256:
            raise RuntimeError("Upstream fixture checksum does not match")
        commit, origin = UPSTREAM_REF, str(fixture.relative_to(ROOT))
    candidate = (ROOT / SOURCE_PATH).read_bytes()
    implementations, provenance = {}, {}
    for label, source in (("baseline", baseline), ("candidate", candidate)):
        module = ModuleType(f"allocator_benchmark_{label}")
        module.__file__ = f"{origin if label == 'baseline' else SOURCE_PATH}"
        exec(compile(source, module.__file__, "exec"), module.__dict__)
        implementations[label] = module.BlockManager
        provenance[label] = {
            "origin": origin if label == "baseline" else SOURCE_PATH,
            "sha256": hashlib.sha256(source).hexdigest(),
        }
    provenance["baseline"]["commit"] = commit
    return sequence, implementations, provenance


def deep_size(root):
    """Reachable Python bytes, counting each object identity once.

    Includes container storage, integer keys, block objects, token lists, and
    instance dictionaries. Shared references inside this graph are deduplicated.
    This is neither process RSS nor exclusively owned memory.
    """
    seen, pending, total = set(), [root], 0
    while pending:
        obj = pending.pop()
        if id(obj) in seen:
            continue
        seen.add(id(obj))
        total += sys.getsizeof(obj)
        if isinstance(obj, dict):
            pending.extend(obj.keys())
            pending.extend(obj.values())
        elif isinstance(obj, (list, tuple, set, frozenset, deque)):
            pending.extend(obj)
        elif hasattr(obj, "__dict__"):
            pending.append(vars(obj))
    return total


def prepare(manager_type, sequence_type, pool_size, prefix_blocks, scenario):
    manager = manager_type(pool_size, sequence_type.block_size)
    # Warm container capacity with cheap, one-block FIFO cycles before introducing
    # any cached prefix. This visits every free ID twice without quadratic scans.
    churn = sequence_type([0])
    for _ in range(2 * pool_size):
        manager.allocate(churn, manager.can_allocate(churn))
        manager.deallocate(churn)
    tokens = list(range(prefix_blocks * sequence_type.block_size))
    seed = None
    if scenario != "cold_miss":
        seed = sequence_type(tokens)
        manager.allocate(seed, manager.can_allocate(seed))
        seed.num_scheduled_tokens = len(tokens)
        manager.hash_blocks(seed)
        if scenario == "reclaimed_prefix":
            manager.deallocate(seed)
    query = sequence_type(tokens + [len(tokens)])
    expected = 0 if scenario == "cold_miss" else prefix_blocks
    if manager.can_allocate(query) != expected:
        raise AssertionError("Workload setup did not create the expected cache state")
    return manager, query, seed, expected


def trial(manager, query, expected, iterations):
    samples = []
    wall_start = perf_counter_ns()
    for _ in range(iterations):
        start = perf_counter_ns()
        cached = manager.can_allocate(query)
        manager.allocate(query, cached)
        manager.deallocate(query)
        elapsed = perf_counter_ns() - start
        if cached != expected:
            raise AssertionError(f"Expected {expected} cached blocks, got {cached}")
        samples.append(elapsed)
    wall_ns = perf_counter_ns() - wall_start
    return {
        "cycle_latency_ns": samples,
        "measured_cycle_ns": sum(samples),
        "loop_wall_ns": wall_ns,
    }


def summarize(trials):
    samples = [sample for trial_result in trials
               for sample in trial_result["cycle_latency_ns"]]
    percentiles = np.percentile(samples, [50, 95, 99], method="linear")
    total_ns = sum(samples)
    return {
        "cycles": len(samples),
        "cycle_latency_us": dict(zip(("p50", "p95", "p99"),
                                     (float(value / 1000) for value in percentiles))),
        "full_cycles_per_second": len(samples) * 1e9 / total_ns,
        "loop_cycles_per_second": len(samples) * 1e9 /
                                  sum(result["loop_wall_ns"] for result in trials),
    }


def benchmark_case(args, sequence_type, implementations, pool, prefix, scenario):
    result = {"pool_blocks": pool, "prefix_blocks": prefix,
              "scenario": scenario, "implementations": {}}
    managers = {}
    for label, manager_type in implementations.items():
        manager, query, seed, expected = prepare(
            manager_type, sequence_type, pool, prefix, scenario
        )
        managers[label] = manager, query, seed, expected
        result["implementations"][label] = {
            "memory_bytes": {"free_pool_shallow": sys.getsizeof(manager.free_block_ids),
                             "free_pool_reachable": deep_size(manager.free_block_ids),
                             "manager_reachable": deep_size(manager)},
            "trials": [],
        }
    for repeat in range(args.repeats):
        order = ("baseline", "candidate") if repeat % 2 == 0 else ("candidate", "baseline")
        for position, label in enumerate(order):
            manager, query, _, expected = managers[label]
            trial(manager, query, expected, args.warmup)
            # Both versions use the same GC policy; collect outside the measured
            # boundary, then keep cyclic GC out of short latency samples.
            gc.collect()
            gc_was_enabled = gc.isenabled()
            gc.disable()
            try:
                observation = trial(manager, query, expected, args.iterations)
            finally:
                if gc_was_enabled:
                    gc.enable()
            observation.update(repeat=repeat, order_position=position)
            result["implementations"][label]["trials"].append(observation)
    for label, (manager, query, seed, _) in managers.items():
        entry = result["implementations"][label]
        entry["memory_bytes"].update(
            free_pool_reachable_after_trials=deep_size(manager.free_block_ids),
            manager_reachable_after_trials=deep_size(manager),
        )
        if query.block_table or query.num_cached_tokens:
            raise AssertionError("Query retained blocks after deallocation")
        if seed is not None and seed.block_table:
            manager.deallocate(seed)
        if len(manager.free_block_ids) != pool or manager.used_block_ids:
            raise AssertionError("Allocator leaked blocks during benchmark")
        entry["summary"] = summarize(entry["trials"])
    baseline = result["implementations"]["baseline"]["summary"]
    candidate = result["implementations"]["candidate"]["summary"]
    result["candidate_speedup"] = {
        "p50": baseline["cycle_latency_us"]["p50"] / candidate["cycle_latency_us"]["p50"],
        "throughput": candidate["full_cycles_per_second"] / baseline["full_cycles_per_second"],
    }
    return result


def positive_int(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return number


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-ref", default=UPSTREAM_REF)
    parser.add_argument("--pools", nargs="+", type=positive_int, default=[1024, 8192, 32768])
    parser.add_argument("--prefix-blocks", nargs="+", type=positive_int, default=[1, 8, 32])
    parser.add_argument("--scenarios", nargs="+", choices=SCENARIOS, default=list(SCENARIOS))
    parser.add_argument("--iterations", type=positive_int, default=200)
    parser.add_argument("--repeats", type=positive_int, default=3)
    parser.add_argument("--warmup", type=positive_int, default=25)
    parser.add_argument("--output", type=Path, default=Path("benchmarks/results/allocator.json"))
    args = parser.parse_args()
    if min(args.pools) <= max(args.prefix_blocks):
        parser.error("every pool must fit the cached prefix and one additional block")
    sequence_type, implementations, provenance = load_allocators(args.baseline_ref)
    try:
        checkout = git("rev-parse", "HEAD")
        source_status = git("status", "--porcelain", "--", SOURCE_PATH,
                            "nanovllm/engine/sequence.py", "nanovllm/sampling_params.py")
    except (subprocess.CalledProcessError, FileNotFoundError):
        checkout, source_status = None, None
    for source_path in ("nanovllm/engine/sequence.py", "nanovllm/sampling_params.py",
                        "benchmarks/allocator.py"):
        provenance[source_path] = {"sha256": hashlib.sha256(
            (ROOT / source_path).read_bytes()).hexdigest()}
    report = {
        "schema_version": 1,
        "environment": {"python": sys.version, "platform": platform.platform(),
                        "machine": platform.machine(), "processor": platform.processor(),
                        "logical_cpu_count": os.cpu_count(),
                        "numpy": version("numpy"), "xxhash": version("xxhash"),
                        "checkout_commit": checkout, "source_status": source_status,
                        "cyclic_gc_during_measurement": False},
        "provenance": provenance,
        "parameters": {**vars(args), "output": str(args.output),
                       "block_size": sequence_type.block_size},
        "measurement": {
            "preconditioning": "2 * pool_blocks untimed one-block cold FIFO cycles per "
                               "implementation, before prefix setup; then warmup before each trial",
            "latency_boundary": "can_allocate + allocate + deallocate; sequence construction, "
                                "prefix hashing during setup, GC, and sample storage excluded",
            "percentiles": "numpy linear percentiles across all measured individual cycles",
            "full_cycles_per_second": "total measured cycles / sum of individual cycle durations",
            "loop_cycles_per_second": "total cycles / loop wall time, including sample recording",
            "memory": "sys.getsizeof recursively over reachable Python objects; identity "
                      "deduplicated; includes integer keys and instance dictionaries; before "
                      "timed trials and after timed trials, before releasing pinned prefixes; not RSS",
            "scope": "CPU allocator microbenchmark; does not measure tokens/s, TTFT, "
                     "GPU latency, or end-to-end serving throughput",
        },
        "cases": [],
    }
    print("scenario          pool prefix   baseline p50 us candidate p50 us   speedup", flush=True)
    for pool in args.pools:
        for prefix in args.prefix_blocks:
            for scenario in args.scenarios:
                case = benchmark_case(args, sequence_type, implementations, pool, prefix, scenario)
                report["cases"].append(case)
                baseline = case["implementations"]["baseline"]["summary"]["cycle_latency_us"]["p50"]
                candidate = case["implementations"]["candidate"]["summary"]["cycle_latency_us"]["p50"]
                print(f"{scenario:<17} {pool:>6} {prefix:>6} {baseline:>17.2f} "
                      f"{candidate:>16.2f} {case['candidate_speedup']['p50']:>9.2f}x", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Raw trials and metadata: {args.output}")


if __name__ == "__main__":
    main()
