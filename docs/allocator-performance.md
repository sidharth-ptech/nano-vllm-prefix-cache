# Removing the free-list scan from prefix-cache reuse

This fork changes Nano-vLLM's CPU block allocator. When a completed request's
prefix is reused, the allocator now removes its cached blocks from an ordered
mapping rather than searching a deque. Eviction order, prefix hashes, reference
counts, and GPU execution remain the same.

The measured benefit depends on the cache workload. Reclaiming an eight-block
prefix was **1.38× faster with 1,024 blocks** and **12.56× faster with 32,768
blocks** in the CPU allocator microbenchmark. Cold allocation became slightly
slower and the mapping uses more host memory.

These results measure allocator cycles, not model token throughput, time to
first token (TTFT), inter-token latency, or HTTP serving performance.

## Why this project and this change

[Nano-vLLM](https://github.com/GeeeekExplorer/nano-vllm) provides a small,
readable inference engine with prefix caching and chunked prefill. Its allocator
is directly involved in admitting work to the scheduler, making it a useful
place to investigate host-side latency. It is an offline inference engine;
this fork does not add an HTTP server.

The starting point is upstream commit
[`bb823b3e06983d71485a8e1f23715ebd87d98ef8`](https://github.com/GeeeekExplorer/nano-vllm/commit/bb823b3e06983d71485a8e1f23715ebd87d98ef8),
dated April 26, 2026. The MIT license and upstream authorship are retained. The
new contribution is the allocator change, regression tests, benchmark, and
this report.

In the original implementation, a free cached block is reclaimed with
`deque.remove(block_id)`. Finding that block takes time proportional to its
position in the free queue. Recently freed blocks sit near the tail, so a
request reusing several such blocks can repeatedly scan a large pool.

`OrderedDict` supports removal by block ID in expected constant time while
retaining insertion order. `popitem(last=False)` preserves FIFO eviction, and
the existing reverse deallocation order remains intact. Ordinary `dict` was
not chosen because repeatedly finding its oldest surviving key can scan
deleted entries. No new scheduling policy or custom linked-list code is
needed.

## Recorded experiment

The run used macOS 14.4 on arm64, CPython 3.12.11, NumPy 2.5.4, and xxHash 4.0.1.
The host exposed 11 logical CPUs. No model or GPU was loaded. Exact machine
details and source hashes are recorded in the
[raw JSON](../benchmarks/results/macos-arm64-python312.json).

- Pool sizes: 1,024, 8,192, and 32,768 blocks.
- Prefix lengths: 1, 8, and 32 blocks; 256 tokens per block.
- Three workloads: cold miss, reclaimed free prefix, and an active shared prefix.
- Five trials of 200 measured cycles per implementation and workload, with 25
  warmup cycles before each trial: 54,000 measured cycles in total.
- The implementation order alternates between trials. Cyclic garbage
  collection is disabled for both implementations inside the timed trial.
- An untimed FIFO prelude rotates the pool twice before prefix setup, exposing
  the ordered mapping's capacity growth before timing and memory accounting.

A cycle is `can_allocate → allocate → deallocate` for a preconstructed sequence.
It includes prefix lookup, hashing during lookup/allocation, reference updates,
and queue maintenance. Setup hashing, sequence construction, garbage
collection, and storing the timing sample are outside that boundary.

Each prefix-hit request contains the shared prefix plus one uncached token.
The reclaimed workload repeatedly releases and reacquires that prefix near the
tail of the free queue. The shared workload keeps a seed request alive, so its
prefix blocks already have owners and need no free-list removal. The cold
workload never populates the prefix cache.

The percentiles below pool all 1,000 individual cycle samples. They describe
this run's latency distribution; the trials are not independent machines or
a confidence interval. CPU affinity, host background activity, and power
state were not controlled. See the [benchmark protocol](../benchmarks/README.md)
for full details and definitions.

## Prefix reuse results

Eight cached blocks represent a 2,048-token prefix, followed by one uncached
token. All latencies below are microseconds per complete allocator cycle.

| Pool blocks | Upstream p50 | Updated p50 | p50 speedup | Upstream p95 | Updated p95 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1,024 | 159.08 | 114.90 | 1.38× | 170.57 | 127.93 |
| 8,192 | 464.08 | 120.17 | 3.86× | 485.80 | 130.35 |
| 32,768 | 1,478.50 | 117.71 | 12.56× | 1,529.28 | 129.80 |

At 32,768 blocks, p99 fell from 1,574.83 to 146.83 microseconds. Measured
allocator-cycle throughput rose from 675.75 to 8,418.34 cycles/second. This
throughput divides cycle count by the sum of cycle durations; the raw file also
reports throughput including loop and sample-recording overhead.

The 32,768-block case is a scaling stress test: at 256 tokens per block, it
represents capacity for 8,388,608 cached tokens. Its speedup should not be
presented as a typical single-GPU result. The smaller pools show how the
benefit changes with available KV capacity.

## Controls and memory cost

Controls below use the same 32,768-block pool and eight-block prefix length.

| Workload | Upstream p50 | Updated p50 | Upstream p99 | Updated p99 |
| --- | ---: | ---: | ---: | ---: |
| Cold miss | 12.79 µs | 13.71 µs | 16.67 µs | 31.71 µs |
| Active shared prefix | 115.54 µs | 114.75 µs | 141.46 µs | 139.39 µs |

Cold-miss median latency increased by 7.2% (about 0.92 microseconds), and its
p99 was worse in this run. The shared-prefix control was approximately
unchanged. The change therefore targets reuse of **free** cached blocks; it
is not a universal allocator speedup.

For the reclaimed-prefix case, reachable Python memory for the free-pool
container increased from **1.134 MiB to 5.375 MiB**. For the whole block
manager it increased from **8.832 MiB to 13.073 MiB**, an extra **4.242 MiB**
(48.0%). These are recursive, identity-deduplicated Python object sizes,
including integer IDs and instance dictionaries. They are not process RSS,
GPU memory, or exclusively owned bytes. The JSON includes pre- and post-trial
sizes; free-pool size was stable across these timed trials.

## Correctness and reproduction

The CPU tests cover explicit FIFO and ownership invariants, shared references,
cache eviction and collisions, block boundaries, chunked prefill, and
preemption/resumption through the real scheduler. Five fixed random seeds
exercise 4,000 lifecycle operations against a byte-exact upstream allocator
snapshot. A lifecycle test also uses the production 256-token block size.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-cpu-lock.txt
python -m pytest -q
ruff check tests benchmarks --exclude tests/fixtures
python benchmarks/allocator.py \
  --iterations 200 --repeats 5 --warmup 25 \
  --output benchmarks/results/reproduction.json
```

The measured candidate was an uncommitted change on top of the pinned upstream
revision. Its SHA-256 is
`98f8714be03f273a88554e4d7bea912460674ce7d88efb5c69c949a8a1ce0c51`;
the raw report records hashes for both allocators, sequence code, sampling
parameters, and the benchmark itself. This makes the measured code identifiable
independently of the later publication commit.

## Remaining inference validation

GPU generation has not been tested for this contribution. Before claiming a
serving gain, run the pinned upstream and this fork on the same NVIDIA GPU,
model revision, software environment, and request trace. Warm model kernels
and prefix caches consistently, alternate run order, and keep GPU memory
utilization, batch limits, and generated lengths fixed.

Include cold prompts, repeated completed prefixes, concurrent active prefixes,
and mixed traffic at several arrival rates. Record actual KV-pool size, cache
hit rate, scheduler CPU time, output tokens/second, TTFT, inter-token latency,
completion-latency percentiles, and host/GPU memory. Compare token sequences
with controlled sampling and exercise cache pressure. A model-bound workload
may show little end-to-end benefit even when allocator time drops substantially.

This experiment supports a narrow conclusion: constant-time removal eliminates
the growing free-list scan in the tested prefix-reuse workloads, at a measured
host-memory and cold-allocation cost.
