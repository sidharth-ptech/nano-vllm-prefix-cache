# CPU regression tests

Run from the repository root with `python -m pytest tests` after installing
`pytest`, `numpy`, and `xxhash`. No model download or GPU is required.

The public `nanovllm` initializer eagerly imports the CUDA engine. The test
configuration exposes namespace packages to import the unchanged CPU modules
directly. Scheduler tests replace only its annotation-only `Config` import with
`SimpleNamespace` during module loading. They execute the real scheduler, sequence,
block manager, NumPy, and xxhash code. These tests do not validate CUDA execution
or model outputs.

`fixtures/block_manager_upstream.py` is an unmodified snapshot of
`nanovllm/engine/block_manager.py` from
[GeeeekExplorer/nano-vllm commit bb823b3](https://github.com/GeeeekExplorer/nano-vllm/blob/bb823b3e06983d71485a8e1f23715ebd87d98ef8/nanovllm/engine/block_manager.py).
It provides an independent, offline differential oracle for the optimization.
Its SHA-256 is
`d0240374f589eaa85c4c7923b8970f2635ab7bc73adc5044fa2d273955b54242`.
The snapshot is copyright (c) 2025 Xingkai Yu and is distributed under the
repository's [MIT license](../LICENSE). Do not reformat or update this fixture
without deliberately changing the pinned baseline and its integrity check.

Focused cases cover FIFO allocation and recycling, cache reclamation from the
head, middle, and tail of the free queue, repeated warm-cache cycles,
reference-counted sharing, cache eviction, hash-collision guards, capacity
pressure, and partial/full block boundaries. Small four-token blocks make
boundary cases readable; an additional lifecycle test uses the production
256-token block size. Real scheduler cases cover chunked prefill, request
completion, prefix reuse, and preemption/resumption under decode pressure.
Five fixed random seeds each run 800 lifecycle steps against both implementations,
checking identical decisions and state plus independent ownership invariants.

Benchmark tests verify percentile and throughput calculations against known
samples, deduplication of aliased memory, all three workload cache states with
both implementations, and complete block release after tiny benchmark runs.
They check measurement mechanics without asserting a performance threshold.
