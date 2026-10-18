# Working on this fork

This fork studies the CPU cost of prefix-cache reuse in
[Nano-vLLM](https://github.com/GeeeekExplorer/nano-vllm). The upstream author,
license, GPU execution code, and eviction policy are retained. Keep changes
focused and make performance claims reproducible.

## CPU development

Use Python 3.10–3.12. A CUDA installation and model weights are unnecessary for
the allocator tests and microbenchmark.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-cpu.txt
python -m pytest -q
ruff check tests benchmarks --exclude tests/fixtures
python benchmarks/allocator.py --output benchmarks/results/local.json
```

The published macOS run uses the exact Python 3.12 dependencies in
`requirements-cpu-lock.txt`. The broader requirements file lets pip select
compatible versions for the other supported interpreters.

Tests load the real CPU allocator, sequence implementation, NumPy, and xxHash.
Their package bootstrap skips the public initializer because it eagerly loads
the CUDA engine. This does not establish that GPU generation works on a CPU.

## Reviewing a performance change

- Preserve free-block order, reference counts, and cache invalidation behavior.
- Compare against the pinned upstream fixture as well as explicit invariants.
- Include cache misses and active shared prefixes as controls.
- Report absolute latency, sample counts, runtime versions, and host-memory
  costs alongside speedup ratios.
- Keep allocator cycles per second separate from generated tokens per second.

The GitHub workflow runs the CPU suite on Python 3.10, 3.11, and 3.12. It has no
GPU job. See the performance report for the remaining inference validation.
