<p align="center">
<img width="300" src="assets/logo.png">
</p>

<p align="center">
<a href="https://trendshift.io/repositories/15323" target="_blank"><img src="https://trendshift.io/api/badge/repositories/15323" alt="GeeeekExplorer%2Fnano-vllm | Trendshift" style="width: 250px; height: 55px;" width="250" height="55"/></a>
</p>

# Nano-vLLM: faster prefix-cache block reuse

A focused performance fork of [GeeeekExplorer/nano-vllm](https://github.com/GeeeekExplorer/nano-vllm),
the lightweight inference engine by Xingkai Yu. Upstream code and the MIT license
are retained; this fork contributes a small allocator optimization, regression
tests, and reproducible CPU measurements.

When a request reuses a completed prefix, reclaiming its cached blocks used to
scan the free-block queue. An ordered mapping now removes those blocks in
expected constant time while preserving FIFO eviction and reference counts.
No additional production dependency is introduced.

**Measured CPU allocator results** for an eight-block cached prefix:

| Pool size | Upstream median cycle | Updated median cycle | Speedup |
| ---: | ---: | ---: | ---: |
| 1,024 blocks | 159.08 µs | 114.90 µs | 1.38× |
| 8,192 blocks | 464.08 µs | 120.17 µs | 3.86× |
| 32,768 blocks | 1,478.50 µs | 117.71 µs | 12.56× |

The largest pool is a scaling stress test. These numbers measure the complete
CPU `can_allocate → allocate → deallocate` cycle, **not GPU tokens/second or
end-to-end serving latency**. At 32,768 blocks, the corresponding cold-miss
control was 7.2% slower and manager metadata used 4.24 MiB more host memory.
GPU inference validation remains outstanding.

Read the [performance report](docs/allocator-performance.md),
[benchmark methodology](benchmarks/README.md), and
[raw measurements](benchmarks/results/macos-arm64-python312.json).
The [test suite](tests/README.md) covers ownership invariants, FIFO behavior,
cache sharing, scheduler preemption, 4,000 differential lifecycle operations,
and benchmark calculations.

## Run the CPU checks

Python 3.10–3.12 is supported. No model download or CUDA installation is needed.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-cpu.txt
python -m pytest -q
python benchmarks/allocator.py --output benchmarks/results/local.json
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for the exact recorded environment and lint checks.

## Upstream engine features

* 🚀 **Fast offline inference** - Comparable inference speeds to vLLM
* 📖 **Readable codebase** - Clean implementation in ~ 1,200 lines of Python code
* ⚡ **Optimization Suite** - Prefix caching, Tensor Parallelism, Torch compilation, CUDA graph, etc.

## Installation

```bash
pip install git+https://github.com/sidharth-ptech/nano-vllm-prefix-cache.git
```

The full engine uses NVIDIA CUDA, PyTorch, Triton, and FlashAttention. The CPU
checks above test allocator/scheduler logic; they do not run model inference.

## Model Download

To download the model weights manually, use the following command:
```bash
huggingface-cli download --resume-download Qwen/Qwen3-0.6B \
  --local-dir ~/huggingface/Qwen3-0.6B/ \
  --local-dir-use-symlinks False
```

## Quick Start

See `example.py` for usage. The API mirrors vLLM's interface with minor differences in the `LLM.generate` method:
```python
from nanovllm import LLM, SamplingParams
llm = LLM("/YOUR/MODEL/PATH", enforce_eager=True, tensor_parallel_size=1)
sampling_params = SamplingParams(temperature=0.6, max_tokens=256)
prompts = ["Hello, Nano-vLLM."]
outputs = llm.generate(prompts, sampling_params)
outputs[0]["text"]
```

## Original upstream GPU benchmark

The following results are reproduced from the upstream README. They predate
this fork and are not evidence of its allocator improvement. See `bench.py`
for the original offline model benchmark.

**Test Configuration:**
- Hardware: RTX 4070 Laptop (8GB)
- Model: Qwen3-0.6B
- Total Requests: 256 sequences
- Input Length: Randomly sampled between 100–1024 tokens
- Output Length: Randomly sampled between 100–1024 tokens

**Performance Results:**
| Inference Engine | Output Tokens | Time (s) | Throughput (tokens/s) |
|----------------|-------------|----------|-----------------------|
| vLLM           | 133,966     | 98.37    | 1361.84               |
| Nano-vLLM      | 133,966     | 93.41    | 1434.13               |


## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=GeeeekExplorer/nano-vllm&type=Date)](https://www.star-history.com/#GeeeekExplorer/nano-vllm&Date)
