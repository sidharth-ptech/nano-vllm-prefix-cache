# CPU block allocator benchmark

Run from the repository root, using Python 3.10–3.12 with `numpy` and `xxhash`
installed:

```sh
python benchmarks/allocator.py
```

The default matrix covers 1,024, 8,192, and 32,768 KV blocks; prefixes of 1, 8,
and 32 blocks; and three cache states. Blocks contain 256 tokens, matching the
actual `Sequence` implementation. Each case has three repetitions of 200 measured
cycles with 25 warmup cycles before each repetition. The execution order alternates
between baseline first and candidate first. Run on an otherwise idle machine;
record additional independent runs when assessing small differences.

For a quick smoke check:

```sh
python benchmarks/allocator.py --pools 1024 --prefix-blocks 1 8 \
  --iterations 10 --repeats 2 --warmup 2 --output /tmp/allocator-smoke.json
```

For longer runs or additional pool sizes, override `--iterations`, `--repeats`,
`--warmup`, `--pools`, or `--prefix-blocks`. For example, `--pools 65536` tests a
larger pool. `--scenarios` accepts any subset of the scenario names below.

## Workloads and measurement boundary

Every measured cycle runs the real `BlockManager.can_allocate`, `allocate`, and
`deallocate` methods on a real `Sequence`. The request contains the configured
number of complete prefix blocks followed by one token in an additional block.
Only the following cache state changes between scenarios:

| Scenario | Initial cache state | Purpose |
| --- | --- | --- |
| `cold_miss` | No blocks have been hashed | FIFO allocation and deallocation control; all request blocks are fresh |
| `reclaimed_prefix` | A seed request is hashed and deallocated | Cached blocks sit near the end of the free pool; reclaiming them exercises arbitrary free-ID removal |
| `shared_prefix` | A hashed seed request remains allocated | Cached blocks are shared with a live request; exercises reference counts without removing their IDs from the free pool |

Prefix construction and `hash_blocks` run during setup, outside the measurement.
The measured request is reused after deallocation clears its block table. Cold
requests are deliberately not hashed between cycles so that they remain cold.
Reclaimed prefixes return to the free-pool tail after each cycle. Shared prefixes
stay pinned until all repetitions finish. The benchmark checks the expected hit
count and verifies that all blocks are free after cleanup.

Before creating each scenario's prefix, both implementations run twice the pool
size in untimed one-block cold FIFO cycles. This cheap linear pass rotates every
free ID twice and exercises container growth from deletion and reinsertion. It
avoids comparing only freshly allocated container capacity. Cyclic garbage
collection runs before measured trials and is disabled during them; normal
reference counting still runs. Timings include two monotonic clock calls per
cycle and exclude sample recording and the subsequent hit-count assertion.

The JSON report includes every measured cycle in nanoseconds, each repetition's
execution order and loop duration, and these summaries:

- **p50/p95/p99 cycle latency:** NumPy linear percentiles of all individual cycle
  samples across the repetitions, reported in microseconds. These are not
  percentiles of repetition averages.
- **Full cycles per second:** total measured cycles divided by the sum of their
  durations. One cycle allocates one request and releases it; it does not mean one
  KV block or one generated token.
- **Loop cycles per second:** total cycles divided by loop wall time, including
  sample recording and validation overhead.
- **Memory:** shallow free-container bytes plus recursive reachable bytes for
  the free container and the complete block manager. Recursive measurements
  include integer keys, container storage, block objects, token lists, and
  instance dictionaries, deduplicating object identities within each traversal.
  Measurements are taken after setup and after trials, before releasing pinned
  prefixes. They are not process RSS, peak memory, or exclusively owned bytes;
  the container and manager numbers overlap and must not be added together.

These are CPU allocator measurements. They establish neither GPU speedup nor
tokens/second, time to first token, decode latency, or end-to-end serving gains.
The reclaimed-prefix scenario intentionally stresses the scan that this change
removes; the cold and shared controls expose the costs paid elsewhere. Ordered
mapping metadata can be substantially larger than deque metadata, and reported
results should include that tradeoff. Occasional resizes and host scheduling can
affect tail samples; inspect raw trials rather than presenting the best run.

## Source and environment provenance

The baseline is the exact block manager at upstream commit
`bb823b3e06983d71485a8e1f23715ebd87d98ef8`. The script obtains it with `git show`
and compiles it into an isolated module. Use `--baseline-ref REVISION` to compare
another locally available revision. A missing default revision falls back to
`tests/fixtures/block_manager_upstream.py` only after verifying its SHA-256 digest
against the pinned upstream source. A different requested revision never silently
falls back.

The candidate is the current working-tree block manager. Both implementations
use the current repository's real `Sequence` and `SamplingParams`, NumPy, and
xxhash. The script creates namespace packages only to avoid the public package's
eager GPU-engine import; it does not replace these dependencies with mocks. The
report records source SHA-256 digests, the baseline revision and origin, working
tree status for supporting runtime files, Python and dependency versions, host
platform, and CPU count. No model download or CUDA installation is required.

Results default to `benchmarks/results/allocator.json`; override `--output` to
retain several runs. Report the source digests and environment alongside any
published timing table.
