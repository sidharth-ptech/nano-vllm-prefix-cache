from types import SimpleNamespace

from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.sampling_params import SamplingParams


def make_scheduler(scheduler_type, *, blocks=5, token_budget=3):
    return scheduler_type(
        SimpleNamespace(
            max_num_seqs=2,
            max_num_batched_tokens=token_budget,
            eos=-1,
            kvcache_block_size=4,
            num_kvcache_blocks=blocks,
        )
    )


def test_chunked_prefill_completion_and_warm_prefix_reuse(scheduler_type):
    scheduler = make_scheduler(scheduler_type)
    first = Sequence(list(range(9)), SamplingParams(max_tokens=1))
    scheduler.add(first)

    for expected_cached in (3, 6, 0):
        scheduled, is_prefill = scheduler.schedule()
        assert scheduled == [first]
        assert is_prefill
        assert first.num_scheduled_tokens == 3
        scheduler.postprocess(scheduled, [99], is_prefill)
        assert first.num_cached_tokens == expected_cached

    assert first.is_finished
    assert scheduler.is_finished()
    assert list(scheduler.block_manager.free_block_ids) == [3, 4, 2, 1, 0]

    second = Sequence(list(range(9)), SamplingParams(max_tokens=1))
    scheduler.add(second)
    scheduled, is_prefill = scheduler.schedule()
    assert is_prefill
    assert second.block_table == [0, 1, 3]
    assert second.num_cached_tokens == 8
    assert second.num_scheduled_tokens == 1
    scheduler.postprocess(scheduled, [99], is_prefill)
    assert second.is_finished
    assert scheduler.is_finished()
    assert list(scheduler.block_manager.free_block_ids) == [4, 2, 3, 1, 0]
    assert not scheduler.block_manager.used_block_ids
    assert all(block.ref_count == 0 for block in scheduler.block_manager.blocks)


def test_decode_pressure_preempts_and_resumes_with_cached_prefix(scheduler_type):
    scheduler = make_scheduler(scheduler_type, blocks=3, token_budget=8)
    first = Sequence([1] * 4, SamplingParams(max_tokens=4))
    second = Sequence([2] * 4, SamplingParams(max_tokens=4))
    scheduler.add(first)
    scheduler.add(second)

    scheduled, is_prefill = scheduler.schedule()
    assert scheduled == [first, second]
    scheduler.postprocess(scheduled, [10, 20], is_prefill)
    scheduled, is_prefill = scheduler.schedule()
    assert scheduled == [first]
    assert not is_prefill
    assert first.block_table == [0, 2]
    assert second.status is SequenceStatus.WAITING
    assert second.block_table == []
    assert list(scheduler.waiting) == [second]
    assert list(scheduler.block_manager.free_block_ids) == [1]
    scheduler.postprocess(scheduled, [11], is_prefill)

    for token in (12, 13):
        scheduled, is_prefill = scheduler.schedule()
        assert scheduled == [first]
        assert not is_prefill
        scheduler.postprocess(scheduled, [token], is_prefill)
    assert first.is_finished

    scheduled, is_prefill = scheduler.schedule()
    assert scheduled == [second]
    assert is_prefill
    assert second.num_cached_tokens == 4
    assert second.num_scheduled_tokens == 1
    assert second.block_table == [1, 2]
    scheduler.postprocess(scheduled, [21], is_prefill)
    for token in (22, 23):
        scheduled, is_prefill = scheduler.schedule()
        assert scheduled == [second]
        assert not is_prefill
        scheduler.postprocess(scheduled, [token], is_prefill)
    assert scheduler.is_finished()
    assert second.is_finished
    assert sorted(scheduler.block_manager.free_block_ids) == [0, 1, 2]
    assert not scheduler.block_manager.used_block_ids
    assert all(block.ref_count == 0 for block in scheduler.block_manager.blocks)
