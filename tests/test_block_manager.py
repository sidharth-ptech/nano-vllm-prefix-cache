import hashlib
import random
from collections import Counter
from copy import deepcopy
from pathlib import Path

import pytest

from nanovllm.engine.block_manager import BlockManager
from nanovllm.engine.sequence import Sequence


def allocate(manager, token_ids):
    seq = Sequence(token_ids)
    cached = manager.can_allocate(seq)
    assert cached >= 0
    manager.allocate(seq, cached)
    return seq


def process_tokens(manager, seq, count=None):
    """Commit a prefill or decode chunk to the cache as Scheduler would."""
    seq.num_scheduled_tokens = (
        len(seq) - seq.num_cached_tokens if count is None else count
    )
    manager.hash_blocks(seq)
    seq.num_cached_tokens += seq.num_scheduled_tokens
    seq.num_scheduled_tokens = 0


def manager_state(manager):
    return (
        list(manager.free_block_ids),
        set(manager.used_block_ids),
        dict(manager.hash_to_block_id),
        [
            (block.block_id, block.ref_count, block.hash, list(block.token_ids))
            for block in manager.blocks
        ],
    )


def assert_invariants(manager, sequences):
    free = list(manager.free_block_ids)
    used = manager.used_block_ids
    references = Counter(block_id for seq in sequences for block_id in seq.block_table)
    assert len(free) == len(set(free))
    assert set(free).isdisjoint(used)
    assert set(free) | used == set(range(len(manager.blocks)))
    assert used == set(references)
    for block in manager.blocks:
        assert block.ref_count == references[block.block_id]
    for block_hash, block_id in manager.hash_to_block_id.items():
        assert manager.blocks[block_id].hash == block_hash


def test_fresh_and_recycled_blocks_follow_fifo_order():
    manager = BlockManager(4, 4)
    first = allocate(manager, [1] * 5)
    second = allocate(manager, [2])
    assert first.block_table == [0, 1]
    assert second.block_table == [2]

    manager.deallocate(first)
    assert first.block_table == []
    assert first.num_cached_tokens == 0
    assert list(manager.free_block_ids) == [3, 1, 0]

    third = allocate(manager, [3] * 9)
    assert third.block_table == [3, 1, 0]
    manager.deallocate(second)
    manager.deallocate(third)
    assert list(manager.free_block_ids) == [2, 0, 1, 3]
    assert_invariants(manager, [])


@pytest.mark.parametrize("cache_position", ["middle", "tail"])
def test_reclaiming_a_cached_block_preserves_remaining_fifo_order(cache_position):
    manager = BlockManager(5, 4)
    cached = allocate(manager, [10, 11, 12, 13, 14])
    process_tokens(manager, cached)
    unrelated = allocate(manager, [20] * 5)
    release_order = (
        (cached, unrelated) if cache_position == "middle" else (unrelated, cached)
    )
    for seq in release_order:
        manager.deallocate(seq)

    before = list(manager.free_block_ids)
    assert before.index(0) == (2 if cache_position == "middle" else 4)
    candidate = Sequence([10, 11, 12, 13, 99])
    assert manager.can_allocate(candidate) == 1
    manager.allocate(candidate, 1)

    assert candidate.block_table == [0, 4]
    assert candidate.num_cached_tokens == 4
    assert list(manager.free_block_ids) == [i for i in before if i not in (0, 4)]
    assert_invariants(manager, [candidate])


def test_used_prefix_is_shared_until_last_reference_is_released():
    manager = BlockManager(3, 4)
    first = allocate(manager, [1, 2, 3, 4, 5])
    process_tokens(manager, first)
    second = allocate(manager, [1, 2, 3, 4, 6])
    assert second.block_table == [0, 2]
    assert manager.blocks[0].ref_count == 2
    assert manager.can_allocate(Sequence([1, 2, 3, 4, 7])) == -1

    manager.deallocate(first)
    assert manager.blocks[0].ref_count == 1
    assert list(manager.free_block_ids) == [1]
    assert_invariants(manager, [second])
    manager.deallocate(second)
    assert list(manager.free_block_ids) == [1, 2, 0]
    assert_invariants(manager, [])


def test_reclaiming_head_of_free_queue_preserves_fifo_order():
    manager = BlockManager(4, 4)
    cached = allocate(manager, [1, 2, 3, 4, 5])
    process_tokens(manager, cached)
    manager.deallocate(cached)
    temporary = allocate(manager, [9] * 9)
    manager.deallocate(temporary)
    assert list(manager.free_block_ids) == [0, 1, 3, 2]

    restored = allocate(manager, [1, 2, 3, 4, 6])
    assert restored.block_table == [0, 1]
    assert restored.num_cached_tokens == 4
    assert list(manager.free_block_ids) == [3, 2]
    assert_invariants(manager, [restored])


def test_repeated_warm_cache_cycles_preserve_order_and_references():
    manager = BlockManager(3, 4)
    for cycle in range(40):
        seq = allocate(manager, list(range(9)))
        assert seq.block_table == [0, 1, 2]
        assert seq.num_cached_tokens == (8 if cycle else 0)
        assert not manager.free_block_ids
        process_tokens(manager, seq)
        assert_invariants(manager, [seq])
        manager.deallocate(seq)
        assert list(manager.free_block_ids) == [2, 1, 0]
        assert len(manager.hash_to_block_id) == 2
        assert_invariants(manager, [])


@pytest.mark.parametrize("block_size", [4, 256])
def test_prefix_reclaim_sharing_and_decode_with_supported_block_size(
    block_size, monkeypatch
):
    monkeypatch.setattr(Sequence, "block_size", block_size)
    manager = BlockManager(4, block_size)
    tokens = list(range(block_size)) + [1000]
    original = allocate(manager, tokens)
    process_tokens(manager, original)
    manager.deallocate(original)
    restored = allocate(manager, tokens)
    shared = allocate(manager, tokens[:-1] + [2000])
    assert restored.block_table == [0, 2]
    assert shared.block_table == [0, 3]
    assert restored.num_cached_tokens == shared.num_cached_tokens == block_size
    assert manager.blocks[0].ref_count == 2
    process_tokens(manager, restored)
    process_tokens(manager, shared)

    while len(restored) < 2 * block_size + 1:
        restored.append_token(3000 + len(restored))
        assert manager.can_append(restored)
        manager.may_append(restored)
        process_tokens(manager, restored, 1)
    assert restored.block_table == [0, 2, 1]
    assert not manager.free_block_ids
    assert_invariants(manager, [restored, shared])
    manager.deallocate(restored)
    assert manager.blocks[0].ref_count == 1
    manager.deallocate(shared)
    assert list(manager.free_block_ids) == [1, 2, 3, 0]
    assert_invariants(manager, [])


def test_free_cached_blocks_still_consume_capacity_when_reclaimed():
    manager = BlockManager(3, 4)
    cached = allocate(manager, list(range(9)))
    process_tokens(manager, cached)
    manager.deallocate(cached)
    unrelated = allocate(manager, [99])
    assert unrelated.block_table == [2]
    assert len(manager.hash_to_block_id) == 2
    # Two cached blocks plus one uncached block need three free blocks.
    assert manager.can_allocate(Sequence(list(range(9)))) == -1
    manager.deallocate(unrelated)
    restored = allocate(manager, list(range(9)))
    assert restored.block_table == [0, 1, 2]
    assert restored.num_cached_tokens == 8
    assert_invariants(manager, [restored])


def test_eviction_removes_old_hash_before_block_reuse():
    manager = BlockManager(1, 4)
    cached = allocate(manager, [1, 2, 3, 4])
    process_tokens(manager, cached)
    old_hash = manager.blocks[0].hash
    manager.deallocate(cached)
    replacement = allocate(manager, [9])
    assert old_hash not in manager.hash_to_block_id
    assert manager.blocks[0].hash == -1
    assert manager.blocks[0].token_ids == []
    assert_invariants(manager, [replacement])


def test_eviction_does_not_remove_a_newer_mapping_for_the_same_hash(monkeypatch):
    monkeypatch.setattr(
        BlockManager, "compute_hash", staticmethod(lambda token_ids, prefix=-1: 7)
    )
    manager = BlockManager(2, 4)
    first = allocate(manager, [1] * 4)
    second = allocate(manager, [2] * 4)
    process_tokens(manager, first)
    process_tokens(manager, second)
    assert manager.hash_to_block_id == {7: 1}
    manager.deallocate(first)
    manager.deallocate(second)

    replacement = allocate(manager, [9])
    assert replacement.block_table == [0]
    assert manager.hash_to_block_id == {7: 1}
    assert manager.blocks[1].token_ids == [2] * 4
    assert_invariants(manager, [replacement])


def test_hash_collision_with_different_tokens_is_not_a_cache_hit(monkeypatch):
    monkeypatch.setattr(
        BlockManager, "compute_hash", staticmethod(lambda token_ids, prefix=-1: 7)
    )
    manager = BlockManager(3, 4)
    cached = allocate(manager, [1] * 5)
    process_tokens(manager, cached)
    manager.deallocate(cached)
    assert manager.can_allocate(Sequence([1] * 4 + [8])) == 1
    assert manager.can_allocate(Sequence([2] * 4 + [8])) == 0


def test_hashing_waits_for_full_blocks_across_prefill_chunks():
    manager = BlockManager(2, 4)
    seq = allocate(manager, list(range(7)))
    process_tokens(manager, seq, 3)
    assert manager.hash_to_block_id == {}
    process_tokens(manager, seq, 2)
    first_hash = manager.compute_hash([0, 1, 2, 3])
    assert manager.hash_to_block_id == {first_hash: 0}
    process_tokens(manager, seq, 2)
    assert manager.blocks[1].hash == -1
    seq.append_token(7)
    assert manager.can_append(seq)
    manager.may_append(seq)
    process_tokens(manager, seq, 1)
    second_hash = manager.compute_hash([4, 5, 6, 7], first_hash)
    assert manager.hash_to_block_id == {first_hash: 0, second_hash: 1}
    assert seq.block_table == [0, 1]


@pytest.mark.parametrize("length", [1, 3, 4, 5, 7, 8])
def test_append_needs_a_free_block_only_at_a_new_block_boundary(length):
    block_count = (length + 3) // 4
    manager = BlockManager(block_count, 4)
    seq = allocate(manager, list(range(length)))
    seq.append_token(99)
    needs_block = length % 4 == 0
    assert manager.can_append(seq) is not needs_block
    if not needs_block:
        before = list(seq.block_table)
        manager.may_append(seq)
        assert seq.block_table == before
        assert_invariants(manager, [seq])


def test_append_reuses_the_oldest_released_block():
    manager = BlockManager(3, 4)
    seq = allocate(manager, [1, 2, 3, 4])
    other = allocate(manager, [9])
    manager.deallocate(other)
    assert list(manager.free_block_ids) == [2, 1]
    seq.append_token(5)
    assert manager.can_append(seq)
    manager.may_append(seq)
    assert seq.block_table == [0, 2]
    assert list(manager.free_block_ids) == [1]
    assert_invariants(manager, [seq])


def test_last_prompt_block_is_recomputed_even_when_full():
    manager = BlockManager(4, 4)
    first = allocate(manager, list(range(8)))
    process_tokens(manager, first)
    manager.deallocate(first)
    seq = Sequence(list(range(8)))
    assert manager.can_allocate(seq) == 1
    manager.allocate(seq, 1)
    assert seq.block_table == [0, 2]
    assert seq.num_cached_tokens == 4


def test_upstream_fixture_matches_pinned_source():
    source = Path(__file__).parent / "fixtures/block_manager_upstream.py"
    assert hashlib.sha256(source.read_bytes()).hexdigest() == (
        "d0240374f589eaa85c4c7923b8970f2635ab7bc73adc5044fa2d273955b54242"
    )


@pytest.mark.parametrize("seed", [0, 7, 42, 2026, 65537])
def test_randomized_lifecycle_matches_upstream(seed, upstream_manager_type):
    rng = random.Random(seed)
    managers = (BlockManager(24, 4), upstream_manager_type(24, 4))
    active = []
    prompts = [[family] * 8 for family in range(4)]
    operations = Counter()

    for step in range(800):
        action = rng.choices(("allocate", "append", "release"), (5, 3, 3))[0]
        if not active:
            action = "allocate"
        if action == "allocate":
            tokens = rng.choice(prompts)[: rng.randint(1, 8)]
            tokens += [rng.randrange(4, 16) for _ in range(rng.randrange(9))]
            sequences = (Sequence(tokens), Sequence(tokens))
            decisions = [
                manager.can_allocate(seq) for manager, seq in zip(managers, sequences)
            ]
            assert decisions[0] == decisions[1], (seed, step, action)
            if decisions[0] >= 0:
                for manager, seq, cached in zip(managers, sequences, decisions):
                    manager.allocate(seq, cached)
                    process_tokens(manager, seq)
                active.append(sequences)
                operations["cache_hit"] += decisions[0] > 0
                operations[action] += 1
            else:
                operations["capacity_rejection"] += 1
        elif action == "release":
            sequences = active.pop(rng.randrange(len(active)))
            for manager, seq in zip(managers, sequences):
                manager.deallocate(seq)
                assert seq.block_table == []
                assert seq.num_cached_tokens == 0
            operations[action] += 1
        else:
            sequences = rng.choice(active)
            token = rng.randrange(16)
            probes = [deepcopy(seq) for seq in sequences]
            for probe in probes:
                probe.append_token(token)
            decisions = [
                manager.can_append(seq) for manager, seq in zip(managers, probes)
            ]
            assert decisions[0] == decisions[1], (seed, step, action)
            if decisions[0]:
                for manager, seq in zip(managers, sequences):
                    seq.append_token(token)
                    manager.may_append(seq)
                    process_tokens(manager, seq, 1)
                operations[action] += 1

        assert manager_state(managers[0]) == manager_state(managers[1]), (
            seed,
            step,
            action,
        )
        for left, right in active:
            assert left.block_table == right.block_table
            assert left.num_cached_tokens == right.num_cached_tokens
            assert left.token_ids == right.token_ids
        for index, manager in enumerate(managers):
            assert_invariants(manager, [pair[index] for pair in active])

    # Ensure every seeded trace actually exercises the intended paths.
    assert all(
        operations[name] > 0
        for name in ("allocate", "append", "release", "cache_hit", "capacity_rejection")
    )
    for sequences in active:
        for manager, seq in zip(managers, sequences):
            manager.deallocate(seq)
    assert manager_state(managers[0]) == manager_state(managers[1])
    for manager in managers:
        assert_invariants(manager, [])
