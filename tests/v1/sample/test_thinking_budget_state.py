# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for ThinkingBudgetStateHolder."""

import pytest
import torch

from tests.v1.sample.utils import create_mock_reasoning_config
from vllm.sampling_params import SamplingParams
from vllm.v1.sample.logits_processor.interface import (
    BatchUpdate,
    MoveDirectionality,
)
from vllm.v1.sample.thinking_budget_state import ThinkingBudgetStateHolder


def _make_holder() -> ThinkingBudgetStateHolder:
    return ThinkingBudgetStateHolder(
        create_mock_reasoning_config([151667], [151668]),
        8,
        0,
        torch.device("cpu"),
        False,
    )


def test_swap_budgeted_with_unbudgeted_clears_empty_side():
    """Asymmetric SWAP must not leave the empty index sharing state."""
    h = _make_holder()
    h.sync_batch(
        BatchUpdate(
            batch_size=2,
            removed=(),
            added=[
                (0, SamplingParams(thinking_token_budget=5), None, []),
                (1, SamplingParams(), None, []),
            ],
            moved=(),
        )
    )
    assert list(h._state.keys()) == [0]
    budget_state = h._state[0]

    h.sync_batch(
        BatchUpdate(
            batch_size=2,
            removed=(),
            added=(),
            moved=[(0, 1, MoveDirectionality.SWAP)],
        )
    )
    assert list(h._state.keys()) == [1]
    assert h._state[1] is budget_state
    assert h._state[1]["thinking_token_budget"] == 5

    h.sync_batch(
        BatchUpdate(
            batch_size=2,
            removed=(),
            added=(),
            moved=[(0, 1, MoveDirectionality.SWAP)],
        )
    )
    assert list(h._state.keys()) == [0]
    assert h._state[0] is budget_state


def test_swap_exchanges_two_budgeted_states():
    h = _make_holder()
    h.sync_batch(
        BatchUpdate(
            batch_size=2,
            removed=(),
            added=[
                (0, SamplingParams(thinking_token_budget=3), None, []),
                (1, SamplingParams(thinking_token_budget=7), None, []),
            ],
            moved=(),
        )
    )
    b0 = h._state[0]["thinking_token_budget"]
    b1 = h._state[1]["thinking_token_budget"]
    h.sync_batch(
        BatchUpdate(
            batch_size=2,
            removed=(),
            added=(),
            moved=[(0, 1, MoveDirectionality.SWAP)],
        )
    )
    assert h._state[0]["thinking_token_budget"] == b1
    assert h._state[1]["thinking_token_budget"] == b0


THINK_START = 151667
THINK_END = 151668


def _make_spec_holder(
    budget: int, end_ids: tuple[int, ...] = (THINK_END,)
) -> ThinkingBudgetStateHolder:
    h = ThinkingBudgetStateHolder(
        create_mock_reasoning_config([THINK_START], list(end_ids)),
        8,
        2,
        torch.device("cpu"),
        False,
    )
    h.sync_batch(
        BatchUpdate(
            batch_size=1,
            removed=(),
            added=[(0, SamplingParams(thinking_token_budget=budget), None, [])],
            moved=(),
        )
    )
    return h


def _spec_step(
    h: ThinkingBudgetStateHolder, output: list[int], drafts: list[int]
) -> tuple[list[int], list[int]]:
    """Run one spec decode step; return the bonus and target rows forced to end."""
    h.update_state([output], [drafts])
    vocab_size = THINK_END + 1
    bonus = h.apply_to_logits(torch.zeros(1, vocab_size), True, [drafts])
    target = h.apply_to_logits(torch.zeros(len(drafts), vocab_size), False, [drafts])
    return (
        bonus[:, THINK_END].nonzero().flatten().tolist(),
        target[:, THINK_END].nonzero().flatten().tolist(),
    )


@pytest.mark.parametrize(
    "budget,drafts,expected_bonus,expected_target",
    [
        # A draft end before the budget row ends thinking naturally if accepted.
        (2, [THINK_END, 11], [], []),
        (3, [THINK_END, 11], [], []),
        # A draft end at the budget row, or no draft end, still forces.
        (2, [11, THINK_END], [], [1]),
        (3, [11, 12], [0], []),
    ],
)
def test_spec_budget_transition_respects_draft_end(
    budget, drafts, expected_bonus, expected_target
):
    h = _make_spec_holder(budget)
    assert _spec_step(h, [THINK_START, 10], drafts) == (
        expected_bonus,
        expected_target,
    )


def test_spec_budget_enforced_after_draft_end_rejected():
    h = _make_spec_holder(2)
    assert _spec_step(h, [THINK_START, 10], [THINK_END, 11]) == ([], [])
    # The draft end was rejected at row 0; thinking continues to the budget.
    assert _spec_step(h, [THINK_START, 10, 12], [13, 14]) == ([], [0])


def test_spec_budget_enforced_in_second_think_after_draft_end_rejected():
    h = _make_spec_holder(2)
    # The first think section ends on a forced bonus row.
    assert _spec_step(h, [THINK_START, 10], [11]) == ([0], [])
    assert _spec_step(h, [THINK_START, 10, 11, THINK_END], [20]) == ([], [])
    # In a second think section the draft end is rejected at row 0.
    output = [THINK_START, 10, 11, THINK_END, 20, THINK_START, 12]
    assert _spec_step(h, output, [THINK_END]) == ([], [])
    assert _spec_step(h, output + [15], [16]) == ([], [0])


def test_spec_budget_transition_respects_end_split_across_output_and_drafts():
    # A multi-token end started by the last output token, finished by drafts.
    h = _make_spec_holder(3, (THINK_END, 21))
    assert _spec_step(h, [THINK_START, 10, THINK_END], [21, 11]) == ([], [])
