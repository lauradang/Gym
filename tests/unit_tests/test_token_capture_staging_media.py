# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The compact-space delta a Megatron VLM call stages beside its expanded token delta."""

from typing import Any

import pytest

from nemo_gym.token_id_capture.staging import compute_extras_digest
from nemo_gym.token_id_capture.staging.media import COMPACT_TOKEN_IDS_DELTA_FIELD, build_compact_token_ids_delta


@pytest.mark.parametrize(
    ("compact_prev_len", "expected"),
    [(0, [1, 9, 2, 3]), (1, [9, 2, 3]), (4, "outside the compact prompt length")],
)
def test_compact_delta_mirrors_the_expanded_delta_rule(compact_prev_len: int, expected: Any) -> None:
    if isinstance(expected, str):
        with pytest.raises(ValueError, match=expected):
            build_compact_token_ids_delta([1, 9, 2], [3], compact_prev_len=compact_prev_len)
    else:
        assert build_compact_token_ids_delta([1, 9, 2], [3], compact_prev_len=compact_prev_len) == expected


def test_compact_delta_extras_pin_the_digest() -> None:
    # The envelope is a plain JSON list of ints, so it is digest-safe and stable.
    extras = {COMPACT_TOKEN_IDS_DELTA_FIELD: build_compact_token_ids_delta([1, 9, 2], [3], compact_prev_len=1)}
    assert compute_extras_digest(extras) == "49b3524383448e564b355361eeb80a1e9d1f39cdaf9507539c959a93ef38d35a"
