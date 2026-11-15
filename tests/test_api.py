"""End-to-end and algorithm tests for the trajectory matching API."""

from __future__ import annotations

import copy

import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def point(ts: int, *candidates: tuple[str, int]) -> dict:
    return {
        "timestamp": f"2026-01-01T00:{ts:02d}:00Z",
        "candidates": [
            {"segment_id": segment_id, "offset": offset}
            for segment_id, offset in candidates
        ],
    }


def segment(segment_id: str, start: str, end: str, length: int) -> dict:
    return {
        "id": segment_id,
        "start_node": start,
        "end_node": end,
        "length": length,
    }


def line() -> list[dict]:
    return [
        segment("A", "N1", "N2", 100),
        segment("B", "N2", "N3", 200),
        segment("C", "N3", "N4", 300),
    ]


def post(payload: dict):
    return client.post("/match", json=payload)


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_unique_optimal_route():
    payload = {
        "points": [point(0, ("A", 1)), point(5, ("B", 2))],
        "segments": line(),
    }
    response = post(payload)
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["status"] == "matched"
    assert body["ambiguous"] is False
    assert body["segment_sequence"] == ["A", "B"]
    assert body["assignments"] == ["A", "B"]
    assert body["offset_cost"] == 3
    # Entering A costs 100, entering B costs 200.
    assert body["length_cost"] == 300
    assert body["optimal_cost"] == 303
    assert body["witnesses"] == []


def test_candidate_offsets_drive_choice():
    # S1 (len 50) and S2 (len 60) are both reachable from the first point;
    # the second point can be served by the same segment or by another one.
    segments = [
        segment("S1", "N1", "N2", 90),
        segment("S2", "N1", "N2", 60),
    ]
    payload = {
        "points": [
            point(0, ("S1", 10), ("S2", 1)),
            point(1, ("S1", 1), ("S2", 10)),
        ],
        "segments": segments,
    }
    body = post(payload).json()
    assert body["status"] == "matched"
    # Staying on S2 for both points: offsets 1+10, length 60 once.
    assert body["segment_sequence"] == ["S2"]
    assert body["optimal_cost"] == 71


def test_same_segment_is_not_counted_twice():
    segments = [segment("S1", "N1", "N2", 40)]
    payload = {
        "points": [
            point(0, ("S1", 0)),
            point(1, ("S1", 0)),
            point(2, ("S1", 0)),
        ],
        "segments": segments,
    }
    body = post(payload).json()
    assert body["status"] == "matched"
    assert body["segment_sequence"] == ["S1"]
    assert body["length_cost"] == 40
    assert body["optimal_cost"] == 40


def test_up_to_two_unlocated_segments_between_points():
    segments = [
        segment("A", "N1", "N2", 10),
        segment("X", "N2", "N3", 20),
        segment("Y", "N3", "N4", 30),
        segment("B", "N4", "N5", 40),
    ]
    payload = {
        "points": [point(0, ("A", 0)), point(1, ("B", 0))],
        "segments": segments,
    }
    body = post(payload).json()
    assert body["status"] == "matched"
    assert body["segment_sequence"] == ["A", "X", "Y", "B"]
    assert body["length_cost"] == 100
    assert body["assignments"] == ["A", "B"]


def test_three_unlocated_segments_is_unmatched():
    segments = [
        segment("A", "N1", "N2", 10),
        segment("X", "N2", "N3", 20),
        segment("Y", "N3", "N4", 30),
        segment("Z", "N4", "N5", 40),
        segment("B", "N5", "N6", 50),
    ]
    payload = {
        "points": [point(0, ("A", 0)), point(1, ("B", 0))],
        "segments": segments,
    }
    body = post(payload).json()
    assert body["status"] == "unmatched"
    assert body["ambiguous"] is False
    assert body["segment_sequence"] is None
    assert body["assignments"] is None
    assert body["optimal_cost"] is None
    assert body["witnesses"] == []
    assert body["reason"] == "no_connected_route"


def test_direction_is_respected():
    segments = [
        segment("A", "N1", "N2", 10),
        segment("B", "N3", "N2", 20),  # points back toward N2
    ]
    payload = {
        "points": [point(0, ("A", 0)), point(1, ("B", 0))],
        "segments": segments,
    }
    body = post(payload).json()
    assert body["status"] == "unmatched"


def test_ambiguity_reports_two_distinct_witnesses():
    # Two parallel segments with identical lengths and offsets tie exactly.
    segments = [
        segment("P", "N1", "N2", 50),
        segment("Q", "N1", "N2", 50),
        segment("R", "N2", "N3", 70),
    ]
    payload = {
        "points": [
            point(0, ("P", 0), ("Q", 0)),
            point(1, ("R", 0)),
        ],
        "segments": segments,
    }
    body = post(payload).json()
    assert body["status"] == "ambiguous"
    assert body["ambiguous"] is True
    assert body["segment_sequence"] is None
    assert body["optimal_cost"] == 120
    sequences = sorted(
        tuple(witness["segment_sequence"]) for witness in body["witnesses"]
    )
    assert sequences == [("P", "R"), ("Q", "R")]
    assert all(witness["total_cost"] == 120 for witness in body["witnesses"])


def test_ambiguity_between_intermediate_routes():
    # Two different two-hop connections with equal total length.
    segments = [
        segment("A", "N1", "N2", 10),
        segment("X", "N2", "N3", 20),
        segment("Y", "N3", "N4", 30),
        segment("B", "N4", "N5", 40),
        segment("X2", "N2", "N3", 20),
    ]
    payload = {
        "points": [point(0, ("A", 0)), point(1, ("B", 0))],
        "segments": segments,
    }
    body = post(payload).json()
    assert body["status"] == "ambiguous"
    sequences = sorted(
        tuple(witness["segment_sequence"]) for witness in body["witnesses"]
    )
    assert sequences == [("A", "X", "Y", "B"), ("A", "X2", "Y", "B")]


def test_unique_optimum_beats_tied_candidate_start():
    segments = [
        segment("P", "N1", "N2", 30),
        segment("Q", "N1", "N2", 30),
    ]
    payload = {
        "points": [point(0, ("P", 0), ("Q", 0)), point(1, ("Q", 0))],
        "segments": segments,
    }
    body = post(payload).json()
    # P,Q and Q are distinct sequences, but Q is strictly cheaper because
    # the shared segment is entered only once.
    assert body["status"] == "matched"
    assert body["segment_sequence"] == ["Q"]
    assert body["optimal_cost"] == 30


def test_duplicate_candidates_keep_smallest_offset():
    segments = [segment("S", "N1", "N2", 10)]
    payload = {
        "points": [
            {
                "timestamp": "2026-01-01T00:00:00Z",
                "candidates": [
                    {"segment_id": "S", "offset": 9},
                    {"segment_id": "S", "offset": 2},
                ],
            },
            point(1, ("S", 1)),
        ],
        "segments": segments,
    }
    body = post(payload).json()
    assert body["status"] == "matched"
    assert body["offset_cost"] == 3
    assert body["length_cost"] == 10


@pytest.mark.parametrize(
    "mutate",
    [
        lambda p: p.update(points=p["points"][:1]),
        lambda p: p.update(points=[point(i, ("A", 0)) for i in range(13)]),
        lambda p: p["points"][0].update(
            candidates=[{"segment_id": "A", "offset": 0}] * 5
        ),
        lambda p: p["points"][1].update(
            timestamp=p["points"][0]["timestamp"]
        ),
        lambda p: p["points"][0]["candidates"][0].update(offset=-1),
        lambda p: p["points"][0]["candidates"][0].update(offset=1.5),
        lambda p: p["segments"].append(
            segment("Z", "N1", "N2", 0)
        ),
        lambda p: p.update(segments=p["segments"] * 2),
        lambda p: p["points"][0]["candidates"][0].update(
            segment_id="MISSING"
        ),
    ],
)
def test_invalid_requests_are_rejected(mutate):
    # Deep copy so mutations such as appending a segment stay local.
    payload = copy.deepcopy({
        "points": [point(0, ("A", 1)), point(1, ("B", 1))],
        "segments": line(),
    })
    mutate(payload)
    response = post(payload)
    assert response.status_code == 422


def test_timestamps_are_compared_on_utc_timeline():
    payload = {
        "points": [
            {"timestamp": "2026-01-01T01:00:00+01:00",
             "candidates": [{"segment_id": "A", "offset": 0}]},
            {"timestamp": "2026-01-01T00:30:00Z",
             "candidates": [{"segment_id": "B", "offset": 0}]},
        ],
        "segments": line(),
    }
    # 01:00+01:00 == 00:00Z, which is before 00:30Z: valid increasing order.
    response = post(payload)
    assert response.status_code == 200, response.text
    assert response.json()["status"] == "matched"


def test_alternate_field_names_are_accepted():
    payload = {
        "points": [
            {"time": "2026-01-01T00:00:00Z",
             "roads": [{"id": "A", "deviation": 1}]},
            {"time": "2026-01-01T00:01:00Z",
             "roads": [{"id": "B", "deviation": 2}]},
        ],
        "roads": [
            {"id": "A", "from": "N1", "to": "N2", "length_m": 100},
            {"id": "B", "from": "N2", "to": "N3", "length_m": 200},
        ],
    }
    body = post(payload).json()
    assert body["status"] == "matched"
    assert body["segment_sequence"] == ["A", "B"]


def test_extra_fields_are_ignored():
    payload = {
        "points": [
            {"timestamp": "2026-01-01T00:00:00Z",
             "candidates": [{"segment_id": "A", "offset": 0, "extra": 1}],
             "ignored": True},
            point(1, ("B", 0)),
        ],
        "segments": line(),
        "unused": "ok",
    }
    assert post(payload).status_code == 200


def test_unmatched_never_returns_partial_route_even_when_prefix_exists():
    segments = [
        segment("A", "N1", "N2", 10),
        segment("B", "N2", "N3", 20),
        segment("C", "N9", "N10", 30),  # disconnected island
    ]
    payload = {
        "points": [point(0, ("A", 0)), point(1, ("C", 0))],
        "segments": segments,
    }
    body = post(payload).json()
    assert body["status"] == "unmatched"
    for key in (
        "segment_sequence",
        "assignments",
        "offset_cost",
        "length_cost",
        "optimal_cost",
    ):
        assert body[key] is None


def test_loop_and_self_loop_network():
    segments = [
        segment("A", "N1", "N2", 10),
        segment("B", "N2", "N2", 5),
        segment("C", "N2", "N3", 10),
    ]
    payload = {
        "points": [point(0, ("A", 0)), point(1, ("C", 0))],
        "segments": segments,
    }
    body = post(payload).json()
    assert body["status"] == "matched"
    # Direct A->C is cheaper (10+10) than A->B->C (10+5+10).
    assert body["segment_sequence"] == ["A", "C"]
    assert body["optimal_cost"] == 20


def test_same_sequence_with_different_placements_is_not_ambiguous():
    # p1 can sit on B or on C; either way the full traversed sequence is
    # A,B,C and the total cost is identical.  The contract keys ambiguity on
    # the road sequence, not on which point was placed where, so this must be
    # reported as a single optimal sequence.
    segments = [
        segment("A", "N1", "N2", 10),
        segment("B", "N2", "N3", 10),
        segment("C", "N3", "N4", 10),
    ]
    payload = {
        "points": [
            point(0, ("A", 0)),
            point(1, ("B", 5), ("C", 5)),
            point(2, ("C", 0)),
        ],
        "segments": segments,
    }
    body = post(payload).json()
    assert body["status"] == "matched"
    assert body["segment_sequence"] == ["A", "B", "C"]
    assert body["optimal_cost"] == 35
    assert tuple(body["assignments"]) in {("A", "B", "C"), ("A", "C", "C")}


def test_boolean_numbers_are_rejected():
    payload = {
        "points": [
            {"timestamp": "2026-01-01T00:00:00Z",
             "candidates": [{"segment_id": "A", "offset": True}]},
            point(1, ("B", 0)),
        ],
        "segments": line(),
    }
    assert post(payload).status_code == 422


def test_integral_float_numbers_are_accepted():
    payload = {
        "points": [
            {"timestamp": "2026-01-01T00:00:00Z",
             "candidates": [{"segment_id": "A", "offset": 2.0}]},
            point(1, ("B", 0)),
        ],
        "segments": line(),
    }
    body = post(payload).json()
    assert body["status"] == "matched"
    assert body["offset_cost"] == 2


def test_compact_candidate_forms_are_accepted():
    payload = {
        "points": [
            {"timestamp": "2026-01-01T00:00:00Z", "candidates": ["A", ["B", 4]]},
            {"timestamp": "2026-01-01T00:01:00Z", "candidates": [["C", 1]]},
        ],
        "segments": line(),
    }
    body = post(payload).json()
    assert body["status"] == "matched"
    # A at the first point would force A->B->C (600); placing the first point
    # on B is reachable directly (200+300) and wins despite the offset of 4.
    assert body["segment_sequence"] == ["B", "C"]
    assert body["offset_cost"] == 5


@pytest.mark.parametrize(
    "first,second",
    [
        (1767225600, 1767225660),
        (1767225600000, 1767225660000),
        ("1767225600", "1767225660"),
    ],
)
def test_epoch_timestamps_are_accepted(first, second):
    payload = {
        "points": [
            {"timestamp": first, "candidates": ["A"]},
            {"timestamp": second, "candidates": ["B"]},
        ],
        "segments": line(),
    }
    body = post(payload).json()
    assert body["status"] == "matched"
    assert body["segment_sequence"] == ["A", "B"]


def test_negative_epoch_and_bad_numeric_timestamps():
    payload = {
        "points": [
            {"timestamp": -2, "candidates": ["A"]},
            {"timestamp": -1, "candidates": ["B"]},
        ],
        "segments": line(),
    }
    assert post(payload).status_code == 200
    bad = {
        "points": [
            {"timestamp": "12.3.4", "candidates": ["A"]},
            {"timestamp": "2026-01-01T00:01:00Z", "candidates": ["B"]},
        ],
        "segments": line(),
    }
    assert post(bad).status_code == 422


def test_ambiguity_witnesses_are_deterministic_across_requests():
    segments = [
        segment("P", "N1", "N2", 50),
        segment("Q", "N1", "N2", 50),
        segment("R", "N2", "N3", 70),
    ]
    payload = {
        "points": [
            point(0, ("P", 0), ("Q", 0)),
            point(1, ("R", 0)),
        ],
        "segments": segments,
    }
    first = post(payload).json()
    for _ in range(5):
        assert post(payload).json() == first


def test_witness_sequences_are_full_and_include_unlocated_segments():
    segments = [
        segment("A", "N1", "N2", 10),
        segment("X", "N2", "N3", 10),
        segment("Y", "N3", "N4", 10),
        segment("B", "N4", "N5", 10),
        segment("X2", "N2", "N3", 10),
    ]
    payload = {
        "points": [point(0, ("A", 0)), point(1, ("B", 0))],
        "segments": segments,
    }
    body = post(payload).json()
    assert body["status"] == "ambiguous"
    for witness in body["witnesses"]:
        assert witness["segment_sequence"][0] == "A"
        assert witness["segment_sequence"][-1] == "B"
        assert len(witness["segment_sequence"]) == 4
        assert witness["assignments"] == ["A", "B"]
        assert witness["total_cost"] == 40
        assert (
            witness["offset_cost"] + witness["length_cost"]
            == witness["total_cost"]
        )
