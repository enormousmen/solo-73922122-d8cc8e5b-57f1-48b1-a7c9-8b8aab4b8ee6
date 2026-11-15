"""Cross-check the dynamic program against exhaustive enumeration.

The reference solver enumerates every assignment of points to candidate
segments and every directed walk of at most three edges between consecutive
assignments (two intermediate segments).  It deliberately *includes* walks
that revisit the source, the target or an intermediate segment; since all
lengths are positive those cannot beat a simpler walk, so agreement confirms
that the optimized matcher prunes only dominated continuations.
"""

from __future__ import annotations

import random

from app.matcher import match
from app.schemas import MatchRequest


def reference(points, segments):
    """Return (best_cost, distinct optimal full sequences) or (None, set())."""

    roads = {s["id"]: s for s in segments}
    outgoing = {}
    for road in segments:
        outgoing.setdefault(road["start_node"], []).append(road["id"])

    def walks(a: str, b: str):
        """All (path, cost) with path the newly entered segments."""

        results = []
        if a == b:
            results.append(((), 0))

        # Exactly one edge a->b.
        if roads[a]["end_node"] == roads[b]["start_node"]:
            results.append(((b,), roads[b]["length"]))

        # Two edges a->x->b (single unlocated segment).
        for x in outgoing.get(roads[a]["end_node"], []):
            if roads[x]["end_node"] == roads[b]["start_node"]:
                results.append(((x, b), roads[x]["length"] + roads[b]["length"]))

        # Three edges a->x->y->b (two unlocated segments).
        for x in outgoing.get(roads[a]["end_node"], []):
            for y in outgoing.get(roads[x]["end_node"], []):
                if roads[y]["end_node"] == roads[b]["start_node"]:
                    results.append(
                        (
                            (x, y, b),
                            roads[x]["length"]
                            + roads[y]["length"]
                            + roads[b]["length"],
                        )
                    )
        return results

    candidate_sets = []
    for point in points:
        best_offsets: dict[str, int] = {}
        for candidate in point["candidates"]:
            segment_id = candidate["segment_id"]
            offset = candidate["offset"]
            if segment_id not in best_offsets or offset < best_offsets[segment_id]:
                best_offsets[segment_id] = offset
        candidate_sets.append(sorted(best_offsets.items()))

    best_cost: int | None = None
    best_sequences: set[tuple[str, ...]] = set()

    def explore(index: int, sequence, cost, current):
        nonlocal best_cost, best_sequences
        if index == len(candidate_sets):
            if best_cost is None or cost < best_cost:
                best_cost = cost
                best_sequences = {tuple(sequence)}
            elif cost == best_cost:
                best_sequences.add(tuple(sequence))
            return

        for segment_id, offset in candidate_sets[index]:
            if current is None:
                explore(
                    index + 1,
                    [segment_id],
                    offset + roads[segment_id]["length"],
                    segment_id,
                )
                continue
            for path, path_cost in walks(current, segment_id):
                explore(
                    index + 1,
                    list(sequence) + list(path),
                    cost + offset + path_cost,
                    segment_id,
                )

    explore(0, [], 0, None)
    return best_cost, best_sequences


def random_case(rng: random.Random):
    segment_count = rng.randint(2, 7)
    node_count = rng.randint(2, 5)
    nodes = [f"N{i}" for i in range(node_count)]
    segments = []
    for index in range(segment_count):
        start, end = rng.sample(nodes, 2) if rng.random() < 0.8 else (
            rng.choice(nodes),
            rng.choice(nodes),
        )
        segments.append(
            {
                "id": f"S{index}",
                "start_node": start,
                "end_node": end,
                "length": rng.randint(1, 12),
            }
        )
    point_count = rng.randint(2, 5)
    points = []
    for index in range(point_count):
        candidate_count = rng.randint(1, min(4, segment_count))
        chosen = rng.sample(segments, candidate_count)
        points.append(
            {
                "timestamp": f"2026-01-01T00:{index:02d}:00Z",
                "candidates": [
                    {
                        "segment_id": road["id"],
                        "offset": rng.randint(0, 8),
                    }
                    for road in chosen
                ],
            }
        )
    return points, segments


def test_matcher_agrees_with_exhaustive_reference():
    rng = random.Random(20261115)
    for case_index in range(600):
        points, segments = random_case(rng)
        request = MatchRequest.model_validate(
            {"points": points, "segments": segments}
        )
        outcome = match(request)

        expected_cost, expected_sequences = reference(points, segments)

        if expected_cost is None:
            assert outcome.status == "unmatched", (case_index, points, segments)
            assert outcome.best is None
            assert outcome.witnesses == ()
            continue

        assert outcome.optimal_cost == expected_cost, (
            case_index, points, segments, expected_cost, outcome
        )

        if len(expected_sequences) == 1:
            (expected_sequence,) = expected_sequences
            assert outcome.status == "matched", (case_index, outcome)
            assert outcome.best is not None
            assert outcome.best.segment_sequence == expected_sequence
            assert outcome.best.total_cost == expected_cost
        else:
            assert outcome.status == "ambiguous", (case_index, outcome)
            reported = {
                witness.segment_sequence for witness in outcome.witnesses
            }
            assert len(reported) == 2
            assert reported <= expected_sequences
            for witness in outcome.witnesses:
                assert witness.total_cost == expected_cost


def test_ambiguity_counts_match_reference_in_tied_network():
    segments = []
    for index in range(6):
        segments.append(
            {
                "id": f"T{index}",
                "start_node": "N1",
                "end_node": "N2",
                "length": 7,
            }
        )
    points = [
        {
            "timestamp": "2026-01-01T00:00:00Z",
            "candidates": [
                {"segment_id": f"T{index}", "offset": 0} for index in range(4)
            ],
        },
        {
            "timestamp": "2026-01-01T00:01:00Z",
            "candidates": [
                {"segment_id": f"T{index}", "offset": 0} for index in range(4)
            ],
        },
    ]
    request = MatchRequest.model_validate(
        {"points": points, "segments": segments}
    )
    outcome = match(request)
    assert outcome.status == "ambiguous"
    assert outcome.optimal_cost == 7
    assert len(outcome.witnesses) == 2
    # Both optimal witnesses stay on one parallel segment for both points.
    for witness in outcome.witnesses:
        assert len(witness.segment_sequence) == 1
        assert witness.assignments == (witness.segment_sequence[0],) * 2
