"""Core trajectory-to-directed-road matching.

Semantics implemented here:

* Every location point must be placed on exactly one of its candidate
  segments.  The chosen candidate's offset is part of the cost.
* Consecutive located segments must be connected following the direction of
  the road network (``A.end_node == B.start_node``).
* Between two consecutive located segments at most two segments that carry no
  location point may be traversed.  This is evaluated independently for each
  gap between consecutive points.
* Every time a segment is entered its positive length is paid once.  If two
  consecutive location points share a segment, that segment is entered only
  once (no duplicate length).
* The objective is the sum of the selected candidates' offsets plus the
  lengths of all entered segments.  Connection paths between consecutive
  points are represented by their cheapest possible continuation; this is
  safe because all lengths are positive and the cost is additive.
* All distinct full segment sequences achieving the minimum are considered.
  If at least two differ, the result is ambiguous and two optimal witnesses
  are reported.  A sequence is the complete list of entered segments,
  including segments that carry no location point.

The algorithm is an exact dynamic program over (point index, chosen segment).
Its state records the minimum cost and up to two distinct optimal expansions,
which is sufficient to decide whether the global optimum is unique.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from .schemas import MatchRequest

MAX_UNLOCATED_SEGMENTS_PER_GAP = 2
MAX_WITNESSES = 2


@dataclass(frozen=True, slots=True)
class Road:
    id: str
    start_node: str
    end_node: str
    length: int


@dataclass(frozen=True, slots=True)
class MatchWitness:
    """One optimal route.

    ``assignments`` has one entry per location point; ``segment_sequence`` is
    the complete traversed sequence (points plus unlocated intermediates).
    """

    segment_sequence: tuple[str, ...]
    assignments: tuple[str, ...]
    total_cost: int
    offset_cost: int
    length_cost: int


@dataclass(frozen=True, slots=True)
class MatchOutcome:
    status: str  # "matched" | "ambiguous" | "unmatched"
    message: str
    reason_code: str | None
    best: MatchWitness | None
    witnesses: tuple[MatchWitness, ...]
    optimal_cost: int | None


@dataclass(slots=True)
class _State:
    cost: int
    witnesses: list[MatchWitness]


def _variants(
    a: str,
    b: str,
    roads: dict[str, Road],
    outgoing: dict[str, tuple[str, ...]],
) -> tuple[tuple[tuple[str, ...], int], ...]:
    """Cheapest continuation from segment ``a`` to segment ``b``.

    Returns up to two distinct cheapest paths.  Each path is the tuple of
    *newly entered* segments (intermediates plus ``b``); the empty tuple means
    both points are on ``a`` and nothing new is entered.
    """

    if a == b:
        return (((), 0),)

    target_start = roads[b].start_node
    costs: dict[tuple[str, ...], int] = {}

    def consider(prefix: tuple[str, ...], extra_length: int) -> None:
        costs[prefix] = extra_length

    # Direct continuation: a -> b.
    if roads[a].end_node == target_start:
        consider((b,), roads[b].length)

    # One unlocated segment: a -> x -> b.
    for x in outgoing[a]:
        if x in (a, b):
            continue
        if roads[x].end_node != target_start:
            continue
        consider((x, b), roads[x].length + roads[b].length)

    # Two unlocated segments: a -> x -> y -> b.
    for x in outgoing[a]:
        if x in (a, b):
            continue
        for y in outgoing[x]:
            if y in (a, b, x):
                continue
            if roads[y].end_node != target_start:
                continue
            consider(
                (x, y, b),
                roads[x].length + roads[y].length + roads[b].length,
            )

    if not costs:
        return ()

    best_cost = min(costs.values())
    # Iterate deterministically so witness selection is reproducible.
    seen_paths: set[tuple[str, ...]] = set()
    selected: list[tuple[tuple[str, ...], int]] = []
    for path in sorted(costs):
        if costs[path] != best_cost or path in seen_paths:
            continue
        seen_paths.add(path)
        selected.append((path, best_cost))
        if len(selected) == MAX_WITNESSES:
            break
    return tuple(selected)


def _dedupe(witnesses: Iterable[MatchWitness]) -> list[MatchWitness]:
    """Keep up to two witnesses with distinct full sequences."""

    seen: set[tuple[str, ...]] = set()
    result: list[MatchWitness] = []
    for witness in witnesses:
        if witness.segment_sequence in seen:
            continue
        seen.add(witness.segment_sequence)
        result.append(witness)
        if len(result) == MAX_WITNESSES:
            break
    return result


def _unmatched(reason: str, message: str) -> MatchOutcome:
    return MatchOutcome(
        status="unmatched",
        message=message,
        reason_code=reason,
        best=None,
        witnesses=(),
        optimal_cost=None,
    )


def match(request: MatchRequest) -> MatchOutcome:
    """Match a validated request, returning a complete or ambiguous result."""

    roads = {
        segment.id: Road(
            id=segment.id,
            start_node=segment.start_node,
            end_node=segment.end_node,
            length=segment.length,
        )
        for segment in request.segments
    }

    by_start: dict[str, list[str]] = {}
    for road in roads.values():
        by_start.setdefault(road.start_node, []).append(road.id)
    outgoing = {
        road.id: tuple(sorted(by_start.get(road.end_node, ())))
        for road in roads.values()
    }

    variant_cache: dict[
        tuple[str, str], tuple[tuple[tuple[str, ...], int], ...]
    ] = {}

    def variants(a: str, b: str):
        key = (a, b)
        cached = variant_cache.get(key)
        if cached is None:
            cached = _variants(a, b, roads, outgoing)
            variant_cache[key] = cached
        return cached

    candidates = [point.normalized_candidates() for point in request.points]

    states: dict[str, _State] = {}
    for candidate in candidates[0]:
        road = roads[candidate.segment_id]
        witness = MatchWitness(
            segment_sequence=(candidate.segment_id,),
            assignments=(candidate.segment_id,),
            total_cost=candidate.offset + road.length,
            offset_cost=candidate.offset,
            length_cost=road.length,
        )
        states[candidate.segment_id] = _State(
            cost=witness.total_cost, witnesses=[witness]
        )

    for point_index in range(1, len(candidates)):
        next_states: dict[str, _State] = {}

        for candidate in candidates[point_index]:
            target = candidate.segment_id
            best_cost: int | None = None
            collected: list[MatchWitness] = []

            for source, state in states.items():
                for path, path_length in variants(source, target):
                    cost = state.cost + candidate.offset + path_length
                    if best_cost is None or cost < best_cost:
                        best_cost = cost
                        collected = []
                    if cost == best_cost:
                        for witness in state.witnesses:
                            collected.append(
                                MatchWitness(
                                    segment_sequence=(
                                        witness.segment_sequence + path
                                    ),
                                    assignments=(
                                        witness.assignments + (target,)
                                    ),
                                    total_cost=(
                                        witness.offset_cost
                                        + candidate.offset
                                        + witness.length_cost
                                        + path_length
                                    ),
                                    offset_cost=(
                                        witness.offset_cost + candidate.offset
                                    ),
                                    length_cost=(
                                        witness.length_cost + path_length
                                    ),
                                )
                            )

            if best_cost is not None:
                next_states[target] = _State(
                    cost=best_cost, witnesses=_dedupe(collected)
                )

        states = next_states
        if not states:
            return _unmatched(
                "no_connected_route",
                "No directed road sequence connects all location points while "
                "traversing at most two road segments without a location "
                "point between consecutive points.",
            )

    best_cost = min(state.cost for state in states.values())
    final_witnesses: list[MatchWitness] = []
    for state in states.values():
        if state.cost == best_cost:
            final_witnesses.extend(state.witnesses)
    distinct = _dedupe(final_witnesses)

    if not distinct:
        # Defensive: with non-empty candidates this cannot happen.
        return _unmatched(
            "no_connected_route",
            "No directed road sequence connects all location points.",
        )

    if len(distinct) == 1:
        best = distinct[0]
        return MatchOutcome(
            status="matched",
            message=(
                "Unique optimal directed road sequence found for all "
                f"{len(candidates)} location points."
            ),
            reason_code=None,
            best=best,
            witnesses=(),
            optimal_cost=best.total_cost,
        )

    return MatchOutcome(
        status="ambiguous",
        message=(
            "Multiple distinct directed road sequences achieve the minimum "
            "total cost; two optimal witnesses are returned."
        ),
        reason_code=None,
        best=None,
        witnesses=tuple(distinct[:MAX_WITNESSES]),
        optimal_cost=best_cost,
    )
