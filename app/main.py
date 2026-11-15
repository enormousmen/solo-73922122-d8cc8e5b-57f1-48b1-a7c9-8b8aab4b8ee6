"""FastAPI application exposing the trajectory-to-road matching service."""

from __future__ import annotations

from typing import Literal

from fastapi import FastAPI
from pydantic import BaseModel, Field

from .matcher import MatchOutcome, MatchWitness, match
from .schemas import MatchRequest

API_VERSION = "1.0.0"


class WitnessResponse(BaseModel):
    """One optimal route, including unlocated intermediate segments."""

    segment_sequence: list[str] = Field(
        description="Complete directed segment sequence, in traversal order."
    )
    assignments: list[str] = Field(
        description="Segment chosen for each location point, in input order."
    )
    offset_cost: int = Field(
        description="Sum of the chosen candidates' lateral offsets."
    )
    length_cost: int = Field(
        description="Sum of every entered segment length, counted once per entry."
    )
    total_cost: int = Field(
        description="offset_cost + length_cost."
    )


class MatchResponse(BaseModel):
    status: Literal["matched", "ambiguous", "unmatched"]
    ambiguous: bool
    optimal_cost: int | None = Field(
        default=None,
        description="Minimum achievable total cost, null when no route exists.",
    )
    segment_sequence: list[str] | None = Field(
        default=None,
        description="The unique optimal sequence; null unless status=matched.",
    )
    assignments: list[str] | None = Field(
        default=None,
        description="Assignments for the unique optimum; null unless matched.",
    )
    offset_cost: int | None = None
    length_cost: int | None = None
    witnesses: list[WitnessResponse] = Field(
        default_factory=list,
        description="Two optimal witnesses when status=ambiguous.",
    )
    reason: str | None = Field(
        default=None,
        description="Machine-readable failure reason when status=unmatched.",
    )
    message: str = Field(description="Human-readable explanation.")


def _witness(witness: MatchWitness) -> WitnessResponse:
    return WitnessResponse(
        segment_sequence=list(witness.segment_sequence),
        assignments=list(witness.assignments),
        offset_cost=witness.offset_cost,
        length_cost=witness.length_cost,
        total_cost=witness.total_cost,
    )


def _respond(outcome: MatchOutcome) -> MatchResponse:
    if outcome.status == "matched":
        best = outcome.best
        assert best is not None
        return MatchResponse(
            status="matched",
            ambiguous=False,
            optimal_cost=best.total_cost,
            segment_sequence=list(best.segment_sequence),
            assignments=list(best.assignments),
            offset_cost=best.offset_cost,
            length_cost=best.length_cost,
            witnesses=[],
            message=outcome.message,
        )

    if outcome.status == "ambiguous":
        return MatchResponse(
            status="ambiguous",
            ambiguous=True,
            optimal_cost=outcome.optimal_cost,
            segment_sequence=None,
            assignments=None,
            offset_cost=None,
            length_cost=None,
            witnesses=[_witness(witness) for witness in outcome.witnesses],
            message=outcome.message,
        )

    return MatchResponse(
        status="unmatched",
        ambiguous=False,
        optimal_cost=None,
        segment_sequence=None,
        assignments=None,
        offset_cost=None,
        length_cost=None,
        witnesses=[],
        reason=outcome.reason_code,
        message=outcome.message,
    )


app = FastAPI(
    title="Directed Road Trajectory Matcher",
    version=API_VERSION,
    summary=(
        "Match a short UTC location stream to a directed road network, "
        "minimizing lateral offsets plus entered segment lengths."
    ),
    description=(
        "POST /match with 2-12 timestamped points (1-4 candidate directed "
        "segments each, non-negative integer lateral offsets) and up to 30 "
        "directed segments. Consecutive points are connected along the "
        "direction of travel, traversing at most two segments without a "
        "location point per gap. Consecutive hits on the same segment are "
        "charged once. The response is `matched` with the unique optimal "
        "full segment sequence, `ambiguous` with two optimal witnesses when "
        "distinct sequences tie for the minimum cost, or `unmatched` with no "
        "partial route."
    ),
)


@app.get("/health", tags=["ops"])
def health() -> dict[str, str]:
    return {"status": "ok", "version": API_VERSION}


@app.post("/match", response_model=MatchResponse, tags=["matching"])
def match_trajectory(request: MatchRequest) -> MatchResponse:
    """Match a trajectory to the directed road network.

    All costs are integers.  When no complete directed route exists the
    outcome is `unmatched`; no partial or best-effort sequence is returned.
    """

    return _respond(match(request))
