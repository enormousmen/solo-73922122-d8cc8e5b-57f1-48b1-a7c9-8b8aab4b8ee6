"""Request models for the trajectory-to-road matching API.

The public wire format is intentionally small and self-describing:

    {
      "points": [
        {"timestamp": "2026-01-01T00:00:00Z",
         "candidates": [{"segment_id": "S1", "offset": 3}]},
        ...
      ],
      "segments": [
        {"id": "S1", "start_node": "N1", "end_node": "N2", "length": 120},
        ...
      ]
    }

A few common alternative field names are accepted through AliasChoices so
callers do not need to reshape otherwise reasonable payloads.
"""

from __future__ import annotations

from collections import Counter
from datetime import UTC, datetime
from typing import Annotated, Any

from pydantic import (
    AliasChoices,
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)


def _coerce_integral(value: Any) -> int:
    """Accept integral JSON numbers, rejecting booleans and fractions.

    Booleans are rejected explicitly because ``True`` is not a count or a
    length.  ``3.0`` is accepted as ``3`` so clients whose JSON serializer
    emits floats are not needlessly rejected; ``3.5`` and ``"3"`` are not.
    """

    if isinstance(value, bool):
        raise ValueError("value must be an integer, not a boolean")
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    raise ValueError("value must be an integer")


IntegralInt = Annotated[int, BeforeValidator(_coerce_integral)]
NonNegativeInt = Annotated[IntegralInt, Field(ge=0)]
PositiveInt = Annotated[IntegralInt, Field(gt=0)]


def _clean_identifier(value: Any, *, what: str) -> str:
    """Normalize road/node identifiers to non-empty strings.

    Integer identifiers are accepted and stringified (``7`` -> ``"7"``).
    Booleans are rejected because ``True`` is not a meaningful identifier.
    """

    if isinstance(value, bool) or value is None:
        raise ValueError(f"{what} must be a non-empty string or an integer")
    if isinstance(value, int):
        return str(value)
    if isinstance(value, str):
        cleaned = value.strip()
        if not cleaned:
            raise ValueError(f"{what} must not be empty")
        return cleaned
    raise ValueError(f"{what} must be a string or an integer")


class Candidate(BaseModel):
    """One possible road segment for a location point, plus its offset."""

    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    segment_id: str = Field(
        validation_alias=AliasChoices(
            "segment_id", "id", "segment", "road_id", "road"
        )
    )
    offset: NonNegativeInt = Field(
        validation_alias=AliasChoices(
            "offset", "deviation", "lateral_offset", "lateral_deviation",
            "distance",
        )
    )

    @field_validator("segment_id", mode="before")
    @classmethod
    def _normalize_id(cls, value: Any) -> str:
        return _clean_identifier(value, what="candidate segment_id")


def _is_signed_number(text: str) -> bool:
    """True for strings such as ``-3`` or ``-3.5`` but not ``-"``."""

    body = text[1:] if text[:1] in {"+", "-"} else text
    if not body:
        return False
    if body.count(".") > 1:
        return False
    return body.replace(".", "", 1).isdigit()


class LocationPoint(BaseModel):
    """One UTC timestamp with 1..4 candidate directed road segments."""

    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    timestamp: datetime = Field(
        validation_alias=AliasChoices(
            "timestamp", "time", "ts", "utc_time", "recorded_at"
        )
    )
    candidates: list[Candidate] = Field(
        min_length=1,
        max_length=4,
        validation_alias=AliasChoices(
            "candidates",
            "candidate_segments",
            "segments",
            "roads",
        ),
    )

    @field_validator("timestamp", mode="before")
    @classmethod
    def _coerce_epoch(cls, value: Any) -> Any:
        """Accept epoch seconds or milliseconds alongside ISO-8601 strings.

        Values with a magnitude of at least 1e11 are treated as milliseconds
        (that is roughly year 5138 in seconds), which keeps both conventions
        unambiguous for realistic fleet data.
        """

        if isinstance(value, bool):
            raise ValueError("timestamp must not be a boolean")
        if isinstance(value, (int, float)):
            seconds = value / 1000.0 if abs(value) >= 1e11 else float(value)
            try:
                return datetime.fromtimestamp(seconds, tz=UTC)
            except (OverflowError, OSError, ValueError) as error:
                raise ValueError(f"invalid epoch timestamp {value!r}") from error
        if isinstance(value, str):
            text = value.strip()
            if text and (text.isdigit() or _is_signed_number(text)):
                return cls._coerce_epoch(float(text))
        return value

    @field_validator("timestamp")
    @classmethod
    def _normalize_utc(cls, value: datetime) -> datetime:
        # Naive timestamps are interpreted as UTC; aware timestamps are
        # converted to UTC so ordering is compared on one timeline.
        if value.tzinfo is None:
            return value.replace(tzinfo=UTC)
        return value.astimezone(UTC)

    def normalized_candidates(self) -> list[Candidate]:
        """Drop duplicate segment entries, keeping the smallest offset."""

        seen: dict[str, Candidate] = {}
        order: list[str] = []
        for candidate in self.candidates:
            previous = seen.get(candidate.segment_id)
            if previous is None:
                seen[candidate.segment_id] = candidate
                order.append(candidate.segment_id)
            elif candidate.offset < previous.offset:
                seen[candidate.segment_id] = candidate
        return [seen[segment_id] for segment_id in order]


class DirectedSegment(BaseModel):
    """A directed road segment with a positive integer length."""

    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    id: str = Field(
        validation_alias=AliasChoices("id", "segment_id", "road_id")
    )
    start_node: str = Field(
        validation_alias=AliasChoices(
            "start_node", "start", "from", "from_node", "source"
        )
    )
    end_node: str = Field(
        validation_alias=AliasChoices(
            "end_node", "end", "to", "to_node", "target"
        )
    )
    length: PositiveInt = Field(
        validation_alias=AliasChoices("length", "length_m", "meters")
    )

    @field_validator("id", "start_node", "end_node", mode="before")
    @classmethod
    def _normalize_identifiers(cls, value: Any, info: Any) -> str:
        return _clean_identifier(value, what=str(info.field_name))


class MatchRequest(BaseModel):
    """A trajectory and the directed road network it should be matched to."""

    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    points: list[LocationPoint] = Field(
        min_length=2,
        max_length=12,
        validation_alias=AliasChoices("points", "locations", "positions"),
    )
    segments: list[DirectedSegment] = Field(
        max_length=30,
        validation_alias=AliasChoices("segments", "roads", "network"),
    )

    @model_validator(mode="before")
    @classmethod
    def _accept_compact_candidates(cls, data: Any) -> Any:
        """Allow ``["S1", 3]`` or ``[["S1", 3]]`` alongside objects.

        A candidate may be given as ``{"segment_id": ..., "offset": ...}``,
        as a two-item ``[segment_id, offset]`` pair, or as a bare segment id
        with offset ``0``.
        """

        if not isinstance(data, dict):
            return data
        points = next(
            (data[key] for key in ("points", "locations", "positions")
             if key in data),
            None,
        )
        if not isinstance(points, list):
            return data

        normalized_points = []
        for point in points:
            if not isinstance(point, dict):
                normalized_points.append(point)
                continue
            point = dict(point)
            alias = next(
                (
                    key
                    for key in ("candidates", "candidate_segments",
                                "segments", "roads")
                    if key in point
                ),
                None,
            )
            candidates = point[alias] if alias is not None else None
            if isinstance(candidates, list):
                point_candidates = []
                for candidate in candidates:
                    if isinstance(candidate, str) or isinstance(candidate, int):
                        point_candidates.append(
                            {"segment_id": candidate, "offset": 0}
                        )
                    elif isinstance(candidate, (list, tuple)):
                        if len(candidate) == 1:
                            point_candidates.append(
                                {"segment_id": candidate[0], "offset": 0}
                            )
                        elif len(candidate) == 2:
                            point_candidates.append(
                                {
                                    "segment_id": candidate[0],
                                    "offset": candidate[1],
                                }
                            )
                        else:
                            point_candidates.append(candidate)
                    else:
                        point_candidates.append(candidate)
                # Update the exact key the caller used so validation aliases
                # never see two competing representations.
                if point_candidates or alias is not None:
                    point[alias] = point_candidates
            normalized_points.append(point)

        data = dict(data)
        for key in ("points", "locations", "positions"):
            if key in data:
                data[key] = normalized_points
                break
        return data

    @model_validator(mode="after")
    def _validate_consistency(self) -> MatchRequest:
        previous: datetime | None = None
        for index, point in enumerate(self.points):
            if previous is not None and point.timestamp <= previous:
                raise ValueError(
                    f"points[{index}].timestamp must be strictly later than "
                    f"points[{index - 1}].timestamp"
                )
            previous = point.timestamp

        segment_ids = [segment.id for segment in self.segments]
        duplicates = sorted(
            segment_id
            for segment_id, count in Counter(segment_ids).items()
            if count > 1
        )
        if duplicates:
            raise ValueError(
                "duplicate segment id(s) in network: " + ", ".join(duplicates)
            )

        known = set(segment_ids)
        for point_index, point in enumerate(self.points):
            for candidate_index, candidate in enumerate(point.candidates):
                if candidate.segment_id not in known:
                    raise ValueError(
                        f"points[{point_index}].candidates[{candidate_index}] "
                        f"references unknown segment "
                        f"{candidate.segment_id!r}"
                    )
        return self
