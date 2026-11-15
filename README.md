# Trajectory → Directed Road Matcher

FastAPI service that assigns a short vehicle trajectory to a directed road
network. It is meant for a fleet data warehouse: given a few GPS fixes with
candidate directed segments and lateral offsets, it returns the complete
ordered segment sequence so downstream jobs can aggregate traffic by road
segment.

## Run it

```bash
docker compose up --build -d
curl -s http://localhost:8000/health
curl -s -X POST http://localhost:8000/match \
  -H 'content-type: application/json' \
  --data-binary @examples/request.json
```

Interactive OpenAPI docs: <http://localhost:8000/docs>.
Stop with `docker compose down`.

Without Docker:

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements-dev.txt
python -m app            # or: uvicorn app.main:app --port 8000
pytest
```

Three ready-to-post payloads live in `examples/`: `request.json` (unique
optimum), `ambiguous_request.json`, and `unmatched_request.json`.

## Request

`POST /match`

```json
{
  "points": [
    {
      "timestamp": "2026-01-01T00:00:00Z",
      "candidates": [
        {"segment_id": "W1", "offset": 4},
        {"segment_id": "E1", "offset": 9}
      ]
    },
    {
      "timestamp": "2026-01-01T00:00:40Z",
      "candidates": [{"segment_id": "W2", "offset": 2}]
    }
  ],
  "segments": [
    {"id": "W1", "start_node": "N1", "end_node": "N2", "length": 120},
    {"id": "W2", "start_node": "N2", "end_node": "N3", "length": 150},
    {"id": "E1", "start_node": "N1", "end_node": "N5", "length": 60}
  ]
}
```

| Field | Rules |
| --- | --- |
| `points` | 2–12 items, strictly increasing UTC timestamps |
| `points[].candidates` | 1–4 candidate segments per point |
| `candidates[].segment_id` | must exist in `segments` |
| `candidates[].offset` | non-negative integer lateral deviation |
| `segments` | at most 30 directed segments, unique ids |
| `length` | positive integer; the cost unit for traversal |

Naive timestamps are treated as UTC and aware timestamps are normalized to
UTC before ordering, so `01:00+01:00` correctly precedes `00:30Z`.
Extra JSON fields are ignored. A candidate can also be written compactly as
`["W1", 4]` or just `"W1"` (offset 0). Boolean values are rejected for
integer fields; integral floats such as `4.0` are accepted.

## Cost model

`total = Σ chosen candidate offsets + Σ length of every entered segment`

* Consecutive points are joined along segment direction
  (`seg.end_node == next.start_node`).
* Between two consecutive points, at most **two segments without a location
  point** may be traversed.
* Two consecutive points on the same segment count that segment's length
  **once**; leaving and re-entering later counts it again.
* All lengths are positive, so only the cheapest continuation for each
  (from, to) segment pair can participate in an optimum. The matcher caches
  those continuations and runs an exact dynamic program over
  `(point, chosen segment)` states.

## Response

```json
{
  "status": "matched",
  "ambiguous": false,
  "optimal_cost": 369,
  "segment_sequence": ["W1", "W2", "W3"],
  "assignments": ["W1", "W2", "W3"],
  "offset_cost": 9,
  "length_cost": 360,
  "witnesses": [],
  "reason": null,
  "message": "Unique optimal directed road sequence found for all 3 location points."
}
```

`status` has three values:

| `status` | Meaning | Payload |
| --- | --- | --- |
| `matched` | exactly one optimal sequence | `segment_sequence`, `assignments`, cost split |
| `ambiguous` | ≥2 distinct sequences tie for the minimum | `witnesses` with two optimal sequences |
| `unmatched` | no directed route satisfies the constraints | `reason`, no partial result |

For `unmatched`, every route field is `null` (or `[]`) — the service never
returns a best-effort prefix. `reason` is currently
`no_connected_route`.

An ambiguous witness contains its full `segment_sequence`, the per-point
`assignments`, and the `offset_cost` / `length_cost` / `total_cost` split.
Ambiguity is judged on the **road sequence**, not on which candidate a point
landed on: if the traversed sequence is identical, the result stays
`matched`.

## Tests

`pytest` runs 37 tests, including:

* a randomized cross-check of the dynamic program against an independent
  exhaustive solver over 600 generated networks (the reference solver also
  enumerates self-intersecting walks, confirming that pruning them is safe);
* direction, two-intermediate-segment limit, same-segment deduplication,
  loop handling, ambiguity witnesses, versioned input validation, and
  no-partial-result guarantees.

## Layout

```
app/schemas.py    request validation and input normalization
app/matcher.py    exact dynamic program and witness extraction
app/main.py       FastAPI routes and response models
tests/            API and brute-force equivalence tests
examples/         ready-to-post payloads (matched/ambiguous/unmatched)
Dockerfile        slim, non-root, health-checked image
docker-compose.yml
```

## Notes and limits

* The service is stateless; each request carries its own network.
* Work is polynomial: `O(points × candidates² × 30²)` with tiny constants;
  a maximum-size dense request matches in a few milliseconds.
* Ambiguity is reported when **any** two distinct optimal full sequences
  exist, even if they already diverge in an earlier gap.
* `offset` and `length` share one integer cost unit, as specified. Clients
  should normalize units before sending if their lateral deviation and road
  length use different scales.
