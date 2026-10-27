"""Score offline atomic-claim reviews without changing the memory admission policy.

Gold judgments and locations must be independently annotated. Predicted judgments are verifier
outputs, while admission and lifecycle fields are observed SDK outcomes. Neither a legal citation
nor a verifier's own confidence supplies ground truth. This module makes no model or store calls.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StrictBool, StringConstraints, model_validator

Label = Literal["entailment", "contradiction", "unknown"]
_Text = Annotated[str, StringConstraints(strict=True, strip_whitespace=True, min_length=1)]
_Count = Annotated[int, Field(strict=True, ge=0)]
_Milliseconds = Annotated[float, Field(strict=True, ge=0, allow_inf_nan=False)]
_LABELS: tuple[Label, ...] = ("entailment", "contradiction", "unknown")


class _ReviewValue(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class EvidenceLocation(_ReviewValue):
    """Half-open Unicode offsets, zero-based frame indices, or audio milliseconds.

    Frame intervals name exactly one frame (a still image uses frame zero). Media coordinates
    also require the original asset ID, so two attachments of one memory cannot match by accident.
    """

    source_id: _Text
    kind: Literal["text", "frame", "audio"]
    start: _Count
    end: _Count
    asset_id: _Text | None = None

    @model_validator(mode="after")
    def _coordinates(self) -> EvidenceLocation:
        if self.end <= self.start:
            raise ValueError("location intervals must be non-empty")
        if self.kind == "text":
            if self.asset_id is not None:
                raise ValueError("text locations name the source record, not an asset")
        elif self.asset_id is None:
            raise ValueError("media locations require an asset ID")
        if self.kind == "frame" and self.end != self.start + 1:
            raise ValueError("a frame location must name exactly one frame")
        return self


class LifecycleReview(_ReviewValue):
    """A reviewed post-operation visibility obligation, distinct from cognitive forgetting."""

    kind: Literal["withdrawal", "soft_forgetting"]
    expected_absent: StrictBool
    visible_after: StrictBool | None = None


class ClaimReview(_ReviewValue):
    """One independently annotated atomic claim and its actual admission outcome.

    Each inner gold-location tuple is a complete required set (AND); outer alternatives are OR.
    Unknown has no implied contradiction. Gold labels concern the supplied evidence snapshot,
    not world truth, and a source span's existence does not establish entailment.
    """

    claim_id: _Text
    gold: Label
    admitted: StrictBool | None
    predicted: Label | None = None
    gold_locations: tuple[tuple[EvidenceLocation, ...], ...] = ()
    predicted_locations: tuple[EvidenceLocation, ...] | None = None
    lifecycle: LifecycleReview | None = None
    error_code: _Text | None = None

    @model_validator(mode="after")
    def _observations(self) -> ClaimReview:
        if (self.error_code is None) != (self.admitted is not None):
            raise ValueError("failed claims require an error code and unknown admission")
        if self.lifecycle is not None and self.admitted is not True:
            raise ValueError("lifecycle tests require an initially admitted claim")
        for locations in self.gold_locations:
            if not locations or len(set(locations)) != len(locations):
                raise ValueError("gold alternatives must be non-empty sets of unique locations")
        predicted = self.predicted_locations
        if predicted is not None and len(set(predicted)) != len(predicted):
            raise ValueError("predicted locations must be unique")
        return self


class AuditCost(_ReviewValue):
    """Measured totals for one whole run; omitted measurements remain unknown.

    Record shared/batched requests once here, never once per atomic claim. Latency is wall time
    for the stated phase; tokens and requests count actual attempts, including retries.
    """

    latency_ms: _Milliseconds | None = None
    requests: _Count | None = None
    input_tokens: _Count | None = None
    output_tokens: _Count | None = None


class AuditRun(_ReviewValue):
    fixture_version: _Text
    producer_revision: _Text | None = None
    reviewer_revision: _Text | None = None
    claims: Annotated[tuple[ClaimReview, ...], Field(min_length=1)]
    write_cost: AuditCost | None = None
    verification_cost: AuditCost | None = None

    @model_validator(mode="after")
    def _identities(self) -> AuditRun:
        if len({claim.claim_id for claim in self.claims}) != len(self.claims):
            raise ValueError("claim IDs must be unique within a run")
        if self.reviewer_revision is None and any(
            claim.predicted is not None or claim.predicted_locations is not None
            for claim in self.claims
        ):
            raise ValueError("verifier predictions require a reviewer revision")
        return self


def _rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def score_audit(run: AuditRun) -> dict[str, object]:
    """Return aggregate metrics and denominators without echoing content or source IDs.

    Missing predictions reduce coverage; if no predictions exist, verifier quality is unmeasured.
    Failed admission is not safe rejection. Missing lifecycle observations never count as absence.
    """
    claims = run.claims
    admitted = [claim for claim in claims if claim.admitted is True]
    supported = [claim for claim in claims if claim.gold == "entailment"]
    contradictions = [claim for claim in claims if claim.gold == "contradiction"]
    judged = [claim for claim in claims if claim.predicted is not None]
    located = [claim for claim in claims if claim.gold_locations]
    attempted_locations = [claim for claim in located if claim.predicted_locations is not None]
    correct_locations = sum(
        frozenset(claim.predicted_locations or ())
        in {frozenset(alternative) for alternative in claim.gold_locations}
        for claim in attempted_locations
    )
    withdrawals = [
        claim.lifecycle
        for claim in claims
        if claim.lifecycle is not None and claim.lifecycle.kind == "withdrawal"
    ]
    should_disappear = [event for event in withdrawals if event.expected_absent]
    checked_absence = [event for event in should_disappear if event.visible_after is not None]
    should_survive = [event for event in withdrawals if not event.expected_absent]
    checked_presence = [event for event in should_survive if event.visible_after is not None]
    return {
        "schema_version": 1,
        "evaluation_scope": "offline_annotated_atomic_claims",
        "fixture_version": run.fixture_version,
        "producer_revision": run.producer_revision,
        "reviewer_revision": run.reviewer_revision,
        "claims": len(claims),
        "admitted_claims": len(admitted),
        "failed_claims": sum(claim.error_code is not None for claim in claims),
        "admission_coverage": _rate(
            sum(claim.admitted is not None for claim in claims), len(claims)
        ),
        "unsupported_assertion_rate": _rate(
            sum(claim.gold != "entailment" for claim in admitted), len(admitted)
        ),
        "supported_retention_rate": _rate(
            sum(claim.admitted is True for claim in supported), len(supported)
        ),
        "classification_accuracy": _rate(
            sum(claim.predicted == claim.gold for claim in judged), len(judged)
        ),
        "classification_coverage": _rate(len(judged), len(claims)),
        "contradiction_recall": (
            _rate(
                sum(claim.predicted == "contradiction" for claim in contradictions),
                len(contradictions),
            )
            if judged
            else None
        ),
        "confusion": {
            gold: {
                predicted: sum(
                    claim.gold == gold and (claim.predicted or "unreviewed") == predicted
                    for claim in claims
                )
                for predicted in (*_LABELS, "unreviewed")
            }
            for gold in _LABELS
        },
        "localization_exact_match_rate": (
            _rate(correct_locations, len(located)) if attempted_locations else None
        ),
        "localization_coverage": _rate(len(attempted_locations), len(located)),
        "localization_annotated_claims": len(located),
        "withdrawal_residual_rate": _rate(
            sum(event.visible_after is True for event in checked_absence), len(checked_absence)
        ),
        "withdrawal_coverage": _rate(len(checked_absence), len(should_disappear)),
        "withdrawal_eligible_claims": len(should_disappear),
        "withdrawal_preservation_rate": _rate(
            sum(event.visible_after is True for event in checked_presence), len(checked_presence)
        ),
        "withdrawal_preservation_coverage": _rate(len(checked_presence), len(should_survive)),
        "soft_forgetting_cases": sum(
            claim.lifecycle is not None and claim.lifecycle.kind == "soft_forgetting"
            for claim in claims
        ),
        "write_cost": None if run.write_cost is None else run.write_cost.model_dump(),
        "verification_cost": (
            None if run.verification_cost is None else run.verification_cost.model_dump()
        ),
    }


def main(argv: Sequence[str] | None = None) -> int:
    """Score a caller-owned JSON review file; no provider calls or product writes occur."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    try:
        run = AuditRun.model_validate_json(args.input.read_text(encoding="utf-8"))
        payload = json.dumps(score_audit(run), indent=2, sort_keys=True, allow_nan=False) + "\n"
        if args.output is not None:
            args.output.write_text(payload, encoding="utf-8")
    except (OSError, ValueError):
        # Validation errors can include private input values; keep them out of ordinary output.
        print("Cannot score audit: invalid review data or inaccessible file.", file=sys.stderr)
        return 2
    print(payload, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
