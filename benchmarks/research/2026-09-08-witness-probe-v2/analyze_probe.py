"""Build a compact, mechanical adjudication view of the frozen witness probe."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
ARTIFACT = Path(__file__).resolve().parent
OUTPUT = ROOT / ".benchmarks/results/e2e-witness-probe-v2"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sse_content(body: str) -> tuple[str, str | None, dict[str, object] | None]:
    parts: list[str] = []
    finish_reason = None
    usage = None
    for line in body.splitlines():
        if not line.startswith("data: ") or line == "data: [DONE]":
            continue
        value = json.loads(line.removeprefix("data: "))
        if value.get("usage") is not None:
            usage = value["usage"]
        for choice in value.get("choices") or ():
            delta = choice.get("delta") or {}
            if isinstance(delta.get("content"), str):
                parts.append(delta["content"])
            if choice.get("finish_reason") is not None:
                finish_reason = choice["finish_reason"]
    return "".join(parts), finish_reason, usage


def _formation_transport(path: Path) -> dict[str, object]:
    for line in path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if row["client"] != "generation":
            continue
        request = json.loads(row["request_body"])
        system = str(request["messages"][0]["content"])
        if not system.startswith("Form typed memories"):
            continue
        content, finish_reason, usage = _sse_content(row["response_body"])
        return {
            "request_sha256": row["request_sha256"],
            "response_sha256": row["response_sha256"],
            "raw_assistant_content": content,
            "finish_reason": finish_reason,
            "usage": usage,
        }
    raise RuntimeError(f"formation transport absent: {path}")


def _case(path: Path) -> dict[str, object]:
    source = json.loads(path.read_text(encoding="utf-8"))
    tables = source["store"]["tables"]
    source_ids = source["source_record_ids"]
    sources = set(source_ids)
    semantics = {row["memory_id"]: row for row in tables["memory_semantics"]}
    records = {row["memory_id"]: row for row in tables["memory_records"]}
    clauses: dict[str, list[dict[str, object]]] = {}
    members_by_clause: dict[tuple[str, str], list[dict[str, object]]] = {}
    for row in tables.get("memory_evidence_clause_members", []):
        members_by_clause.setdefault((row["memory_id"], row["clause_id"]), []).append(row)
    for row in tables.get("memory_evidence_clauses", []):
        if row["retired_at"] is not None:
            continue
        members = sorted(
            members_by_clause[(row["memory_id"], row["clause_id"])],
            key=lambda value: value["position"],
        )
        clauses.setdefault(row["memory_id"], []).append(
            {
                "clause_id": row["clause_id"],
                "primary_witness_id": members[0]["source_memory_id"],
                "extra_witness_ids": [value["source_memory_id"] for value in members[1:]],
                "all_witness_ids": [value["source_memory_id"] for value in members],
                "confidence": row["confidence"],
            }
        )
    if not clauses:
        for row in tables["memory_evidence"]:
            if row["retired_at"] is None:
                clauses.setdefault(row["memory_id"], []).append(
                    {
                        "clause_id": None,
                        "primary_witness_id": row["source_memory_id"],
                        "extra_witness_ids": [],
                        "all_witness_ids": [row["source_memory_id"]],
                        "confidence": row["confidence"],
                    }
                )
    claims = []
    targets = {source_ids[index] for index in source["target_dependencies"]}
    for memory_id, record in records.items():
        if memory_id in sources:
            continue
        claim_clauses = clauses.get(memory_id, [])
        claims.append(
            {
                "memory_id": memory_id,
                "content": record["content"],
                "kind": semantics[memory_id]["kind"],
                "subject": semantics[memory_id]["subject"],
                "predicate": semantics[memory_id]["predicate"],
                "value": semantics[memory_id]["value"],
                "witness_clauses": claim_clauses,
                "has_target_dependency_clause": any(
                    targets.issubset(set(clause["all_witness_ids"])) for clause in claim_clauses
                ),
            }
        )
    bundle_ids = {hit["id"] for hit in source["compile"]["bundle_hits"]}
    incomplete = [
        hit["id"]
        for hit in source["compile"]["bundle_hits"]
        if not set(hit["evidence_ids"]).issubset(bundle_ids)
    ]
    transport = Path(source["transport_journal"])
    return {
        "arm": source["arm"],
        "case_id": source["case_id"],
        "question": source["question"],
        "observations": [
            {"index": index, "source_memory_id": memory_id, "text": text}
            for index, (memory_id, text) in enumerate(
                zip(source_ids, source["observations"], strict=True)
            )
        ],
        "target_dependencies": source["target_dependencies"],
        "target_source_ids": sorted(targets),
        "claims": claims,
        "claim_count": len(claims),
        "claims_with_target_dependency_clause": sum(
            bool(claim["has_target_dependency_clause"]) for claim in claims
        ),
        "raw_ask": source["raw_ask"],
        "compile_answer": source["compile"]["answer"],
        "compile_bundle_hit_ids": [hit["id"] for hit in source["compile"]["bundle_hits"]],
        "compile_bundle_incomplete_derived_ids": incomplete,
        "formation_transport": _formation_transport(transport),
        "result_sha256": _sha256(path),
        "transport_sha256": _sha256(transport),
    }


def main() -> None:
    cases = [
        _case(path)
        for path in sorted((OUTPUT / "cases").glob("*/*.json"))
        if not path.name.endswith(".failure.json")
    ]
    if len(cases) != 16:
        raise SystemExit(f"expected 16 complete case/arm outputs, got {len(cases)}")
    payload = {
        "schema_version": 1,
        "interpretation": {
            "mechanical_only": True,
            "target_clause_note": (
                "A target-dependency clause proves declared joint attribution only. Separate "
                "correctly attributed atomic claims can still support a correct answer; manual "
                "semantic adjudication decides whether any claim actually resolves the question."
            ),
        },
        "cases": cases,
        "counts": {
            arm: {
                "cases": sum(case["arm"] == arm for case in cases),
                "claims": sum(case["claim_count"] for case in cases if case["arm"] == arm),
                "claims_with_target_dependency_clause": sum(
                    case["claims_with_target_dependency_clause"]
                    for case in cases
                    if case["arm"] == arm
                ),
                "cases_with_target_dependency_clause": sum(
                    case["claims_with_target_dependency_clause"] > 0
                    for case in cases
                    if case["arm"] == arm
                ),
                "compile_bundles_with_incomplete_derived": sum(
                    bool(case["compile_bundle_incomplete_derived_ids"])
                    for case in cases
                    if case["arm"] == arm
                ),
            }
            for arm in ("pre_witness", "witness")
        },
    }
    destination = OUTPUT / "adjudication.json"
    destination.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps({"path": str(destination), "sha256": _sha256(destination), **payload["counts"]})
    )


if __name__ == "__main__":
    main()
