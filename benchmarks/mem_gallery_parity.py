"""Rescore saved answers against the pinned official evaluator without calling a model."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import string
from collections import Counter, defaultdict
from pathlib import Path
from statistics import fmean
from typing import Any

import regex
from nltk.stem import PorterStemmer
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu

from mindbridge.benchmarks.official_scorers import local_scores, scorer_protocol

UPSTREAM_COMMIT = "a93959e1e978a6a7d77798ae92c2ffe41c538c62"
EVALUATOR_SHA256 = "8ed8fb242ba0a9bf7b0029a138864ebd4f0d520528e6ffdc5492543221f5ba00"
FUNCTIONS = frozenset(
    {
        "normalize_answer_universal",
        "f1_score",
        "bleu_score",
        "exact_match_score",
        "calculate_retrieval_metrics",
    }
)


def _official(path: Path) -> dict[str, Any]:
    # Verify the entire source before executing only its five audited pure functions. Imports,
    # CLI setup, judge construction and heavyweight learned scorers are never executed.
    source = path.read_bytes().rstrip(b"\n") + b"\n"
    if hashlib.sha256(source).hexdigest() != EVALUATOR_SHA256:
        raise ValueError("official evaluator differs from the pinned source")
    functions = [
        node
        for node in ast.parse(source).body
        if isinstance(node, ast.FunctionDef) and node.name in FUNCTIONS
    ]
    namespace: dict[str, Any] = {
        "regex": regex,
        "string": string,
        "Counter": Counter,
        "ps": PorterStemmer(),
        "SmoothingFunction": SmoothingFunction,
        "sentence_bleu": sentence_bleu,
        "List": list,
        "Dict": dict,
    }
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(path), "exec"), namespace)
    return namespace


def _official_scores(row: dict[str, Any], official: dict[str, Any]) -> dict[str, float]:
    prediction, reference = row["prediction"], row["references"][0]
    scores = {
        "f1": official["f1_score"](prediction, reference),
        "bleu": official["bleu_score"](prediction, reference),
        "bleu_1": official["bleu_score"](prediction, reference, weights=(1, 0, 0, 0)),
        "bleu_2": official["bleu_score"](prediction, reference, weights=(0.5, 0.5, 0, 0)),
        "exact_match": float(official["exact_match_score"](prediction, reference)),
    }
    retrieval = official["calculate_retrieval_metrics"](
        row["ranked_source_ids"], row["metadata"]["clue_ids"], k=10
    )
    scores.update(
        {
            "retrieval_precision@10": retrieval["precision_k"],
            "retrieval_recall@10": retrieval["recall_k"],
            "retrieval_hit@10": retrieval["hitrate_k"],
        }
    )
    return scores


def audit(samples: Path, evaluator: Path, expected_count: int) -> dict[str, Any]:
    official = _official(evaluator)
    with samples.open(encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream]
    rows = [r for r in rows if r["task"] == "mem-gallery" and r["arm"] == "mindbridge"]
    identities = {(r["unit_id"], r["question_id"]) for r in rows}
    if len(rows) != expected_count or len(identities) != len(rows):
        raise ValueError("missing or duplicate Mem-Gallery answers")
    current_mismatches: Counter[str] = Counter()
    stored_mismatches: Counter[str] = Counter()
    stored_missing: Counter[str] = Counter()
    values: dict[str, list[float]] = defaultdict(list)
    categories: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        expected = _official_scores(row, official)
        current = local_scores(
            "mem-gallery",
            score_kind="text",
            prediction=row["prediction"],
            references=tuple(row["references"]),
            parsed_choice=None,
            expected_choice=None,
            question=row["source_question"],
            metadata=row["metadata"],
            evidence_source_ids=row["ranked_source_ids"],
        )
        categories[row["metadata"]["point"]].append(expected["f1"])
        for name, score in expected.items():
            values[name].append(score)
            if name not in current or not math.isclose(score, current[name], abs_tol=1e-12):
                current_mismatches[name] += 1
            if name not in row["metrics"]:
                stored_missing[name] += 1
            elif not math.isclose(score, row["metrics"][name], abs_tol=1e-12):
                stored_mismatches[name] += 1
    return {
        "scope": "Fixed predictions; eight deterministic metrics; no judge or model calls",
        "upstream_commit": UPSTREAM_COMMIT,
        "official_evaluator_sha256": EVALUATOR_SHA256,
        "samples_sha256": hashlib.sha256(samples.read_bytes()).hexdigest(),
        "scorer_protocol": scorer_protocol("mem-gallery"),
        "dataset_sha256": sorted({r["dataset_sha256"] for r in rows}),
        "evaluation_sha256": sorted({r["evaluation_sha256"] for r in rows}),
        "questions": len(rows),
        "topics": len({r["unit_id"] for r in rows}),
        "answer_errors": sum(r["error_code"] is not None for r in rows),
        "current_mismatches": dict(current_mismatches),
        "stored_mismatches": dict(stored_mismatches),
        "stored_missing": dict(stored_missing),
        "official_means": {name: fmean(scores) for name, scores in values.items()},
        "categories": {
            point: {"questions": len(scores), "f1": fmean(scores)}
            for point, scores in sorted(categories.items())
        },
        "passed": not current_mismatches,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--official-evaluator", type=Path, required=True)
    parser.add_argument("--expected-count", type=int, default=1711)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = audit(args.samples, args.official_evaluator, args.expected_count)
    args.output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps({key: report[key] for key in ("questions", "current_mismatches", "passed")}))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
