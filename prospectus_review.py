"""Human review persistence for the indexed Task 2 benchmark."""
import csv
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import prospectus_evaluation as ev
import prospectus_task2 as task2

KEYS = {
    "passages": ("question_id", "method", "rank", "chunk_id"),
    "filing_truth": ("question_id",),
    "answers": ("question_id", "case_id"),
    "heldout": ("question_id", "case_id"),
}
FIELDS = {
    "passages": ("relevance_label", "review_notes"),
    "filing_truth": ("verified_relevant_files", "complete_verified", "review_notes"),
    "answers": (*ev.LABELS, "review_notes"),
    "heldout": ("decision", "review_notes"),
}


def latest_run(directory, index_digest):
    required = ("questions.json", "passages.csv", "filing_truth.csv", "answers.csv",
                "candidate_evidence.csv", "rank_comparison.csv", "retrieval_metrics.csv")
    for path in sorted(directory.glob("run_*/summary.json"),
                       key=lambda p: p.stat().st_mtime_ns, reverse=True):
        try:
            summary = json.loads(path.read_text())
            if (summary["index"]["faiss_sha256"] == index_digest
                    and all((path.parent / name).is_file() for name in required)):
                return path.parent, summary
        except (OSError, ValueError, KeyError, TypeError):
            continue
    return None, None


def key(kind, row):
    return tuple(str(row[field]) for field in KEYS[kind])


def inputs(directory, kind):
    rows = ev.review_rows(directory / f"{kind}.csv")
    result = {key(kind, row): row for row in rows}
    if len(result) != len(rows):
        raise ValueError("Duplicate human review identities")
    return result


def fingerprint(kind, row):
    fields = {
        "passages": ("question", "file_name", "filing_date", "doc_url", "text"),
        "filing_truth": ("question", "proposed_files"),
        "answers": ("question", "status", "answer", "evidence_json"),
        "heldout": ("question", "file", "evidence"),
    }
    return ev.digest([str(row.get(field, "")) for field in (*KEYS[kind], *fields[kind])])


def save(directory, kind, row, values, expected, current_rows, complete_files):
    """Explicit optimistic revisions with a durable record of the replaced judgment."""
    identity = key(kind, row)
    with ev.advisory_lock(directory, ".review.lock"):
        current = next((candidate for candidate in current_rows() if key(kind, candidate) == identity), None)
        source_hash = fingerprint(kind, row)
        if current is None or fingerprint(kind, current) != source_hash:
            raise ValueError("This review item changed. Reload the current evaluation and review it again.")
        if kind == "answers" and current["status"] != "ready_for_review":
            raise ValueError("This answer is still awaiting generation.")
        previous = inputs(directory, kind)
        old = previous.get(identity)
        if old != expected:
            raise ValueError("Another submission changed this review. Reload it before revising.")
        if old and old.get("source_digest") and old["source_digest"] != source_hash:
            raise ValueError("The saved judgment belongs to a different evidence version.")
        normalized = {field: str(values.get(field, "")).strip() for field in FIELDS[kind]}
        if kind == "passages" and normalized["relevance_label"] not in {"0", "1", "2"}:
            raise ValueError("Choose a relevance label.")
        if kind == "filing_truth":
            files = {f.strip() for f in normalized["verified_relevant_files"].split(";") if f.strip()}
            if not files <= complete_files or normalized["complete_verified"] not in {"yes", "no"}:
                raise ValueError("Choose valid indexed filenames and a completeness status.")
            if normalized["complete_verified"] == "yes" and not files:
                raise ValueError("Complete positive ground truth requires at least one verified filing.")
            normalized["verified_relevant_files"] = "; ".join(sorted(files))
        if kind == "answers":
            for field in ev.LABELS:
                allowed = {"PASS", "PARTIAL", "FAIL"} | ({"NA"} if field == "appropriate_abstention" else set())
                if normalized[field] not in allowed:
                    raise ValueError("Score every rubric dimension.")
        if kind == "heldout" and normalized["decision"] not in {"approved", "needs_revision"}:
            raise ValueError("Select a held-out approval decision.")
        updated = dict(zip(KEYS[kind], identity)) | normalized | {"source_digest": source_hash}
        if old == updated:
            return
        stamp = datetime.now(timezone.utc).isoformat()
        revision = {"at": stamp, "before": old, "after": updated}
        ev.write_once(directory / "revisions" / f"{kind}_{ev.digest(revision)}.json",
                      json.dumps(revision, ensure_ascii=False, indent=2))
        previous[identity] = updated
        fields = tuple(dict.fromkeys((*KEYS[kind], *FIELDS[kind], "source_digest",
                                     *(field for record in previous.values() for field in record))))
        target = directory / f"{kind}.csv"
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="", dir=directory,
                                         delete=False) as stream:
            temporary = Path(stream.name)
            writer = csv.DictWriter(stream, fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(previous.values())
        os.replace(temporary, target)


def check_input_versions(directory, kind, rows):
    expected = {key(kind, row): fingerprint(kind, row) for row in rows}
    for identity, row in inputs(directory, kind).items():
        if identity in expected and row.get("source_digest") not in (None, "", expected[identity]):
            raise ValueError("A human judgment refers to a different evidence version. Restore its matching run before scoring.")


def export_review(run, summary, complete_files):
    cases = json.loads((run / "questions.json").read_text())
    passages = ev.review_rows(run / "passages.csv")
    answers = ev.review_rows(run / "answers.csv")
    human = run.parent / "human"
    check_input_versions(human, "passages", passages)
    check_input_versions(human, "filing_truth", ev.review_rows(run / "filing_truth.csv"))
    check_input_versions(human, "answers", answers)
    for row in passages:
        row["rank"] = int(row["rank"])
        row["primary"] = row["primary"] == "True"
    return task2.export(run.parent, summary["index"], summary["benchmark_version"], cases, passages,
                        ev.review_rows(run / "candidate_evidence.csv"),
                        ev.review_rows(run / "rank_comparison.csv"),
                        answers, complete_files)


def approved_heldout(directory, version, rows, index, lookup):
    decisions = inputs(directory / "human", "heldout")
    if not rows or any(
        decisions.get(key("heldout", row), {}).get("decision") != "approved"
        or decisions.get(key("heldout", row), {}).get("source_digest") != fingerprint("heldout", row)
        for row in rows
    ):
        raise ValueError("All proposed questions and their evidence need human approval.")
    cases = []
    for row in rows:
        files = sorted({lookup[cid]["file_name"] for cid in row["evidence"]})
        cases.append(dict(id=row["question_id"], question=row["question"],
                          file=files[0] if len(files) == 1 else None,
                          approved_candidate_chunks=row["evidence"]))
    payload = dict(version=version, index=index, cases=cases)
    path = directory / f"approved_{ev.digest(payload)[:16]}.json"
    return ev.write_once(path, json.dumps(payload, ensure_ascii=False, indent=2))


def load_ai_passage_reviews(path, passages, cases):
    """Join offline suggestions to exactly the primary positive benchmark evidence."""
    positive = {case["id"] for case in cases if not case["abstention"]}
    primary = [row for row in passages if row["primary"] in ("True", True)
               and row["question_id"] in positive and int(row["rank"]) <= 5]
    by_key = {key("passages", row): row for row in primary}
    suggestions = ev.review_rows(path)
    ai_keys = [key("passages", row) for row in suggestions]
    if len(ai_keys) != len(set(ai_keys)) or set(ai_keys) != set(by_key) or len(primary) != len(by_key):
        raise ValueError("The AI suggestions do not match the current primary passage queue.")
    for item, identity in zip(suggestions, ai_keys):
        passage = by_key[identity]
        if (item["source_digest"] != fingerprint("passages", passage)
                or any(item[field] != passage[field] for field in
                       ("question", "file_name", "deal_name", "filing_date", "doc_url"))
                or item["relevance_score"] not in {"0", "1", "2"}
                or item["confidence"] not in {"High", "Medium", "Low"}
                or item["human_review_recommended"] not in {"yes", "no"}
                or not item["reasoning"].strip()
                or not item["supporting_excerpt"]
                or item["supporting_excerpt"] not in passage["text"]):
            raise ValueError(f"AI suggestion has invalid evidence or metadata: {identity}")
    return [(item, by_key[identity]) for item, identity in zip(suggestions, ai_keys)]


def ai_precision(suggestions):
    # Use the existing strict Precision@5 scorer, in memory only.
    return ev.reviewed_precision([
        item | {"relevance_label": item["relevance_score"]} for item in suggestions
    ])
