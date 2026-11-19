"""Scoped Task 2 benchmark using the existing notebook index and evaluation helpers."""
import json
from pathlib import Path

import prospectus_evaluation as ev


def load_cases(path, questions, target_file, gold):
    config = json.loads(Path(path).read_text())
    original = {row["id"]: row for row in questions}
    cases = []
    for qid in config["original_ids"]:
        source = original[qid]
        cases.append(dict(id=qid, question=source["question"],
                          file=target_file if qid in gold else None,
                          evidence={target_file: sorted(gold[qid])} if qid in gold else {},
                          issuers=[x.strip() for x in source["expected_issuers"].split(";") if x.strip()],
                          abstention=qid == "q19"))
    cases.extend(dict(case, issuers=[], abstention=False) for case in config["supplemental"])
    if not 15 <= len(cases) <= 20 or len({case["id"] for case in cases}) != len(cases):
        raise ValueError("Task 2 requires 15–20 unique question IDs")
    return config["version"], cases


def _check_rows(rows, lookup, scope):
    for row in rows:
        chunk = lookup[row["chunk_id"]]
        if chunk["file_name"] not in scope or any(
            row[k] != chunk[k] for k in ("file_name", "doc_url", "text")
        ):
            raise ValueError(f"Invalid retrieval source: {row['chunk_id']}")


def replay(cases, chunks, complete_files, vectorizer, matrix, get_issuer, baseline, improved, semantic=None):
    lookup = {chunk["chunk_id"]: chunk for chunk in chunks}
    numbered = {(chunk["file_name"], chunk["chunk_index"]): chunk for chunk in chunks}
    items, proposals, comparisons, chosen = [], [], [], {}
    for case in cases:
        qid, question, target = (case[k] for k in ("id", "question", "file"))
        scope = {target} if target else complete_files
        if not scope <= complete_files:
            raise ValueError(f"Benchmark filing is outside the complete index: {qid}")
        eligible = [i for i, chunk in enumerate(chunks) if chunk["file_name"] in scope]
        order = ev.keyword_order(vectorizer, matrix[eligible], question)
        lexical = [eligible[i] for i in order]
        revised = ev.true_sale_order(question, target, lexical, vectorizer, matrix, eligible)
        methods = {"tfidf": [chunks[i] for i in lexical]}
        if revised != lexical:
            methods["tfidf_true_sale"] = [chunks[i] for i in revised[:8]]
        if qid in baseline:
            methods["cached_hybrid"] = baseline[qid]
        if semantic and qid in semantic:
            methods["approved_semantic"] = semantic[qid]
        if qid == "q18":
            methods["cached_hybrid_true_sale_replay"] = improved[qid]
        primary = next(
            method for method in ("cached_hybrid_true_sale_replay", "cached_hybrid",
                                  "tfidf", "tfidf_true_sale") if method in methods
        )
        chosen[qid] = methods[primary] if primary == "cached_hybrid" else methods[primary][:8]
        proposed = [numbered[file, number] for file, numbers in case["evidence"].items() for number in numbers]
        _check_rows(proposed, lookup, scope)
        candidate_ids = {row["chunk_id"] for row in proposed}
        candidate_files = set(case["evidence"]) | {
            file for file in scope if get_issuer(file) in case["issuers"]
        }
        for file in sorted(candidate_files):
            evidence = [row for row in proposed if row["file_name"] == file]
            for row in evidence or [None]:
                proposals.append(dict(
                    question_id=qid, question=question, file_name=file,
                    chunk_id=row["chunk_id"] if row else "",
                    doc_url=(row or next(chunk for chunk in chunks if chunk["file_name"] == file))["doc_url"],
                    passage=row["text"] if row else "", human_verified="",
                ))
        for method, hits in methods.items():
            _check_rows(hits, lookup, scope)
            if len({row["chunk_id"] for row in hits}) != len(hits):
                raise ValueError(f"Duplicate retrieval IDs: {qid} {method}")
            top_files = list(dict.fromkeys(row["file_name"] for row in hits))[:5]
            for rank, row in enumerate(ev.evidence_rows(hits[:8]), 1):
                items.append(dict(question_id=qid, question=question, method=method,
                                  primary=method == primary, rank=rank,
                                  filing_top5="; ".join(top_files), **row,
                                  relevance_label="", review_notes=""))
            if candidate_ids:
                comparisons.append(dict(
                    question_id=qid, method=method,
                    evidence_status="existing_gold" if qid in {"q17", "q18"} else "proposal_unverified",
                    matched_at_5=len(candidate_ids & {row["chunk_id"] for row in hits[:5]}),
                    matched_at_8=len(candidate_ids & {row["chunk_id"] for row in hits[:8]}),
                    candidate_count=len(candidate_ids),
                ))
    return items, proposals, comparisons, chosen


def review_key(row):
    return tuple(str(row[field]) for field in ("question_id", "method", "rank", "chunk_id"))


def score(cases, review, truths, complete_files):
    """Only complete human filing sets and five ranked human labels enter scores."""
    case_by_id = {case["id"]: case for case in cases}
    truth_by_id = {}
    seen = set()
    for row in truths:
        qid = row["question_id"]
        if qid not in case_by_id or qid in seen:
            raise ValueError(f"Unknown or duplicate filing truth: {qid}")
        seen.add(qid)
        if row["complete_verified"].strip().lower() != "yes":
            continue
        files = {f.strip() for f in row["verified_relevant_files"].split(";") if f.strip()}
        if case_by_id[qid]["file"] or case_by_id[qid]["abstention"] or not files or not files <= complete_files:
            raise ValueError(f"Invalid verified filing set: {qid}")
        truth_by_id[qid] = files
    groups = {}
    for row in review:
        groups.setdefault((row["method"], row["question_id"]), []).append(row)
    output = []
    for (method, qid), rows in sorted(groups.items()):
        ranked = sorted(rows, key=lambda row: int(row["rank"]))
        precision = None if case_by_id[qid]["abstention"] else ev.reviewed_precision(ranked).get(qid)
        evidence_hit = None
        if precision is not None:
            evidence_hit = int(precision > 0)
        truth = truth_by_id.get(qid)
        files = ranked[0]["filing_top5"].split("; ")
        matched = set(files) & truth if truth else set()
        output.append(dict(question_id=qid, method=method, primary=ranked[0]["primary"],
                           precision_at_5=precision, evidence_hit_at_5=evidence_hit,
                           filing_hit_at_5=int(bool(matched)) if truth else None,
                           filing_recall_at_5=len(matched) / len(truth) if truth else None))
    return output


def aggregate(rows):
    result = {}
    for field in ("precision_at_5", "evidence_hit_at_5", "filing_hit_at_5", "filing_recall_at_5"):
        values = [row[field] for row in rows if row["primary"] and row[field] is not None]
        result[field] = dict(n=len(values), value=sum(values) / len(values) if values else None)
    return result


def run_semantic_plan(cases, index_id, directory, retrieve, lookup, complete_files,
                      *, approved=False, question_ids=(), max_calls=0):
    """Each authorized ID consumes at most one query embedding; reuse exact snapshots."""
    if approved is not True:
        raise PermissionError("Live semantic retrieval requires explicit approval")
    cases = {case["id"]: case for case in cases}
    requested = tuple(dict.fromkeys(question_ids))
    if not requested or not len(requested) <= max_calls <= 3 or set(requested) - cases.keys():
        raise ValueError("Choose valid question IDs and an explicit budget of at most three")
    outputs = []
    for qid in requested:
        case = cases[qid]
        plan = dict(question_id=qid, question=case["question"], file=case["file"],
                    top_k=8, index=index_id, method="semantic",
                    retrieval_version="semantic_top8_v1")
        target = Path(directory) / f"{qid}_{ev.digest(plan)[:20]}.json"
        with ev.advisory_lock(target.parent, ".semantic_requests.lock"):
            if target.exists():
                payload = json.loads(target.read_text())
                if payload["plan"] != plan:
                    raise ValueError(f"Semantic snapshot identity changed: {qid}")
                _check_rows(payload["hits"], lookup, {case["file"]} if case["file"] else complete_files)
            else:
                try:
                    hits = retrieve(case["question"], case["file"])
                except Exception as error:
                    raise RuntimeError(f"Semantic request failed for {qid} ({type(error).__name__})") from None
                _check_rows(hits, lookup, {case["file"]} if case["file"] else complete_files)
                ev.write_once(target, json.dumps(dict(plan=plan, hits=hits), indent=2))
        outputs.append(target)
    return outputs


def semantic_snapshots(cases, index_id, directory, lookup, complete_files):
    loaded = {}
    for case in cases:
        qid = case["id"]
        plan = dict(question_id=qid, question=case["question"], file=case["file"],
                    top_k=8, index=index_id, method="semantic",
                    retrieval_version="semantic_top8_v1")
        target = Path(directory) / f"{qid}_{ev.digest(plan)[:20]}.json"
        if not target.exists():
            continue
        payload = json.loads(target.read_text())
        if payload["plan"] != plan:
            raise ValueError(f"Semantic snapshot identity changed: {qid}")
        _check_rows(payload["hits"], lookup, {case["file"]} if case["file"] else complete_files)
        loaded[qid] = payload["hits"]
    return loaded


def export(directory, index_id, version, cases, review, proposals, comparisons, answer_rows, complete_files):
    directory = Path(directory)
    human = directory / "human"
    review = ev.merge_review_labels(
        review, ev.review_rows(human / "passages.csv"),
        ("question_id", "method", "rank", "chunk_id"), ("relevance_label", "review_notes"),
    )
    truths = ev.review_rows(human / "filing_truth.csv")
    metrics = score(cases, review, truths, complete_files)
    truth_template = [
        dict(question_id=c["id"], question=c["question"],
             proposed_files="; ".join(sorted({p["file_name"] for p in proposals if p["question_id"] == c["id"]})),
             verified_relevant_files="", complete_verified="", review_notes="")
        for c in cases if not c["file"] and not c["abstention"]
    ]
    truth_by_id = {row["question_id"]: row for row in truths}
    for row in truth_template:
        if row["question_id"] in truth_by_id:
            for field in ("verified_relevant_files", "complete_verified", "review_notes"):
                row[field] = truth_by_id[row["question_id"]].get(field, "")
    ready = {row["case_id"]: row for row in answer_rows if row["status"] == "ready_for_review"}
    human_answers = [row for row in ev.review_rows(human / "answers.csv") if row["case_id"] in ready]
    merged = ev.merge_review_labels(
        list(ready.values()), human_answers, ("question_id", "case_id"), (*ev.LABELS, "review_notes"),
    )
    reviewed = {row["case_id"]: row for row in merged}
    answer_rows = [reviewed.get(row["case_id"], row) for row in answer_rows]
    answer_scores = {}
    for field in ev.LABELS:
        values = [row[field] for row in merged if row[field] and row[field] != "NA"]
        answer_scores[field] = {"n": len(values), **{
            choice: values.count(choice) for choice in ("PASS", "PARTIAL", "FAIL")
        }}
    abstentions = {c["id"] for c in cases if c["abstention"]}
    summary = dict(benchmark_version=version, index=index_id, complete_filings=len(complete_files),
                   questions=len(cases), offline_ranked=len({row["question_id"] for row in review}),
                   reviewed_metrics=aggregate(metrics), answer_cases_ready=len(merged),
                   answer_cases_prepared=sum(bool(row["case_id"]) for row in answer_rows),
                   answer_scores=answer_scores,
                   abstention_reviews={row["question_id"]: row["appropriate_abstention"] or None
                                       for row in answer_rows if row["question_id"] in abstentions})
    run = directory / f"run_{ev.digest([summary, cases, review, proposals, comparisons, metrics, truth_template, answer_rows])[:16]}"
    ev.write_once(run / "summary.json", json.dumps(summary, indent=2))
    ev.write_once(run / "questions.json", json.dumps(cases, indent=2, ensure_ascii=False))
    for name, rows in (("passages", review), ("candidate_evidence", proposals), ("rank_comparison", comparisons),
                       ("retrieval_metrics", metrics), ("filing_truth", truth_template), ("answers", answer_rows)):
        if rows:
            ev.write_csv(run / f"{name}.csv", rows, list(rows[0]))
    return run, summary
