"""Offline benchmark replay and opt-in answer review for the notebook."""
import csv
import fcntl
import hashlib
import json
import re
from pathlib import Path
from contextlib import contextmanager
from contextlib import nullcontext

TRUE_SALE_KEYWORDS = "true sale sell and assign pledge"
TRUE_SALE_SLOTS = 5
RETRIEVAL_VERSION = "true_sale_lexical_v1"
LABELS = ("factual_correctness", "evidence_relevance", "citation_support",
          "completeness", "ambiguity_uncertainty", "appropriate_abstention")


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def file_digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def keyword_order(vectorizer, matrix, question):
    scores = (matrix @ vectorizer.transform([question]).T).toarray().ravel()
    return [int(i) for i in scores.argsort(kind="stable")[::-1] if scores[i] > 0]


def true_sale_order(question, file_name, hybrid_ids, vectorizer, matrix, eligible):
    """Reserve evidence for a transfer-chain question in a single filing."""
    if not file_name or not re.search(r"\btrue[\s-]+sale\b", question, re.I):
        return list(hybrid_ids)
    indices = keyword_order(vectorizer, matrix[eligible], TRUE_SALE_KEYWORDS)
    lexical_ids = [int(eligible[i]) for i in indices[:TRUE_SALE_SLOTS]]
    return list(dict.fromkeys([*lexical_ids, *hybrid_ids]))


@contextmanager
def advisory_lock(directory, name):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / name).open("a") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        yield


def write_once(path, content):
    """Use immutable exports, including when two notebook sessions race."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x", encoding="utf-8", newline="") as stream:
            stream.write(content)
    except FileExistsError:
        if path.read_text(encoding="utf-8") != content.replace("\r\n", "\n"):
            raise RuntimeError(f"Existing export differs: {path}")
    return path


def write_csv(path, rows, fields):
    import io
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return write_once(path, stream.getvalue())


def review_rows(path):
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def reviewed_precision(reviews):
    """Only score five ranked rows with valid human labels and unique IDs."""
    groups = {}
    for row in reviews:
        groups.setdefault(row["question_id"], []).append(row)
    results = {}
    for qid, rows in groups.items():
        top = sorted(rows, key=lambda row: int(row["rank"]))[:5]
        labels = [str(row.get("relevance_label", "")).strip() for row in top]
        if ([int(row["rank"]) for row in top] == list(range(1, 6))
                and len({row["chunk_id"] for row in top}) == 5
                and set(labels) <= {"0", "1", "2"}):
            results[qid] = labels.count("2") / 5
    return results


def _metrics(rows, expected, gold, get_issuer):
    if gold:
        retrieved = {row["chunk_id"] for row in rows[:8]}
        matches = retrieved & gold
    elif expected:
        retrieved = list(dict.fromkeys(get_issuer(row["file_name"]) for row in rows))[:5]
        matches = set(retrieved) & expected
    else:
        return None, None
    return int(bool(matches)), len(matches) / len(gold or expected)


def benchmark_replay(coverage, raw, chunks, vectorizer, matrix, get_issuer,
                     indexed_issuers, target_file, gold_numbers):
    lookup = {chunk["chunk_id"]: chunk for chunk in chunks}
    position = {chunk["chunk_id"]: i for i, chunk in enumerate(chunks)}
    toyota = [i for i, chunk in enumerate(chunks) if chunk["file_name"] == target_file]
    rows, candidates, debug = [], {}, {}
    for row in coverage:
        qid = row["id"]
        target = target_file if qid in gold_numbers else None
        gold = {chunks[i]["chunk_id"] for i in toyota
                if chunks[i]["chunk_index"] in gold_numbers.get(qid, set())}
        expected = set(row["expected_issuers"].split("; ")) & indexed_issuers
        metric = dict(question_id=qid, question=row["question"],
                      coverage_status=row["coverage_status"], evaluated=qid in raw,
                      metric="gold_passage@8" if target else "covered_issuer@5",
                      ground_truth_count=len(gold or expected), before_hit=None,
                      before_recall=None, after_hit=None, after_recall=None)
        if qid in raw:
            baseline = raw[qid]
            for hit in baseline:
                chunk = lookup[hit["chunk_id"]]
                if any(hit[k] != chunk[k] for k in ("file_name", "doc_url", "text")):
                    raise ValueError(f"Cached source mismatch for {qid}: {hit['chunk_id']}")
            replay = true_sale_order(row["question"], target,
                                     [position[hit["chunk_id"]] for hit in baseline],
                                     vectorizer, matrix, toyota)
            after = [chunks[i] for i in replay[:len(baseline)]]
            candidates[qid] = after
            metric["before_hit"], metric["before_recall"] = _metrics(baseline, expected, gold, get_issuer)
            metric["after_hit"], metric["after_recall"] = _metrics(after, expected, gold, get_issuer)
            if qid == "q18":
                ranks = {}
                for name, query in (("exact_keyword", row["question"]), ("expanded_keyword", TRUE_SALE_KEYWORDS)):
                    order = keyword_order(vectorizer, matrix[toyota], query)
                    ranks[name] = {chunks[toyota[i]]["chunk_id"]: rank for rank, i in enumerate(order, 1)}
                debug = {cid: {name: rank.get(cid) for name, rank in ranks.items()} |
                         {"cached_hybrid": next((i for i, r in enumerate(baseline, 1)
                                                if r["chunk_id"] == cid), None)}
                         for cid in sorted(gold)}
        rows.append(metric)
    return rows, candidates, debug


def passage_review(original, metrics, raw, chunks, get_issuer):
    """Reconcile the original review with exactly the rows it was built from."""
    lookup = {chunk["chunk_id"]: chunk for chunk in chunks}
    questions = {row["question_id"]: row["question"] for row in metrics}
    expected_rows = {}
    for qid, hits in raw.items():
        if qid in {"q17", "q18"}:
            chosen = hits[:8]
        else:
            chosen = []
            seen = set()
            for hit in hits:
                issuer = get_issuer(hit["file_name"])
                if issuer not in seen:
                    chosen.append(hit)
                    seen.add(issuer)
                if len(chosen) == 5:
                    break
        expected_rows.update({(qid, rank): hit["chunk_id"] for rank, hit in enumerate(chosen, 1)})
    output = []
    for row in original:
        qid, rank = row["question_id"], int(row["rank"])
        chunk = lookup[row["chunk_id"]]
        if (expected_rows.pop((qid, rank), None) != row["chunk_id"]
                or row["question"] != questions[qid]
                or row["passage"] != chunk["text"]
                or any(row[k] != chunk[k] for k in ("file_name", "doc_url"))
                or row["issuer"] != get_issuer(chunk["file_name"])):
            raise ValueError(f"Review does not match cached evidence: {qid} rank {rank}")
        output.append({**row, "source_ref": f"S{rank}", "deal_name": chunk["deal_name"],
                       "filing_date": chunk["filing_date"], "accession": chunk["accession"]})
    if expected_rows:
        raise ValueError("Original passage review is incomplete")
    return output


def evidence_rows(hits):
    return [{"source_ref": f"S{rank}", **{k: chunk[k] for k in
             ("chunk_id", "file_name", "deal_name", "filing_date", "doc_url", "text")}}
            for rank, chunk in enumerate(hits, 1)]


def citation_checks(result, evidence):
    core_fields = ("chunk_id", "file_name", "doc_url")
    metadata = ("deal_name", "filing_date")
    actual = {source["reference"]: source for source in result.get("sources", [])}
    expected = {row["source_ref"]: row for row in evidence}
    used = set(re.findall(r"\[(S\d+)\]", result["answer"]))
    core_match = (
        actual.keys() == expected.keys()
        and len(actual) == len(result.get("sources", []))
        and all(all(source.get(k) == expected[ref][k] for k in core_fields)
                for ref, source in actual.items())
    )
    return {
        "all_sources_match": core_match,
        "missing_source_metadata": sorted({k for source in actual.values()
                                           for k in metadata if k not in source}),
        "source_metadata_mismatches": sorted({ref for ref, source in actual.items()
                                               if ref in expected and any(
                                                   k in source and source[k] != expected[ref][k]
                                                   for k in metadata)}),
        "has_citations": bool(used),
        "unknown_citations": sorted(used - actual.keys()),
    }


def answer_plan(metrics, candidates, abstention_hits, index_id, model, prompt_version, system_prompt):
    plans = {}
    for row in metrics:
        qid = row["question_id"]
        hits = candidates.get(qid, [])
        if qid == "q19":
            hits = abstention_hits
            scope = "Only the supplied excerpts from fully indexed filings. Coverage is limited."
        elif not hits:
            continue
        else:
            scope = "Evidence from the current fully indexed filing scope. Coverage is limited."
        if qid not in {"q17", "q18", "q19"}:
            best_ranked = {}
            for hit in hits:
                best_ranked.setdefault(hit["file_name"], hit)
            hits = list(best_ranked.values())
        evidence = evidence_rows(hits[:8])
        settings = dict(question=row["question"], index=index_id, model=model,
                        prompt_version=prompt_version, prompt_hash=digest(system_prompt),
                        retrieval_version=RETRIEVAL_VERSION, scope=scope, evidence=evidence)
        plans[qid] = {**settings, "cache_id": digest(settings)}
    return plans


def evaluate_answers(plans, metrics, output_dir, generate, *, question_ids=(), max_calls=0):
    """Exact-match cache reuse by default. Explicit IDs and a positive cap enable calls."""
    requested = tuple(dict.fromkeys(question_ids))
    if set(requested) - plans.keys() or not 0 <= max_calls <= 3 or len(requested) > max_calls:
        raise ValueError("Select prepared question IDs and a matching call budget of at most three")
    output_dir = Path(output_dir)
    rows = []
    # Honor the caller's requested order and exclude all other rows from live execution.
    results = {}
    for qid in dict.fromkeys([*requested, *plans]):
        plan = plans[qid]
        path = output_dir / f"{qid}_{plan['cache_id'][:20]}.json"
        guard = advisory_lock(output_dir, ".answer_requests.lock") if qid in requested else nullcontext()
        with guard:
            if path.exists():
                saved = json.loads(path.read_text())
                if saved.get("plan") != plan:
                    raise ValueError(f"Answer cache identity mismatch: {qid}")
                results[qid] = saved["result"]
            elif qid in requested:
                try:
                    result = generate(plan)
                except Exception as error:
                    raise RuntimeError(f"Answer request failed for {qid} ({type(error).__name__})") from None
                if result.get("question") != plan["question"] or result.get("model") != plan["model"] or result.get("prompt_version") != plan["prompt_version"]:
                    raise ValueError(f"Answer identity mismatch: {qid}")
                write_once(path, json.dumps({"plan": plan, "result": result}, indent=2))
                results[qid] = result
    for metric in metrics:
        qid = metric["question_id"]
        plan = plans.get(qid)
        result = results.get(qid)
        row = dict(question_id=qid, question=metric["question"],
                   status="ready_for_review" if result else
                   ("awaiting_generation" if plan else "no_evaluation_evidence"),
                   case_id=plan["cache_id"] if plan else "",
                   evidence_json=json.dumps(plan["evidence"], ensure_ascii=False) if plan else "",
                   answer=result["answer"] if result else "",
                   citation_checks=json.dumps(citation_checks(result, plan["evidence"])) if result else "")
        rows.append(row | dict.fromkeys((*LABELS, "review_notes"), ""))
    return rows


def merge_review_labels(template, submissions, key_fields, label_fields):
    """Merge explicit human input by evidence identity and reject stale/duplicate rows."""
    by_key = {tuple(str(row[k]) for k in key_fields): row.copy() for row in template}
    seen = set()
    for row in submissions:
        key = tuple(str(row[k]) for k in key_fields)
        if key not in by_key or key in seen:
            raise ValueError(f"Review row is unknown or duplicated: {key}")
        seen.add(key)
        for field in label_fields:
            value = row.get(field, "").strip()
            if field == "relevance_label" and value not in {"", "0", "1", "2"}:
                raise ValueError(f"Invalid relevance label: {key}")
            if field in LABELS and value and value not in {"PASS", "PARTIAL", "FAIL", "NA"}:
                raise ValueError(f"Invalid {field} review label: {key}")
            if value:
                by_key[key][field] = value
    return list(by_key.values())


def summarize_retrieval(metrics, review):
    precision = reviewed_precision(review)
    output = {}
    for name in ("covered_issuer@5", "gold_passage@8"):
        scored = [row for row in metrics if row["metric"] == name and row["before_hit"] is not None]
        output[name] = {"evaluated": len(scored)}
        for phase in ("before", "after"):
            for field in ("hit", "recall"):
                output[name][f"{phase}_mean_{field}"] = (
                    sum(row[f"{phase}_{field}"] for row in scored) / len(scored) if scored else None
                )
    evaluated = sum(row["evaluated"] for row in metrics)
    return dict(evaluated=evaluated, unevaluated=len(metrics) - evaluated,
                precision_at_5=precision,
                mean_precision_at_5=sum(precision.values()) / len(precision) if precision else None,
                retrieval=output)


def smoke_test(output_dir, index_id, scopes, retrieve, generate, *, approved=False):
    """One embedding and at most one completion per scope; persist each completed case."""
    if approved is not True:
        raise PermissionError("Live smoke test requires explicit approval")
    if len(scopes) != 2 or sorted(file is None for _, file in scopes) != [False, True]:
        raise ValueError("Select one specific-filing and one all-indexed question")
    from datetime import datetime, timezone
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    paths = []
    for i, (question, file_name) in enumerate(scopes, 1):
        try:
            hits = retrieve(question, file_name)
            evidence = evidence_rows(hits)
            result = generate(question, file_name, evidence) if evidence else None
        except Exception as error:
            raise RuntimeError(f"Smoke-test case {i} failed ({type(error).__name__})") from None
        checks = citation_checks(result, evidence) if result else None
        report = dict(question=question, file_name=file_name, index=index_id,
                      evidence=evidence, result=result, citation_checks=checks,
                      manual_claim_support="pending" if result else "no_evidence")
        paths.append(write_once(Path(output_dir) / f"{stamp}_{i}.json", json.dumps(report, indent=2)))
    return paths


def export_benchmark(output_dir, index_id, metrics, reviews, candidates, debug,
                     plans, answer_rows, historical, collisions, semantic_pilot, weaknesses):
    root = Path(output_dir)
    baseline_key = digest([[row[k] for k in ("question_id", "rank", "chunk_id")]
                           for row in reviews])[:16]
    passage_fields = ("question_id", "rank", "chunk_id")
    # Human input is separate from immutable, versioned exports.
    reviews = merge_review_labels(
        reviews, review_rows(root / "human" / f"passages_{baseline_key}.csv"),
        passage_fields, ("relevance_label", "review_notes"))
    q18_review = [
        {"question_id": "q18", "question": next(row["question"] for row in metrics if row["question_id"] == "q18"),
         "rank": rank, **hit, "relevance_label": "", "review_notes": ""}
        for rank, hit in enumerate(evidence_rows(candidates.get("q18", [])[:8]), 1)
    ]
    if q18_review:
        improved_key = digest([[row[k] for k in passage_fields] for row in q18_review])[:16]
        q18_review = merge_review_labels(
            q18_review, review_rows(root / "human" / f"passages_{improved_key}.csv"),
            passage_fields, ("relevance_label", "review_notes"))
    else:
        improved_key = None
    ready = [row for row in answer_rows if row["status"] == "ready_for_review"]
    answer_labels = root / "human" / "answers.csv"
    ready_ids = {row["case_id"] for row in ready}
    relevant_labels = [row for row in review_rows(answer_labels) if row["case_id"] in ready_ids]
    merged = merge_review_labels(
        ready, relevant_labels, ("question_id", "case_id"), (*LABELS, "review_notes"))
    updated = {row["question_id"]: row for row in merged}
    answer_rows = [updated.get(row["question_id"], row) for row in answer_rows]
    summary = summarize_retrieval(metrics, reviews)
    summary["q18_improved_precision_at_5"] = reviewed_precision(q18_review).get("q18")
    summary["answers_awaiting_generation"] = sum(row["status"] == "awaiting_generation" for row in answer_rows)
    summary["answers_ready_for_review"] = len(ready)
    summary["answers_fully_reviewed"] = sum(all(row[field] for field in LABELS) for row in merged)
    payload = dict(index=index_id, retrieval_version=RETRIEVAL_VERSION, metrics=metrics,
                   summary=summary, q18_gold_ranks=debug, q18_semantic_pilot=semantic_pilot,
                   provenance=collisions, weaknesses=weaknesses,
                   passage_label_key=baseline_key, q18_label_key=improved_key,
                   answer_label_file="human/answers.csv")
    run = root / f"benchmark_{digest([payload, reviews, q18_review, answer_rows, historical])[:16]}"
    write_once(run / "summary.json", json.dumps(payload, indent=2))
    write_once(run / "answer_plans.json", json.dumps(plans, indent=2))
    for filename, rows in (("retrieval_metrics.csv", metrics), ("passage_review.csv", reviews),
                           ("q18_after_review.csv", q18_review), ("answer_review.csv", answer_rows),
                           ("historical_answers.csv", historical), ("provenance_collisions.csv", collisions),
                           ("weaknesses.csv", weaknesses)):
        if rows:
            write_csv(run / filename, rows, list(rows[0]))
    return run, summary, reviews, answer_rows
