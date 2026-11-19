"""Streamlit review workspace. Reading and saving reviews is entirely offline."""
import json
import re

import pandas as pd
import streamlit as st

import prospectus_evaluation as ev
import prospectus_review as review
from prospectus_backend import sec_url

RATINGS = ("PASS", "PARTIAL", "FAIL")
RELEVANCE = {"2": "2 · Directly relevant", "1": "1 · Partially relevant", "0": "0 · Irrelevant"}


def proposal_rows(backend):
    config = json.loads((backend.root / "eval" / "heldout_proposals_v1.json").read_text())
    numbered = {(chunk["accession"], chunk["chunk_index"]): chunk for chunk in backend.chunks
                if chunk["file_name"] in backend.complete_files}
    rows = []
    for qid, question, requested in config["questions"]:
        evidence = [numbered[accession, number]["chunk_id"]
                    for accession, numbers in requested.items() for number in numbers]
        case_id = ev.digest([config["version"], backend.index_digest, question, evidence])
        rows.append(dict(question_id=qid, question=question, case_id=case_id, evidence=evidence))
    return config["version"], rows


def chunk_view(chunk):
    st.caption(f"{chunk['deal_name']} · {chunk['filing_date']}")
    st.caption(f"{chunk['file_name']} · {chunk['chunk_id']}")
    if url := sec_url(chunk["doc_url"]):
        st.link_button("Open SEC filing", url)
    st.text(chunk["text"])


def valid_passage(backend, row):
    chunk = backend.resolve_source(row)
    if (chunk and chunk["file_name"] in backend.complete_files
            and all(row[k] == chunk[k] for k in ("filing_date", "text"))
            and row["source_ref"] == f"S{row['rank']}"):
        return chunk
    return None


def save_form(backend, directory, kind, row, widgets, current_rows, benchmark=None, *, form_variant="manual"):
    identity = review.key(kind, row)
    old = review.inputs(directory, kind).get(identity)
    form_key = f"review_{kind}_{form_variant}_{ev.digest([identity, review.fingerprint(kind, row)])[:20]}"
    expected_key = f"{form_key}_expected"
    # Keep the revision the reviewer originally opened, including across widget reruns.
    expected = st.session_state.setdefault(expected_key, old)
    if old != expected:
        st.warning("This judgment changed in another review. Reload before editing.")
        if st.button("Reload judgment", key=f"{form_key}_reload"):
            st.session_state.pop(expected_key, None)
            st.rerun()
        return
    with st.form(form_key):
        values = widgets(old or {})
        has_existing = old is not None or any(
            row.get(field) for field in review.FIELDS[kind] if field != "review_notes"
        )
        confirmed = st.checkbox("Confirm revision of the existing judgment") if has_existing else True
        if st.form_submit_button("Save judgment", type="primary"):
            if not confirmed:
                st.warning("Confirm the revision before saving.")
                return
            try:
                review.save(directory, kind, row, values, expected, current_rows, backend.complete_files)
                st.session_state.pop(expected_key, None)
                if benchmark:
                    try:
                        review.export_review(*benchmark, backend.complete_files)
                    except (ValueError, OSError, RuntimeError) as error:
                        st.warning(f"Judgment saved. Evaluation export needs attention: {error}")
                        return
                st.session_state["review_saved"] = "Judgment saved."
                st.rerun()
            except (ValueError, OSError, RuntimeError) as error:
                st.error(str(error))


def pick(label, rows, key, formatter):
    return rows[st.selectbox(label, range(len(rows)), key=key,
                             format_func=lambda i: formatter(rows[i]))]


def passage_review(backend, run, summary, rows, cases, load):
    positive = {c["id"] for c in cases if not c["abstention"]}
    primary = [r for r in rows if r["primary"] == "True" and int(r["rank"]) <= 5
               and r["question_id"] in positive]
    human = run.parent / "human"
    judgments = review.inputs(human, "passages")
    st.write(f"**{sum(bool((judgments.get(review.key('passages', r)) or r).get('relevance_label')) for r in primary)} / {len(primary)}** primary top-five passages rated")
    mode = st.radio("Passage queue", ("Primary top five", "All methods and ranks"), horizontal=True)
    available = primary if mode == "Primary top five" else rows
    question = pick("Benchmark question", [c for c in cases if any(r["question_id"] == c["id"] for r in available)],
                    f"pr_question_{mode}", lambda c: f"{c['id']} · {c['question']}")
    matches = [r for r in available if r["question_id"] == question["id"]]
    row = pick("Retrieval method and rank", matches, f"pr_item_{mode}_{question['id']}",
               lambda r: f"{r['method']} · rank {r['rank']} · {r['source_ref']}")
    if not (chunk := valid_passage(backend, row)):
        st.error("The passage no longer matches the active index.")
        return
    chunk_view(chunk)

    def widgets(old):
        value = old.get("relevance_label", row.get("relevance_label", ""))
        label = st.radio("Relevance", list(RELEVANCE), index=list(RELEVANCE).index(value) if value in RELEVANCE else None,
                         format_func=RELEVANCE.get)
        return dict(relevance_label=label or "", review_notes=st.text_area("Reviewer notes", old.get("review_notes", row.get("review_notes", ""))))

    save_form(backend, human, "passages", row, widgets, lambda: load("passages"), (run, summary))


def filing_review(backend, run, summary, rows, passages, proposals, load):
    verified = sum(r["complete_verified"] == "yes" for r in rows)
    st.write(f"**{verified} / {len(rows)}** complete filing sets verified")
    st.caption("Mark a set complete only after reviewing the full indexed scope.")
    row = pick("Cross-filing question", rows, "filing_question", lambda r: f"{r['question_id']} · {r['question']}")
    qid = row["question_id"]
    candidates = [p for p in proposals if p["question_id"] == qid]
    for file in dict.fromkeys(p["file_name"] for p in candidates):
        info = backend.files.get(file)
        if not info:
            continue
        with st.expander(info["deal_name"] + " · " + info["filing_date"]):
            st.caption("Proposed filing · needs human verification")
            hits = [backend.lookup[p["chunk_id"]] for p in candidates
                    if p["file_name"] == file and p["chunk_id"] in backend.lookup]
            if not hits:
                hits = [chunk for r in passages if r["question_id"] == qid and r["file_name"] == file
                        and (chunk := valid_passage(backend, r))]
            unique = {c["chunk_id"]: c for c in hits}
            if unique:
                chunk_view(next(iter(unique.values())))
                for chunk in list(unique.values())[1:]:
                    with st.expander(chunk["chunk_id"]):
                        chunk_view(chunk)
            else:
                st.caption("No proposed passage is recorded. Inspect an indexed passage or the SEC filing.")
                chunks = [c for c in backend.chunks if c["file_name"] == file]
                cid = st.selectbox("Inspect passage", [c["chunk_id"] for c in chunks], index=None,
                                   key=f"filing_evidence_{qid}_{file}",
                                   format_func=lambda value: f"Chunk {backend.lookup[value]['chunk_index']}")
                if cid:
                    chunk_view(backend.lookup[cid])
                elif url := sec_url(info["doc_url"]):
                    st.link_button("Open SEC filing", url)

    def widgets(old):
        current = old.get("verified_relevant_files", row.get("verified_relevant_files", ""))
        files = st.multiselect("Verified relevant filings", sorted(backend.complete_files),
                               default=[f for f in current.split("; ") if f in backend.complete_files])
        complete = st.checkbox("I verified the complete relevant filing set", value=old.get("complete_verified", row.get("complete_verified")) == "yes")
        return dict(verified_relevant_files="; ".join(files), complete_verified="yes" if complete else "no",
                    review_notes=st.text_area("Reviewer notes", old.get("review_notes", row.get("review_notes", ""))))

    save_form(backend, run.parent / "human", "filing_truth", row, widgets, lambda: load("filing_truth"), (run, summary))


def answer_review(backend, run, summary, rows, load):
    ready = [r for r in rows if r["status"] == "ready_for_review"]
    judgments = review.inputs(run.parent / "human", "answers")
    complete = sum(all((judgments.get(review.key("answers", r)) or r).get(field) for field in ev.LABELS) for r in ready)
    st.write(f"**{len(ready)} / {len(rows)}** generated · **{complete}** fully rated")
    if not ready:
        st.info("Answer review is pending generation. The prepared q17, q18 and q19 batch requires explicit API approval in the notebook.")
        return
    row = pick("Generated answer", ready, "answer_question", lambda r: f"{r['question_id']} · {r['question']}")
    st.markdown(row["answer"])
    checks = json.loads(row["citation_checks"] or "{}")
    if checks.get("unknown_citations"):
        st.warning("Invalid citation IDs: " + ", ".join(checks["unknown_citations"]))
    if not checks.get("has_citations"):
        st.caption("The answer contains no citation IDs.")
    if checks.get("all_sources_match") is False:
        st.warning("The saved source mappings require review.")
    if row["question_id"] == "q19":
        st.info("Check whether the answer distinguishes background mentions of student loans from evidence of deal collateral.")
    cited = set(re.findall(r"\[(S\d+)\]", row["answer"]))
    for source in json.loads(row["evidence_json"]):
        with st.expander(source["source_ref"] + " · " + source["file_name"]):
            st.caption("Cited in answer" if source["source_ref"] in cited else "Additional retrieved context")
            if (chunk := backend.resolve_source(source)) and source["text"] == chunk["text"]:
                chunk_view(chunk)
            else:
                st.warning("The saved source does not match the current indexed passage.")
    st.caption("For citation support, check each material claim against the passage it cites.")

    def widgets(old):
        values = {}
        for field in ev.LABELS:
            choices = (*RATINGS, "NA") if field == "appropriate_abstention" else RATINGS
            prior = old.get(field, row.get(field))
            values[field] = st.selectbox(field.replace("_", " ").capitalize(), choices,
                                         index=choices.index(prior) if prior in choices else None) or ""
        values["review_notes"] = st.text_area("Reviewer notes", old.get("review_notes", row.get("review_notes", "")))
        return values

    save_form(backend, run.parent / "human", "answers", row, widgets, lambda: load("answers"), (run, summary))


def heldout_review(backend, directory, index):
    version, rows = proposal_rows(backend)
    human = directory.parents[1] / version / backend.index_digest[:16] / "human"
    decisions = review.inputs(human, "heldout")
    st.write(f"**{sum(decisions.get(review.key('heldout', r), {}).get('decision') == 'approved' for r in rows)} / {len(rows)}** proposals approved")
    st.caption("Review questions and proposed evidence before ranking. These team-authored proposals need independent review before they qualify as a held-out benchmark.")
    if st.button("Export approved held-out question set"):
        try:
            path = review.approved_heldout(human.parent, version, rows, index, backend.lookup)
            st.success(f"Approved benchmark exported to {path.name}.")
        except ValueError as error:
            st.warning(str(error))
    row = pick("Proposed question", rows, "heldout_question", lambda r: f"{r['question_id']} · {r['question']}")
    for cid in row["evidence"]:
        chunk_view(backend.lookup[cid])

    def widgets(old):
        choices = ("approved", "needs_revision")
        prior = old.get("decision")
        return dict(decision=st.radio("Question and candidate evidence", choices,
                                      index=choices.index(prior) if prior in choices else None,
                                      format_func=lambda v: "Approve question and evidence" if v == "approved" else "Needs revision") or "",
                    review_notes=st.text_area("Reviewer notes", old.get("review_notes", "")))

    save_form(backend, human, "heldout", row, widgets, lambda: proposal_rows(backend)[1])


def metrics_page(backend, run, summary, cases, rows):
    st.subheader("Development benchmark")
    st.write(f"**{summary['offline_ranked']} / {summary['questions']}** questions ranked · **{summary['complete_filings']}** fully indexed filings")
    metrics = ev.review_rows(run / "retrieval_metrics.csv")
    result = []
    for method in ["Primary mix", *sorted({r["method"] for r in metrics})]:
        selected = [r for r in metrics if r["primary"] == "True"] if method == "Primary mix" else [
            r for r in metrics if r["method"] == method]
        for field in ("precision_at_5", "evidence_hit_at_5", "filing_hit_at_5", "filing_recall_at_5"):
            scores = [float(r[field]) for r in selected if r[field]]
            result.append({"Method": method, "Metric": field.replace("_", " ").replace("at 5", "@5").capitalize(),
                           "Value": f"{sum(scores) / len(scores):.1%}" if scores else "Pending", "Questions scored": len(scores)})
    st.dataframe(pd.DataFrame(result), hide_index=True)
    st.caption("Passage metrics require five human labels and exclude q19. Filing metrics require complete verified ground truth and exclude fixed-filing and abstention cases.")
    gold = ev.review_rows(run / "rank_comparison.csv")
    st.subheader("Gold passages and true-sale comparisons")
    for row in gold:
        row["gold_recall_at_8"] = (
            f"{int(row['matched_at_8']) / int(row['candidate_count']):.0%}"
            if row["evidence_status"] == "existing_gold" and int(row["candidate_count"]) else "—"
        )
    st.dataframe(pd.DataFrame(gold), hide_index=True)
    st.caption("Existing gold is available for q17 and q18. Other candidate overlap counts use unverified proposals.")
    st.subheader("Answer-quality ratings")
    answers = ev.review_rows(run / "answers.csv")
    fully_reviewed = sum(r["status"] == "ready_for_review" and all(r[field] for field in ev.LABELS)
                         for r in answers)
    st.write(f"{summary['answer_cases_ready']}/{len(answers)} generated · {fully_reviewed} fully reviewed")
    for field, result in summary["answer_scores"].items():
        n = result["n"]
        st.write(f"{field.replace('_', ' ').capitalize()} · "
                 + (f"{result['PASS']}/{n} PASS · {result['PARTIAL']} PARTIAL · {result['FAIL']} FAIL" if n else "Pending"))
    st.caption("Each denominator includes human ratings of that dimension. NA and missing scores are excluded.")
    st.write("q19 abstention · " + (summary["abstention_reviews"].get("q19") or "Pending"))
    semantic = {r["question_id"] for r in rows if r["method"] == "approved_semantic"}
    st.write(f"Exact semantic rankings available for {len(semantic)}/{len(cases)} questions.")
    st.caption("The first uncached batch is q19 and s01–s07. Use batches of up to three only after approval. Notebook exports add exact semantic results to the review queue.")
    if st.button("Recalculate and export evaluation"):
        try:
            path, _ = review.export_review(run, summary, backend.complete_files)
            st.session_state["review_saved"] = f"Evaluation exported to {path.name}."
            st.rerun()
        except (ValueError, OSError, RuntimeError) as error:
            st.error(str(error))


def ai_evaluation(backend, run, summary, rows, cases, load):
    st.subheader("AI passage judgments")
    st.caption("Independent AI ratings of the primary top five. Only explicit human acceptance or correction enters human metrics.")
    try:
        pairs = review.load_ai_passage_reviews(
            backend.root / "eval" / "ai_passage_reviews.csv", rows, cases
        )
        scores = review.ai_precision([suggestion for suggestion, _ in pairs])
    except (ValueError, OSError, KeyError) as error:
        st.warning(str(error))
        return
    judgments = review.inputs(run.parent / "human", "passages")
    verified = sum(review.key("passages", row) in judgments for _, row in pairs)
    a, b, c = st.columns(3)
    a.metric("AI-judged passages", len(pairs))
    b.metric("AI mean Precision@5", f"{sum(scores.values()) / len(scores):.1%}" if scores else "Pending")
    c.metric("Human judgments saved", f"{verified}/{len(pairs)}")
    with st.expander("AI Precision@5 by question"):
        st.dataframe(pd.DataFrame([
            {"Question": qid, "AI Precision@5": f"{score:.0%}"} for qid, score in scores.items()
        ]), hide_index=True)
    confidence = st.selectbox("Confidence filter", ("All", "Low", "Medium", "High"))
    flagged = st.checkbox("Only show recommendations for human review")
    available = [(suggestion, row) for suggestion, row in pairs
                 if (confidence == "All" or suggestion["confidence"] == confidence)
                 and (not flagged or suggestion["human_review_recommended"] == "yes")]
    if not available:
        st.info("No judgments match these filters.")
        return
    suggestion, row = pick("AI judgment", available, f"ai_item_{confidence}_{flagged}",
                           lambda pair: f"{pair[1]['question_id']} · rank {pair[1]['rank']} · {pair[1]['method']}")
    st.markdown(f"**{row['question']}**")
    st.write(f"**{RELEVANCE[suggestion['relevance_score']]}** · {suggestion['confidence']} confidence")
    if suggestion["human_review_recommended"] == "yes":
        st.caption("Human review recommended")
    st.write(suggestion["reasoning"])
    st.text(suggestion["supporting_excerpt"])
    if not (chunk := valid_passage(backend, row)):
        st.error("The passage no longer matches the active index.")
        return
    with st.expander("Full passage and SEC source"):
        chunk_view(chunk)

    def widgets(old):
        if old.get("relevance_label") in RELEVANCE:
            st.caption(f"Existing human judgment: {RELEVANCE[old['relevance_label']]}")
        action = st.radio("Human decision", ("Accept AI suggestion", "Override AI suggestion"), index=None)
        # Both form fields must be rendered before submission so an override can be entered in one pass.
        override = st.selectbox("Override score", list(RELEVANCE), index=None, format_func=RELEVANCE.get)
        label = suggestion["relevance_score"] if action == "Accept AI suggestion" else override if action == "Override AI suggestion" else None
        return dict(relevance_label=label or "", review_notes=st.text_area("Reviewer notes", old.get("review_notes", "")))

    save_form(backend, run.parent / "human", "passages", row, widgets,
              lambda: load("passages"), (run, summary), form_variant="ai")


def evaluation_review_page(backend):
    st.title("Evaluation Review")
    st.caption("Human judgments for the development benchmark and proposed held-out set.")
    directory = backend.processed / "evaluation_runs" / "task2_v1" / backend.index_digest[:16]
    run, summary = review.latest_run(directory, backend.index_digest)
    if not run:
        st.error("No complete Task 2 export matches the loaded checkpoint.")
        return
    if message := st.session_state.pop("review_saved", None):
        st.success(message)

    def load(kind):
        latest, _ = review.latest_run(directory, backend.index_digest)
        return ev.review_rows(latest / f"{kind}.csv") if latest else []

    cases = json.loads((run / "questions.json").read_text())
    rows = ev.review_rows(run / "passages.csv")
    section = st.radio("Review workflow", ("Passages", "AI Evaluation", "Filing relevance", "Answers", "Held-out proposals", "Results"), horizontal=True)
    if section == "Passages":
        passage_review(backend, run, summary, rows, cases, load)
    elif section == "AI Evaluation":
        ai_evaluation(backend, run, summary, rows, cases, load)
    elif section == "Filing relevance":
        filing_review(backend, run, summary, load("filing_truth"), rows, load("candidate_evidence"), load)
    elif section == "Answers":
        answer_review(backend, run, summary, load("answers"), load)
    elif section == "Held-out proposals":
        heldout_review(backend, directory, summary["index"])
    else:
        metrics_page(backend, run, summary, cases, rows)
