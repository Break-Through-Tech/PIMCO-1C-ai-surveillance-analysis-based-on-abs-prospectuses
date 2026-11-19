"""Local presentation UI for the ABS surveillance research prototype."""
import json
import re
from pathlib import Path

import pandas as pd
import streamlit as st

from evaluation_review_page import evaluation_review_page
from prospectus_backend import MAX_NEW_QUERIES, ProspectusBackend, QueryError, sec_url


ROOT = Path(__file__).resolve().parent
PAGES = ("Research Assistant", "Overview", "Prospectus Explorer", "Evaluation", "Evaluation Review")
STYLE = """
<style>
[data-testid="stAppViewContainer"] .block-container {
    max-width: 1450px; padding: 2.3rem 2.1rem 2rem;
}
[data-testid="stSidebar"] {
    background: #10243a; border-right: 1px solid #30465d;
}
[data-testid="stSidebar"] :is(p, label, h1, h2, h3, span) { color: #e6eff9; }
[data-testid="stSidebar"] [role="radiogroup"] { gap: .45rem; }
[data-testid="stSidebar"] label[data-baseweb="radio"] { padding: .48rem 0; }
[data-testid="stSidebar"] hr { border-color: #41556c; }
h1 { font-size: 2rem !important; letter-spacing: -.035em; }
h2, h3 { letter-spacing: -.023em; }
.st-key-workspace [data-testid="stChatMessage"] {
    background: #fff; border: 1px solid #dae4ee; border-radius: 13px;
    margin: .15rem 0; padding: .85rem .95rem; gap: .66rem;
}
.st-key-workspace [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
    background: #edf5f5; border-color: #cde2e1;
}
.st-key-workspace [data-testid="stChatMessageAvatarUser"] { background: #28475b; }
.st-key-workspace [data-testid="stChatMessageAvatarAssistant"] { background: #08756f; }
.st-key-workspace [data-testid="stChatMessageContent"] { min-width: 0; }
.st-key-workspace [data-testid="stChatMessageContent"] p { line-height: 1.58; }
[class*="st-key-transcript_"], [class*="st-key-evidence_body_"] { background: #f0f4f8; border-radius: 12px; }
[class*="st-key-evidence_body_"] [data-testid="stVerticalBlockBorderWrapper"] { background: white; }
.st-key-workspace [data-testid="stHeading"] h3 { font-size: 1.14rem; }
[data-testid="stMetric"], [data-testid="stVerticalBlockBorderWrapper"] {
    overflow-wrap: anywhere;
}
[data-testid="stMetric"] {
    background: white; border: 1px solid #dce5ee; border-radius: 12px; padding: 1rem;
}
[data-testid="stMetricValue"] { letter-spacing: -.04em; color: #193b51; }
[data-testid="stExpander"] { background: white; border-radius: 9px; }
.stMarkdown :is(p, li, code), [data-testid="stText"] { overflow-wrap: anywhere; }
[data-testid="stText"] { white-space: pre-wrap !important; }
@media (max-width: 760px) {
    [data-testid="stAppViewContainer"] .block-container {padding: 2rem 1rem;}
    .st-key-workspace [data-testid="stHorizontalBlock"] {flex-wrap: wrap;}
    .st-key-workspace [data-testid="stHorizontalBlock"] > [data-testid="stColumn"] {
        flex: 1 1 100%; min-width: 0;
    }
}
</style>
"""


@st.cache_resource(show_spinner="Validating the saved search index…")
def load_backend():
    return ProspectusBackend(ROOT)


def filing_label(backend, file):
    row = backend.files[file]
    return f"{row['deal_name'].strip()} · {row['filing_date']}"


def clear_live_consent():
    st.session_state["allow_live"] = False


def go_to(page):
    clear_live_consent()
    st.session_state["navigation"] = page


def select_evidence(turn_id):
    st.session_state["selected_evidence"] = turn_id


def append_turn(question, file, *, result=None, origin=None, error=None):
    turns = st.session_state.setdefault("research_log", [])
    turn = {
        "id": len(turns) + 1, "question": question, "file": file,
        "result": result, "origin": origin, "error": error,
    }
    turns.append(turn)
    if result is not None:
        select_evidence(turn["id"])


def load_example(example):
    append_turn(example["result"]["question"], example["file"],
                result=example["result"], origin="Saved example")
    st.session_state["allow_live"] = False


def reset_conversation():
    st.session_state["research_log"] = []
    st.session_state.pop("selected_evidence", None)
    st.session_state.pop("submitted_live_consent", None)
    st.session_state["allow_live"] = False


def open_research(file):
    st.session_state["research_scope"] = file
    st.session_state["scope_picker"] = file
    st.session_state["allow_live"] = False
    go_to("Research Assistant")


def overview(backend):
    st.title("Workspace overview")
    st.write("Research collateral, servicing, custody, and transaction structure in SEC asset-backed securities prospectuses.")
    processed, indexed, passages = st.columns(3)
    processed.metric("Processed prospectuses", f"{len(backend.files):,}")
    indexed.metric("Fully indexed filings", f"{len(backend.complete_files):,}")
    passages.metric("Indexed passages", f"{backend.index.ntotal:,}")
    st.caption("The explorer covers the processed collection. Research searches fully indexed filings.")
    st.divider()
    st.subheader("From question to source")
    st.write("Choose a filing scope, ask a standalone research question, and compare the answer with its cited SEC excerpts.")
    st.button("Open Research Assistant", type="primary", on_click=go_to, args=("Research Assistant",))
    st.button("Browse prospectuses", on_click=go_to, args=("Prospectus Explorer",))


def scope_label(backend, file):
    return filing_label(backend, file) if file else "All fully indexed filings"


def render_turn(backend, turn):
    with st.chat_message("user"):
        st.text(turn["question"])
    with st.chat_message("assistant"):
        st.caption(scope_label(backend, turn["file"]))
        if turn["error"]:
            st.warning(turn["error"])
            return
        result = turn["result"]
        st.caption(turn["origin"])
        st.markdown(result["answer"])
        if result.get("unknown_citations"):
            st.warning("Unresolved citation IDs: " + ", ".join(result["unknown_citations"]))
        if re.search(r"not established by the supplied evidence|no relevant passages", result["answer"], re.I):
            st.caption("The supplied evidence was insufficient for at least part of this question.")
        is_selected = st.session_state.get("selected_evidence") == turn["id"]
        st.button("View evidence" if not is_selected else "Evidence selected",
                  key=f"evidence_turn_{turn['id']}", disabled=is_selected,
                  on_click=select_evidence, args=(turn["id"],))


def render_evidence(backend, turn):
    if turn is None:
        st.caption("Choose an answer to inspect its SEC passages.")
        return
    st.caption(scope_label(backend, turn["file"]))
    st.text(turn["question"])
    result = turn["result"]
    sources = result.get("sources") or []
    if not sources:
        st.caption("No supporting passages were returned for this answer.")
        return
    cited_ids = set(re.findall(r"\[(S\d+)\]", result["answer"]))
    for source in sorted(sources, key=lambda item: item.get("reference") not in cited_ids):
        ref = source.get("reference", "")
        chunk = backend.resolve_source(source)
        if chunk is None:
            st.warning(f"Source {ref or 'reference'} could not be matched to the saved index.")
            continue
        with st.container(border=True):
            designation = "Cited passage" if ref in cited_ids else "Retrieved context"
            st.markdown(f"**[{ref}]** · {designation}")
            st.caption(filing_label(backend, chunk["file_name"]))
            text = chunk["text"]
            preview = text[:190].rsplit(" ", 1)[0] + "…" if len(text) > 190 else text
            st.text(preview)
            if url := sec_url(chunk["doc_url"]):
                st.link_button("Open SEC filing ↗", url)
            with st.expander("Read full passage"):
                st.caption(chunk["file_name"])
                st.caption(f"Chunk {chunk['chunk_id']}")
                st.text(text)


def capture_live_consent():
    st.session_state["submitted_live_consent"] = st.session_state.get("allow_live", False)
    st.session_state["allow_live"] = False


def persist_scope():
    st.session_state["research_scope"] = st.session_state["scope_picker"]
    st.session_state["allow_live"] = False


def submit_question(backend, question, file):
    allow = st.session_state.pop("submitted_live_consent", False)
    try:
        with st.spinner("Retrieving passages and preparing an answer…"):
            result, origin = backend.analyze(question, file, allow, st.session_state)
        append_turn(question, file, result=result, origin=origin)
    except QueryError as error:
        append_turn(question, file, error=str(error))
    except Exception as error:
        message = (
            "The provider’s rate or quota limit was reached. Saved examples are available."
            if getattr(error, "status_code", None) == 429 else
            "The analysis could not be completed. Try a saved example or retry later."
        )
        append_turn(question, file, error=message)


def example_controls(examples, empty):
    if empty:
        st.markdown("**Start with a saved research question**")
        for example in examples:
            if example["key"] in {"collections", "paper", "cross"}:
                st.button(example["label"], key=f"suggest_{example['key']}",
                          on_click=load_example, args=(example,))
        return
    with st.popover("Saved examples"):
        for example in examples:
            st.button(example["label"], key=f"saved_{example['key']}",
                      on_click=load_example, args=(example,))


def research(backend):
    turns = st.session_state.setdefault("research_log", [])
    examples = list(backend.examples())
    title, reset = st.columns([4, 1], vertical_alignment="center")
    title.title("Research Assistant")
    with reset:
        with st.container(horizontal=True, horizontal_alignment="right"):
            st.button("New Conversation", on_click=reset_conversation, disabled=not turns)
    st.caption("Each question is analyzed independently. Include the filing or facts a follow-up needs.")
    with st.container(key="workspace"):
        conversation, evidence = st.columns([65, 35], gap="medium")
        with conversation:
            scope_col, controls = st.columns([3, 2], vertical_alignment="bottom")
            with scope_col:
                files = sorted(backend.complete_files, key=lambda name: filing_label(backend, name))
                if "scope_picker" not in st.session_state:
                    st.session_state["scope_picker"] = st.session_state.get("research_scope")
                file = st.selectbox(
                    "Document scope", [None, *files], key="scope_picker",
                    format_func=lambda name: scope_label(backend, name), on_change=persist_scope,
                )
            with controls:
                with st.container(horizontal=True, horizontal_alignment="right"):
                    if turns and examples:
                        example_controls(examples, empty=False)
            if file:
                row = backend.files[file]
                st.caption(f"{row['filing_type']} · {row['accession']} · {row['indexed_chunks']} indexed passages")
            # Remount on a new turn so Streamlit follows the new message.
            # Selecting evidence or changing scope keeps the existing scroll position.
            with st.container(height=440, border=True, key=f"transcript_{len(turns)}"):
                if turns:
                    for turn in turns:
                        render_turn(backend, turn)
                else:
                    st.subheader("Research SEC prospectuses")
                    st.write("Ask about servicing, collateral custody, or the structure of an ABS deal.")
                    if examples:
                        example_controls(examples, empty=True)
            remaining = MAX_NEW_QUERIES - st.session_state.get("new_attempts", 0)
            st.checkbox("Allow live API calls for this submission", key="allow_live",
                        help="One uncached attempt can use one query embedding and one completion.")
            st.caption(f"{remaining} of {MAX_NEW_QUERIES} live attempts available this session.")
            question = st.chat_input("Ask a standalone research question", key="research_prompt",
                                     on_submit=capture_live_consent)
            if question:
                submit_question(backend, question, file)
                st.rerun()
        with evidence:
            st.subheader("Supporting evidence")
            selected_id = st.session_state.get("selected_evidence")
            selected = next((turn for turn in turns if turn["id"] == selected_id), None)
            with st.container(height=510, border=True, key=f"evidence_body_{selected_id}"):
                render_evidence(backend, selected)


def explorer(backend):
    st.title("Prospectus Explorer")
    st.write("Browse the processed SEC collection and check coverage before choosing a filing for research.")
    rows = list(backend.files.values())
    search = st.text_input("Search filings", placeholder="Issuer, filename, or SEC accession").casefold().strip()
    a, b, c = st.columns(3)
    issuer = a.selectbox("Issuer or deal", [None, *sorted({r["deal_name"].strip() for r in rows})],
                         format_func=lambda x: x or "All issuers")
    year = b.selectbox("Filing year", [None, *sorted({r["filing_year"] for r in rows}, reverse=True)],
                       format_func=lambda x: str(x) if x else "All years")
    statuses = ("Fully indexed", "Partially indexed", "Not indexed")
    status = c.selectbox("Index status", [None, *statuses], format_func=lambda x: x or "All statuses")
    matches = [
        row for row in rows
        if (not issuer or issuer == row["deal_name"].strip())
        and (not year or year == row["filing_year"])
        and (not status or status == row["status"])
        and search in " ".join(str(row[k]) for k in ("deal_name", "file_name", "accession")).casefold()
    ]
    matches.sort(key=lambda row: (row["filing_date"], row["file_name"]), reverse=True)
    st.caption(f"{len(matches)} matching " + ("filing" if len(matches) == 1 else "filings"))
    if not matches:
        st.info("No filings match the current filters.")
        return
    st.dataframe(pd.DataFrame([{
        "Issuer or deal": row["deal_name"].strip(), "Filed": row["filing_date"], "Form": row["filing_type"],
        "Index status": row["status"], "Indexed passages": row["indexed_chunks"],
    } for row in matches]), hide_index=True, width="stretch", height=min(320, 37 + 35 * len(matches)))
    file = st.selectbox("Inspect a filing", [row["file_name"] for row in matches],
                        format_func=lambda name: filing_label(backend, name))
    row = backend.files[file]
    with st.container(border=True):
        st.subheader(row["deal_name"].strip())
        st.caption(row["file_name"])
        st.write(f"**Filed** {row['filing_date']} · **Form** {row['filing_type']} · **{row['status']}**")
        st.write(f"**SEC accession** {row['accession']} · **CIK** {row['cik']}")
        st.write(f"**Passages indexed** {row['indexed_chunks']:,} of {row['total_chunks']:,}")
        if url := sec_url(row["doc_url"]):
            st.link_button("Open SEC filing ↗", url)
        if file in backend.complete_files:
            st.button("Analyze this filing", on_click=open_research, args=(file,))


def read_evaluation(backend):
    base = backend.processed / "evaluation_runs" / "task2_v1" / backend.index_digest[:16]
    for path in sorted(base.glob("run_*/summary.json"), key=lambda p: p.stat().st_mtime, reverse=True):
        try:
            summary = json.loads(path.read_text())
            if summary["index"]["faiss_sha256"] == backend.index_digest:
                return summary, pd.read_csv(path.parent / "rank_comparison.csv"), path.parent
        except (OSError, ValueError, KeyError):
            continue
    return None, None, None


def evaluation_page(backend):
    st.title("Evaluation")
    st.write("Benchmark coverage, source retrieval, and known limitations.")
    summary, ranks, run = read_evaluation(backend)
    if summary is None:
        st.info("No saved evaluation report matches the loaded checkpoint.")
        return
    a, b, c = st.columns(3)
    a.metric("Benchmark questions", summary["questions"])
    b.metric("Offline rankings", summary["offline_ranked"])
    c.metric("Answer cases prepared", summary["answer_cases_prepared"])
    st.caption(f"Scoped to {summary['complete_filings']} fully indexed filings. Offline rankings combine cached hybrid runs and TF-IDF.")
    with st.container(border=True):
        st.subheader("Toyota true-sale retrieval")
        st.caption("q18 · Offline replay · Known gold passages in the top eight")
        q18 = ranks[ranks["question_id"] == "q18"].set_index("method")
        if {"cached_hybrid", "cached_hybrid_true_sale_replay"} <= set(q18.index):
            before, after = st.columns(2)
            for col, label, method in (
                (before, "Original cached hybrid", "cached_hybrid"),
                (after, "Revised offline replay", "cached_hybrid_true_sale_replay"),
            ):
                row = q18.loc[method]
                col.metric(label, f"{row['matched_at_8']}/{row['candidate_count']}")
        st.write("The revised replay surfaced the transfer and pledge sequence and passages on potential recharacterization.")
        try:
            passages = pd.read_csv(run / "passages.csv")
            sample = passages.loc[
                (passages["question_id"] == "q18")
                & (passages["method"] == "cached_hybrid_true_sale_replay")
            ].sort_values("rank")
            if not sample.empty:
                row = sample.iloc[0]
                chunk = backend.resolve_source(row)
                if chunk and chunk["text"] == row["text"]:
                    with st.expander("View the first passage in the revised replay"):
                        st.caption(f"[{row['source_ref']}] {filing_label(backend, chunk['file_name'])}")
                        st.text(chunk["text"])
                        if url := sec_url(chunk["doc_url"]):
                            st.link_button("Open SEC filing ↗", url)
        except (OSError, ValueError, KeyError):
            pass
    st.subheader("Reviewed performance")
    metrics = summary.get("reviewed_metrics", {})
    names = {
        "precision_at_5": "Passage Precision@5", "evidence_hit_at_5": "Evidence Hit Rate@5",
        "filing_hit_at_5": "Filing Hit Rate@5", "filing_recall_at_5": "Filing Recall@5",
    }
    reviewed = [{"Metric": names[k], "Value": f"{v['value']:.1%}", "Reviewed questions": v["n"]}
                for k, v in metrics.items() if k in names and v["n"] > 0 and v["value"] is not None]
    if reviewed:
        st.dataframe(reviewed, hide_index=True, width="stretch")
    else:
        st.info("Passage precision, filing hit rate, and filing recall await human review.")
    if summary["answer_cases_ready"] == 0:
        st.caption("Answer quality and abstention await generation and human review.")
    else:
        scores = summary.get("answer_scores", {})
        ratings = [{"Dimension": key.replace("_", " ").capitalize(), **value}
                   for key, value in scores.items() if value["n"]]
        if ratings:
            st.dataframe(ratings, hide_index=True, width="stretch")
        else:
            st.caption("Generated answers await human review.")
    st.subheader("Current limitations")
    st.markdown(
        "- Search covers fully indexed filings from the processed collection.\n"
        "- The true-sale supplement showed mixed overlap with unverified evidence proposals on two AFS questions.\n"
        "- Citation IDs identify passages. Claim-level support requires human review."
    )


def main():
    st.set_page_config(page_title="ABS Surveillance Intelligence", layout="wide", initial_sidebar_state="expanded")
    st.markdown(STYLE, unsafe_allow_html=True)
    with st.sidebar:
        st.subheader("ABS Surveillance")
        st.caption("Prospectus research")
        st.divider()
        page = st.radio("Navigate", PAGES, key="navigation", label_visibility="collapsed", on_change=clear_live_consent)
        st.divider()
        st.caption("Student AI Studio project developed in collaboration with PIMCO.")
    try:
        backend = load_backend()
    except Exception:
        st.error("The matching saved checkpoint and chunk archive could not be loaded. See the README for the required local files.")
        return
    pages = dict(zip(PAGES, (research, overview, explorer, evaluation_page, evaluation_review_page)))
    pages[page](backend)


if __name__ == "__main__":
    main()
