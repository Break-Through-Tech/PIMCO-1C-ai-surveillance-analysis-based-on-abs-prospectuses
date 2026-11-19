# PIMCO 1C - AI Surveillance Analysis Based on ABS Prospectuses

---

### 👥 **Team Members**

**Example:**

| Name             | GitHub Handle | Contribution                                                             |
|------------------|---------------|--------------------------------------------------------------------------|
| Saivya Telang    | @saivyatelang | Data collection, text extraction, metadata organization  |
| Mahi Sheth       | @     |   |
| Maxi Tran        | @  |                  |

---

## 🎯 **Project Highlights**

**Example:**

- Collected **122 SEC ABS prospectus filings** in HTM/HTML format for analysis.
- Extracted and organized text from the prospectuses using **Python, Pandas, and BeautifulSoup**.
- Created an index containing filing information such as **deal name, filing date, filing type, and SEC metadata**.
- Developed an initial **keyword and regular expression (regex) search** to identify potentially relevant information across prospectuses.
- Working toward an AI-powered search and question-answering tool that can identify relevant deals and provide supporting passages from ABS prospectuses.


---

## 👩🏽‍💻 **Setup and Installation**

Streamlit is the primary local application. From the repository root in the VS Code terminal, select your Python environment and run:

```bash
python -m pip install -r requirements.txt
streamlit run app.py
```

Open the localhost URL shown in the terminal. Restore the team's matching local `data/processed/` artifacts before launch:

- `abs_chunks.jsonl.gz`, `abs_openai_semantic.faiss`, and `abs_openai_semantic_state.json` for the validated chunk and index mapping.
- `task2_answers/` and `toyota_true_sale_rag_answer.json` for offline demonstrations and existing answer caches.
- `evaluation_runs/task2_v1/` for the checkpoint-matched evaluation display.

The app opens in **Research Assistant**. Choose a document scope and load an offline Toyota collections, Toyota custody, or cross-filing electronic custody example from the empty workspace or **Saved examples**. Use **View evidence** on any earlier answer to switch the evidence panel to that turn. The original saved wording and citations are preserved. The session log holds independent research questions. Include any needed deal names and facts in each follow-up. **New Conversation** clears the log and selected evidence while retaining the session’s live-attempt count.

For an authorized uncached question, privately configure `API_KEY` or `OPENAI_API_KEY` in the ignored `.env` and set `OPENAI_BASE_URL` if required by the approved endpoint. Choose a scope and check **Allow live API calls for this submission** before sending a question with the chat input. Consent resets after each submission. Each uncached attempt can make one query-embedding request and one completion. The limit is three attempts per browser session, and saved answers remain available. The server binds to localhost.

The explorer covers 122 processed filings. Live retrieval covers the 12 fully indexed filings and uses the existing 2,808 vectors without regenerating document embeddings. The partially indexed filing remains visible in the explorer. The evaluation page identifies offline results and reports reviewed scores only when labels are available.

For development and evaluation, open `notebooks/btt_ai_fall_studio_pimco_1c.ipynb` in VS Code with the same Python environment. The notebook and Streamlit share retrieval, answer, and cache functions. Gradio remains available as a deprecated fallback. Set `RUN_GRADIO_FALLBACK = True` in its launch cell and run that cell to open it.

---

## 🏗️ **Project Overview**

- This project is part of the **Break Through Tech AI Program** and is being developed with **PIMCO** as the AI Studio host company.
- The project focuses on developing an **AI-powered surveillance and search tool** for ABS prospectuses.
- The goal is to help users efficiently search through large collections of **Asset-Backed Securities (ABS) prospectuses**.
- ABS prospectuses contain large amounts of detailed information that can be time-consuming to review manually.
- The tool aims to allow users to ask questions in **plain language** and identify relevant deals and supporting passages.
- An example question is:
  - **"Which deals appear to use paper (physical) custody?"**
- The project is being developed in stages:
  - Data collection
  - Text extraction
  - Metadata organization
  - Keyword and regex search
  - Semantic search
  - AI-generated responses
  - Supporting citations

---

## 📊 **Data Exploration**

- The current dataset contains:
  - **124 total files**
  - **122 HTM/HTML prospectus filings**
  - **SEC 424H filings**
  - Filing dates ranging from **2024–2026**
- The prospectus files were collected from **SEC EDGAR**.
- The HTM files were processed using **BeautifulSoup** to extract readable text.
- Each filing is represented as a row in a Pandas DataFrame.
- The current index includes:
  - Deal/entity name
  - Filing date
  - Filing type
  - File name
  - Extracted text
  - CIK
  - Accession number
  - Primary document name
  - SEC document URL
- SEC metadata was matched by filename to **122 of 122 local filings**. Five filenames map to multiple manifest accessions. The notebook resolves the current five from local HTML and SEC filing evidence and reports unresolved collisions without selecting a candidate.
- Initial keyword testing showed:
  - `"physical"` → **122 filings**
  - `"custody"` → **94 filings**
  - `"custodian"` → **122 filings**
  - `"paper"` → **122 filings**
  - `"physical custody"` → **0 filings**
- These results showed that broad keyword searches can return too many results.
- The results also showed that relevant information may be described using different wording across prospectuses.
- The project is therefore moving toward **regex and semantic search** to identify related concepts.


**Potential visualizations to include:**

* Plots, charts, heatmaps, feature visualizations, sample dataset images

---

## 🧠 **Model Development**

**You might consider describing the following (as applicable):**

* Model(s) used (e.g., CNN with transfer learning, regression models)
* Feature selection and Hyperparameter tuning strategies
* Training setup (e.g., % of data for training/validation, evaluation metric, baseline performance)


---

## 📈 **Results & Key Findings**

**You might consider describing the following (as applicable):**

* Performance metrics (e.g., Accuracy, F1 score, RMSE)
* How your model performed
* Insights from evaluating model fairness

**Potential visualizations to include:**

* Confusion matrix, precision-recall curve, feature importance plot, prediction distribution, outputs from fairness or explainability tools

---

## 🚀 **Next Steps**

**You might consider addressing the following (as applicable):**

* What are some of the limitations of your model?
* What would you do differently with more time/resources?
* What additional datasets or techniques would you explore?

---

## 📝 **License**

Specify how your project can be used by others. Choose an appropriate license and link it here (e.g., MIT, Apache 2.0). Make sure your Challenge Advisor approves of the selected license type. 

**Example:**
This project is licensed under the MIT License.

---

## 📄 **References** (Optional but encouraged)

Cite relevant papers, articles, or resources that supported your project.

---

## 🙏 **Acknowledgements** (Optional but encouraged)

Thank your Challenge Advisor, host company representatives, TA, and others who supported your project.

## Benchmark and human review

Run the notebook through **Versioned benchmark and answer review** with the saved artifacts in place. The default run requires no API calls. It exports a content-versioned folder under `data/processed/evaluation_runs/` with `summary.json`, retrieval metrics, the original 31 passages with source metadata, a separate q18 candidate review, answer review and evidence plans, historical answer checks, weaknesses, and accession collision evidence. See [Milestone 2 evaluation summary](eval/MILESTONE2_EVALUATION.md) for interpretation.

- In `summary.json`, find `passage_label_key` and `q18_label_key`. Copy `passage_review.csv` and `q18_after_review.csv` to `data/processed/evaluation_runs/human/passages_<key>.csv` using their respective keys. Fill `relevance_label` with 0 (irrelevant), 1 (partially relevant), or 2 (directly relevant). Enter optional `review_notes`, keeping the question ID, rank, and chunk ID. Rerun the export cell to incorporate the labels. Strict Precision@5 counts label 2 and requires five unique, valid labeled ranks. The original issuer-deduplicated lists can contain fewer than five indexed issuers.
- After authorized generation, copy or append the `ready_for_review` rows from `answer_review.csv` to `data/processed/evaluation_runs/human/answers.csv`. Keep `question_id` and `case_id`. Use `PASS`, `PARTIAL`, `FAIL`, or `NA` on each rubric dimension. In notes, identify unsupported claims, missing facts, and citations without supporting text. Keep `NA` for dimensions that genuinely do not apply. For q19, assess whether the answer explains the evidence limits and avoids a corpus-wide negative claim.
- Run the unit tests with `python -m unittest discover -s tests -v`.

### Explicitly authorized live evaluation

Set the team-approved endpoint privately in `OPENAI_BASE_URL` and use the existing credential configuration. Run the notebook normally and execute either call below manually only after budget approval.

```python
# At most three completion attempts per invocation. Uses prepared evidence without embedding calls.
answer_rows = run_benchmark_answers(
    approved=True, question_ids=("q17", "q18", "q19"), max_calls=3
)
benchmark_run, *_ = export_current_evaluation(answer_rows)

# Separate PIMCO smoke test. At most two embeddings and two completions.
smoke_files = run_pimco_smoke_test(approved=True)
```

The smoke test uses the April 2026 Toyota collections question and a cross-filing electronic-custody question. Check each saved JSON's selected scope, source text, citation IDs, and `manual_claim_support` field. Review the material factual claims against their referenced passages. The automated citation check covers IDs, source identity and metadata. Review the claims separately using the evidence saved with each response.

The challenge brief calls for a Milestone 3 benchmark evaluation and describes success on roughly 15–20 held-out questions: relevant deals among the top results and supporting text and sources for each answer. It lists retrieval precision and recall, LLM-as-judge, and manual spot checks under Suggested Approach. The repository has no separately numbered Task 2 specification, numeric acceptance threshold, or judge rubric. The versioned benchmark implements the evaluation deliverable described in the brief. For data sign-off, confirm the intended mix of filing forms and formats, the ticker requirement, and what constitutes a deal type. The brief names downloader libraries as examples and asks for SEC collection. It does not explicitly require a reproducible downloader deliverable.

### Milestone 3 Task 2 scoped benchmark

Run **Milestone 3 Task 2 — Scoped benchmark** after the existing evaluation cells. The notebook prints the latest immutable run folder under `data/processed/evaluation_runs/task2_v1/<index>/`. [Task 2 report](eval/MILESTONE3_TASK2_REPORT.md)

- Copy `passages.csv` into the sibling `human/passages.csv` input. Start with the first five `primary=True` rows for each positive question. Use 2 for directly relevant, 1 for partial, and 0 for irrelevant. Preserve question ID, method, rank, and chunk ID. Label alternate true-sale rankings if comparing variants.
- Use `candidate_evidence.csv` to inspect draft source proposals. For the seven unfiltered positive questions, copy `filing_truth.csv` to `human/filing_truth.csv`. Enter semicolon-separated verified filenames and set `complete_verified=yes` only after checking the complete 12-filing scope. Targeted-filing searches and q19 are excluded from filing discovery metrics.
- After authorized generation, copy or append ready answer rows to `human/answers.csv`. Use `PASS`, `PARTIAL`, `FAIL`, or `NA` with notes for factual correctness, evidence relevance, claim-level citation support, completeness, handling uncertainty, and abstention. Rerun the Task 2 export cell to recalculate scores.

With explicit approval, use batches of up to three. The eight uncached semantic cases are q19 and s01–s07. Existing hybrid snapshots remain a separate baseline. A semantic batch does not generate answers.

```python
run_task2_semantic(approved=True, question_ids=("s01", "s02", "s07"), max_calls=3)
# Rerun the Task 2 offline and export cells to add the saved semantic rankings.

run_task2_answers(approved=True, question_ids=("q17", "q18", "q19"), max_calls=3)
```

### Evaluation Review in Streamlit

Open **Evaluation Review** from the app navigation. **Passages** starts with 75 primary top-five rows. Switch to **All methods and ranks** to compare alternate runs. **Filing relevance** displays proposals and retrieved excerpts. Select only filings you have verified, and mark the set complete after checking the full indexed scope. **Answers** shows exact generated cases when a notebook export makes them available. **Held-out proposals** requires approval of each question and its candidate evidence before exporting an approved set. **Results** shows denominators by method and can recalculate a versioned evaluation export.

Saving writes to the benchmark and index-specific `human/` inputs. Revising a judgment requires confirmation, saves an audit receipt under `human/revisions/`, and rejects a stale evidence version or a conflicting update. The page makes no API requests. After explicit approval, use the existing notebook generation and export functions described above. Keep held-out approvals and results separate from the development benchmark. The approved held-out manifest is a preparation artifact. Held-out retrieval and scoring remain pending.

After all held-out proposals are independently approved, export the manifest from **Evaluation Review**. With explicit API approval, the notebook's `run_heldout_semantic(manifest_path, approved=True, question_ids=("h01", "h02", "h03"), max_calls=3)` writes semantic results to a separate directory for that approved manifest. No held-out results exist yet.

**AI Evaluation** within **Evaluation Review** displays the separate offline AI judgments in `eval/ai_passage_reviews.csv`. Filter by confidence or review recommendation, inspect the excerpt and original passage, and explicitly accept or override each suggestion. Only saved human decisions enter the human review inputs and metrics.
