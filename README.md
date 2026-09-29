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

**Provide step-by-step instructions so someone else can run your code and reproduce your results. Depending on your setup, include:**

* How to clone the repository
* How to install dependencies
* How to set up the environment
* How to access the dataset(s)
* How to run the notebook or scripts

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
- SEC metadata was successfully matched to **116 of the 122 filings**.
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
