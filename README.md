# Personal Research Portal (Local RAG) - Final Exam (otan)

This project utilizes a purely local architecture (`DeepSeek-R1` and `Llama 3.2` running via local Ollama instances). No external API keys, tokens, or internet connections are required for reproduction, supporting both **Offline** and **Replay** modes.

* 
**External APIs:** None.


* **Replay/Offline Support:** Supported. Graders do not need keys to run artifacts.

Due to the assignment restrictions for submission, the file paths might be off. If you would like the full project, please visit the github page: https://github.com/tanoctavius/final_exam (has everything for the entire assignment)

## Command to Reproduce Artifacts

To automatically rebuild `evidence.json` and `eval.json` from the claims, run the following command:

```bash
python generate_artifacts.py --mode offline

```

## Included Cache & Index Files

To support offline reproduction, the following local artifacts are included in the `otan-code.zip`:

* `data/vectorstore_llama/`: FAISS Dense Index containing vectorized paper chunks.
* `data/bm25_retriever.pkl`: BM25 Sparse Index for keyword-based retrieval.
* `data/data_manifest.csv`: Source-of-truth mapping for paper IDs (P1-P10).
* `claims.json`: Extracted research claims used as input for artifact generation.

## Prerequisites

1. 
**Ollama**: Required to run local models (`llama3.2` and `nomic-embed-text`).


2. 
**Python**: Version 3.10 or higher.



## Installation & Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt

```

### 2. Pull Local Models

```bash
ollama pull llama3.2
ollama pull nomic-embed-text

```

### 3. Execution Workflow

If you wish to re-run the entire pipeline from scratch, execute these in order:

1. 
`python ingest.py`: Builds the hybrid search indices.


2. 
`python extract_claims.py`: Generates the 10 core research claims (C1-C10).


3. 
`python generate_artifacts.py --mode offline`: Produces the required `evidence.json` and `eval.json`.



## Required Submission Files (Code ZIP)

This submission contains the following strictly required files for grading:

* 
`evidence.json`: Verbatim quotes and explanations for all 10 claims.


* 
`eval.json`: Self-reported metrics and spot-checks.


* 
`prompts.md`: Log of major prompts used during the exam.


* 
`README.md`: This reproduction guide.


* 
`requirements.txt`: Minimal dependency list.


---
