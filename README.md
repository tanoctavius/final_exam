# Personal Research Portal (Local RAG) - For Final Exam (Octavius Tan otan)

Bcs this project utilizes a purely local architecture (`DeepSeek-R1` and `Llama 3.2` running via local Ollama instances), no external API keys, tokens, or internet connections are required. 

- **External APIs:** None.
- **Cache Files Included:** Local FAISS vector index (`data/vectorstore_llama/`) and BM25 index (`data/bm25_retriever.pkl`) generated during the ingestion phase.

## Command to Reproduce Artifacts
To automatically rebuild `evidence.json` and `eval.json` from the provided claims, run:
```bash
python generate_artifacts.py 

Included Cache & Index Files:
To support the reproduction of the analysis, the following local artifacts are included in the otan-code.zip:
data/vectorstore_llama/: The FAISS Dense Index containing vectorized paper chunks.

data/bm25_retriever.pkl: The BM25 Sparse Index for keyword-based retrieval.

logs/generation_cache.json: Cached LLM outputs for the evaluation suite.

data/data_manifest.csv: The source-of-truth mapping for paper IDs (P1-P10) and citations.

## Prerequisites

To run, you must have the following installed:

1.  **Ollama**: This application runs the Large Language Model locally.
    * Download from: https://ollama.com
    * Verify installation by running `ollama --version` in your terminal.
2.  **Python**: Version 3.10 or higher.
3.  **Git**: To clone this repository.

## Installation

### 1. Clone the Repository

```bash
git clone [https://github.com/tanoctavius/final_exam.git](https://github.com/tanoctavius/final_exam.git)
cd final_exam

```

### 2. Install Dependencies
If desired to replicate what was done during the exam. 

```bash
conda create -n final_exam python=3.10 -y
conda activate final_exam
pip install -r requirements.txt
```

### 3. Pull the AI Models

Open your terminal and run these commands to download Deepseek and the embedding model required for the vector database:

```bash
ollama pull deepseek-r1
ollama pull llama3.2
ollama pull nomic-embed-text

```

## Configuration and Data Setup

### 1. Added the 10 files required for the exam into the data manifest file

### 2. Build the Hybrid Database

This script performs Semantic Chunking and builds both the FAISS (Vector) and BM25 (Keyword) indices.

```bash
python ingest.py


## Project Structure

```text
.
├── data/
│   ├── raw/                  # Source PDFs
│   ├── vectorstore_llama/    # FAISS Dense Index
│   ├── bm25_retriever.pkl    # BM25 Sparse Index
│   └── data_manifest.csv     # Citation Metadata
├── logs/
│   ├── rag_logs.csv          # Interaction history
│   ├── generation_cache.json # Cached answers for Eval
│   └── evaluation_results.csv# Final RAGAs scores
├── src/
│   └── app.py                # Phase 3 Streamlit Web UI
├── ingest.py                 # Semantic + BM25 Ingestion
├── rag_pipeline.py           # Engine (Hybrid + Rerank + Logging)
├── generate_answers.py       # Eval Phase 1 (Generating)
├── evaluation.py             # Eval Phase 2 (Grading)
└── requirements.txt          # Pinned Dependency list
```


```