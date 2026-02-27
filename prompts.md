### **Major Prompts Used During Exam**

#### **Prompt 1: Thematic Synthesis for Literature Summary**

**(a) Tool/Model Name:** Gemini 3 Pro.

**(b) Purpose:** To identify cross-cutting intellectual patterns across the 10-paper corpus and organize them into exactly three distinct themes, as required by the rubric.

**(c) Prompt Text:** > "I have 10 research papers (P1-P10) regarding LLMs for scientific discovery. Analyze the provided abstracts and key findings. Instead of summarizing each paper individually, identify three high-level thematic groups that span multiple papers (e.g., 'Agent Autonomy' or 'Domain-Specific Pretraining'). For each theme, provide a concise synthesis statement and list which [P#] IDs belong to it. Ensure themes do not overlap significantly."

#### **Prompt 2: Gap Analysis for Future Research Directions**

**(a) Tool/Model Name:** Gemini 3 Pro.

**(b) Purpose:** To pinpoint specific limitations or "gaps" in the corpus and propose concrete, testable methodologies to address them.

**(c) Prompt Text:** > "Based on the corpus [P1-P10], identify a specific limitation regarding how these agents handle 'novelty' or 'out-of-distribution scientific tasks.' Define this as a 'Gap.' Then, propose a concrete technical 'Approach' involving a specific method (e.g., iterative symbolic verification) and an 'Evaluation' metric (e.g., a specific benchmark like MLGym). Keep the total word count for this direction between 60-90 words." 


#### **Prompt 3: Critical Comparison for Reflection Section**

**(a) Tool/Model Name:** Gemini 3 Pro.

**(b) Purpose:** To compare the focused "mini-survey" results against a broader, published comparison survey (S1-S4) to demonstrate critical thinking.

**(c) Prompt Text:** > "Compare my mini-survey (focused on the 10-paper corpus P1-P10) against the published survey [S#]. Identify one specific topic that the broader survey [S#] covers which my survey does not, and explain why that omission is justified given the corpus constraint. Conversely, identify one area where my deep-dive into the 10 specific papers reveals a nuance that [S#] glosses over. Provide a concrete evaluation weakness shared by both." 

---
### **Automation and Reproducibility Notes**

#### **Automation Scope**

This submission automates the generation of core grading artifacts to ensure objective evidence grounding and internal consistency:

**Evidence Grounding (`generate_artifacts.py`):** Automates the retrieval of verbatim quotes and metadata (Page #, Section) for the 10 required claims, ensuring 100% traceability to the corpus PDFs.

**Metric Synthesis (`generate_artifacts.py`):** Automatically calculates word counts, paper coverage, and citation counts for `eval.json` to prevent discrepancy penalties.

**Offline Reproducibility:** The entire pipeline is built on a local architecture (Ollama) to ensure the grader can reproduce `evidence.json` without external API keys or internet access.


#### **How to Reproduce Artifacts (Offline Mode)**
Due to the assignment restrictions for submission, the file paths might be off. If you would like the full project, please visit the github page: https://github.com/tanoctavius/final_exam (has everything for the entire assignment)


The following steps allow for the end-to-end reproduction of the submitted artifacts in an environment without internet access or API keys:

1. **Initialize Local Indices:** Build the FAISS and BM25 search indices from the corpus:
```bash
python ingest.py

```

2. **Generate Grading Artifacts:** Run the automation script in offline mode to rebuild `evidence.json` and `eval.json`:
```bash
python generate_artifacts.py --mode offline

```

#### **Replay Mode Support**

The submission includes a `cache/` directory containing the raw LLM responses used to draft the claims and summaries. To reproduce the final artifacts using these cached outputs (skipping local LLM inference):

```bash
python generate_artifacts.py --mode replay

```