A log of the major prompts you used during the exam. Graders use this to understand your workflow, 
not to penalise you for using LLMs.
• Include at least 3 major prompts.
• For each prompt, state: (a) the tool/model name, (b) the purpose (what were you trying to 
achieve?), (c) the prompt text or a close paraphrase.
• If you pursued the automation points (Section 6), add a section titled "Automation and 
Reproducibility Notes" explaining what you automated and how to rerun it offline or in replay 
mode

Automation Scope:
This submission automates the generation of core grading artifacts to ensure objective evidence grounding and internal consistency: 

Evidence Grounding (generate_artifacts.py): Automates the retrieval of verbatim quotes and metadata (Page #, Section) for the 10 required claims, ensuring 100% traceability to the corpus PDFs. 

Metric Synthesis (generate_artifacts.py): Automatically calculates word counts, paper coverage, and citation counts for eval.json to prevent discrepancy penalties. 

Offline Reproducibility: The entire pipeline is built on a local architecture (Ollama) to ensure the grader can reproduce evidence.json without external API keys or internet access. 


How to Reproduce Artifacts (Offline Mode):
The following steps allow for the end-to-end reproduction of the submitted artifacts in an environment without internet access or API keys: 

Initialize Local Indices: Build the FAISS and BM25 search indices from the corpus:

Bash
python ingest.py


Generate Grading Artifacts: Run the automation script in offline mode to rebuild evidence.json and eval.json:

Bash
python generate_artifacts.py --mode offline


Replay Mode Support
The submission includes a cache/ directory containing the raw LLM responses used to draft the claims and summaries. To reproduce the final artifacts using these cached outputs (skipping local LLM inference): 


Bash
python generate_artifacts.py --mode replay