import json
import os
import re
from langchain_community.vectorstores import FAISS
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate

INDEX_PATH = "data/vectorstore_llama"
EMBEDDING_MODEL = "nomic-embed-text"

def extract_claims():
    print("Initializing Claim Extractor...")
    
    # 1. Direct FAISS Load: Bypass the Cross-Encoder to prevent target chunks from being filtered out.
    if not os.path.exists(INDEX_PATH):
        raise FileNotFoundError(f"FAISS index not found at {INDEX_PATH}. Run ingest.py first.")
        
    print(f"Loading direct FAISS index from {INDEX_PATH}...")
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
    vectorstore = FAISS.load_local(INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
    
    # 2. Model Sync: Matches the parameters defined in your generate_answers.py
    llm = ChatOllama(model="deepseek-r1", temperature=0, num_ctx=8192, timeout=600.0)
    
    # 3. Rubric Alignment: Forces the falsifiable, 1-sentence structure needed for maximum points
    system_prompt = """
    You are a rigorously graded Senior Research Scientist. Your task is to extract EXACTLY ONE specific, falsifiable claim from the provided context.
    
    A STRONG CLAIM MUST:
    - Be exactly ONE sentence.
    - Name a specific method, result, number, or comparison (e.g., 'Method X reduces error by 15%').
    - Be falsifiable (provable or disprovable by data).
    
    CRITICAL: Output ONLY the single claim sentence. Do not include any introductory text, concluding remarks, or conversational filler.
    """
    
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("user", "Paper ID: {paper_id}\nContext: {context}\n\nExtract a strong, falsifiable claim for this paper.")
    ])
    chain = prompt_template | llm
    
    claims_output = {}
    
    for i in range(1, 11):
        paper_id = f"P{i}"
        print(f"Extracting claim for {paper_id}...")
        
        # 4. Strict Metadata Filtering: Guarantees context is exclusively from the target paper
        retriever = vectorstore.as_retriever(
            search_kwargs={"k": 5, "filter": {"source_id": paper_id}}
        )
        
        # Broad keywords to grab methodology and results chunks
        docs = retriever.invoke("quantitative results metrics methods findings conclusion")
        context = "\n".join([d.page_content for d in docs])
        
        if not context:
            print(f"Warning: No context found for {paper_id}. Ensure your data_manifest.csv source_ids match exactly.")
            continue
            
        raw_response = chain.invoke({"paper_id": paper_id, "context": context}).content
        
        # 5. Regex Sync: Implements the exact DeepSeek tag scrubber from generate_answers.py
        clean_claim = re.sub(r'<think>.*?</think>', '', raw_response, flags=re.DOTALL).strip()
        
        # Fallback cleanup just in case DeepSeek ignores the "no filler" system prompt
        if clean_claim.lower().startswith("here is"):
            clean_claim = clean_claim.split(":", 1)[-1].strip()
            
        claims_output[f"C{i}"] = clean_claim
        print(f"  -> Extracted: {clean_claim[:60]}...")

    with open("claims.json", "w") as f:
        json.dump(claims_output, f, indent=4)
        
    print(f"\nSuccess! 10 claims saved to claims.json. Ready for generate_artifacts.py.")

if __name__ == "__main__":
    extract_claims()