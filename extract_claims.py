import json
import os
from langchain_community.vectorstores import FAISS
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate

INDEX_PATH = "data/vectorstore_llama"
EMBEDDING_MODEL = "nomic-embed-text"

def extract_claims():
    print("Initializing Claim Extractor...")
    
    if not os.path.exists(INDEX_PATH):
        raise FileNotFoundError(f"FAISS index not found at {INDEX_PATH}. Run ingest.py first.")
        
    print(f"Loading direct FAISS index from {INDEX_PATH}...")
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
    vectorstore = FAISS.load_local(INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
    
    # FIX 1: Swap to Llama 3.2 for massive speed improvements over DeepSeek-R1
    # Reduced num_ctx since we are pre-filtering chunks
    llm = ChatOllama(model="llama3.2", temperature=0, num_ctx=4096, timeout=120.0)
    
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
        ("user", "Context:\n{context}\n\nExtract a strong, falsifiable claim for this paper.")
    ])
    chain = prompt_template | llm
    
    claims_output = {}
    
    # Pre-fetch all documents from FAISS to do manual, robust filtering
    all_docs = list(vectorstore.docstore._dict.values())
    
    for i in range(1, 11):
        paper_id = f"P{i}"
        print(f"Extracting claim for {paper_id}...")
        
        # FIX 2: Case-insensitive, whitespace-stripped filtering bypasses FAISS exact-match errors
        paper_docs = [d for d in all_docs if str(d.metadata.get('source_id', '')).strip().upper() == paper_id]
        
        if not paper_docs:
            print(f"  -> ERROR: No context found for {paper_id}. If this persists, the PDF failed to load during ingest.py.")
            continue
            
        # Sort by length to grab the most substantive chunks for methodology/results
        paper_docs.sort(key=lambda x: len(x.page_content), reverse=True)
        top_docs = paper_docs[:3] 
        
        context = "\n".join([d.page_content for d in top_docs])
        
        raw_response = chain.invoke({"context": context}).content
        
        # Clean up the response
        clean_claim = raw_response.strip()
        if clean_claim.lower().startswith("here is"):
            clean_claim = clean_claim.split(":", 1)[-1].strip()
            
        claims_output[f"C{i}"] = clean_claim
        print(f"  -> Extracted: {clean_claim[:60]}...")

    # Output the required artifacts
    with open("claims.json", "w") as f:
        json.dump(claims_output, f, indent=4)
        
    # Generate the Markdown table for easy copy-pasting into paper.docx
    with open("claims_table.md", "w") as f:
        f.write("| claim id | claim text | paper_ids_used |\n")
        f.write("|---|---|---|\n")
        for i in range(1, 11):
            c_id = f"C{i}"
            p_id = f"P{i}"
            f.write(f"| {c_id} | {claims_output.get(c_id, 'CLAIM MISSING - Check ingest') } | [{p_id}] |\n")
            
    print(f"\nSuccess! Claims saved to claims.json and claims_table.md.")

if __name__ == "__main__":
    extract_claims()