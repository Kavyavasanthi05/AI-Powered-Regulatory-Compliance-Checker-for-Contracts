#!/usr/bin/env python3
"""
main.py - Unified Regulatory Compliance & RAG Contract Analyzer
Option A: Full file with risk-based clause updating, Groq-powered rewrite,
diff highlighting, PDF output containing:
  1) ORIGINAL CONTRACT
  2) UPDATED CLAUSE (improved to reduce risk)
  3) HIGHLIGHTED DIFFERENCES
and the updated clause is also replaced inside the contract (saved as a separate PDF).
"""

import os
import re
from pathlib import Path
from typing import List, Dict
from dotenv import load_dotenv
import requests
import fitz  # PyMuPDF
from fpdf import FPDF
from difflib import ndiff

# LangChain / FAISS imports (assumes installed in your environment)
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# Load environment variables
load_dotenv()

# ----------------------------
# CONFIG
# ----------------------------
DATASET_PDF = r"D:\AI-Powered-Regulatory-Compliance-Checker-for-Contracts\data\Business_Compliance_Dataset.pdf"
INDEX_PATH = Path(os.getenv("INDEX_PATH", "./faiss_index"))
REBUILD_INDEX = os.getenv("REBUILD_INDEX", "True").lower() in ("1", "true", "yes")
TOP_K = int(os.getenv("TOP_K", 4))
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 900))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 150))

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("❌ GROQ_API_KEY missing in environment!")
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"

# ----------------------------
# HELPERS - PDF reading / cleaning
# ----------------------------
def read_pdf_text(path: str) -> str:
    doc = fitz.open(path)
    pages_text = [page.get_text() for page in doc]
    doc.close()
    return "\n".join(pages_text)

def clean_text_for_pdf(text: str) -> str:
    replacements = {
        "“": '"', "”": '"', "‘": "'", "’": "'",
        "•": "-", "–": "-", "—": "-", "…": "...",
        "→": "->", "←": "<-", "©": "(c)", "®": "(R)",
        "\t": " "
    }
    for k, v in replacements.items():
        text = text.replace(k, v)
    return text.encode("latin-1", "ignore").decode("latin-1")

# ----------------------------
# CONTRACT / CLAUSE EXTRACTION
# ----------------------------
def extract_contract_blocks(full_text: str) -> List[str]:
    CONTRACT_SPLIT_PATTERN = r"(Contract\s*#\d+\s*\|[\s\S]*?)(?=\nContract\s*#\d+\s*\||$)"
    matches = re.findall(CONTRACT_SPLIT_PATTERN, full_text, flags=re.IGNORECASE)
    return [m.strip() for m in matches if len(m.strip()) > 80]

def extract_regulations_from_text(contract_text: str) -> List[Dict[str, str]]:
    key_clause_patterns = {
        "Scope of Services": r"1\.\s*Scope of Services[\s\S]*?(?=\n2\.)",
        "Confidentiality": r"2\.\s*Confidentiality[\s\S]*?(?=\n3\.)",
        "Data Protection": r"3\.\s*Data Protection[\s\S]*?(?=\n4\.)",
        "Compliance & Audit Rights": r"4\.\s*Compliance & Audit Rights[\s\S]*?(?=\n5\.)",
        "Termination": r"5\.\s*Termination[\s\S]*?(?=\n6\.)",
        "Liability Limitation": r"6\.\s*Liability Limitation[\s\S]*?(?=\n7\.)",
        "Governing Law": r"7\.\s*Governing Law[\s\S]*?(?=$)"
    }

    clauses = []
    for idx, (title, pattern) in enumerate(key_clause_patterns.items(), 1):
        match = re.search(pattern, contract_text, flags=re.IGNORECASE)
        text = match.group(0).strip() if match else f"{title} clause not found."
        clauses.append({
            "id": idx,
            "title": title,
            "text": text
        })
    return clauses

# ----------------------------
# RELEVANCE / RISK
# ----------------------------
def relevance_between(contract_text: str, regulation_text: str) -> str:
    c = contract_text.lower()
    r = regulation_text.lower()
    r_words = [w for w in re.findall(r"\w+", r) if len(w) > 3]
    if not r_words:
        return "Low"
    matches = sum(1 for w in set(r_words) if w in c)
    score = matches / max(1, len(set(r_words)))
    if score > 0.7:
        return "High"
    elif score > 0.4:
        return "Medium"
    else:
        return "Low"

# ----------------------------
# DIFF GENERATOR
# ----------------------------
def generate_diff_text(old_clause: str, new_clause: str) -> str:
    diff_lines = list(ndiff(old_clause.splitlines(), new_clause.splitlines()))
    output = ["HIGHLIGHTED DIFFERENCES:\n"]
    for line in diff_lines:
        if line.startswith("+ "):
            output.append(f"[ADDED] {line[2:]}")
        elif line.startswith("- "):
            output.append(f"[REMOVED] {line[2:]}")
        elif line.startswith("? "):
            output.append(f"   ^ {line[2:]}")
        else:
            output.append(f"      {line[2:]}")
    return "\n".join(output)

# ----------------------------
# PDF SAVERS
# ----------------------------
def save_contract_with_update_and_diff(original_contract: str, old_clause: str, updated_clause: str, filename: str) -> Path:
    """
    Create a PDF that contains:
      - ORIGINAL CONTRACT (unchanged)
      - UPDATED CLAUSE (the new, regulation-compliant text)
      - HIGHLIGHTED DIFFERENCES (human readable diff)
    """
    diff_text = generate_diff_text(old_clause, updated_clause)

    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)

    # ORIGINAL CONTRACT
    pdf.set_font("Arial", "B", 12)
    pdf.multi_cell(0, 8, "ORIGINAL CONTRACT:\n")
    pdf.set_font("Arial", "", 12)
    pdf.multi_cell(0, 7, clean_text_for_pdf(original_contract))
    pdf.ln(10)

    # UPDATED CLAUSE
    pdf.set_font("Arial", "B", 12)
    pdf.multi_cell(0, 8, "UPDATED CLAUSE:\n")
    pdf.set_font("Arial", "", 12)
    pdf.multi_cell(0, 7, clean_text_for_pdf(updated_clause))
    pdf.ln(10)

    # HIGHLIGHTED DIFFERENCES
    pdf.set_font("Arial", "B", 12)
    pdf.multi_cell(0, 8, "HIGHLIGHTED DIFFERENCES:\n")
    pdf.set_font("Arial", "", 12)
    pdf.multi_cell(0, 7, clean_text_for_pdf(diff_text))

    out = Path(filename).with_suffix(".pdf")
    pdf.output(str(out))
    return out

def save_full_updated_contract(updated_contract_text: str, filename: str) -> Path:
    """
    Save the full contract text (with the clause replaced) as a PDF.
    """
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_font("Arial", "", 12)
    pdf.multi_cell(0, 7, clean_text_for_pdf(updated_contract_text))
    out = Path(filename).with_suffix(".pdf")
    pdf.output(str(out))
    return out

# ----------------------------
# FAISS / RAG helpers
# ----------------------------
def documents_from_contract_texts(contracts: List[str]) -> List[Document]:
    return [Document(page_content=c, metadata={"source": f"contract_{i}"}) for i, c in enumerate(contracts, 1)]

def split_documents(docs: List[Document]) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    return splitter.split_documents(docs)

def build_or_load_faiss(chunks: List[Document], rebuild: bool = REBUILD_INDEX) -> FAISS:
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    if rebuild or not INDEX_PATH.exists():
        print("🔁 Building FAISS index...")
        vs = FAISS.from_documents(chunks, embeddings)
        INDEX_PATH.mkdir(parents=True, exist_ok=True)
        vs.save_local(str(INDEX_PATH))
        print(f"✔ Saved index to {INDEX_PATH}")
    else:
        print("📦 Loading FAISS index...")
        vs = FAISS.load_local(str(INDEX_PATH), embeddings, allow_dangerous_deserialization=True)
    return vs

# ----------------------------
# GROQ CALL
# ----------------------------
def call_groq(prompt: str, max_tokens: int = 1500, temperature: float = 0.2) -> str:
    payload = {
        "model": "llama-3.3-70b-versatile",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens
    }
    headers = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}
    try:
        r = requests.post(GROQ_API_URL, json=payload, headers=headers, timeout=120)
        r.raise_for_status()
        data = r.json()
        # Defensive extraction
        if isinstance(data, dict) and "choices" in data and data["choices"]:
            message = data["choices"][0].get("message", {})
            return message.get("content", "") if isinstance(message, dict) else str(message)
        return str(data)
    except Exception as e:
        return f"Groq Error: {e}"

# ----------------------------
# CONTRACT UPDATE UTIL
# ----------------------------
def update_clause_in_contract(contract_text: str, clause_title: str, new_clause_text: str) -> str:
    """
    Replace clause in the contract by clause number and title.
    Expects clauses numbered 1..7 as in patterns used elsewhere.
    Replaces the whole clause block with the new block (including header).
    """
    number_lookup = {
        "Scope of Services": 1,
        "Confidentiality": 2,
        "Data Protection": 3,
        "Compliance & Audit Rights": 4,
        "Termination": 5,
        "Liability Limitation": 6,
        "Governing Law": 7,
    }
    clause_num = number_lookup.get(clause_title)
    if not clause_num:
        # fallback: try simple replace
        return contract_text.replace(clause_title, new_clause_text)

    # Ensure new_clause_text begins with "N. Title:" so it fits the contract format
    header = f"{clause_num}. {clause_title}:"
    # If user-provided new clause already contains header, keep it; else add header.
    if not re.match(rf"^\s*{clause_num}\.\s*{re.escape(clause_title)}\s*:", new_clause_text):
        new_block = header + "\n" + new_clause_text.strip() + "\n"
    else:
        new_block = new_clause_text.strip() + "\n"

    # Pattern to find the clause block up to the next numbered clause or end of doc
    pattern = rf"{clause_num}\.\s*{re.escape(clause_title)}[\s\S]*?(?=(?:\n{clause_num + 1}\.)|\Z)"
    updated = re.sub(pattern, lambda m: new_block, contract_text, flags=re.IGNORECASE)
    return updated

# ----------------------------
# APPLY REGULATION (RISK-BASED UPDATE)
# ----------------------------
def apply_regulation_to_contract(contracts, regs_list):
    """
    When user selects a clause, this function:
      - Shows original clause
      - Computes relevance/risk
      - Calls Groq with a risk-aware prompt to produce an updated clause that reduces risk
      - Replaces the clause inside the contract text
      - Saves:
          1) PDF with Original Contract + Updated Clause + Diff
          2) Full updated contract PDF (clause replaced)
    """
    cnum = int(input(f"Enter contract number (1-{len(contracts)}): "))
    original_contract = contracts[cnum - 1]
    regs = regs_list[cnum - 1]

    print("\nAvailable Clauses:")
    for r in regs:
        print(f"{r['id']}) {r['title']}")

    rnum = int(input(f"\nClause number: "))
    if rnum < 1 or rnum > len(regs):
        print("Invalid clause number.")
        return

    selected = regs[rnum - 1]
    clause_title = selected["title"]
    original_clause_text = selected["text"]

    # Compute existing risk / relevance
    current_risk = relevance_between(original_contract, original_clause_text)
    print(f"\nCurrent risk level for '{clause_title}': {current_risk}")

    # Build a risk-aware prompt for Groq
    update_prompt = f"""
You are a senior regulatory and risk compliance officer with deep expertise in GDPR, EU AI Act, PCI-DSS,
and global industry best-practices. You will produce an UPDATED clause that materially reduces legal,
privacy, security, and regulatory risk (i.e. move risk from {current_risk} to Low).

Rules:
- Keep the clause numbering/title separate from the body (we will add the header).
- The updated clause must be clearly different from the original, add specific obligations, and be
  actionable and legally enforceable.
- Include technical controls (encryption, access control, MFA, logging), data protection/GDPR elements
  (data minimization, purpose limitation, DPIA if AI/high risk), breach notification timings, and
  auditability requirements where relevant.
- Keep the update concise but thorough (3-10 lines preferred).
- Output ONLY the clause text (no commentary), do NOT repeat the prompt.

ORIGINAL CLAUSE ({clause_title}):
{original_clause_text}

Generate the UPDATED CLAUSE now:
"""

    print("\nGenerating updated clause (this calls Groq)...")
    updated_clause_body = call_groq(update_prompt).strip()

    # If Groq fails or returns an error string, fallback to a simple enhanced template
    if updated_clause_body.lower().startswith("groq error") or len(updated_clause_body) < 30:
        # Basic fallback: produce an expanded clause with common required controls
        updated_clause_body = (
            "Each Party shall maintain confidentiality of proprietary, sensitive, and personal data shared under this Agreement. "
            "The Parties shall implement reasonable and industry-standard administrative, technical and physical safeguards "
            "including encryption at rest and in transit, role-based access control, multi-factor authentication, logging, and "
            "regular security testing. The Parties shall comply with applicable data protection laws (including GDPR) and, "
            "where relevant, obligations under the EU AI Act. Confidential information shall not be disclosed to third parties "
            "except when required by law, and any unauthorized access or breach must be reported to the other Party within 72 hours."
        )

    # Format updated clause with header (so it can replace the block cleanly)
    clause_num_lookup = {
        "Scope of Services": 1,
        "Confidentiality": 2,
        "Data Protection": 3,
        "Compliance & Audit Rights": 4,
        "Termination": 5,
        "Liability Limitation": 6,
        "Governing Law": 7,
    }
    clause_num = clause_num_lookup.get(clause_title, rnum)
    updated_clause_full = f"{clause_num}. {clause_title}:\n{updated_clause_body.strip()}"

    # Replace clause inside the contract to produce the updated full contract
    updated_full_contract = update_clause_in_contract(original_contract, clause_title, updated_clause_full)

    # Prepare a PDF showing original contract, updated clause (body) and diff
    pdf_filename = f"contract_{cnum:03d}_original_and_updated_clause.pdf"
    save_contract_with_update_and_diff(original_contract, original_clause_text, updated_clause_full, pdf_filename)
    print(f"✔ PDF created: {pdf_filename}")

    # Save the full updated contract as a separate PDF
    updated_contract_filename = f"contract_{cnum:03d}_full_updated_contract.pdf"
    save_full_updated_contract(updated_full_contract, updated_contract_filename)
    print(f"✔ Full updated contract saved: {updated_contract_filename}")

    print(f"\nDone. Clause '{clause_title}' updated and saved. The updated clause aims to reduce the risk from {current_risk}.")

# ----------------------------
# MENU UTILITIES
# ----------------------------
def relevance_analysis_by_contract(contracts, regs_list):
    cnum = int(input(f"Enter contract number (1-{len(contracts)}): "))
    contract = contracts[cnum - 1]
    regs = regs_list[cnum - 1]
    print(f"\n=== Relevance for Contract {cnum} ===")
    for r in regs:
        level = relevance_between(contract, r["text"])
        print(f"{r['id']}) {r['title']} - {level}")

def view_extracted_regs_by_contract(contracts, regs_list):
    cnum = int(input(f"Enter contract number (1-{len(contracts)}): "))
    regs = regs_list[cnum - 1]
    print(f"\n--- Extracted Clauses for Contract #{cnum} ---")
    for r in regs:
        print(f"{r['id']}) {r['title']}")

# ----------------------------
# RAG / ANALYSIS
# ----------------------------
def run_full_analysis_on_text(contract_text: str) -> Dict[str, str]:
    analysis_prompt = f"""
You are a senior legal and compliance analyst.
Provide the following for the contract below:
1) FULL detailed interpretation of the contract.
2) Key Clauses summary.
3) Compliance issues with risk levels (High/Medium/Low).
4) Recommendations to mitigate issues.

Contract Text:
{contract_text}
"""
    full_analysis = call_groq(analysis_prompt)
    return {"full_analysis": full_analysis}

# ----------------------------
# MAIN
# ----------------------------
def main():
    print("\n📄 Extracting contracts & regulations from dataset PDF...")
    text = read_pdf_text(DATASET_PDF)
    contracts = extract_contract_blocks(text)
    regs_list = [extract_regulations_from_text(c) for c in contracts]
    print(f"✔ Extracted {len(contracts)} contract(s).")

    # Build FAISS and retriever for RAG
    docs = documents_from_contract_texts(contracts)
    chunks = split_documents(docs)
    retriever = build_or_load_faiss(chunks).as_retriever(search_kwargs={"k": TOP_K})

    cnum = int(input(f"Select contract for full RAG analysis (1-{len(contracts)}): "))
    contract_text = contracts[cnum - 1]

    print(f"\n🔍 Running full RAG + Groq analysis for Contract #{cnum}...")
    try:
        docs = retriever.get_relevant_documents(contract_text)
        context = "\n\n".join(d.page_content for d in docs) if docs else contract_text
    except Exception:
        context = contract_text

    results = run_full_analysis_on_text(context)
    print("\n=== FULL ANALYSIS ===\n")
    print(results["full_analysis"])

    while True:
        print("\n=== MENU ===")
        print("1) Relevance analysis")
        print("2) View extracted clauses")
        print("3) Apply clause update to contract (risk-aware)")
        print("4) Exit")
        choice = input("Choice: ").strip()
        if choice == "1":
            relevance_analysis_by_contract(contracts, regs_list)
        elif choice == "2":
            view_extracted_regs_by_contract(contracts, regs_list)
        elif choice == "3":
            apply_regulation_to_contract(contracts, regs_list)
        elif choice == "4":
            print("Bye!")
            break
        else:
            print("Invalid choice.")

if __name__ == "__main__":
    main()
