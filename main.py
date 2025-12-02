#!/usr/bin/env python3
"""
main.py - Unified Regulatory Compliance & RAG Contract Analyzer
Updated: Shows extracted regulations, relevance risk, and applies improved clauses to full contract PDF.
"""

import os
import re
from pathlib import Path
from typing import List, Dict
from dotenv import load_dotenv  # type: ignore
import requests
import fitz  # type: ignore  # PyMuPDF
from fpdf import FPDF  # type: ignore

# LangChain / FAISS imports
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS  # type: ignore
from langchain_huggingface import HuggingFaceEmbeddings  # type: ignore

# Load env
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
# HELPERS
# ----------------------------
def read_pdf_text(path: str) -> str:
    doc = fitz.open(path)
    pages_text = [page.get_text() for page in doc]
    doc.close()
    return "\n".join(pages_text)


def extract_contract_blocks(full_text: str) -> List[str]:
    CONTRACT_SPLIT_PATTERN = r"(Contract\s*#\d+\s*\|[\s\S]*?)(?=(?:\nContract\s*#\d+\s*\|)|$)"
    matches = re.findall(CONTRACT_SPLIT_PATTERN, full_text, flags=re.IGNORECASE)
    return [m.strip() for m in matches if len(m.strip()) > 80]


# ----------------------------
# EXTRACT REGULATIONS / KEY CLAUSES
# ----------------------------
def extract_regulations_from_text(contract_text: str) -> List[Dict[str, str]]:
    key_clause_patterns = {
        "Scope of Services": r"Scope of Services.*?(?=Confidentiality|Data Protection|$)",
        "Confidentiality": r"Confidentiality.*?(?=Data Protection|Compliance & Audit Rights|$)",
        "Data Protection": r"Data Protection.*?(?=Compliance & Audit Rights|Termination|$)",
        "Compliance & Audit Rights": r"Compliance & Audit Rights.*?(?=Termination|Liability Limitation|$)",
        "Termination": r"Termination.*?(?=Liability Limitation|Governing Law|$)",
        "Liability Limitation": r"Liability Limitation.*?(?=Governing Law|$)",
        "Governing Law": r"Governing Law.*?$"
    }

    clauses = []
    for idx, (title, pattern) in enumerate(key_clause_patterns.items(), 1):
        match = re.search(pattern, contract_text, flags=re.IGNORECASE | re.DOTALL)
        text = match.group(0).strip() if match else f"{title} clause not found."
        clauses.append({
            "id": idx,
            "title": title,
            "text": text
        })
    return clauses


def relevance_between(contract_text: str, regulation_text: str) -> str:
    c = contract_text.lower()
    r = regulation_text.lower()
    r_words = [w for w in re.findall(r"\w+", r) if len(w) > 3]
    if not r_words: return "Low"
    matches = sum(1 for w in set(r_words) if w in c)
    score = matches / max(1, len(set(r_words)))
    if score > 0.7:
        return "High"
    elif score > 0.4:
        return "Medium"
    else:
        return "Low"


def clean_text_for_pdf(text: str) -> str:
    replacements = {
        "“": '"', "”": '"', "‘": "'", "’": "'",
        "•": "-", "–": "-", "—": "-", "…": "...",
        "\t": " "
    }
    for k, v in replacements.items():
        text = text.replace(k, v)
    return text.encode("latin-1", "ignore").decode("latin-1")


# ----------------------------
# FAISS
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
# GROQ API
# ----------------------------
def call_groq(prompt: str, max_tokens: int = 2000, temperature: float = 0.35) -> str:
    payload = {
        "model": "llama-3.3-70b-versatile",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens
    }
    headers = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}
    try:
        r = requests.post(GROQ_API_URL, json=payload, headers=headers, timeout=60)
        r.raise_for_status()
        data = r.json()
        return data["choices"][0]["message"]["content"]
    except Exception as e:
        return f"Groq Error: {e}"


# ----------------------------
# IMPROVE CLAUSE
# ----------------------------
def improve_regulation_clause(original_clause: str, clause_title: str, contract_context: str) -> str:
    prompt = f"""
You are a legal compliance expert. Improve the following clause to significantly LOWER compliance risk.

Rules:
- Keep it realistic and legally enforceable.
- Strengthen obligations, add clarity, improve security, add auditability.
- Do NOT change the section title or meaning.
- Make the clause safer and more compliant.

Contract Context:
{contract_context}

Original Clause ({clause_title}):
{original_clause}

Return ONLY the improved clause text.
"""
    return call_groq(prompt)


# ----------------------------
# SAVE UPDATED CONTRACT PDF
# ----------------------------
def save_full_updated_contract(contract_text: str, clause_title: str,
                               original_clause: str, improved_clause: str,
                               filename: str) -> Path:
    updated = contract_text.replace(original_clause, improved_clause)
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=12)
    pdf.set_font("Arial", size=11)
    updated_clean = clean_text_for_pdf(updated)
    pdf.multi_cell(0, 6, updated_clean)
    out_path = Path(filename).with_suffix(".pdf")
    pdf.output(str(out_path))
    return out_path


# ----------------------------
# APPLY REGULATION
# ----------------------------
def apply_regulation_to_contract(contracts, regs_list):
    cnum = int(input(f"Enter contract number (1-{len(contracts)}): "))
    regs = regs_list[cnum - 1]
    contract = contracts[cnum - 1]

    for r in regs:
        print(f"{r['id']}) {r['title']}")

    rnum = int(input(f"Select regulation number to apply (1-{len(regs)}): "))
    reg = regs[rnum - 1]
    clause_title = reg["title"]
    original_clause = reg["text"]

    print("\n🔧 Generating improved low-risk clause using Groq...")
    improved_clause = improve_regulation_clause(original_clause, clause_title, contract_context=contract)

    updated_filename = f"contract_{cnum:03d}_full_updated_contract.pdf"
    out_updated = save_full_updated_contract(contract, clause_title, original_clause, improved_clause, updated_filename)

    print(f"✔ Full updated contract saved to {out_updated}")


# ----------------------------
# VIEW REGULATIONS
# ----------------------------
def view_extracted_regulations(contracts, regs_list):
    cnum = int(input(f"Select contract number (1-{len(contracts)}): "))
    regs = regs_list[cnum - 1]
    print(f"\n--- Regulations / Key Clauses for Contract #{cnum} ---")
    for r in regs:
        print(f"{r['id']}) {r['title']}")


# ----------------------------
# RELEVANCE ANALYSIS
# ----------------------------
def relevance_analysis(contracts, regs_list):
    cnum = int(input(f"Select contract number (1-{len(contracts)}): "))
    contract = contracts[cnum - 1]
    regs = regs_list[cnum - 1]

    print(f"\n=== RELEVANCE ANALYSIS for Contract #{cnum} ===")
    for r in regs:
        risk = relevance_between(contract, r["text"])
        print(f"{r['id']}) {r['title']} - Risk Level: {risk}")


# ----------------------------
# MAIN
# ----------------------------
def main():
    print("\n📄 Extracting contracts & regulations from dataset PDF...")
    text = read_pdf_text(DATASET_PDF)
    contracts = extract_contract_blocks(text)
    regs_list = [extract_regulations_from_text(c) for c in contracts]
    print(f"✔ Extracted {len(contracts)} contracts.\n")

    docs = documents_from_contract_texts(contracts)
    chunks = split_documents(docs)
    retriever = build_or_load_faiss(chunks).as_retriever(search_kwargs={"k": TOP_K})

    cnum = int(input(f"Select contract for full RAG analysis (1-{len(contracts)}): "))
    contract_text = contracts[cnum - 1]

    print(f"\n🔍 Running full RAG + Groq analysis for Contract #{cnum}...")
    try:
        docs = retriever.get_relevant_documents(contract_text)
        context = "\n\n".join(d.page_content for d in docs) if docs else contract_text
    except:
        context = contract_text

    analysis_prompt = f"""
You are a senior legal and compliance analyst.
Provide full interpretation, key clauses, compliance issues, and recommendations.

Contract Text:
{context}
"""
    analysis = call_groq(analysis_prompt)
    print("\n=== FULL ANALYSIS ===\n")
    print(analysis)

    # MENU LOOP
    while True:
        print("\n=== MENU ===")
        print("1) Relevance analysis")
        print("2) View extracted regulations")
        print("3) Apply a regulation to a contract")
        print("4) Exit")

        choice = input("Choice: ").strip()

        if choice == "1":
            relevance_analysis(contracts, regs_list)
        elif choice == "2":
            view_extracted_regulations(contracts, regs_list)
        elif choice == "3":
            apply_regulation_to_contract(contracts, regs_list)
        elif choice == "4":
            print("Bye!")
            break
        else:
            print("Invalid choice.")


if __name__ == "__main__":
    main()
