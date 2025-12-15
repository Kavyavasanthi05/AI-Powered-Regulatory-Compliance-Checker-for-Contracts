#!/usr/bin/env python3   
"""
main.py - Unified Regulatory Compliance & RAG Contract Analyzer
Updated: GDPR menu option removed, full functionality preserved
Added: Real-time compliance notifications via Google Sheets, Slack, and Email
"""

import os
import re
import json
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv  # type: ignore
import requests
import fitz  # type: ignore  # PyMuPDF
from fpdf import FPDF  # type: ignore
import smtplib
from email.message import EmailMessage

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
DATASET_PDF = os.getenv(
    "DATASET_PDF",
    r"D:\AI-Powered-Regulatory-Compliance-Checker-for-Contracts\data\Business_Compliance_Dataset.pdf"
)
INDEX_PATH = Path(os.getenv("INDEX_PATH", "./faiss_index"))
REBUILD_INDEX = os.getenv("REBUILD_INDEX", "True").lower() in ("1", "true", "yes")
TOP_K = int(os.getenv("TOP_K", 4))
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 900))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 150))

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("❌ GROQ_API_KEY missing in environment!")

GROQ_API_URL = os.getenv("GROQ_API_URL", "https://api.groq.com/openai/v1/chat/completions")
REGULATORY_SOURCES_PATH = Path(os.getenv("REGULATORY_SOURCES_PATH", "./regulatory_sources.json"))

# Notification config
SLACK_WEBHOOK_URL = os.getenv("SLACK_WEBHOOK_URL")
EMAIL_SENDER = os.getenv("EMAIL_SENDER")
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD")
EMAIL_RECEIVER = os.getenv("EMAIL_RECEIVER")
GOOGLE_SHEET_API_URL = os.getenv("GOOGLE_SHEET_API_URL")  # Assumes a webhook or Apps Script endpoint

# ----------------------------
# HELPERS - PDF read/write and text utilities
# ----------------------------
def read_pdf_text(path: str) -> str:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"PDF not found: {path}")
    doc = fitz.open(str(p))
    pages_text = [page.get_text() for page in doc]
    doc.close()
    return "\n".join(pages_text)

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
# CONTRACT SPLITTING & CLAUSE EXTRACTION
# ----------------------------
def extract_contract_blocks(full_text: str) -> List[str]:
    CONTRACT_SPLIT_PATTERN = r"(Contract\s*#\d+\s*\|[\s\S]*?)(?=(?:\nContract\s*#\d+\s*\|)|\Z)"
    matches = re.findall(CONTRACT_SPLIT_PATTERN, full_text, flags=re.IGNORECASE)
    contracts = [m.strip() for m in matches if len(m.strip()) > 80]
    if not contracts and full_text.strip():
        return [full_text.strip()]
    return contracts

KEY_CLAUSE_TITLES = [
    "Scope of Services",
    "Confidentiality",
    "Data Protection",
    "Compliance & Audit Rights",
    "Termination",
    "Liability Limitation",
    "Governing Law"
]

def build_clause_regex_for_title(title: str) -> re.Pattern:
    esc_title = re.escape(title)
    header_pattern = (
        r"(?P<header>^\s*\d+\.\s*" + esc_title + r"(?:\s*:|\s|$).*)"
        r"(?P<body>[\s\S]*?)(?=(?:\n\s*\d+\.\s+[A-Z0-9]|$))"
    )
    return re.compile(header_pattern, flags=re.IGNORECASE | re.MULTILINE)

def extract_regulations_from_text(contract_text: str) -> List[Dict[str, str]]:
    clauses = []
    for idx, title in enumerate(KEY_CLAUSE_TITLES, 1):
        pat = build_clause_regex_for_title(title)
        m = pat.search(contract_text)
        if m:
            header = m.group("header").strip()
            body = m.group("body").strip()
            full_text = (header + "\n" + body).strip()
        else:
            simple_pat = re.compile(
                rf"({re.escape(title)}\s*:?.*?)(?=(?:\n[A-Z][a-z]+:)|$)",
                flags=re.IGNORECASE | re.DOTALL
            )
            m2 = simple_pat.search(contract_text)
            full_text = m2.group(1).strip() if m2 else f"{title} clause not found."
        clauses.append({"id": idx, "title": title, "text": full_text})
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
# FAISS helpers
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
# GROQ API wrapper
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
        if "choices" in data and data["choices"]:
            choice = data["choices"][0]
            if isinstance(choice, dict) and "message" in choice and isinstance(choice["message"], dict):
                return choice["message"].get("content", "") or ""
            return choice.get("text", "") or ""
        return ""
    except Exception as e:
        return f"Groq Error: {str(e)}"

# ----------------------------
# Clause improvement
# ----------------------------
def improve_regulation_clause(original_clause: str, clause_title: str, contract_context: str) -> str:
    prompt = f"""
You are a senior legal compliance expert. Strengthen the following clause to ensure
maximum compliance with GDPR, industry security standards, and global privacy laws.

Requirements:
- Preserve numbering and section title exactly as in the original header line.
- Return ONLY the improved clause text, including the header line.
Contract Context:
{contract_context}
Original Clause ({clause_title}):
{original_clause}
"""
    return call_groq(prompt)

# ----------------------------
# Notifications
# ----------------------------
def send_slack_notification(message: str):
    if not SLACK_WEBHOOK_URL:
        return
    try:
        requests.post(SLACK_WEBHOOK_URL, json={"text": message}, timeout=5)
    except Exception:
        pass

def send_email_notification(subject: str, body: str):
    if not (EMAIL_SENDER and EMAIL_PASSWORD and EMAIL_RECEIVER):
        return
    try:
        msg = EmailMessage()
        msg.set_content(body)
        msg['Subject'] = subject
        msg['From'] = EMAIL_SENDER
        msg['To'] = EMAIL_RECEIVER
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
            smtp.login(EMAIL_SENDER, EMAIL_PASSWORD)
            smtp.send_message(msg)
    except Exception:
        pass

def send_google_sheet_update(payload: dict):
    if not GOOGLE_SHEET_API_URL:
        return
    try:
        requests.post(GOOGLE_SHEET_API_URL, json=payload, timeout=5)
    except Exception:
        pass

def notify_compliance_issue(contract_id: int, clause_title: str, risk_level: str):
    message = f"⚠ Compliance Alert:\nContract #{contract_id}\nClause: {clause_title}\nRisk Level: {risk_level}"
    send_slack_notification(message)
    send_email_notification(f"Compliance Alert - Contract #{contract_id}", message)
    send_google_sheet_update({"contract_id": contract_id, "clause": clause_title, "risk": risk_level})

# ----------------------------
# Clause replacement & PDF save
# ----------------------------
def replace_clause_in_contract(contract_text: str, original_clause_text: str, improved_clause_text: str) -> str:
    first_line = original_clause_text.splitlines()[0].strip() if original_clause_text.strip() else ""
    header_search = re.match(r"^\s*(\d+\.\s*.+?[:]?\s*)$", first_line)
    if header_search:
        header_line = header_search.group(1)
        title_text = re.sub(r"^\s*\d+\.\s*", "", header_line).strip().rstrip(":").strip()
        pat = build_clause_regex_for_title(title_text)
        m = pat.search(contract_text)
        if m:
            start, end = m.span()
            return contract_text[:start] + improved_clause_text.strip() + contract_text[end:]
    if original_clause_text and original_clause_text in contract_text:
        return contract_text.replace(original_clause_text, improved_clause_text)
    return contract_text + "\n\n" + improved_clause_text

def save_full_updated_contract(contract_text: str, updated_contract_text: str, filename: str) -> Path:
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=12)
    pdf.add_page()
    pdf.set_font("Arial", size=11)
    updated_clean = clean_text_for_pdf(updated_contract_text)
    pdf.multi_cell(0, 6, updated_clean)
    out_path = Path(filename).with_suffix(".pdf")
    pdf.output(str(out_path))
    return out_path

# ----------------------------
# Regulatory sources
# ----------------------------
def load_regulatory_sources(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        sample = [
            {"id": "gdpr", "title": "EU GDPR (reference)", "source": "https://eur-lex.europa.eu/eli/reg/2016/679/oj", "type": "url"}
        ]
        path.write_text(json.dumps(sample, indent=2), encoding="utf-8")
        return sample
    return json.loads(path.read_text(encoding="utf-8"))

def fetch_regulatory_text(source: Dict[str, str]) -> Optional[str]:
    try:
        if source.get("type") == "local":
            p = Path(source.get("source", ""))
            if p.exists():
                return p.read_text(encoding="utf-8")
        elif source.get("type") == "url":
            r = requests.get(source.get("source", ""), timeout=10)
            if r.status_code == 200:
                return r.text
        return None
    except Exception:
        return None

# ----------------------------
# CLI actions
# ----------------------------
def apply_regulation_to_contract(contracts: List[str], regs_list: List[List[Dict[str, str]]]):
    if not contracts:
        print("No contracts loaded.")
        return
    try:
        cnum = int(input(f"Enter contract number (1-{len(contracts)}): ").strip())
        if not (1 <= cnum <= len(contracts)):
            cnum = 1
    except ValueError:
        cnum = 1
    regs = regs_list[cnum - 1]
    contract = contracts[cnum - 1]
    try:
        rnum = int(input(f"Select regulation number to apply (1-{len(regs)}): ").strip())
        if not (1 <= rnum <= len(regs)):
            rnum = 1
    except ValueError:
        rnum = 1
    reg = regs[rnum - 1]
    clause_title = reg["title"]
    original_clause = reg["text"]
    improved_clause = improve_regulation_clause(original_clause, clause_title, contract)
    updated_contract_text = replace_clause_in_contract(contract, original_clause, improved_clause)
    updated_filename = f"contract_{cnum:03d}_full_updated_contract.pdf"
    out_updated = save_full_updated_contract(contract, updated_contract_text, updated_filename)
    print(f"✔ Full updated contract saved to {out_updated}")
    # Notify for every clause
    risk_level = relevance_between(contract, original_clause)
    notify_compliance_issue(cnum, clause_title, risk_level)

def relevance_analysis(contracts: List[str], regs_list: List[List[Dict[str, str]]]):
    if not contracts:
        print("No contracts available.")
        return
    try:
        cnum = int(input(f"Select contract number (1-{len(contracts)}): ").strip())
        if not (1 <= cnum <= len(contracts)):
            cnum = 1
    except ValueError:
        cnum = 1
    contract = contracts[cnum - 1]
    regs = regs_list[cnum - 1]
    print(f"\n=== RELEVANCE ANALYSIS for Contract #{cnum} ===")
    for r in regs:
        risk = relevance_between(contract, r["text"])
        print(f"{r['id']}) {r['title']} - Risk Level: {risk}")
        # Notify for every clause
        notify_compliance_issue(cnum, r["title"], risk)

def view_extracted_regulations(contracts: List[str], regs_list: List[List[Dict[str, str]]]):
    if not contracts:
        print("No contracts available.")
        return
    try:
        cnum = int(input(f"Select contract number (1-{len(contracts)}): ").strip())
        if not (1 <= cnum <= len(contracts)):
            cnum = 1
    except ValueError:
        cnum = 1
    regs = regs_list[cnum - 1]
    print(f"\n--- Regulations / Key Clauses for Contract #{cnum} ---")
    for r in regs:
        print(f"{r['id']}) {r['title']}")

# ----------------------------
# MAIN
# ----------------------------
def main():
    if not Path(DATASET_PDF).exists():
        print(f"Dataset PDF not found at: {DATASET_PDF}")
        sys.exit(1)

    print("\n📄 Extracting contracts & regulations from dataset PDF...")
    text = read_pdf_text(DATASET_PDF)
    contracts = extract_contract_blocks(text)
    if not contracts:
        print("No contracts found in dataset.")
        sys.exit(1)

    regs_list = [extract_regulations_from_text(c) for c in contracts]
    print(f"✔ Extracted {len(contracts)} contracts.\n")

    docs = documents_from_contract_texts(contracts)
    chunks = split_documents(docs)
    try:
        retriever = build_or_load_faiss(chunks).as_retriever(search_kwargs={"k": TOP_K})
    except Exception as e:
        print(f"FAISS error: {e}. Continuing without retriever.")
        retriever = None

    regulatory_sources = load_regulatory_sources(REGULATORY_SOURCES_PATH)

    # --- Ask user to select contract for full RAG + Groq analysis ---
    try:
        cnum = int(input(f"Select contract for full RAG + Groq analysis (1-{len(contracts)}): ").strip())
        if not (1 <= cnum <= len(contracts)):
            cnum = 1
    except ValueError:
        cnum = 1

    contract_text = contracts[cnum - 1]

    print(f"\n🔍 Running full RAG + Groq analysis for Contract #{cnum}...")
    try:
        if retriever:
            try:
                docs = retriever.get_relevant_documents(contract_text)
            except AttributeError:
                docs = retriever._get_relevant_documents(contract_text)
            context = "\n\n".join(d.page_content for d in docs) if docs else contract_text
        else:
            context = contract_text
    except Exception:
        context = contract_text

    analysis_prompt = f"""
You are a senior legal and compliance analyst.
Provide full interpretation, key clauses, compliance issues, and recommendations.

Contract Text:
{context}
"""
    analysis = call_groq(analysis_prompt)
    analysis = re.sub(r"\*\*Additional Recommendations[\s\S]*", "", analysis)
    print("\n=== FULL ANALYSIS ===\n")
    print(analysis)

    # --- Menu loop ---
    while True:
        print("\n=== MENU ===")
        print("1) Relevance analysis")
        print("2) View extracted regulations")
        print("3) Apply a regulation to a contract (improve clause) -- saves PDF only")
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
