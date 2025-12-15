# ------------------------------------------------------  
# streamlit_app.py (FINAL VERSION - Modified)
# ------------------------------------------------------

import os
import re
import io
import time
import logging
from pathlib import Path
from typing import List, Dict

import streamlit as st
from dotenv import load_dotenv
import pandas as pd
from fpdf import FPDF
import requests
import smtplib
from email.message import EmailMessage


# Load environment
load_dotenv()
logger = logging.getLogger("app_streamlit")
logger.setLevel(logging.INFO)

DEFAULT_INDEX_DIR = Path(os.getenv("INDEX_PATH", "./faiss_index"))
DEFAULT_TOP_K = int(os.getenv("TOP_K", 4))
DEFAULT_CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 900))
DEFAULT_CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 150))
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_API_URL = os.getenv("GROQ_API_URL", "https://api.groq.com/openai/v1/chat/completions")

EMAIL_SENDER = os.getenv("EMAIL_SENDER", "")
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD", "")
EMAIL_RECEIVER = os.getenv("EMAIL_RECEIVER", "")

# PDF reading
try:
    import fitz
except Exception:
    fitz = None

# Embedding / FAISS
try:
    import faiss
    from sentence_transformers import SentenceTransformer
    FAISS_AVAILABLE = True
except Exception:
    FAISS_AVAILABLE = False

top_k = DEFAULT_TOP_K

# ---------------------------- Utilities ----------------------------
@st.cache_data(show_spinner=False)
def read_pdf_text_bytes(file_bytes: bytes) -> str:
    if not fitz or not file_bytes:
        return ""
    try:
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        pages = [p.get_text() for p in doc]
        doc.close()
        return "\n".join(pages)
    except Exception as e:
        logger.exception("Failed to read PDF bytes: %s", e)
        return ""

@st.cache_data(show_spinner=False)
def extract_contract_blocks(full_text: str) -> List[str]:
    CONTRACT_SPLIT_PATTERN = r"(Contract\s*#\d+\s*\|[\s\S]*?)(?=(?:\nContract\s*#\d+\s*\|)|\Z)"
    matches = re.findall(CONTRACT_SPLIT_PATTERN, full_text, flags=re.IGNORECASE)
    blocks = [m.strip() for m in matches if len(m.strip()) > 80]
    if not blocks:
        parts = [p.strip() for p in re.split(r"\f|\n-{5,}\n|\n\n", full_text) if len(p.strip()) > 80]
        if parts:
            return parts
    return blocks

def clean_text_for_pdf(text: str) -> str:
    replacements = {"“": '"', "”": '"', "‘": "'", "’": "'", "•": "-", "–": "-", "—": "-", "…": "...", "\t": " "}
    for k, v in replacements.items():
        text = text.replace(k, v)
    return text

def save_text_as_pdf_bytes(text: str) -> bytes:
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=12)
    base_dir = os.path.dirname(__file__)
    font_path = os.path.join(base_dir, "fonts", "DejaVuSans.ttf")

    safe_text = clean_text_for_pdf(text)
    try:
        if os.path.exists(font_path):
            pdf.add_font("DejaVu", "", font_path, uni=True)
            pdf.set_font("DejaVu", size=11)
        else:
            pdf.set_font("Arial", size=11)
    except Exception:
        pdf.set_font("Arial", size=11)

    pdf.add_page()
    pdf.multi_cell(0, 6, safe_text)
    try:
        s = pdf.output(dest='S').encode('latin-1')
        return s
    except Exception as e:
        logger.exception("PDF generation error: %s", e)
        bio = io.BytesIO()
        pdf.output(bio)
        bio.seek(0)
        return bio.read()

# ---------------------------- Clause extraction ----------------------------
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
        pattern = build_clause_regex_for_title(title)
        m = pattern.search(contract_text)
        if m:
            header = m.group("header").strip()
            body = m.group("body").strip()
            full = (header + "\n" + body).strip()
        else:
            full = f"{title} clause not found."
        clauses.append({"id": idx, "title": title, "text": full})
    return clauses

# ---------------------------- Relevance ----------------------------
def relevance_between(contract_text: str, regulation_text: str) -> float:
    c = (contract_text or "").lower()
    r = (regulation_text or "").lower()
    if "clause not found" in r:
        return 0.0
    if len(r.strip()) < 80:
        return 0.15
    r_words = [w for w in re.findall(r"\w+", r) if len(w) > 4]
    if not r_words:
        return 0.1
    matches = sum(1 for w in set(r_words) if w in c)
    coverage = matches / max(1, len(set(r_words)))
    if matches >= 6 and coverage >= 0.45:
        score = 0.75
    elif matches >= 4:
        score = 0.55
    elif matches >= 2:
        score = 0.35
    else:
        score = 0.15
    return float(score)

def relevance_label(score: float) -> str:
    if score > 0.6:
        return "High"
    if score > 0.3:
        return "Medium"
    return "Low"

# ---------------------------- Retriever ----------------------------
@st.cache_data(show_spinner=False)
def load_sentence_transformer_model(name="all-mpnet-base-v2"):
    try:
        return SentenceTransformer(name)
    except Exception as e:
        logger.warning("SentenceTransformer unavailable: %s", e)
        return None

class SimpleInMemoryRetriever:
    def __init__(self, texts, model=None):
        self.texts = texts
        self.model = model
        if model:
            try:
                self.embs = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
            except Exception:
                self.embs = None
        else:
            self.embs = None

    def get_relevant_documents(self, query: str, k=4):
        if self.embs is None:
            scored = []
            q = query.lower()
            for t in self.texts:
                score = sum(1 for w in set(re.findall(r"\w+", q)) if w in t.lower())
                scored.append((score, t))
            scored.sort(reverse=True)
            return [type("D", (), {"page_content": doc}) for _, doc in scored[:k]]
        import numpy as np
        qv = self.model.encode([query], convert_to_numpy=True)[0]
        sims = (self.embs @ qv) / (np.linalg.norm(self.embs, axis=1) * np.linalg.norm(qv) + 1e-10)
        idx = np.argsort(-sims)[:k]
        return [type("D", (), {"page_content": self.texts[i]}) for i in idx]

# ---------------------------- LLM ----------------------------
def call_groq(prompt: str, max_tokens=1500, temperature=0.25) -> str:
    if not GROQ_API_KEY:
        return "Missing GROQ_API_KEY."
    payload = {
        "model": "llama-3.3-70b-versatile",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature
    }
    headers = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}
    try:
        r = requests.post(GROQ_API_URL, json=payload, headers=headers, timeout=60)
        r.raise_for_status()
        data = r.json()
        if isinstance(data, dict) and data.get('choices'):
            return data['choices'][0]['message'].get('content', '')
        return str(data)
    except Exception as e:
        logger.exception("LLM call failed: %s", e)
        return f"Model error: {e}"

def improve_regulation_clause(original, title, context):
    prompt = f"""
You are a senior legal compliance expert.
Strengthen this clause for regulatory compliance.

Contract Context:
{context}

Original Clause ({title}):
{original}

Return ONLY the improved clause (header + body).
"""
    return call_groq(prompt).replace("**", "")

# ---------------------------- Email ----------------------------
def send_email_notification(subject, body, sender, password, receiver, attachment_bytes=None, attachment_name="contract.pdf"):
    if not (sender and password and receiver):
        logger.warning("Email credentials incomplete.")
        return False
    try:
        msg = EmailMessage()
        msg.set_content(body)
        msg["Subject"] = subject
        msg["From"] = sender
        msg["To"] = receiver

        if attachment_bytes:
            msg.add_attachment(attachment_bytes, maintype="application", subtype="pdf", filename=attachment_name)

        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
            smtp.login(sender, password)
            smtp.send_message(msg)
        return True
    except Exception as e:
        logger.exception("Failed to send email: %s", e)
        return False

# ---------------------------- Streamlit Layout ----------------------------
st.set_page_config(page_title="Compliance Dashboard", layout="wide")

st.markdown("""
<style>
textarea:disabled {
    cursor: text !important;
    pointer-events: auto !important; 
}
</style>
""", unsafe_allow_html=True)

# Session state init
for key, default in {
    "contracts": [], "regs_list": [], "retriever": None, "notif_log": [], "updated_contracts": [],
    "last_pdf_bytes": None, "last_analysis": "", "last_improved": None, "weak_clauses": [],
    "overall_risk": 0, "risk_label": "🟢 Low Risk", "updated_versions": {},
}.items():
    st.session_state.setdefault(key, default)

# Sidebar
st.sidebar.title("app")
page = st.sidebar.radio("Navigation", [
    "Dashboard", "Upload Contract", "Risk Analysis",
    "Regulatory Updates", "Amendment System", "AI Chatbot"
])

# ---------------------------- Dashboard ----------------------------
if page == "Dashboard":
    st.markdown("""
    <h1 style="text-align:center; font-size:40px;">
        ⚖️ AI-Powered Regulatory Compliance System
    </h1>
    <p style="text-align:center; opacity:0.8;">
        Monitor contract compliance, apply automatic amendments, and interact with an AI assistant.
    </p>
    """, unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("""
    <div style="background:#0b1220; padding:18px; border-radius:10px;">
      <h3 style="color:#E5E7EB;">How to use</h3>
      <ol style="color:#D1D5DB;">
        <li>Go to <strong>📤 Upload Contract</strong> to upload PDF contracts or paste contract text.</li>
        <li>Use <strong>🛡️ Risk Analysis</strong> for RAG + relevance scoring.</li>
        <li>Use <strong>📝 Amendment System</strong> to generate and apply improved clauses.</li>
        <li>Use <strong>📘 Regulatory Updates</strong> to view key extracted clauses.</li>
        <li>Use <strong>🤖 AI Chatbot</strong> to query the LLM over contract data.</li>
      </ol>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")
    st.caption("Tip: Configure email ENV variables for notifications.")

# ---------------------------- Upload Contract ----------------------------
elif page == "Upload Contract":
    st.header("📤 Upload Contract")
    uploaded = st.file_uploader("Upload a multi-contract PDF", type=["pdf"])
    if st.button("Load Uploaded Document"):
        text = ""
        if uploaded:
            try:
                file_bytes = uploaded.read()
                text = read_pdf_text_bytes(file_bytes) if fitz else ""
            except Exception as e:
                st.warning(f"Failed to read uploaded PDF: {e}")
                text = ""
        if text and text.strip():
            contracts = extract_contract_blocks(text)
            if not contracts:
                contracts = [text]
            st.session_state.contracts = contracts
            st.session_state.regs_list = [extract_regulations_from_text(c) for c in contracts]
            st.session_state.updated_contracts = list(contracts)
            st.success(f"Loaded {len(contracts)} contract(s).")
        else:
            st.warning("No text found. Upload a PDF or paste text.")

    if st.session_state.contracts:
        show_preview = st.checkbox("Show text preview", value=False)
        if show_preview:
            for i, c in enumerate(st.session_state.contracts, 1):
                with st.expander(f"Contract #{i}"):
                    st.text_area("", value=c[:3000], height=180)

# ---------------------------- Risk Analysis ----------------------------
elif page == "Risk Analysis":
    st.header("🛡️ Compliance Risk Analysis")

    if not st.session_state.contracts:
        st.info("Upload a contract first.")
    else:
        # Load the contract
        c_choice = st.selectbox(
            "Select a contract to analyze:", 
            list(range(1, len(st.session_state.contracts) + 1))
        )
        idx = c_choice - 1
        contract = st.session_state.contracts[idx]

        # Helper to explain weak clauses
        def explain_weak_clause(title, text):
            prompt = f"""
            The following clause is weak or missing in a contract:

            Title: {title}
            Clause Text: {text}

            Explain briefly in 2–3 sentences WHY this clause is considered weak or missing.
            """
            try:
                return call_groq(prompt)
            except:
                return "Explanation unavailable."

        # Helper to score relevance between contract and regulation
        def relevance_between_risk_page(contract_text: str, regulation_text: str) -> float:
            c = (contract_text or "").lower()
            r = (regulation_text or "").lower()
            if "clause not found" in r or len(r.strip()) < 50:
                return 0.0
            r_words = [w for w in re.findall(r"\w+", r) if len(w) > 4] 
            if not r_words:
                return 0.0
            matches = sum(1 for w in set(r_words) if w in c)
            coverage = matches / max(1, len(set(r_words)))
            if coverage >= 0.6:
                return 1.0
            elif coverage >= 0.35:
                return 0.6
            elif coverage > 0.1:
                return 0.4
            else:
                return 0.0

        # Run Compliance Risk Check
        if st.button("Run Compliance Risk Check"):
            weak_clauses = []
            scores = []

            for reg in st.session_state.regs_list[idx]:
                score = relevance_between_risk_page(contract, reg["text"]) 
                scores.append(score)
                if score < 1.0:
                    explanation = explain_weak_clause(reg["title"], reg["text"])
                    weak_clauses.append({
                        "title": reg["title"], 
                        "text": reg["text"], 
                        "score": score, 
                        "reason": explanation
                    })

            # Calculate overall risk
            risk_percent = int((1 - sum(scores)/len(scores)) * 100) if scores else 0
            if risk_percent >= 70:
                risk_label = "🔴 High Risk"
            elif risk_percent >= 30:
                risk_label = "🟡 Medium Risk"
            else:
                risk_label = "🟢 Low Risk"

            # Store in session state
            st.session_state.overall_risk = risk_percent
            st.session_state.risk_label = risk_label
            st.session_state.weak_clauses = weak_clauses
            st.session_state.show_risk_summary = True

            st.success("✅ Compliance Risk Check completed.")

        # Display Risk Summary only after running
        if st.session_state.get("show_risk_summary"):
            st.subheader("📊 Risk Summary")
            st.markdown(
                f"### Overall Risk Level \n**{st.session_state.overall_risk}%** — {st.session_state.risk_label}"
            )
            st.markdown("---")
            st.subheader("⚠️ Weak / Missing Clauses")
            if st.session_state.weak_clauses:
                for clause in st.session_state.weak_clauses:
                    st.markdown(f"### 🟠 {clause['title']}")
                    st.text_area(f"Clause Text — {clause['title']}", value=clause['text'], height=120)
                    st.markdown(f"**Reason:** {clause['reason']}")
            else:
                st.success("No weak or missing clauses. Contract meets all regulatory requirements.")


# ---------------------------- Regulatory Updates ----------------------------
elif page == "Regulatory Updates":
    st.header("📘 Regulatory Update Monitor")
    st.write("This page shows only regulation changes, not full internal JSON structure.")
    if not st.session_state.contracts:
        st.info("Nothing loaded.")
    else:
        c_choice = st.selectbox("Contract", list(range(1, len(st.session_state.contracts) + 1)))
        idx = c_choice - 1
        current_regs = st.session_state.regs_list[idx]
        if "prev_regs_list" not in st.session_state:
            st.session_state.prev_regs_list = [None] * len(st.session_state.contracts)
        prev_regs = st.session_state.prev_regs_list[idx]

        with st.expander("📄 View current regulation titles"):
            for r in current_regs:
                st.markdown(f"- **{r['title']}**")

        st.markdown("---")
        st.subheader("⚠️ Detected Regulation Changes")
        def compare_regulations(old, new):
            if old is None: return []
            changes = []
            old_titles = {o["title"]: o["text"] for o in old}
            new_titles = {n["title"]: n["text"] for n in new}
            for title, text in old_titles.items():
                if title not in new_titles:
                    changes.append(f"❌ Removed: **{title}**")
                elif new_titles[title] != text:
                    changes.append(f"🔄 Updated: **{title}**")
            for title in new_titles:
                if title not in old_titles:
                    changes.append(f"➕ Added: **{title}**")
            return changes

        changes = compare_regulations(prev_regs, current_regs)
        if changes:
            for c in changes:
                st.warning(c)
        else:
            st.success("No changes detected between previous and current regulations.")

        if st.button("Refresh Regulations"):
            st.session_state.prev_regs_list[idx] = [r.copy() for r in current_regs]
            st.session_state.regs_list[idx] = extract_regulations_from_text(st.session_state.contracts[idx])
            st.success("Regulations refreshed!")

# ---------------------------- Amendment System ----------------------------
elif page == "Amendment System":
    st.header("🛠️ Amendment System")
    if not st.session_state.contracts:
        st.info("Upload a contract first.")
    else:
        c_choice = st.selectbox("Select a contract:", list(range(1, len(st.session_state.contracts) + 1)))
        idx = c_choice - 1
        contract_text = st.session_state.contracts[idx]
        regs = st.session_state.regs_list[idx]
        st.subheader("📄 Original Contract")
        st.text_area("Contract preview", value=contract_text, height=240, disabled=True)
        st.markdown("---")
        st.subheader("⚙️ Run Auto-Amendment Based on Regulations")
        st.session_state.updated_versions.setdefault(idx, [])
        st.session_state.setdefault("last_pdf_bytes", None)
        st.session_state.setdefault("last_pdf_name", None)
        weak_clauses = st.session_state.get("weak_clauses", [])
        if not weak_clauses:
            st.warning("No weak/missing clauses found. Run 'Risk Analysis' first to identify amendments.")
        else:
            with st.expander("⚠️ View Clauses Flagged for Amendment"):
                for i, mc in enumerate(weak_clauses, start=1):
                    title = mc.get("title", "(no title)")
                    text = mc.get("text", "")
                    reason = mc.get("reason", "No reason provided.")
                    st.markdown(f"**{i}. {title}**")
                    st.text_area(f"Original text — {title}", value=text or "[no original text found]", key=f"orig_amend_{idx}_{i}", height=100, disabled=True)
                    st.markdown(f"**Reason:** *{reason}*")
                    st.markdown("---")
            if st.button("Apply Amendments Now"):
                updated_contract = contract_text
                action_log = []
                changes = []
                for mc in weak_clauses:
                    title = mc.get("title", "Untitled Clause")
                    orig_reg_text = mc.get("text", "")
                    reason = mc.get("reason", "")
                    with st.spinner(f"Generating amendment for: {title}..."):
                        try:
                            improved = improve_regulation_clause(orig_reg_text, title, updated_contract) 
                            improved = str(improved).strip() or orig_reg_text
                        except Exception as e:
                            logger.exception("LLM amendment failed: %s", e)
                            improved = orig_reg_text or f"[Improvement failed for {title}]"
                    action = "appended"
                    if orig_reg_text and orig_reg_text in updated_contract:
                        updated_contract = updated_contract.replace(orig_reg_text, improved, 1)
                        action = "replaced"
                    else:
                        updated_contract += f"\n\n### AMENDMENT: {title}\n{improved}\n"
                        action = "appended"
                    changes.append({"title": title, "action": action, "before": orig_reg_text, "after": improved, "reason": reason})
                    action_log.append(f"{action.capitalize()}: {title}")
                st.session_state.contracts[idx] = updated_contract
                try:
                    st.session_state.regs_list[idx] = extract_regulations_from_text(updated_contract) 
                except Exception:
                    st.warning("Could not re-extract regulations from the updated contract.")
                version_num = len(st.session_state.updated_versions[idx]) + 1
                timestamp = int(time.time())
                version_pdf_name = f"contract_v{version_num}_{timestamp}.pdf"
                st.session_state.updated_versions[idx].append({"name": version_pdf_name, "text": updated_contract, "timestamp": timestamp, "changes": changes})
                try:
                    pdf_bytes = save_text_as_pdf_bytes(updated_contract) 
                    st.session_state.last_pdf_bytes = pdf_bytes
                    st.session_state.last_pdf_name = version_pdf_name
                except Exception as e:
                    pdf_bytes = None
                    st.warning(f"PDF generation failed: {e}")
                st.success(f"New updated contract created: {version_pdf_name}")
                st.subheader("Actions performed:")
                for a in action_log:
                    st.markdown(f"- {a}")
                if pdf_bytes:
                    st.download_button("⬇️ Download updated contract PDF", data=pdf_bytes, file_name=version_pdf_name, mime="application/pdf")

        st.markdown("---")
        st.subheader("📁 View Updated Versions")
        version_list = st.session_state.updated_versions.get(idx, [])
        if version_list:
            v_choice = st.selectbox("Select updated version to view:", [v["name"] for v in version_list], key=f"ver_{idx}")
            selected = next((v for v in version_list if v["name"] == v_choice), None)
            if selected:
                st.text_area("Updated contract text", value=selected["text"], height=300) 
                with st.expander("View Amendment Details"):
                    st.markdown("**Amendments in this version:**")
                    for ch in selected.get("changes", []):
                        st.markdown(f"- **{ch['title']}** — {ch['action']}. Reason: {ch.get('reason','')}")
        else:
            st.info("No updated versions yet. Apply amendments to create one.")
        st.markdown("---")
        st.subheader("📧 Email Updated Contract")
        if st.button("Send Latest Updated Contract via Email"):
            if not st.session_state.get("last_pdf_bytes"):
                st.warning("No PDF available. Generate an updated version first.")
            else:
                subject = f"Compliance Updated – Contract "
                body = f"Hello,\n\nThe contract has been updated.\n\nPlease review the attached updated contract PDF.\n\n"
                sent = send_email_notification(subject, body, EMAIL_SENDER, EMAIL_PASSWORD, EMAIL_RECEIVER, attachment_bytes=st.session_state.last_pdf_bytes, attachment_name=st.session_state.last_pdf_name)
                if sent:
                    st.success(f"Email sent successfully to {EMAIL_RECEIVER}!")
                else:
                    st.error("Failed to send email. Check your email configuration.")

# ---------------------------- AI Chatbot ----------------------------
elif page == "AI Chatbot":
    st.header("🤖 AI Chatbot — Query Contracts")
    st.write("Ask the LLM a question about the loaded contracts. Responses use the configured LLM if available.")
    if not st.session_state.get("contracts"):
        st.info("Upload a contract first.")
    else:
        q = st.text_input("Enter your question:")
        c_choice = st.selectbox("Choose contract context:", ["All"] + [f"Contract {i+1}" for i in range(len(st.session_state.contracts))])
        if st.button("Ask") and q.strip():
            if c_choice == "All":
                context = "\n\n".join(st.session_state.contracts)
            else:
                idx = int(c_choice.split()[-1]) - 1
                context = st.session_state.contracts[idx]
            prompt = f"User question: {q}\n\nContext:\n{context}\n\nAnswer concisely."
            ans = call_groq(prompt)
            st.text_area("Response:", value=ans, height=200)

# ---------------------------- End ----------------------------
else:
    st.error("Unknown page selected.")
