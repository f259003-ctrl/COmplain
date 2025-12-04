import streamlit as st
import pandas as pd
import json
from pathlib import Path
from sentence_transformers import SentenceTransformer
from langchain.llms import OpenAI
import faiss

st.set_page_config(page_title="Contract Compliance Checker", layout="wide")
st.title("Contract Compliance Checker - Full Scan")
st.caption("Automated compliance analysis using OpenAI & FAISS RAG")

# --- Load FAISS index and metadata ---
index = faiss.read_index("faiss_index/index.faiss")
with open("faiss_index/chunk_texts.json", "r") as f:
    chunk_texts = json.load(f)
with open("faiss_index/metadatas.json", "r") as f:
    metadatas = json.load(f)

embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# OpenAI LLM
llm = OpenAI(temperature=0, model_name="gpt-4")  # API key from environment or Streamlit secrets

# Load compliance rules
with open("compliance_rules.json", "r") as f:
    rules = json.load(f)

# --- Helper functions ---
def search_faiss(query, top_k=3):
    q_emb = embedder.encode([query])
    q_emb = q_emb / ((q_emb**2).sum()**0.5)  # normalize
    D, I = index.search(q_emb, top_k)
    results = []
    for score, idx in zip(D[0], I[0]):
        results.append({
            "text": chunk_texts[idx],
            "source": metadatas[idx]["source"],
            "score": float(score)
        })
    return results

def check_rule(rule_id, rule_data):
    retrieved = search_faiss(rule_data["rule"], top_k=3)
    context = "\n\n".join([r["text"] for r in retrieved])
    prompt = f"""
You are a contract compliance checker. Rule: {rule_data['rule']}
Keywords: {', '.join(rule_data['keywords'])}
Contract Sections: {context}

Answer:
1. COMPLIANT or NON-COMPLIANT
2. Evidence from contract
3. If non-compliant, provide remediation steps
"""
    resp = llm(prompt)
    status = "COMPLIANT" if "COMPLIANT" in resp.upper() else "NON-COMPLIANT"
    return {
        "rule_id": rule_id,
        "rule": rule_data["rule"],
        "category": rule_data["category"],
        "status": status,
        "evidence": retrieved[0]["text"][:300] if retrieved else "",
        "analysis": resp[:500],
        "source": retrieved[0]["source"] if retrieved else ""
    }

# --- Run Full Scan ---
if st.button("Run Full Compliance Scan"):
    st.info("Running compliance check...")
    results = []
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, (rule_id, rule_data) in enumerate(rules.items(), 1):
        status_text.text(f"Checking {rule_id} ({i}/{len(rules)})")
        r = check_rule(rule_id, rule_data)
        results.append(r)
        progress_bar.progress(i/len(rules))
    
    st.success("Full compliance scan completed!")
    
    df = pd.DataFrame(results)
    
    # --- Summary Metrics ---
    compliant = len(df[df['status']=='COMPLIANT'])
    st.markdown(f"**Compliance Summary:** {compliant}/{len(df)} rules compliant ({compliant/len(df)*100:.1f}%)")
    
    # --- Detailed Results Table ---
    def color_status(val):
        color = 'green' if val=="COMPLIANT" else 'red'
        return f'background-color: {color}; color:white'
    
    display_df = df[['rule_id','category','rule','status']].copy()
    display_df['rule'] = display_df['rule'].str[:60] + "..."
    st.dataframe(display_df.style.applymap(color_status, subset=['status']), use_container_width=True)
    
    # --- Download CSV ---
    csv = df.to_csv(index=False)
    st.download_button(
        "Download Full Results (CSV)",
        data=csv,
        file_name="compliance_scan_results.csv",
        mime="text/csv"
    )
