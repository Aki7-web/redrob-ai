"""
Minimal sandbox app (Streamlit) satisfying submission_spec.md Section 10.5.
Deploy this for free on Streamlit Cloud (streamlit.io/cloud) by connecting
your GitHub repo -- no code changes needed.

Run locally with: streamlit run sandbox_app.py
"""
import json
import streamlit as st
import pandas as pd

import tfidf
import jd_requirements as req
import features as feat
from rank import score_candidate, build_reasoning

st.title("Redrob Candidate Ranker — Sandbox")
st.write(
    "Upload a small candidate sample (JSONL, one JSON object per line, "
    "matching candidate_schema.json) to see the ranker run end-to-end."
)

uploaded = st.file_uploader("candidate sample (.jsonl)", type=["jsonl"])

if uploaded:
    candidates = [json.loads(line) for line in uploaded.read().decode("utf-8").splitlines() if line.strip()]
    st.write(f"Loaded {len(candidates)} candidates.")

    texts = [feat._text_blob(c) for c in candidates]
    idf, _ = tfidf.build_idf(texts, max_vocab=30000)
    query_weights, query_norm = tfidf.tfidf_vector(req.JD_TEXT, idf)

    rows = []
    for cand, text in zip(candidates, texts):
        sim = tfidf.cosine_sim_to_query(text, idf, query_weights, query_norm)
        composite, detail = score_candidate(cand, sim)
        rows.append({
            "candidate_id": cand["candidate_id"],
            "score": round(composite, 4),
            "reasoning": build_reasoning(cand, detail),
        })
    df = pd.DataFrame(rows).sort_values("score", ascending=False).reset_index(drop=True)
    df.insert(0, "rank", range(1, len(df) + 1))
    st.dataframe(df)
    st.download_button("Download ranked CSV", df.to_csv(index=False), "ranked_sample.csv")
