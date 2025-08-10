# frontend/app.py
import streamlit as st
import requests
import os
API_BASE = os.getenv("API_BASE", "http://127.0.0.1:8000")

st.set_page_config(page_title="HR Query Chatbot", layout="centered")
st.title("🤖 HR Resource Query Chatbot")
st.markdown("Ask natural language queries to find employees. Uses semantic search + RAG.")

with st.form("query_form"):
    query = st.text_input("What do you need?", placeholder="e.g. Find Python devs with 3+ years for healthcare project")
    col1, col2 = st.columns(2)
    with col1:
        min_exp = st.number_input("Min experience (years)", min_value=0, max_value=50, value=0)
    with col2:
        availability = st.selectbox("Availability", options=["any", "available", "busy"], index=0)
    skills = st.text_input("Required skills (comma-separated)", placeholder="e.g. AWS, Docker")
    submitted = st.form_submit_button("Search")

if submitted:
    if not query.strip():
        st.warning("Please enter a query.")
    else:
        payload = {
            "query": query,
            "top_k": 5,
            "min_experience": int(min_exp),
            "availability": None if availability == "any" else availability,
            "required_skills": [s.strip() for s in skills.split(",") if s.strip()] or None
        }
        try:
            with st.spinner("Contacting backend..."):
                resp = requests.post(f"{API_BASE}/chat", json=payload, timeout=30)
            if resp.status_code == 200:
                data = resp.json()
                st.subheader("Generated Answer")
                st.write(data.get("generated_answer", ""))
                st.subheader("Matches")
                matches = data.get("matches", [])
                if not matches:
                    st.info("No matches found.")
                else:
                    for m in matches:
                        st.markdown(f"**{m['name']}** — {m['experience_years']} yrs | Availability: {m['availability']}")
                        st.write(f"Skills: {', '.join(m['skills'])}")
                        st.write(f"Projects: {', '.join(m['projects'])}")
                        st.divider()
            else:
                st.error(f"Backend error: {resp.status_code} - {resp.text}")
        except Exception as e:
            st.error(f"Failed to contact backend: {e}")
