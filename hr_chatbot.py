import os
import pandas as pd
import requests
import streamlit as st
from dotenv import load_dotenv
from typing import Optional, Dict, Any
from sentence_transformers import SentenceTransformer
import faiss

# ------------------ SETUP & ENV ------------------
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")
API_URL = "https://router.huggingface.co/v1/chat/completions"
MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct" 
DATA_PATH = "HR-Employee-Attrition.csv"

# Page Config - Centered layout without sidebar
st.set_page_config(page_title="HR Intelligence Hub", layout="centered", initial_sidebar_state="collapsed")

# ------------------ CUSTOM STYLING (THE PURPLE EDIT) ------------------
st.markdown("""
<style>
    /* Global Background */
    .stApp { background-color: #FDFBFF; }
    
    /* Hide Sidebar */
    [data-testid="stSidebar"] { display: none; }
    
    /* Main Header Container */
    .header-container {
        background: linear-gradient(135deg, #4A148C 0%, #7B1FA2 100%);
        padding: 3rem 2rem;
        border-radius: 20px;
        color: white;
        text-align: center;
        margin-bottom: 2.5rem;
        box-shadow: 0 10px 30px rgba(74, 20, 140, 0.2);
    }

    /* KPI Cards */
    .metric-row { display: flex; gap: 15px; margin-bottom: 25px; }
    .metric-card {
        flex: 1;
        background: white;
        padding: 1.5rem;
        border-radius: 15px;
        text-align: center;
        border: 1px solid #E1BEE7;
        box-shadow: 0 4px 6px rgba(0,0,0,0.02);
        transition: transform 0.3s ease;
    }
    .metric-card:hover { transform: translateY(-5px); border-color: #7B1FA2; }
    .metric-card h4 { color: #6A1B9A; font-size: 0.9rem; margin-bottom: 10px; text-transform: uppercase; letter-spacing: 1px; }
    .metric-card h2 { color: #4A148C; margin: 0; font-size: 2rem; }

    /* Chat Styling */
    .stChatMessage { 
        background-color: white !important; 
        border: 1px solid #F3E5F5 !important; 
        border-radius: 15px !important;
        box-shadow: 0 2px 5px rgba(0,0,0,0.02);
    }
    .stChatInputContainer { padding-bottom: 30px; }
    
    /* Purple Buttons */
    .stButton>button {
        background-color: #7B1FA2;
        color: white;
        border-radius: 25px;
        padding: 0.5rem 2rem;
        border: none;
        font-weight: 600;
    }
    .stButton>button:hover { background-color: #4A148C; color: white; }
</style>
""", unsafe_allow_html=True)

# ------------------ DATA & ADVANCED ANALYTICS ------------------
@st.cache_data
def load_hr_data(path):
    if not os.path.exists(path): return None
    df = pd.read_csv(path)
    return df.drop(columns=[c for c in ['EmployeeCount', 'Over18', 'StandardHours'] if c in df.columns])

@st.cache_data
def compute_executive_metrics(df: pd.DataFrame) -> Dict[str, Any]:
    total = len(df)
    attr_rate = (df['Attrition'] == "Yes").mean() * 100
    
    # Advanced Dept Breakdown
    dept_stats = df.groupby('Department').agg({
        'Attrition': lambda x: (x == 'Yes').mean() * 100,
        'MonthlyIncome': 'mean'
    }).to_dict(orient='index')
    
    # High Risk segment logic
    risk_count = ((df['JobSatisfaction'] <= 2) & (df['OverTime'] == 'Yes')).sum() if 'JobSatisfaction' in df.columns else 0

    return {
        "total": total,
        "rate": f"{attr_rate:.1f}%",
        "risk": risk_count,
        "avg_age": f"{df['Age'].mean():.1f}",
        "dept_stats": dept_stats
    }

# ------------------ RAG VECTOR ENGINE ------------------
@st.cache_resource
def build_vector_index(df: pd.DataFrame):
    model = SentenceTransformer("all-MiniLM-L6-v2")
    # Structured narratives for better retrieval
    texts = df.apply(lambda r: f"Employee in {r['Department']} as {r['JobRole']}. Income: {r['MonthlyIncome']}. Satisfaction: {r['JobSatisfaction']}. Attrition: {r['Attrition']}", axis=1).tolist()
    embeddings = model.encode(texts, convert_to_numpy=True)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    return index, texts, model

# ------------------ INTELLIGENT MEMORY RESPONSE ------------------
def generate_response(question: str, context: str, metrics: dict, history: list) -> str:
    # Format the last 3 messages for history
    history_context = "\n".join([f"{m['role'].upper()}: {m['content']}" for m in history[-3:]])
    
    system_prompt = f"""
    You are the Strategic HR Director Assistant. 
    
    [SNAPSHOT]
    Total Headcount: {metrics['total']} | Attrition: {metrics['rate']}
    High Burnout Risk: {metrics['risk']} | Average Age: {metrics['avg_age']}
    
    [HISTORY]
    {history_context}

    [DATABASE CONTEXT]
    {context}

    RULES:
    1. Be executive and data-driven.
    2. If the user says 'Hi', greet them and mention the {metrics['rate']} attrition rate.
    3. Use the HISTORY to understand follow-up questions.
    4. Provide one 'Strategic Recommendation' in every answer.
    """

    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    payload = {
        "model": MODEL_NAME,
        "messages": [{"role": "system", "content": system_prompt}, {"role": "user", "content": question}],
        "temperature": 0.2
    }

    try:
        r = requests.post(API_URL, headers=headers, json=payload, timeout=40)
        return r.json()["choices"][0]["message"]["content"]
    except:
        return "❌ Connection error with Intelligence Engine."

# ------------------ MAIN INTERFACE ------------------
def main():
    # Header
    st.markdown('<div class="header-container"><h1>🚀 Strategic HR Intelligence</h1><p>Workforce Decision Support System</p></div>', unsafe_allow_html=True)
    
    df = load_hr_data(DATA_PATH)
    if df is not None:
        metrics = compute_executive_metrics(df)
        index, texts, model = build_vector_index(df)

        # Purple Metric Bar
        st.markdown(f"""
        <div class="metric-row">
            <div class="metric-card"><h4>Headcount</h4><h2>{metrics['total']}</h2></div>
            <div class="metric-card"><h4>Attrition</h4><h2>{metrics['rate']}</h2></div>
            <div class="metric-card"><h4>High Risk</h4><h2>{metrics['risk']}</h2></div>
            <div class="metric-card"><h4>Avg Age</h4><h2>{metrics['avg_age']}</h2></div>
        </div>
        """, unsafe_allow_html=True)

        # Conversation State
        if "messages" not in st.session_state: st.session_state.messages = []
        
        for m in st.session_state.messages:
            with st.chat_message(m["role"]): st.markdown(m["content"])

        if user_input := st.chat_input("Ask about attrition, risks, or specific departments..."):
            st.session_state.messages.append({"role": "user", "content": user_input})
            with st.chat_message("user"): st.markdown(user_input)

            with st.chat_message("assistant"):
                # Intelligent Retrieval
                is_greet = any(x in user_input.lower() for x in ["hi", "hello", "hey"])
                if is_greet:
                    context = "Greeting sequence initiated."
                else:
                    with st.spinner("Analyzing workforce data..."):
                        _, I = index.search(model.encode([user_input]), 5)
                        context = "\n".join([texts[i] for i in I[0]])
                
                # AI Logic with History
                response = generate_response(user_input, context, metrics, st.session_state.messages[:-1])
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})

        # Reset Option
        if st.button("Clear Chat History"):
            st.session_state.messages = []
            st.rerun()
            
    else:
        st.error(f"Critical Error: {DATA_PATH} not found.")

if __name__ == "__main__":
    main()