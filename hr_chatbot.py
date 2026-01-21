import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import pandas as pd
import streamlit as st
import faiss
import requests
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer

# --- SYSTEM CONFIG ---
load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")
API_URL = os.getenv("API_URL", "https://router.huggingface.co/v1/chat/completions")
CLOUD_MODEL = os.getenv("CLOUD_MODEL", "Qwen/Qwen2.5-7B-Instruct")

MODEL_DIR = "./models/qwen_1_5b"
EMBED_DIR = "./models/embeddings"
DATA_PATH = "HR-Employee-Attrition.csv"

st.set_page_config(page_title="Strategic HR Intelligence", layout="centered", initial_sidebar_state="collapsed")

# ------------------ STYLING ------------------
# STC Primary Purple: #4F008C
st.markdown("""
<style>
    /* Main App Background */
    .stApp { 
        background-color: #FFFFFF; 
    }
    
    /* Hide Sidebar */
    [data-testid="stSidebar"] { 
        display: none; 
    }

    /* Header Container - STC Deep Purple */
    .header-container {
        background-color: #4F008C;
        padding: 2.5rem 2rem; 
        border-radius: 0px 0px 30px 30px; /* Modern curved bottom */
        color: white; 
        text-align: center; 
        margin-bottom: 1.5rem; /* Reduced margin under header as requested */
        box-shadow: 0 10px 30px rgba(79, 0, 140, 0.15);
        margin-top: -60px; /* Pull up to top of page */
        margin-left: -20%;
        margin-right: -20%;
    }
    
    .header-container h1 {
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    .header-container p {
        font-size: 1.1rem;
        opacity: 0.85;
        font-weight: 300;
    }

    /* Metric Cards - STC Accents */
    .metric-row { 
        display: flex; 
        gap: 20px; 
        margin-bottom: 30px; 
    }
    .metric-card {
        flex: 1; 
        background: #F9F5FF; 
        padding: 1.5rem; 
        border-radius: 20px; 
        text-align: center;
        border: 1px solid #E9D5FF; 
        transition: all 0.3s ease;
    }
    .metric-card:hover {
        border-color: #4F008C;
        background: #F3E8FF;
        transform: translateY(-3px);
    }
    .metric-card h4 { 
        color: #4F008C; 
        font-size: 0.85rem; 
        margin-bottom: 8px; 
        text-transform: uppercase; 
        font-weight: 600;
        letter-spacing: 0.5px;
    }
    .metric-card h2 { 
        color: #1A0033; 
        margin: 0; 
        font-size: 2rem; 
        font-weight: 800; 
    }

    /* Chat Messages */
    .stChatMessage { 
        border-radius: 20px !important; 
        border: 1px solid #F3E8FF !important; 
        padding: 15px !important; 
        margin-bottom: 15px !important; 
    }
    
    /* User Message - Light STC Purple Tint */
    [data-testid="stChatMessageUser"] {
        background-color: #F3E8FF !important;
    }

    /* Radio Buttons */
    .stRadio>div { 
        background: #F9F5FF; 
        padding: 15px; 
        border-radius: 20px; 
        border: 1px solid #E9D5FF; 
    }
    .stRadio label {
        color: #4F008C !important;
        font-weight: 600 !important;
    }

    /* Buttons - STC Signature Purple */
    .stButton>button { 
        background-color: #4F008C; 
        color: white; 
        border-radius: 50px; 
        width: 100%; 
        font-weight: 600;
        border: none;
        padding: 0.75rem;
        transition: all 0.3s ease;
        margin-top: 10px;
    }
    .stButton>button:hover {
        background-color: #3A0066;
        box-shadow: 0 8px 20px rgba(79, 0, 140, 0.25);
    }
    
    /* Chat Input Styling */
    .stChatInputContainer {
        padding-bottom: 30px !important;
    }
</style>
""", unsafe_allow_html=True)

# ------------------ ENGINES & DATA ------------------
@st.cache_resource
def load_engines():
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct", cache_dir=MODEL_DIR)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-1.5B-Instruct", cache_dir=MODEL_DIR,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto"
    )
    embed_model = SentenceTransformer("all-MiniLM-L6-v2", cache_folder=EMBED_DIR)
    return tokenizer, model, embed_model

@st.cache_data
def get_data_and_metrics(path):
    if not os.path.exists(path): return None, None
    df = pd.read_csv(path)
    df = df.drop(columns=[c for c in ['EmployeeCount', 'Over18', 'StandardHours'] if c in df.columns])
    
    depts = df['Department'].unique().tolist()
    dept_stats = df.groupby('Department')['Attrition'].apply(lambda x: f"{(x == 'Yes').mean() * 100:.1f}%").to_dict()
    
    metrics = {
        "total": len(df),
        "rate": f"{(df['Attrition'] == 'Yes').mean() * 100:.1f}%",
        "risk": ((df['JobSatisfaction'] <= 2) & (df['OverTime'] == 'Yes')).sum(),
        "avg_age": f"{df['Age'].mean():.1f}",
        "dept_list": depts,
        "dept_info": dept_stats
    }
    return df, metrics

@st.cache_resource
def build_rag_index(_df, _embed_model):
    texts = _df.apply(lambda r: f"Dept: {r['Department']} | Role: {r['JobRole']} | Attrition: {r['Attrition']} | Income: {r['MonthlyIncome']}", axis=1).tolist()
    embeddings = _embed_model.encode(texts, convert_to_numpy=True)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    return index, texts

# ------------------ SMART INFERENCE LOGIC ------------------
def generate_smart_response(question, context, metrics, mode, tokenizer=None, model=None):
    # If the user asks about employee names or anything not in the dataset
    if "name" in question.lower() and "employee" in question.lower():
        return "I don’t know the names of employees as they are not in the dataset."
    
    dept_str = ", ".join(metrics['dept_list'])
    system_instruction = f"""
You are the Lead HR Intelligence Agent. 
[DATA SNAPSHOT]
- Total Headcount: {metrics['total']}
- Overall Attrition: {metrics['rate']}
- Total High-Risk Employees: {metrics['risk']}
- Departments in Database: {dept_str} ({len(metrics['dept_list'])} departments total)
- Departmental Attrition Breakdown: {metrics['dept_info']}

[RELEVANT DATABASE RECORDS]
{context}

RULES:
1. Grounding: If asked "how many departments", refer to the [DATA SNAPSHOT].
2. Strategy: Always provide one 'Strategic Recommendation'.
3. Tone: Executive, data-driven, concise.
"""

    if "Cloud" in mode:
        headers = {"Authorization": f"Bearer {HF_TOKEN}"}
        payload = {
            "model": CLOUD_MODEL,
            "messages": [
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": question}
            ],
            "temperature": 0.1
        }
        r = requests.post(API_URL, headers=headers, json=payload, timeout=40)
        return r.json()["choices"][0]["message"]["content"]
    else:
        full_prompt = f"<|im_start|>system\n{system_instruction}<|im_end|>\n<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n"
        inputs = tokenizer(full_prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=300, temperature=0.1, pad_token_id=tokenizer.eos_token_id)
        answer = tokenizer.decode(outputs[0], skip_special_tokens=False).split("<|im_start|>assistant")[-1]
        return answer.replace("<|im_end|>", "").strip()

# ------------------ MAIN INTERFACE ------------------
def main():
    st.markdown('<div class="header-container"><h1> Strategic HR Intelligence</h1><p>Executive Decision Hub • Data-Grounded</p></div>', unsafe_allow_html=True)
    
    mode = st.radio("Intelligence Engine:", ["Cloud (7B API)", "Local (1.5B Private)"], horizontal=True)
    
    df, metrics = get_data_and_metrics(DATA_PATH)
    if df is not None:
        tokenizer, model, embed_model = load_engines()
        idx, texts = build_rag_index(df, embed_model)

        # KPI Row
        c1, c2, c3, c4 = st.columns(4)
        c1.markdown(f'<div class="metric-card"><h4>Headcount</h4><h2>{metrics["total"]}</h2></div>', unsafe_allow_html=True)
        c2.markdown(f'<div class="metric-card"><h4>Attrition</h4><h2>{metrics["rate"]}</h2></div>', unsafe_allow_html=True)
        c3.markdown(f'<div class="metric-card"><h4>High Risk</h4><h2>{metrics["risk"]}</h2></div>', unsafe_allow_html=True)
        c4.markdown(f'<div class="metric-card"><h4>Avg Age</h4><h2>{metrics["avg_age"]}</h2></div>', unsafe_allow_html=True)

        # Spacing between KPIs and the rest of the content
        st.markdown("<div style='margin-bottom: 20px;'></div>", unsafe_allow_html=True)

        if "messages" not in st.session_state: st.session_state.messages = []
        for m in st.session_state.messages:
            with st.chat_message(m["role"]): st.markdown(m["content"])

        if user_input := st.chat_input("Ask about department counts, attrition, or strategy..."):
            st.session_state.messages.append({"role": "user", "content": user_input})
            with st.chat_message("user"): st.markdown(user_input)

            with st.chat_message("assistant"):
                # Search depth of 5 for better accuracy
                query_emb = embed_model.encode([user_input], convert_to_numpy=True)
                _, I = idx.search(query_emb, 5)
                context = "\n".join([texts[i] for i in I[0]])

                with st.spinner("Processing..."):
                    try:
                        response = generate_smart_response(user_input, context, metrics, mode, tokenizer, model)
                        st.markdown(response)
                        st.session_state.messages.append({"role": "assistant", "content": response})
                    except Exception as e:
                        st.error("Engine Connection Error. Please verify your API URL and Token.")

        # Space before the clear button
        st.markdown("<div style='margin-bottom: 15px;'></div>", unsafe_allow_html=True)
        if st.button("🗑️ Clear Executive Session"):
            st.session_state.messages = []
            st.rerun()
    else:
        st.error(f"Missing {DATA_PATH}. Please ensure the CSV is in the root folder.")

if __name__ == "__main__":
    main()