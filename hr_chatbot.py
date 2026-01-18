import os
import pandas as pd
import requests
import streamlit as st
import json
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

st.set_page_config(page_title="Smart HR Data Assistant", layout="wide")

# ---CSS---
st.markdown(
    """
    <style>
    .stApp {
        background-color: #f8f4ff;
    }
    [data-testid="stSidebar"] {
        background-color: #2e004f;
        color: white;
    }
    [data-testid="stSidebar"] .stMarkdown, [data-testid="stSidebar"] p, [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2 {
        color: #e1bee7;
    }
    h1, h2, h3 {
        color: #4a148c ;
    }
    .stChatMessage {
        background-color: #ffffff;
        border: 1px solid #d1c4e9;
        border-radius: 15px;
    }
    .stButton>button {
        background-color: #6a1b9a;
        color: white;
        border-radius: 8px;
        border: none;
    }
    .stButton>button:hover {
        background-color: #8e24aa;
        color: white;
    }
    .stSpinner > div {
        border-top-color: #7b1fa2 !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# --- CONFIGURATION ---
HF_TOKEN = os.getenv("HF_TOKEN")
API_URL = "https://router.huggingface.co/v1/chat/completions"

# --- DATA LOADING ---
@st.cache_data
def load_data():
    file_path = "HR-Employee-Attrition.csv"
    if not os.path.exists(file_path):
        # Create a dummy dataframe if file doesn't exist for demonstration
        return None
    return pd.read_csv(file_path)

df = load_data()

# --- SMART REASONING ENGINE ---
def ask_smart_assistant(question, df):
    if not HF_TOKEN:
        return "Error: Hugging Face token (HF_TOKEN) is missing in environment variables."
    
    if df is None:
        return "Error: Dataset not found. Please ensure 'HR-Employee-Attrition.csv' is in the directory."

    # Step 1: Provide context about the data
    columns_info = df.dtypes.to_string()
    sample_data = df.head(3).to_string()
    summary_stats = df.describe(include='all').to_string()

    prompt = f"""
You are an expert HR Data Scientist. You have access to a dataset with the following characteristics:

COLUMNS AND TYPES:
{columns_info}

SUMMARY STATISTICS:
{summary_stats}

SAMPLE DATA:
{sample_data}

USER QUESTION: "{question}"

INSTRUCTIONS:
1. Analyze the question and the data context provided.
2. If the question requires a specific calculation (like averages, counts, or correlations), perform a mental simulation of the analysis based on the summary stats.
3. Provide a "Smart Insight": Don't just give a number; explain *why* it matters or what the trend suggests (e.g., "The attrition rate is 16%, which is slightly high for the Research & Development department").
4. If you cannot answer precisely from the summary, explain what you see in the trends.
5. Be professional, helpful, and concise.

RESPONSE FORMAT:
- **Analysis**: Your reasoning process.
- **Answer**: The direct answer to the user.
- **Insight**: A deeper observation based on the data.
"""

    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    payload = {
        "model":"Qwen/Qwen2.5-1.5B-Instruct:featherless-ai" , # Using a larger model for better reasoning
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.3,
        "max_tokens": 800,
    }

    try:
        response = requests.post(API_URL, headers=headers, json=payload, timeout=60)
        response.raise_for_status()
        result = response.json()
        return result["choices"][0]["message"]["content"]
    except Exception as e:
        return f"I encountered an error while thinking: {str(e)}"

# --- UI LAYOUT ---
st.title(" Smart HR Analytics Assistant")
st.markdown("I can analyze your HR data, find trends, and provide strategic insights.")

# Sidebar
with st.sidebar:
    st.header(" Dataset Overview")
    if df is not None:
        st.write(f"**Total Records:** {len(df)}")
        st.write(f"**Features:** {len(df.columns)}")
        if st.checkbox("Show Raw Data"):
            st.dataframe(df.head(20))
    else:
        st.error("Dataset 'HR-Employee-Attrition.csv' not found.")

# Chat Interface
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask me anything about the HR data..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Analyzing data and generating insights..."):
            response = ask_smart_assistant(prompt, df)
            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})