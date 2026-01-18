import streamlit as st
import pandas as pd
import requests
import os
from dotenv import load_dotenv

# Load Hugging Face token
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

# Load HR dataset
df = pd.read_csv("HR-Employee-Attrition.csv")

# Hugging Face router endpoint
API_URL = "https://router.huggingface.co/v1/chat/completions"
headers = {"Authorization": f"Bearer {HF_TOKEN}"}

# Ask Qwen and compute answer
def ask_qwen(question):
    # Tell Qwen to interpret the question in plain English
    prompt = f"""
You are an HR data assistant. You have access to a dataset called df.
Columns in the dataset: {', '.join(df.columns)}.

Answer the user's question in plain English. Compute counts, averages, or summaries directly using Python/pandas if needed.
Return only the final answer in natural language.

Question: {question}
"""
    payload = {
        "model": "Qwen/Qwen2.5-1.5B-Instruct:featherless-ai",
        "messages": [{"role": "user", "content": prompt}]
    }
    response = requests.post(API_URL, headers=headers, json=payload)
    try:
        return response.json()["choices"][0]["message"]["content"]
    except:
        return "Error: cannot get response from Qwen."

# Streamlit UI
st.title("HR Dataset Chatbot (Qwen2.5 1.5B Instruct)")

question = st.chat_input("Ask a question in plain English about your HR data")

if question:
    answer = ask_qwen(question)
    st.markdown(answer)
