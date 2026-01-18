import os
import pandas as pd
import requests
from dotenv import load_dotenv

# Load Hugging Face token
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

# Load dataset
df = pd.read_csv("HR-Employee-Attrition.csv")
columns_desc = ", ".join(df.columns)

# Hugging Face API endpoint
API_URL = "https://router.huggingface.co/v1/chat/completions"
headers = {"Authorization": f"Bearer {HF_TOKEN}"}

# Function to ask Qwen
def ask_qwen(question):
    prompt = f"""
You are an HR data assistant with access to a dataset called df.
Dataset columns: {columns_desc}.

Answer the user's question in **plain English**. Use Python/pandas to compute counts, averages, or summaries if needed.
Return only the final answer, do not add extra explanation.

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

# Command-line chatbot loop
print("HR Chatbot v3 (type 'exit' to quit)")
while True:
    question = input("\nAsk a question: ")
    if question.lower() in ["exit", "quit"]:
        print("Goodbye!")
        break
    answer = ask_qwen(question)
    print("Answer:", answer)
