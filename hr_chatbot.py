import os
import pandas as pd
import requests
from dotenv import load_dotenv

#Load Hugging Face token
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

# Load dataset
df = pd.read_csv("HR-Employee-Attrition.csv")
print("Dataset loaded. Number of rows:", len(df))

# Test Hugging Face API
API_URL = "https://router.huggingface.co/v1/chat/completions"
headers = {"Authorization": f"Bearer {HF_TOKEN}"}

payload = {
    "model": "Qwen/Qwen2.5-1.5B-Instruct:featherless-ai",
    "messages": [{"role": "user", "content": "hello from Qwen!"}]
}

response = requests.post(API_URL, headers=headers, json=payload)
print("API status code:", response.status_code)
print("API response:", response.json()["choices"][0]["message"]["content"])
