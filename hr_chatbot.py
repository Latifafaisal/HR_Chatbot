import os
import pandas as pd
import requests
import streamlit as st
from dotenv import load_dotenv
from typing import Optional, Tuple, Dict, Any

# Load environment variables
load_dotenv()

# --- CONFIGURATION ---
HF_TOKEN = os.getenv("HF_TOKEN")
API_URL = "https://router.huggingface.co/v1/chat/completions"
MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct" 
DATA_PATH = "HR-Employee-Attrition.csv"

# --- PAGE SETUP ---
st.set_page_config(
    page_title="Strategic HR Intelligence Hub",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- ENHANCED UI STYLING ---
st.markdown("""
<style>
    :root {
        --primary-color: #4A148C;
        --secondary-color: #7B1FA2;
        --bg-color: #F4F7F9;
        --sidebar-bg: #1E1E2F;
    }
    .stApp { background-color: var(--bg-color); }
    [data-testid="stSidebar"] { background-color: var(--sidebar-bg); color: #FFFFFF; }
    [data-testid="stSidebar"] * { color: #E0E0E0 !important; }
    
    .metric-card {
        background-color: #FFFFFF;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        border-left: 5px solid var(--primary-color);
    }
    
    .stChatMessage {
        border-radius: 12px;
        margin-bottom: 1rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.03);
    }
    
    .stButton>button {
        width: 100%;
        border-radius: 8px;
        background-color: var(--primary-color);
        color: white;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: var(--secondary-color);
        transform: translateY(-1px);
    }
</style>
""", unsafe_allow_html=True)

# --- DATA ENGINE ---
@st.cache_data(show_spinner="Loading organizational data...")
def load_and_preprocess_data(path: str) -> Optional[pd.DataFrame]:
    if not os.path.exists(path):
        return None
    try:
        df = pd.read_csv(path)
        # Basic cleaning: remove constants or irrelevant columns if they exist
        cols_to_drop = ['EmployeeCount', 'Over18', 'StandardHours']
        df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])
        return df
    except Exception as e:
        st.error(f"Data Loading Error: {e}")
        return None

@st.cache_data
def compute_advanced_metrics(df: pd.DataFrame) -> Dict[str, Any]:
    """Pre-calculates a rich set of metrics to provide better context to the LLM."""
    total = len(df)
    attrition_mask = df['Attrition'] == "Yes"
    overall_attrition_rate = attrition_mask.mean() * 100
    
    # Departmental breakdown
    dept_stats = df.groupby('Department').agg({
        'Attrition': lambda x: (x == 'Yes').mean() * 100,
        'MonthlyIncome': 'mean',
        'YearsAtCompany': 'mean'
    }).rename(columns={'Attrition': 'AttritionRate', 'MonthlyIncome': 'AvgIncome', 'YearsAtCompany': 'AvgTenure'})
    
    # High-risk segments (e.g., low job satisfaction + high overtime)
    if 'JobSatisfaction' in df.columns and 'OverTime' in df.columns:
        high_risk_mask = (df['JobSatisfaction'] <= 2) & (df['OverTime'] == 'Yes')
        high_risk_count = high_risk_mask.sum()
    else:
        high_risk_count = "N/A"

    return {
        "total_employees": total,
        "overall_attrition_rate": f"{overall_attrition_rate:.2f}%",
        "dept_stats": dept_stats.to_dict(orient='index'),
        "high_risk_count": high_risk_count,
        "avg_age": df['Age'].mean() if 'Age' in df.columns else "N/A",
        "top_roles": df['JobRole'].value_counts().head(3).to_dict()
    }

# --- AI REASONING CORE ---
def generate_strategic_response(question: str, df: pd.DataFrame, metrics: Dict[str, Any]) -> str:
    if not HF_TOKEN:
        return "⚠️ System Configuration Error: API Token missing."

    # Construct a highly structured context
    context_summary = f"""
    ORGANIZATIONAL SNAPSHOT:
    - Total Workforce: {metrics['total_employees']}
    - Global Attrition Rate: {metrics['overall_attrition_rate']}
    - High-Risk Employees (Low Satisfaction + Overtime): {metrics['high_risk_count']}
    - Top 3 Roles: {', '.join(metrics['top_roles'].keys())}
    
    DEPARTMENTAL PERFORMANCE:
    {pd.DataFrame(metrics['dept_stats']).T.to_string()}
    
    DATA SCHEMA:
    {df.dtypes.to_string()}
    """

    # Memory Management: Keep last 5 interactions for better continuity
    history = ""
    if "messages" in st.session_state:
        history = "\n".join([f"{m['role'].upper()}: {m['content'][:200]}..." for m in st.session_state.messages[-5:]])

    system_prompt = f"""
You are a Strategic HR Intelligence Assistant.

Your ONLY goal is to answer the user's specific question using the provided data.
You must stay on-topic. Do NOT introduce unrelated metrics, themes, or KPIs.

CONTEXT (use only what is relevant):
{context_summary}

RECENT HISTORY (for continuity only, do NOT change topic):
{history}

USER QUERY:
"{question}"

STRICT RULES (VERY IMPORTANT):
- Every sentence must directly relate to the USER QUERY.
- If a section cannot be answered using the data relevant to the query, explicitly say:
  "The available data does not provide enough information on this aspect."
- Do NOT default to attrition, income, tenure, or satisfaction unless the question explicitly asks about them.
- Do NOT generalize beyond the topic of the question.
- Do NOT add extra insights just to sound strategic.

RESPONSE STRUCTURE (MANDATORY):

[Direct Answer]
- One clear, factual answer to the question.
- Use exact numbers when possible.
- No introductions or filler text.

### 🔍 Deep Dive Analysis
- Analyze ONLY the variables directly related to the question.
- Explain patterns or comparisons strictly within that scope.
- If analysis is limited, clearly state the limitation.

### 💡 Strategic Recommendation
- Provide ONE actionable recommendation.
- The recommendation must be logically derived from the analysis above.
- It must match the topic of the question exactly.
  Examples:
  - Departments → org structure, workload balance, staffing
  - Income → compensation bands, pay equity
  - Age/Tenure → succession planning, mentoring
- If no recommendation can be justified, say so clearly.
"""


    headers = {"Authorization": f"Bearer {HF_TOKEN}", "Content-Type": "application/json"}
    payload = {
        "model": "Qwen/Qwen2.5-7B-Instruct", # Using a larger model for better reasoning
        "messages": [
            {"role": "system", "content": "You are a Strategic HR Intelligence Assistant."},
            {"role": "user", "content": system_prompt}
        ],
        "temperature": 0.2, # Lower temperature for more factual consistency
        "max_tokens": 1000
    }

    try:
        response = requests.post(API_URL, headers=headers, json=payload, timeout=45)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
    except Exception as e:
        return f"❌ Intelligence Engine Error: {str(e)}"

# --- MAIN APPLICATION ---
def main():
    st.title("🚀 Strategic HR Intelligence Hub")
    st.caption("Transforming raw workforce data into executive-level strategy.")

    df = load_and_preprocess_data(DATA_PATH)

    if df is not None:
        metrics = compute_advanced_metrics(df)
        
        # Sidebar Dashboard
        with st.sidebar:
            st.image("https://img.icons8.com/fluency/96/business-conference.png", width=80)
            st.header("Workforce Overview")
            
            col1, col2 = st.columns(2)
            col1.metric("Headcount", metrics['total_employees'])
            col2.metric("Attrition", metrics['overall_attrition_rate'], delta_color="inverse")
            
            st.divider()
            st.subheader("Data Explorer")
            if st.checkbox("Preview Dataset"):
                st.dataframe(df.head(10), use_container_width=True)
            
            if st.button("Clear Conversation"):
                st.session_state.messages = []
                st.rerun()

        # Chat Interface
        if "messages" not in st.session_state:
            st.session_state.messages = [{"role": "assistant", "content": "Welcome, Leader. I've analyzed the HR dataset. How can I assist with your workforce strategy today?"}]

        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        if user_input := st.chat_input("Analyze attrition trends in Sales..."):
            st.session_state.messages.append({"role": "user", "content": user_input})
            with st.chat_message("user"):
                st.markdown(user_input)

            with st.chat_message("assistant"):
                with st.spinner("Consulting the intelligence engine..."):
                    response = generate_strategic_response(user_input, df, metrics)
                    st.markdown(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})
    else:
        st.error(f"Critical Error: '{DATA_PATH}' not found. Please ensure the dataset is in the root directory.")
        st.info("Expected columns: Attrition, Department, MonthlyIncome, Age, etc.")

if __name__ == "__main__":
    main()