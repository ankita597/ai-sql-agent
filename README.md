# 🧠 AI SQL Data Analyst Agent

An AI-powered application that allows users to upload CSV files, ask questions in plain English, and automatically generate SQL queries, insights, and visualizations.

## 🚀 Features
✅ Upload CSV files  
✅ Natural Language → SQL conversion  
✅ Automatic SQL query execution  
✅ Interactive visualizations  
✅ AI-powered insights generation  
✅ Supports multiple LLM models  

## 🛠 Tech Stack
- Frontend: Streamlit
- Backend: Python
- Database: SQLite (in-memory)
- LLM: Groq (Llama 3 / Mixtral)
- Data Processing: Pandas
- Visualization: Plotly

## 📊 Sample Questions
- What are the top 5 products by total sales?
- Show monthly revenue trend
- Which category has the highest average price?
- Count records grouped by region

## ▶️ Run Locally

Install dependencies:

```bash
pip install -r requirements.txt
```

Run application:

```bash
streamlit run app.py
```

Open browser:

```text
http://localhost:8501
```

## 🔑 Groq API Key
1. Create a free account on Groq
2. Generate an API key
3. Paste the API key into the application

## 📌 Project Flow

CSV Upload → Pandas Processing → SQLite Database → Groq LLM → SQL Generation → Query Execution → Results + Visualization
