# 🧠 AI SQL Data Analyst Agent

> Upload any CSV → Ask questions in plain English → Get SQL + Answers + Visualizations

---

## 🚀 Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the app
```bash
streamlit run app.py
```

### 3. Open in browser
Go to `http://localhost:8501`

---

## 🔑 Get a Free Groq API Key
1. Visit [console.groq.com](https://console.groq.com)
2. Sign up for a free account
3. Create an API key
4. Paste it into the sidebar

---

## 🧩 Tech Stack

| Component       | Technology              |
|----------------|-------------------------|
| Frontend        | Streamlit               |
| LLM             | Groq (Llama 3 / Mixtral)|
| Database        | SQLite (in-memory)      |
| Data Handling   | Pandas                  |
| Visualization   | Plotly                  |

---

## 🎯 Features

- ✅ Upload any CSV file
- ✅ Auto-detects schema and column types
- ✅ Natural language → SQL conversion using Groq LLM
- ✅ Executes SQL on an in-memory SQLite database
- ✅ Returns: Answer + SQL Query + Chart
- ✅ Auto-selects best chart type (bar, line, pie, scatter)
- ✅ Direct SQL editor for advanced users
- ✅ Supports Llama 3 (70B, 8B) and Mixtral models

---

## 📊 Example Questions

- "What are the top 5 products by total sales?"
- "Show monthly revenue trend"
- "Which category has the highest average price?"
- "Count records grouped by region"

---

## 🏗️ Architecture

```
User Input (CSV + Question)
       ↓
Data Loader (Pandas)
       ↓
SQLite Database (in-memory)
       ↓
Groq LLM (Llama 3)
       ↓
SQL Query Generation
       ↓
Execution Engine (SQLite)
       ↓
Final Answer + Chart (Plotly)
```
