# GraphRAG Dialysis Patient Search

An end-to-end GraphRAG (Graph Retrieval-Augmented Generation) system for clinical reasoning over dialysis patient data. The system builds a knowledge graph from synthetic patient records and performs multi-hop reasoning to identify effective treatments under complex medical conditions.

---

## 🚀 Problem Statement

Dialysis patients often have multiple co-morbidities (e.g., fluid overload + cardiac history). Identifying effective treatments requires reasoning across similar patient cases.

This project simulates that process using:
- Knowledge graphs
- Multi-hop traversal
- LLM-assisted reasoning

---

## 🧠 Key Features

- Synthetic dataset generation (20+ patients)
- Hybrid entity extraction (LLM + rule-based fallback)
- Multi-relational knowledge graph (NetworkX)
- Multi-hop GraphRAG reasoning:
  patient → similar cases → treatments → outcomes
- Missing link handling with intelligent fallback strategies
- Full reasoning path (explainability)
- RAGAS-style evaluation metrics:
  - Answer relevance
  - Faithfulness
  - Context precision
- Local LLM integration using Ollama (no external APIs)

---

## 🏗️ Architecture Overview

User Query  
↓  
Query Understanding  
↓  
Graph Traversal Engine  
↓  
Similarity Matching  
↓  
Multi-hop Reasoning  
↓  
Fallback Handling (if needed)  
↓  
Final Answer + Reasoning Path + Confidence  

---

## 🔍 Example Query

What treatments worked for patients with fluid overload who also had prior cardiac events?

---

## ✅ Example Output

Recommended Treatments:
- Ultrafiltration → improved (3 patients)
- Diuretics → stable (2 patients)

Reasoning Path:
P3 → similar(P7, P12) → treatments → outcomes

Confidence Score: 0.82

Fallback Used: Partial condition matching

---

## 🧩 Tech Stack

- Python
- NetworkX (graph construction)
- Pandas (data handling)
- Ollama (local LLM inference)
- Requests (API calls)

---

## ⚙️ Setup Instructions

1. Clone Repository
git clone <your-repo-link>
cd GraphRAG_DialysisSearch

2. Install Dependencies
pip install -r requirements.txt

3. Start Ollama
ollama serve

Default endpoint:
http://127.0.0.1:11434

4. Run the Application
python app.py

---

## 🧪 Evaluation Metrics

The system includes lightweight RAGAS-style evaluation:

- Answer Relevance → Does the answer match the query intent?
- Faithfulness → Is the answer grounded in retrieved data?
- Context Precision → Are retrieved nodes relevant?

---

## 🛡️ Robustness Features

- Handles missing data gracefully
- Falls back when LLM fails
- Supports partial condition matching
- Prevents graph traversal loops

---

## 📁 Project Structure

GraphRAG_DialysisSearch/
│
├── app.py
├── data_gen.py
├── extractor.py
├── graph_builder.py
├── reasoning.py
├── similarity.py
├── evaluator.py
├── utils.py
├── config.py
├── requirements.txt
└── README.md

---

## 🔮 Future Improvements

- Graph embeddings (Node2Vec / GNN)
- Hybrid Vector + Graph RAG
- Streamlit UI for visualization
- Real-world clinical dataset integration
- Multi-agent reasoning system

---

## 👤 Author

Prem Raga

---

## ⭐ If you found this useful

Star the repo and share!