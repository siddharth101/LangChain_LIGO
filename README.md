# LangChain-LIGO: Question Answering on LIGO Research Papers

A domain-specific QA system built using **LangChain** and **Retrieval-Augmented Generation (RAG)** that enables users to ask natural language questions about gravitational-wave science and receive answers extracted from a corpus of LIGO publications.

## 🔍 Overview

This project leverages:
- **LangChain** for orchestrating LLM-based QA pipelines
- **Vector databases** (e.g., FAISS or Chroma) for document retrieval
- **LLMs** (e.g., OpenAI GPT, HuggingFace models) for answer synthesis
- **LIGO publications** as the knowledge base

Users can ask open-ended questions like:
> "What is scattered light noise in LIGO and how is it mitigated?"

And receive answers pulled directly from the scientific literature.

---

## 🧠 Features

- 📚 Ingests and indexes PDFs of LIGO scientific papers
- 🔎 Semantic search using vector embeddings
- 💬 Natural language Q&A powered by LLMs
- 🔁 Modular architecture using LangChain chains and tools
- 🧪 Extensible to new domains or datasets

---

## 🏗️ Architecture

1. **Document Loader** – Loads and chunks PDFs into manageable text blocks
2. **Embedding & Indexing** – Computes vector embeddings and stores in FAISS/Chroma
3. **Retriever** – Retrieves relevant chunks based on user query
4. **LLM Chain** – Synthesizes context-aware answers from retrieved chunks

---

## 🛠️ Setup

### 1. Clone the repository
```bash
git clone https://github.com/siddharth101/LangChain_LIGO.git
cd LangChain_LIGO

pip install -r requirements.txt

Create a .env file in the project root:
Add your OpenAI key
OPENAI_API_KEY=your_openai_key_here

python train_multiple.py
```
