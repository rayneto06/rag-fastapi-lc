# 🧠 RAG FastAPI (LangChain + LangSmith + Clean Architecture)

A fully modular **Retrieval-Augmented Generation (RAG)** API built with **FastAPI**, following **Clean Architecture**, and powered by **LangChain**, **LangSmith**, and local or cloud LLMs (OpenAI / Ollama).  
Designed as a didactic, production-quality example for study and portfolio use.

---

## 🚀 Overview

This project demonstrates an end-to-end RAG pipeline:

```
Upload PDF → Split into Chunks → Embed → Store in Chroma → Query → Retrieve Context → Generate Answer
```

### Core Capabilities

| Feature | Description |
|----------|-------------|
| 📁 Document Upload | Upload PDFs and persist embeddings in Chroma |
| 🔍 Context Retrieval | Query Chroma using similarity / MMR search |
| 🧠 Generation | Deterministic or real generation using OpenAI / Ollama |
| ⚙️ Architecture | Clean Architecture (Domain, Use Cases, Infrastructure, Web) |
| 🧪 Testing | Full unit + integration suite with fake and real LLM providers |
| 📊 Observability | Optional LangSmith tracing and metrics |

---

## 🧩 Project Structure

```
rag-fastapi-lc/
├── app/                        # FastAPI entrypoint and settings
├── domain/                     # Core interfaces (LLMProvider, etc.)
├── use_cases/                  # Business logic (Ingest, QueryRAG)
├── infrastructure/             # Adapters (Chroma, LangChain LLMs, Embeddings)
├── interface_adapters/         # FastAPI routers (v1/documents, v1/rag)
├── tests/                      # Unit & integration tests
├── .env.example                # Template environment variables
├── pyproject.toml              # Dependencies and dev tools
└── README.md                   # You are here
```

---

## ⚙️ Installation

```bash
git clone https://github.com/rayneto06/rag-fastapi-lc.git
cd rag-fastapi-lc
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -e .[dev]
cp .env.example .env
```

---

## 🔧 Configuration (.env)

Edit your `.env` file to choose providers and settings.

### Example for local Ollama:

```dotenv
LLM_PROVIDER=ollama
LLM_MODEL=llama3.1
EMBEDDINGS_PROVIDER=ollama
EMBEDDINGS_MODEL=nomic-embed-text
CHROMA_COLLECTION=documents_ollama
LANGCHAIN_TRACING_V2=false
```

### Example for OpenAI:

```dotenv
LLM_PROVIDER=openai
LLM_MODEL=gpt-4o-mini
EMBEDDINGS_PROVIDER=openai
OPENAI_API_KEY=sk-...
```

### Example for fake (deterministic testing):

```dotenv
LLM_PROVIDER=fake
EMBEDDINGS_PROVIDER=fake
```

---

## 🧠 Running Locally

### Start the API

```bash
uvicorn app.main:app --reload
```

Docs available at: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

### Start Ollama (if local)

```bash
ollama serve
ollama pull llama3.1
ollama pull nomic-embed-text
```

---

## 📬 API Endpoints

| Method | Endpoint | Description |
|---------|-----------|-------------|
| `POST` | `/v1/echo` | Simple echo test |
| `POST` | `/v1/documents` | Upload and ingest a PDF into Chroma |
| `GET`  | `/v1/documents` | Get collection stats |
| `POST` | `/v1/rag/query` | Perform retrieval and (optional) generation |

### Example Query (JSON)

```json
{
  "question": "Resuma o documento",
  "generate": true,
  "k": 4,
  "search_type": "mmr"
}
```

---

## 🧪 Testing

### Run all tests

```bash
pytest -q
```

### Run deterministic (fake providers only)

```bash
pytest -q -m "not integration"
```

### Run real integration tests (OpenAI or Ollama)

```bash
pytest -q -m integration
```

---

## 🧱 Clean Architecture Principles

- **Domain**: pure business rules (interfaces only)  
- **Use Cases**: application logic (no external dependencies)  
- **Infrastructure**: adapters for LangChain, Chroma, LLMs, embeddings  
- **Interface Adapters**: FastAPI controllers and DTOs  

Each layer depends **only inward**, enabling testing, replacement, and extension without coupling.

---

## 📊 LangSmith Tracing (optional)

Enable full tracing and experiment logging:

```dotenv
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=sk-...
LANGSMITH_PROJECT=rag-fastapi-lc
```

View runs at [https://smith.langchain.com](https://smith.langchain.com).

---

## 🧰 Makefile (optional)

If you use `make`, create a `Makefile` with shortcuts like:

```makefile
serve:
	uvicorn app.main:app --reload

ollama:
	ollama serve

test:
	pytest -q -m "not integration"

test-int:
	pytest -q -m integration
```

---

## 🧭 Next Steps

- Add `GET /v1/rag/summarize` (full-document summaries)
- Include citation metadata (`source`, `page`, `chunk_id`) in query hits
- Add Dockerfile and CI (GitHub Actions)

---

© 2025 Raymundo Neto — Educational RAG Project
