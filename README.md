# RAG FastAPI (LangChain + LangSmith + Clean Architecture)

Este projeto demonstra **Retrieval-Augmented Generation (RAG)** com:
- **LangChain** (LCEL, loaders, splitters, retrievers, embeddings)
- **Chroma** como VectorStore persistente local
- **LangSmith** para tracing/observabilidade (opcional via env)
- **FastAPI** como camada web
- **Clean Architecture** (domínio e casos de uso independentes de frameworks)

---

## 🧱 Arquitetura (pastas)

```
app/                       # settings, container e app factory
domain/                    # contratos (protocols) e modelos de domínio
use_cases/                 # aplicação (ingest, query_rag, etc.)
infrastructure/            # adapters (LangChain, Chroma, loaders, observability)
interface_adapters/        # web (FastAPI routers)
tests/                     # testes unitários e E2E mínimos
```

Fluxo (simplificado): **/v1/documents → ingest (loader + splitter + Chroma)** → **/v1/rag/query → retriever (MMR/k) → (opcional) geração determinística**.

---

## ✅ Step 1 — Echo + LangSmith

- Rota de sanidade: `POST /v1/echo` usando LCEL (`RunnableLambda`) que apenas ecoa a pergunta.
- LangSmith habilitado via `.env` (opcional). Sem chave, o app não quebra.

## 📥 Step 2 — Ingestão com Chroma (LangChain)

- Loader de PDF: `PyMuPDFLoader`.
- Chunking: `RecursiveCharacterTextSplitter`.
- Persistência: `Chroma` com `chromadb.PersistentClient`.
- Endpoint: `POST /v1/documents` (form-data `file` PDF).
- Teste: `tests/test_ingest.py` (gera PDF sintético e valida ingest).

## 🔎 Step 3 — Retrieval-first RAG + “G” opcional (determinístico)

- Provider de **embeddings** (default: `FakeEmbeddings`) injetado no `Chroma`.
- `as_retriever(search_type="mmr", k=5)` para consulta.
- Caso de uso `QueryRAGUseCase` retorna `hits` e, se `generate=true`, formata resposta determinística a partir do contexto.
- Endpoint: `POST /v1/rag/query`.
- Teste: `tests/test_rag_retrieval.py`.

> **Step 4 (próximo)**: plugar um `LLMProvider` real (Ollama/OpenAI) mantendo a interface no domínio e um teste de integração *skippable* por env.

---

## 🛠️ Setup

```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -e .[dev]
cp .env.example .env
uvicorn app.main:app --reload
# Docs: http://127.0.0.1:8000/docs
```

### Variáveis de ambiente (.env)

```dotenv
# Observabilidade (LangSmith) — opcionais
LANGCHAIN_TRACING_V2=false
LANGCHAIN_API_KEY=
LANGSMITH_PROJECT=rag-fastapi-lc

# Armazenamento / Chroma
VECTOR_STORE_PROVIDER=chroma
CHROMA_DIR=.chroma
RAW_DIR=data/raw
CHROMA_COLLECTION=documents

# Embeddings
EMBEDDINGS_PROVIDER=fake     # fake | ollama | openai
EMBEDDINGS_MODEL=fake        # ex.: nomic-embed-text (Ollama)
# OLLAMA_BASE_URL=http://localhost:11434
# OPENAI_API_KEY=
```

---

## 🧪 Testes

```bash
pytest -q
# ou arquivos específicos:
pytest -q tests/test_echo.py
pytest -q tests/test_ingest.py
pytest -q tests/test_rag_retrieval.py
```

---

## 🧭 Endpoints

### `POST /v1/echo`
```json
{ "question": "ping" } -> { "answer": "echo: ping" }
```

### `POST /v1/documents`
- Form-data: `file` (PDF)
- Resposta 201: `{ "filename": "...", "uploaded_at": "...", "num_docs": 1, "num_chunks": 12, "collection": "documents" }`

### `POST /v1/rag/query`
- Payload: `{ "question": "…", "generate": false, "k": 5, "search_type": "mmr" }`
- Resposta: `{ "answer": null|"...", "hits": [{ "content": "...", "metadata": {...} }, ...] }`

---

## 🧭 Roadmap sugerido (pequenos & testáveis)
1. **LLMProvider real (Ollama/OpenAI)** com adapter LCEL e teste de integração skipável.
2. **Citações** nos resultados (ID/URI do chunk) e ordenação por score.
3. **Streaming** da geração via Server-Sent Events (SSE).
4. **Avaliações no LangSmith** (datasets / runs com metadados de chunking e k).
