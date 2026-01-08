# Clean Architecture - RAG FastAPI (Cheat Sheet)

## 1) Dependency Rules (Layer Overview)

```mermaid
flowchart LR
    subgraph Interface_Adapters["Interface Adapters - FastAPI Controllers and DTOs"]
        IA1[Routers v1 documents and v1 rag]
        IA2[DTOs Request and Response]
    end

    subgraph Use_Cases["Use Cases - Application"]
        UC1[IngestDocument]
        UC2[QueryRAG]
    end

    subgraph Domain["Domain - Business Rules"]
        D1[LLMProvider interface]
        D2[EmbeddingsProvider interface]
        D3[VectorStoreRepository interface]
        D4[Domain Entities and DTOs]
    end

    subgraph Infrastructure["Infrastructure - Implementations"]
        I1[ChromaVectorStore]
        I2[OpenAIEmbeddings and OllamaEmbeddings]
        I3[OpenAILLMProvider and OllamaLLMProvider and FakeLLM]
        I4[TextSplitter and PDF Loader]
        I5[LangSmith optional and Observability]
    end

    IA1 --> UC1
    IA1 --> UC2
    IA2 --> UC1
    IA2 --> UC2

    UC1 --> D2
    UC1 --> D3
    UC2 --> D3
    UC2 --> D1

    I1 -.implements.-> D3
    I2 -.implements.-> D2
    I3 -.implements.-> D1
    I4 -.supports.-> UC1
    I5 -.traces.-> UC1
    I5 -.traces.-> UC2

    classDef layer fill:#111,stroke:#555,color:#eee;
    class Interface_Adapters,Use_Cases,Domain,Infrastructure layer;
```
Rule: dependencies always point inward. Use cases depend only on domain interfaces. Infrastructure implements those interfaces.

---

## 2) Flow - Upload and Ingest POST v1 documents

```mermaid
sequenceDiagram
    autonumber
    participant Client
    participant Router as FastAPI Router IA
    participant UC as Use Case IngestDocument
    participant VS as VectorStoreRepository iface
    participant EMB as EmbeddingsProvider iface
    participant TXT as TextSplitter and PDF Loader infra
    participant CH as ChromaVectorStore impl VS
    participant EImpl as EmbeddingsImpl OpenAI or Ollama impl EMB

    Client->>Router: POST v1 documents with PDF
    Router->>UC: validated DTO file OK
    UC->>TXT: extract text and chunking
    UC->>EImpl: embed chunks
    EImpl-->>UC: embeddings
    UC->>CH: upsert chunks and embeddings
    CH-->>UC: ids and statistics
    UC-->>Router: result collection stats
    Router-->>Client: 201 Created with payload
```

Quick Notes:
- Validation happens in the adapter layer with FastAPI and DTO.
- IngestDocument does not know Chroma or OpenAI. It only talks to interfaces.
- Real implementations such as Chroma OpenAI and Ollama are injected at the composition root.

---

## 3) Flow - Query RAG POST v1 rag query

```mermaid
sequenceDiagram
    autonumber
    participant Client
    participant Router as FastAPI Router IA
    participant UC as Use Case QueryRAG
    participant VS as VectorStoreRepository iface
    participant LLM as LLMProvider iface
    participant CH as ChromaVectorStore impl VS
    participant LImpl as LLMImpl OpenAI or Ollama or Fake impl LLM

    Client->>Router: POST v1 rag query with question k search_type generate
    Router->>UC: validated DTO
    UC->>CH: search with question k search_type
    CH-->>UC: hits with chunks and metadata
    alt generate is true
        UC->>LImpl: generate with context prompt
        LImpl-->>UC: answer
        UC-->>Router: answer and hits
    else
        UC-->>Router: hits
    end
    Router-->>Client: 200 OK JSON
```

Key Params:
- k controls the number of chunks retrieved
- search_type can be similarity or mmr
- generate when true triggers the LLM call otherwise the API returns only the hits

---

## 4) Testability - Fixtures and DI
- Dependency Injection: use cases receive LLMProvider EmbeddingsProvider and VectorStoreRepository through constructor injection
- Fixtures with pytest: create fake implementations such as FakeLLMProvider and FakeVectorStore for deterministic and fast tests
- Integration tests: run with real Chroma and LLM based on env and -m integration markers

---

## 5) Observability optional
- LangSmith enabled via env for chain tracing prompt logs and latency metrics
- Helps debug hallucinations and measure prompt and context quality

---

## 6) Main Routes - Adapters
- POST v1 documents for ingestion embeddings and upsert
- GET v1 documents for collection stats
- POST v1 rag query for retrieval and optional generation

---

## 7) Architecture Benefits
- Low coupling use cases know nothing about external tools
- High testability with mocks and fakes without network calls
- Easy evolution swap Chroma for PGVector swap OpenAI for Ollama or adjust chunking without touching domain logic
