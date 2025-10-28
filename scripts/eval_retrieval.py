import csv
import datetime
import math
import os
import textwrap
from collections.abc import Iterable
from dataclasses import dataclass

# ===================== USER CONFIG (no .env) =====================
RAW_DIR = "data/raw"
EVAL_CHROMA_DIR = ".chroma_eval"

# Split & retrieval params (all models)
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
TOP_K = 5

# Diagnostics: print top-K candidate chunks per question (per model)
EVAL_DUMP = False
EVAL_DUMP_K = 10
PREVIEW_CHARS = 220

# Gold set with (question, relevant_chunk_ids ; separated)
GOLD_PATH = "data/eval/eval_gold.csv"

# Models to evaluate
# type: "ollama" or "hf"
MODELS = [
    # Generic baseline (multilingual-ish)
    {
        "tag": "generic_ollama",
        "type": "ollama",
        "model": "nomic-embed-text",
        "base_url": "http://localhost:11434",
    },
    # PubMed (English scientific STS/NLI) — for English biomedical data
    {
        "tag": "pubmed_sbiomed",
        "type": "hf",
        "model": "pritamdeka/S-Biomed-Roberta-snli-multinli-stsb",
    },
    # Strong multilingual retrieval models to try with PT-BR corpus:
    {"tag": "bge_m3", "type": "hf", "model": "BAAI/bge-m3"},
    {
        "tag": "mpnet_multi",
        "type": "hf",
        "model": "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
    },
]

# Optional: run only a subset by tag (leave empty to run all)
INCLUDE_TAGS: list[str] = []  # e.g., ["generic_ollama", "bge_m3"]
# ================================================================

# LangChain
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


# -------------------- helpers: loading & chunking --------------------
def load_pdfs_as_docs(raw_dir: str) -> list[Document]:
    docs: list[Document] = []
    for name in sorted(os.listdir(raw_dir)):
        if name.lower().endswith(".pdf"):
            pages = PyPDFLoader(os.path.join(raw_dir, name)).load()
            for d in pages:
                d.metadata = d.metadata or {}
                d.metadata["doc_id"] = name
            docs.extend(pages)
    if not docs:
        raise FileNotFoundError(f"No PDFs found under {raw_dir}")
    return docs


def split_docs_with_chunk_ids(docs: list[Document]) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    counters: dict[str, int] = {}
    for c in chunks:
        doc_id = c.metadata.get("doc_id", c.metadata.get("source", "unknown.pdf"))
        idx = counters.get(doc_id, 0)
        c.metadata["chunk_id"] = f"{doc_id}#c{idx}"
        counters[doc_id] = idx + 1
    return chunks


# -------------------- metrics & gold --------------------
@dataclass
class QueryCase:
    question: str
    relevant_ids: set[str]


def load_gold_csv(path: str) -> list[QueryCase]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Gold file not found: {path}")
    out: list[QueryCase] = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rel = [x.strip() for x in (row["relevant_chunk_ids"] or "").split(";") if x.strip()]
            out.append(QueryCase(question=row["question"].strip(), relevant_ids=set(rel)))
    if not out:
        raise ValueError(f"Gold file is empty: {path}")
    return out


def recall_at_k(ranked: Iterable[str], relevant: set[str], k: int) -> float:
    top = list(ranked)[:k]
    return 1.0 if any(r in relevant for r in top) else 0.0


def mrr_at_k(ranked: Iterable[str], relevant: set[str], k: int) -> float:
    for i, rid in enumerate(list(ranked)[:k], start=1):
        if rid in relevant:
            return 1.0 / i
    return 0.0


def ndcg_at_k(ranked: Iterable[str], relevant: set[str], k: int) -> float:
    ranked = list(ranked)[:k]

    def dcg(ids):
        s = 0.0
        for i, rid in enumerate(ids, start=1):
            rel = 1.0 if rid in relevant else 0.0
            s += rel if i == 1 else (rel / math.log2(i))
        return s

    ideal = dcg(sorted(ranked, key=lambda x: (x in relevant), reverse=True))
    actual = dcg(ranked)
    return (actual / ideal) if ideal > 0 else 0.0


# -------------------- vector store plumbing --------------------
def rebuild_index(persist_dir: str, collection: str, embeddings, chunks: list[Document]) -> Chroma:
    vs = Chroma(
        collection_name=collection, embedding_function=embeddings, persist_directory=persist_dir
    )
    try:
        vs.delete_collection()
    except Exception:
        pass
    vs = Chroma(
        collection_name=collection, embedding_function=embeddings, persist_directory=persist_dir
    )
    if chunks:
        vs.add_documents(chunks)
    return vs


def retrieve_docs(vs: Chroma, question: str, k: int) -> list[Document]:
    return vs.as_retriever(search_type="mmr", search_kwargs={"k": k}).invoke(question)


def retrieve_ids(vs: Chroma, question: str, k: int) -> list[str]:
    return [
        (d.metadata or {}).get("chunk_id", "unknown#c?") for d in retrieve_docs(vs, question, k)
    ]


# -------------------- embeddings factory --------------------
def build_embeddings(entry: dict) -> object:
    kind = entry["type"]
    if kind == "ollama":
        base = entry.get("base_url", "http://localhost:11434")
        return OllamaEmbeddings(model=entry["model"], base_url=base)
    elif kind == "hf":
        return HuggingFaceEmbeddings(model_name=entry["model"], cache_folder=".hf_cache")
    else:
        raise ValueError(f"Unknown embeddings type: {kind}")


# -------------------- pretty printing & CSV report --------------------
def dump_candidates(tag: str, vs: Chroma, question: str, k: int) -> None:
    docs = retrieve_docs(vs, question, k)
    print(f"\n--- Candidates [{tag}] — Top {k}\nQ: {question}\n")
    for i, d in enumerate(docs, 1):
        cid = (d.metadata or {}).get("chunk_id", "unknown#c?")
        src = (d.metadata or {}).get("doc_id", (d.metadata or {}).get("source", ""))
        txt = (d.page_content or "").strip().replace("\n", " ")
        txt = textwrap.shorten(txt, width=PREVIEW_CHARS, placeholder=" …")
        print(f"{i:02d}. {cid} [{src}]")
        print(f"    {txt}")
    print("-" * 70)


def write_metrics_csv(out_path: str, rows: list[dict[str, str]]) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fieldnames = [
        "tag",
        "model_id",
        "recall_at_k",
        "mrr_at_k",
        "ndcg_at_k",
        "k",
        "chunk_size",
        "chunk_overlap",
    ]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def print_summary_table(rows: list[dict[str, str]]) -> None:
    # minimal pretty table
    print("\n=== Retrieval Metrics Summary ===")
    print(
        f"{'Tag':<18} {'Model':<48} {'Recall@' + str(TOP_K):>10} {'MRR@' + str(TOP_K):>10} {'NDCG@' + str(TOP_K):>10}"
    )
    for r in rows:
        print(
            f"{r['tag']:<18} {r['model_id']:<48} {float(r['recall_at_k']):>10.3f} {float(r['mrr_at_k']):>10.3f} {float(r['ndcg_at_k']):>10.3f}"
        )
    print("=" * 86)


# -------------------- main --------------------
def main():
    # Filter models if INCLUDE_TAGS is set
    selected = [m for m in MODELS if not INCLUDE_TAGS or m["tag"] in INCLUDE_TAGS]
    if not selected:
        raise ValueError("No models selected. Check INCLUDE_TAGS/MODELS.")

    print(f"[eval] RAW_DIR={RAW_DIR}  EVAL_CHROMA_DIR={EVAL_CHROMA_DIR}")
    base_docs = load_pdfs_as_docs(RAW_DIR)
    chunks = split_docs_with_chunk_ids(base_docs)

    print("[eval] Example chunk_ids:")
    for d in chunks[:10]:
        print("  -", d.metadata["chunk_id"])

    gold = load_gold_csv(GOLD_PATH)
    print(f"[eval] Gold cases: {len(gold)}")

    results_rows: list[dict[str, str]] = []

    for entry in selected:
        tag = entry["tag"]
        model_id = entry["model"]
        print(f"\n[eval] Building embeddings: {tag} -> {model_id}")
        emb = build_embeddings(entry)

        collection = f"eval_{tag}"
        vs = rebuild_index(EVAL_CHROMA_DIR, collection, emb, chunks)

        if EVAL_DUMP:
            for case in gold:
                dump_candidates(tag, vs, case.question, EVAL_DUMP_K)

        # Score metrics
        recalls, mrrs, ndcgs = [], [], []
        for case in gold:
            ranked = retrieve_ids(vs, case.question, TOP_K)
            recalls.append(recall_at_k(ranked, case.relevant_ids, TOP_K))
            mrrs.append(mrr_at_k(ranked, case.relevant_ids, TOP_K))
            ndcgs.append(ndcg_at_k(ranked, case.relevant_ids, TOP_K))

        import statistics as st

        row = {
            "tag": tag,
            "model_id": model_id,
            "recall_at_k": f"{st.mean(recalls) if recalls else 0.0:.6f}",
            "mrr_at_k": f"{st.mean(mrrs) if mrrs else 0.0:.6f}",
            "ndcg_at_k": f"{st.mean(ndcgs) if ndcgs else 0.0:.6f}",
            "k": str(TOP_K),
            "chunk_size": str(CHUNK_SIZE),
            "chunk_overlap": str(CHUNK_OVERLAP),
        }
        results_rows.append(row)

    # Print and save summary
    print_summary_table(results_rows)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_csv = f"data/eval/reports/metrics_{ts}.csv"
    write_metrics_csv(out_csv, results_rows)
    print(f"\n[eval] Saved metrics CSV -> {out_csv}\n")


if __name__ == "__main__":
    main()
