# AI RAG API (FastAPI + Milvus + LLaMA3)

A ready-to-extend Retrieval-Augmented Generation (RAG) API using FastAPI, Milvus for vector storage, SentenceTransformers for embeddings, and LLaMA3 (via Ollama) for querying.

## Features

- FastAPI backend with modern endpoint structure
- Document embedding (with SentenceTransformers, easily extendable)
- Milvus vector database integration for fast dense search
- RAG pipeline for semantic document retrieval and chat
- Modular architecture (`routers`, `db`, `models`, `utils`)
- Example endpoints for chat and batch document additions

## Endpoints

- `POST /query` — Asks a question, retrieves answer using RAG (Milvus+LLaMA3)
- `POST /chat` — (Sample) Receives a message and returns a fake reply (for demo)
- `POST /add_documents` — Batch document upload + auto-embedding

## Example Request

### Ask a Question

```
POST /query
{
  "question": "What is retrieval-augmented generation?"
}
```

### Add Documents

```
POST /add_documents
{
  "texts": ["Document one.", "Document two."]
}
```

## Installation

1. **Clone this repository:**
   ```
   git clone https://github.com/your-repo/rag_app.git
   cd rag_app
   ```

2. **Install dependencies:**
   ```
   pip install -r requirements.txt
   ```

3. **Ensure Milvus is running:**  
   (Default: `localhost:19530`)

4. **Run your API:**
   ```
   uvicorn app.main:app --reload
   ```

## Project Structure

```
app/
 ├── main.py           # FastAPI entry, includes router
 ├── routers/
 │     └── chat.py     # Chat and document upload endpoints
 ├── db/
 │     └── milvus_client.py   # Milvus connection & bulk/insert helpers
 ├── utils/
 │     └── embedder.py # SentenceTransformer wrapper
 ├── models/
 │     └── schemas.py  # Request/response models
 └── rag_pipeline.py   # RAG core logic
```

## Notes

- Collection is named `documents` and expects fields for content, id, and embedding.
- Embeddings default to `all-MiniLM-L6-v2` model.
- Example Turkish docstrings, suitable for local/testing use.

## License

MIT
