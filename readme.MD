# RAG Pipeline (Milvus + SentenceTransformers + LLaMA3)

This project applies semantic retrieval-augmented generation (RAG) using the Milvus vector database and a sentence-transformers-based embedding model in Python, combined with LLaMA3 (via Ollama).

## Features
- Direct Milvus connection (via pymilvus)
- Clear and well-commented code
- Vector-based semantic search (cosine similarity)
- Context-aware response generation using LLaMA3
- Standalone sample scripts and functions

## Basic Flow (see `app/rag_pipeline.py`)

1. **Connect to Milvus:**
```python
connections.connect("default", host="localhost", port="19530")
collection = Collection("my_collection")
```

2. **Load Embedding Model:**
```python
model = SentenceTransformer("all-MiniLM-L6-v2")
```

3. **Semantic Query Function:**
```python
def query_milvus_with_cosine(user_query, top_k=2):
    # Embed the query
    query_vec = model.encode([user_query])
    # Retrieve all embeddings and texts from Milvus
    results = collection.query(expr="id >= 0", output_fields=["id", "text", "embedding"])
    vectors = np.array([r["embedding"] for r in results])
    texts = [r["text"] for r in results]
    sims = cosine_similarity(query_vec, vectors)[0]
    # Return the top_k most similar texts
    top_indices = np.argsort(sims)[::-1][:top_k]
    return [texts[i] for i in top_indices]
```

4. **RAG Pipeline:**
```python
def rag_pipeline(user_query):
    context_docs = query_milvus_with_cosine(user_query)
    context = "\n".join(context_docs)
    prompt = f"""
    You are an AI assistant and you have access to the documents provided below.
    Please answer the user's question based on these documents.
    Documents: {context}\nUser Question: {user_query}\nYour Answer:
    """
    response = chat(model="llama3", messages=[{"role": "user", "content": prompt}])
    return response["message"]["content"]
```

5. **Command Line Usage:**
```python
if __name__ == "__main__":
    query = input("Enter your question: ")
    answer = rag_pipeline(query)
    print("\nLLaMA3 Response:\n", answer)
```

## Requirements
- Python 3.8+
- milvus, sentence-transformers, sklearn, ollama
- A Milvus collection named "my_collection" should already exist with embedded vectors
- LLaMA3 model should be running locally via Ollama

## Example Workflow

1. **Add Data:**
- Set up and populate the Milvus collection with embeddings (see: `db/milvus_client.py` or setup scripts)

2. **Script Q&A:**
```
Enter your question: What is Milvus?

LLaMA3 Response:
... (automatic response generated based on retrieved documents and LLaMA3 model)
```

## Structure (Summary)
```
app/
 ├── rag_pipeline.py      # Main RAG pipeline example
 ├── db/milvus_client.py  # Collection and add functions
 └── ... (other modules)
```

## License
MIT
