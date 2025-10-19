from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pymilvus import connections, Collection
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from ollama import chat
import numpy as np

app = FastAPI(title="RAG API", description="Milvus + LLaMA3 ile çalışan RAG Pipeline")

# Milvus bağlantısı kuruluyor
connections.connect("default", host="localhost", port="19530")
collection = Collection("my_collection")

# Embedding modeli yükleniyor
model = SentenceTransformer("all-MiniLM-L6-v2")

class QueryRequest(BaseModel):
    question: str

def query_milvus_with_cosine(user_query, top_k=2):
    # Sorguyu embed ediyoruz ve Milvus'taki vektörlerle kıyaslıyoruz
    query_vec = model.encode([user_query])
    results = collection.query(expr="id >= 0", output_fields=["id", "text", "embedding"])
    if not results:
        raise HTTPException(status_code=404, detail="Milvus'ta döküman bulunamadı.")
    vectors = np.array([r["embedding"] for r in results])
    texts = [r["text"] for r in results]
    sims = cosine_similarity(query_vec, vectors)[0]
    top_indices = np.argsort(sims)[::-1][:top_k]
    top_texts = [texts[i] for i in top_indices]
    return top_texts

def rag_pipeline(user_query):
    # Milvus'tan döküman çek + LLaMA3 ile yanıt oluştur
    context_docs = query_milvus_with_cosine(user_query)
    context = "\n".join(context_docs)
    prompt = f"""
Sen yardımcı bir yapay zeka asistanısın ve aşağıda sana verilen dökümanlara erişebiliyorsun.
Kullanıcı sorusunu bu dökümanlara dayanarak yanıtla.

Dökümanlar:
{context}

Kullanıcı Sorusu: {user_query}

Yanıtın:
"""
    response = chat(model="llama3", messages=[{"role": "user", "content": prompt}])
    return response["message"]["content"]

@app.post("/query")
async def query_api(req: QueryRequest):
    try:
        answer = rag_pipeline(req.question)
        return {"question": req.question, "answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
