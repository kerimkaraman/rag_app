from pymilvus import connections, Collection
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from ollama import chat  # LLaMA3’ü localden çağıracağız
import numpy as np

# 1. Milvus bağlantısı
connections.connect("default", host="localhost", port="19530")
collection = Collection("my_collection")

# 2. Embedding modeli
model = SentenceTransformer("all-MiniLM-L6-v2")

def query_milvus_with_cosine(user_query, top_k=2):
    """
    Kullanıcının sorusunu embed edip Milvus'taki vektörlerle cosine similarity'e göre karşılaştırır.
    """
    # Query’yi embedle
    query_vec = model.encode([user_query])
    # Milvus'taki tüm embedding’leri ve metadata’yı al
    results = collection.query(
        expr="id >= 0",
        output_fields=["id", "text", "embedding"]
    )
    # Embedding’leri numpy array’e dönüştür
    vectors = np.array([r["embedding"] for r in results])
    texts = [r["text"] for r in results]
    # Cosine benzerliğini hesapla
    sims = cosine_similarity(query_vec, vectors)[0]
    # En benzer top_k dokümanı seç
    top_indices = np.argsort(sims)[::-1][:top_k]
    top_texts = [texts[i] for i in top_indices]
    return top_texts


def rag_pipeline(user_query):
    """
    RAG zinciri:
    1. Milvus'tan ilgili dokümanları alır
    2. Dokümanları birleştirip LLaMA3'e bağlam olarak yollar
    3. Nihai cevabı üretir
    """
    context_docs = query_milvus_with_cosine(user_query)
    context = "\n".join(context_docs)
    prompt = f"""
You are a helpful AI assistant with access to retrieved context documents.
Use the information below to answer the user's question accurately.

Context:
{context}

User Question: {user_query}

Answer:
"""
    # LLaMA3 ile yanıt üretimi (local)
    response = chat(model="llama3", messages=[{"role": "user", "content": prompt}])
    return response["message"]["content"]


if __name__ == "__main__":
    query = input("Ask something: ")
    answer = rag_pipeline(query)
    print("\nLLaMA3 Response:\n", answer)
