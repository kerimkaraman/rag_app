from pymilvus import connections, Collection
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from ollama import chat  # LLaMA3’ü lokalde çağıracağız
import numpy as np

# 1. Milvus bağlantısı
connections.connect("default", host="localhost", port="19530")
collection = Collection("my_collection")

# 2. Embedding modeli
model = SentenceTransformer("all-MiniLM-L6-v2")

def query_milvus_with_cosine(user_query, top_k=2):
    """
    Kullanıcıdan gelen soruyu embed edip Milvus'taki vektörlerle cosine similarity kullanarak karşılaştırır.
    """
    # Sorguyu embedle
    query_vec = model.encode([user_query])
    # Milvus'tan tüm embedding ve meta veriyi al
    results = collection.query(
        expr="id >= 0",
        output_fields=["id", "text", "embedding"]
    )
    # Embedding'leri numpy array'e dönüştür
    vectors = np.array([r["embedding"] for r in results])
    texts = [r["text"] for r in results]
    # Cosine benzerliği hesapla
    sims = cosine_similarity(query_vec, vectors)[0]
    # En ilgili top_k dökümanı sırala
    top_indices = np.argsort(sims)[::-1][:top_k]
    top_texts = [texts[i] for i in top_indices]
    return top_texts


def rag_pipeline(user_query):
    """
    RAG mantığı:
    1. Milvus'tan alakalı dökümanları bulur
    2. Dökümanları prompta ekler
    3. LLaMA3 ile bağlamlı cevap üretir
    """
    context_docs = query_milvus_with_cosine(user_query)
    context = "\n".join(context_docs)
    prompt = f"""
Sen yardımcı bir yapay zeka asistanısın ve aşağıda sana verilen dökümanlara erişimin var.
Buna göre kullanıcının sorusunu cevapla.

Dökümanlar:
{context}

Kullanıcı Sorusu: {user_query}

Yanıtın:
"""
    # LLaMA3 ile yanıt oluşturuluyor
    response = chat(model="llama3", messages=[{"role": "user", "content": prompt}])
    return response["message"]["content"]

if __name__ == "__main__":
    query = input("Bir soru yazınız: ")
    answer = rag_pipeline(query)
    print("\nLLaMA3 Cevabı:\n", answer)
