from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType
from sentence_transformers import SentenceTransformer
import requests
import time

# Milvus bağlantısı
connections.connect("default", host="localhost", port="19530")
print("Milvus'a bağlanıldı.")

# Koleksiyon oluştur veya mevcut olanı kontrol et
collection_name = "documents"

try:
    collection = Collection(collection_name)
    print(f"Koleksiyon bulundu: {collection_name}")
except:
    print(f"Koleksiyon bulunamadı, oluşturuluyor: {collection_name}")
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=384),
        FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535)
    ]
    schema = CollectionSchema(fields, "doküman embeddingleri")
    collection = Collection(collection_name, schema)

# Embedding modeli yükleniyor
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

# Doküman ekleme fonksiyonu
def add_document(doc_id, text):
    embedding = embed_model.encode(text).tolist()
    collection.insert([[doc_id], [embedding], [text]])
    collection.flush()
    print(f"Doküman eklendi: {doc_id} -> {text[:40]}...")

# Arama fonksiyonu
def query_documents(question, top_k=3):
    q_emb = embed_model.encode(question).tolist()
    collection.load()
    print("Koleksiyon arama için belleğe alındı.")
    results = collection.search(
        data=[q_emb],
        anns_field="embedding",
        param={"metric_type": "COSINE", "params": {"nprobe": 10}},
        limit=top_k,
        output_fields=["text"]
    )
    docs = []
    for hits in results:
        for hit in hits:
            docs.append(hit.entity.get("text"))
    return docs

# Ollama API (Llama3) ile cevaplama
def ask_llama3(context, question):
    prompt = f"Bağlam:\n{context}\n\nSoru: {question}\nCevap:"
    url = "http://localhost:11434/api/generate"
    data = {
        "model": "llama3",
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.7}
    }
    response = requests.post(url, json=data)
    if response.status_code != 200:
        raise Exception(f"Ollama isteği başarısız: {response.text}")
    result = response.json()
    return result.get("response", "").strip()

# Örnek doküman ekleme
num_entities = collection.num_entities
if not num_entities:
    add_document(1, "Python bir programlama dilidir.")
    add_document(2, "Milvus bir vektör veritabanıdır.")
    add_document(3, "Ollama, yerel dil modellerini çalıştırmak için kullanılır.")
    time.sleep(1)

# Test sorgu
question = "Milvus nedir?"
docs = query_documents(question)
context = "\n".join(docs)
answer = ask_llama3(context, question)

print("\nDokümanlar:", docs)
print("\nCevap:", answer)
