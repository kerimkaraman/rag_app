# RAG Pipeline (Milvus + SentenceTransformers + LLaMA3)

Bu proje, Python ile Milvus vektör veritabanı ve sentence-transformers tabanlı embedding modelini kullanarak, LLaMA3 (Ollama) ile semantic retrieval-augmented generation uygular.

## Özellikler
- Milvus'a doğrudan bağlantı (pymilvus)
- Türkçe yorumlar ve temiz kod
- Vektör temelli semantic arama (cosine similarity)
- LLaMA3 ile bağlamı (context) doğrudan kullanıp çıktı üretme
- Bağımsız çalışabilen örnek script ve fonksiyonlar

## Temel Akış (app/rag_pipeline.py)

1. **Milvus'a bağlantı kurulur:**
```python
connections.connect("default", host="localhost", port="19530")
collection = Collection("my_collection")
```

2. **Embedding Modeli yüklenir:**
```python
model = SentenceTransformer("all-MiniLM-L6-v2")
```

3. **Semantic Query Fonksiyonu:**
```python
def query_milvus_with_cosine(user_query, top_k=2):
    # Sorguyu embedle
    query_vec = model.encode([user_query])
    # Milvus'tan tüm embedding ve textleri al
    results = collection.query(expr="id >= 0", output_fields=["id", "text", "embedding"])
    vectors = np.array([r["embedding"] for r in results])
    texts = [r["text"] for r in results]
    sims = cosine_similarity(query_vec, vectors)[0]
    # En yakın top_k metni döndür
    top_indices = np.argsort(sims)[::-1][:top_k]
    return [texts[i] for i in top_indices]
```

4. **RAG Pipeline:**
```python
def rag_pipeline(user_query):
    context_docs = query_milvus_with_cosine(user_query)
    context = "\n".join(context_docs)
    prompt = f"""
    Sen yardımcı bir yapay zeka asistanısın ve aşağıda sana verilen dökümanlara erişimin var.
    Buna göre kullanıcının sorusunu cevapla.
    Dökümanlar: {context}\nKullanıcı Sorusu: {user_query}\nYanıtın:
    """
    response = chat(model="llama3", messages=[{"role": "user", "content": prompt}])
    return response["message"]["content"]
```

5. **Komut Satırı Kullanımı:**
```python
if __name__ == "__main__":
    query = input("Bir soru yazınız: ")
    answer = rag_pipeline(query)
    print("\nLLaMA3 Cevabı:\n", answer)
```

## Gereksinimler
- Python 3.8+
- milvus, sentence-transformers, sklearn, ollama
- Milvus üzerinde "my_collection" isimli koleksiyon oluşturulmuş olmalı ve embeddingler yuklenmiş olmalı
- LLaMA3 modeli Ollama üzerinden localde çalışıyor olmalı

## Örnek Çalışma Akışı

1. **Veri ekleyin:**
- Milvus koleksiyonunu ve embeddingleri oluşturup doldurun (bkz: db/milvus_client.py veya setup scriptleri)

2. **Script ile soru-cevap:**
```
Bir soru yazınız: Milvus nedir?

LLaMA3 Cevabı:
... (toplanan dökümanlara ve LLaMA3 modeline göre otomatik cevap)
```

## Yapı (Kısa)
```
app/
 ├── rag_pipeline.py      # Ana RAG pipeline örneği
 ├── db/milvus_client.py  # Koleksiyon ve ekleme fonksiyonları
 └── ... (diğer modüller)
```

## Lisans
MIT
