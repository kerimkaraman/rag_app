from pymilvus import connections, utility, FieldSchema, DataType, Collection, CollectionSchema

def connect_to_milvus():
    """Milvus'a bağlanır ve bağlantı başarılıysa uyarı verir."""
    connections.connect(alias="default", host="localhost", port="19530")
    print("Milvus'a bağlantı başarılı.")

def check_collections():
    """Milvus'taki mevcut koleksiyonları listeler."""
    print("Mevcut koleksiyonlar:", utility.list_collections())

def create_collections():
    """
    'documents' adında bir koleksiyon oluşturur.
    - id: benzersiz döküman numarası
    - content: dökümanın metni
    - embedding: dökümanın vektör embedding'i (float listesidir)
    """
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=500),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=384)
    ]
    schema = CollectionSchema(fields, description="RAG Dökümanlar Koleksiyonu")
    Collection(name="documents", schema=schema)
    print("'documents' koleksiyonu başarıyla oluşturuldu.")

def insert_document(content: str, embedding: list[float]):
    """
    Tek doküman ekleme işlevi
    """
    collection = Collection("documents")
    collection.insert([[content], [embedding]])
    print("Tek döküman kaydedildi.")

def insert_documents(contents: list[str], embeddings: list[list[float]]):
    """
    Çoklu döküman ekleme işlemi
    """
    collection = Collection("documents")
    collection.insert([contents, embeddings])
    print(f"{len(contents)} döküman kaydedildi.")
