import json
import os
import time
import socket
import math
from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility
from sentence_transformers import SentenceTransformer

# 1. AYARLAR
JSON_PATH = "hotels_big_db.json"
COLLECTION_NAME = "oteller_milvus_v3"

def get_milvus_host():
    env_host = os.environ.get("MILVUS_HOST")
    if env_host: return env_host
    try:
        socket.gethostbyname("host.docker.internal")
        return "host.docker.internal"
    except (socket.gaierror, socket.error):
        return "localhost"

MILVUS_HOST = get_milvus_host()

def main():
    print(f"🚀 Zenginleştirilmiş Milvus Yükleme Başlatılıyor... (Host: {MILVUS_HOST})")
    
    # Embedding modelini yükle
    print("⏳ Vektör modeli yükleniyor (MiniLM-L12-v2)...")
    model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

    # Milvus Bağlantısı
    print(f"⏳ Milvus'a bağlanılıyor...")
    try:
        connections.connect("default", host=MILVUS_HOST, port="19530")
    except Exception as e:
        print(f"❌ Bağlantı hatası: {e}.")
        return

    # Koleksiyon Temizliği
    if utility.has_collection(COLLECTION_NAME):
        utility.drop_collection(COLLECTION_NAME)
        print(f"🧹 Eski koleksiyon '{COLLECTION_NAME}' silindi.")

    # 2. ŞEMA TANIMI
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=False),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=384),
        FieldSchema(name="name", dtype=DataType.VARCHAR, max_length=500),
        FieldSchema(name="city", dtype=DataType.VARCHAR, max_length=100),
        FieldSchema(name="district", dtype=DataType.VARCHAR, max_length=100),
        FieldSchema(name="region", dtype=DataType.VARCHAR, max_length=100),
        FieldSchema(name="vibe", dtype=DataType.VARCHAR, max_length=100),
        FieldSchema(name="segment", dtype=DataType.VARCHAR, max_length=100),
        FieldSchema(name="stars", dtype=DataType.INT64),
        FieldSchema(name="min_price", dtype=DataType.INT64),
        FieldSchema(name="concept", dtype=DataType.VARCHAR, max_length=200),
        FieldSchema(name="suitable_months", dtype=DataType.VARCHAR, max_length=500),
        FieldSchema(name="document", dtype=DataType.VARCHAR, max_length=4000), # 🚨 Kapasite 4000'e çıkarıldı
        FieldSchema(name="rooms", dtype=DataType.VARCHAR, max_length=65535)
    ]
    schema = CollectionSchema(fields, description="ValizGPT Zenginleştirilmiş Otel DB")
    collection = Collection(COLLECTION_NAME, schema)

    # 3. VERİ HAZIRLAMA VE ZENGİNLEŞTİRME
    print(f"⏳ '{JSON_PATH}' okunuyor...")
    with open(JSON_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    ids, names, cities, districts, regions, vibes, segments = [], [], [], [], [], [], []
    stars, min_prices, concepts, months, documents, rooms_data = [], [], [], [], [], []

    print(f"🧠 {len(data)} otel için zengin içerik ve vektör hazırlanıyor...")
    
    for idx, otel in enumerate(data):
        # A) Fiyat Hesaplama
        rooms_list = otel.get("rooms", [])
        prices = [r.get("price", 0) for r in rooms_list if r.get("price")]
        min_p = min(prices) if prices else 0

        # B) VERİ ZENGİNLEŞTİRME (Placeholder metinlerin yerine gerçek detayları koyuyoruz)
        # JSON'daki tüm teknik alanları birleştiriyoruz
        base_desc = otel['rag_data'].get('description', '')
        proximity = otel['location'].get('proximity', 'Bilgi Yok')
        amenities = ", ".join(otel.get('amenities', []))
        target_audiences = ", ".join(otel['info'].get('target_audiences', []))
        concept = otel['details'].get('concept', 'Standart')
        
        # Bu metin hem LLM tarafından okunacak hem de Vektör aramasına temel olacak
        rich_doc = (
            f"{otel['info']['name']} - {otel['location']['city']}, {otel['location']['district']}. "
            f"{base_desc} "
            f"📍 Konum Detayı: {proximity}. "
            f"🏨 Konsept: {concept}. "
            f"✨ Sunulan İmkanlar: {amenities}. "
            f"👥 Hitap Ettiği Kitle: {target_audiences}."
        ).replace("\n", " ").strip()

        ids.append(int(otel.get("id", idx)))
        names.append(otel["info"].get("name", ""))
        cities.append(otel["location"].get("city", ""))
        districts.append(otel["location"].get("district", ""))
        regions.append(otel["location"].get("region", ""))
        vibes.append(otel["info"].get("vibe", ""))
        segments.append(otel["info"].get("segment", "")) 
        stars.append(int(otel["info"].get("stars", 0)))
        min_prices.append(int(min_p))
        concepts.append(concept)
        months.append(otel.get("suitable_months", ""))
        documents.append(rich_doc[:3900]) # Sınıra takılmaması için
        rooms_data.append(json.dumps(rooms_list, ensure_ascii=False))

        if (idx + 1) % 1000 == 0:
            print(f"   -> {idx + 1} otel işlendi.")

    # 4. TOPLU EMBEDDING ÜRETİMİ (Artık zengin döküman üzerinden!)
    print(f"🧠 Zenginleştirilmiş vektörler üretiliyor (bu işlem arama kalitesini %100 artıracak)...")
    all_embeddings = model.encode(documents, batch_size=64, show_progress_bar=True).tolist()

    # 5. INSERT VE INDEX
    print("💾 Milvus'a yazılıyor...")
    collection.insert([
        ids, all_embeddings, names, cities, districts, regions, vibes, segments,
        stars, min_prices, concepts, months, documents, rooms_data
    ])
    collection.flush()

    print("⚡ Index oluşturuluyor (HNSW)...")
    collection.create_index(field_name="embedding", index_params={
        "index_type": "HNSW", "metric_type": "L2", "params": {"M": 8, "efConstruction": 64}
    })
    print("🎉 İŞLEM TAMAM! Artık otellerin tüm özellikleri (Havuz, Spa, Konum) sistem tarafından biliniyor.")

if __name__ == "__main__":
    main()