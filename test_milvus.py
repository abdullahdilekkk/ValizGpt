import os
import json
from pymilvus import connections, Collection, utility

# 1. Koleksiyon Adı
COLLECTION_NAME = "oteller_milvus_v3"

def check_database():
    # Bağlantı Ayarları
    milvus_host = os.environ.get("MILVUS_HOST", "host.docker.internal")
    
    # Bağlantı Denemesi
    try:
        print(f"⏳ Milvus'a bağlanılıyor ({milvus_host})...")
        connections.connect("default", host=milvus_host, port="19530", timeout=5)
    except Exception as first_e:
        print(f"⚠️ {milvus_host} başarısız, 'localhost' deneniyor...")
        try:
            connections.connect("default", host="localhost", port="19530", timeout=5)
        except Exception as second_e:
            print(f"❌ Bağlantı hatası: {second_e}")
            return

    if not utility.has_collection(COLLECTION_NAME):
        print(f"❌ HATA: '{COLLECTION_NAME}' koleksiyonu bulunamadı.")
        return

    collection = Collection(COLLECTION_NAME)
    collection.load()

    print(f"\n=== 📊 KOLEKSİYON ÖZETİ: {COLLECTION_NAME} ===")
    print(f"Toplam Kayıt Sayısı: {collection.num_entities}")
    
    # Şemayı yazdır
    print("\n--- 🏗️ Tablo Yapısı (Schema) ---")
    for field in collection.schema.fields:
        print(f"Field: {field.name.ljust(15)} | Type: {field.dtype}")

    # Örnek 3 oteli detaylı incele
    print("\n--- 🔍 Örnek Kayıt İncelemesi (İlk 3 Otel) ---")
    try:
        results = collection.query(
            expr="id >= 0", 
            output_fields=["name", "city", "vibe", "concept", "document", "rooms"],
            limit=3
        )

        for i, res in enumerate(results, 1):
            print(f"\n[{i}] OTEL: {res.get('name')}")
            print(f"Şehir: {res.get('city')}")
            print(f"Vibe/Concept: {res.get('vibe')} / {res.get('concept')}")
            
            doc = res.get('document', '')
            print(f"Açıklama (Document) İlk 200 Karakter:\n{doc[:200]}...")
            
            rooms_str = res.get('rooms', '[]')
            print(f"Oda Verisi (Rooms): {str(rooms_str)[:100]}...")
    except Exception as e:
        print(f"❌ Sorgu hatası: {e}")
        
    collection.release()

if __name__ == "__main__":
    check_database()