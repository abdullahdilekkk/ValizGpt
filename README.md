# 🧳 ValizGPT 

**ValizGPT**, tatil ve seyahat planlaması için geliştirilmiş, Yapay Zeka (AI) destekli, RAG (Retrieval-Augmented Generation) mimarisine sahip akıllı bir seyahat asistanı ve otel tavsiye sistemidir. 

Kullanıcıların seyahat kriterlerini (bütçe, süre, kişi sayısı, bölge) doğal dille analiz eden bu sistem, **Milvus** vektör veritabanını ve yerel LLM modellerini kullanarak en doğru otel ve seyahat seçeneklerini sunar.

---

## 🚀 Teknolojik Altyapı

* **Backend:** Python (FastAPI / LangGraph)
* **Frontend:** React / Vite
* **Vektör Veritabanı:** Milvus (Yerel makinede çalıştırılan)
* **Yerel LLM Sağlayıcısı:** Ollama
* **Embeddings & NLP:** Hugging Face (Sentence Transformers)
* **Altyapı:** Tamamen Dockerize edilmiştir.

---

## ⚙️ Kurulum Öncesi Gereksinimler

Projenin sorunsuz çalışabilmesi için bilgisayarınızda aşağıdaki araçların kurulu ve **çalışır durumda** olması gerekmektedir:

1. **Docker ve Docker Compose**
2. **Ollama:** Yerel model sağlayıcısı. (Proje varsayılan olarak `http://host.docker.internal:11434` adresinde Ollama arar).
3. **Milvus:** Vektör veritabanınız bilgisayarınızda (Docker dışında veya bağımsız bir konteyner olarak) çalışıyor olmalıdır. Proje Milvus'a `host.docker.internal` adresi üzerinden bağlanır.

---

## 🛠️ Kurulum Adımları

Projeyi kendi bilgisayarınızda ayağa kaldırmak için aşağıdaki adımları sırasıyla uygulayın:

### 1. Projeyi Klonlayın
```bash
git clone https://github.com/KULLANICI_ADINIZ/ValizGpt.git
cd ValizGpt
```

### 2. Çevre Değişkenlerini (Environment Variables) Ayarlayın
Güvenlik sebebiyle API şifreleri kodun içinde barınmaz. Proje ana dizininde `.env` isimli bir dosya oluşturun ve içine kendi Hugging Face güvenlik anahtarınızı ekleyin:

```env
# .env dosyası
HF_TOKEN=sizin_huggingface_tokeniniz_buraya_gelecek
```

*(Not: Eğer `OLLAMA_HOST` veya `MILVUS_HOST` adreslerinizi değiştirmek isterseniz `docker-compose.yml` içindeki environment kısmından kendi yerel ayarlarınıza göre uyarlayabilirsiniz).*

### 3. Projeyi Çalıştırın
Tüm backend ve frontend servislerini ayağa kaldırmak için terminale şu komutu yazın:

```bash
docker-compose up --build
```

Bu komut:
* Python Backend sunucusunu `http://localhost:9005` portunda,
* React Frontend arayüzünü `http://localhost:5173` portunda başlatacaktır.

---

## ⚠️ Önemli Notlar ve Veritabanı Durumu

* **Büyük Veriler Dahildir:** Proje klasöründeki `hotels_big_db.json` dosyası sistemin veritabanını oluşturduğu için silinmemelidir.
* **Sanal Ortamlar Yoksayıldı:** `.venv` veya `node_modules` gibi gereksiz büyüklükteki dosyalar `.gitignore` ve `.dockerignore` listelerine eklendiği için GitHub'a gönderilmemiştir. Docker, sistemi ayağa kaldırırken bu gereksinimleri kendi içinde otomatik kuracaktır.
* Mac kullanıcıları için Pytorch thread çakışmalarına (Deadlock) karşı gerekli önlemler `docker-compose` değişkenlerinde alınmıştır (`OMP_NUM_THREADS=1` vb.).

---
*Geliştirme ve RAG optimizasyonları devam etmektedir...*
