# 🧳 ValizGPT 

**ValizGPT**, tatil ve seyahat planlaması için geliştirilmiş, Yapay Zeka (AI) destekli, RAG (Retrieval-Augmented Generation) mimarisine sahip akıllı bir seyahat asistanı ve otel tavsiye sistemidir. 

Kullanıcıların seyahat kriterlerini (bütçe, süre, kişi sayısı, bölge) doğal dille analiz eden bu sistem, **Milvus** vektör veritabanını ve yerel LLM modellerini kullanarak en doğru otel ve seyahat seçeneklerini sunar.

---

## 🚀 Teknolojik AltyapıTabii, hemen aşağıya kopyalayıp doğrudan GitHub depona yapıştırabileceğin düz metin (Markdown) formatında ekliyorum.

-----

# 🧳 ValizGPT: AI-Powered RAG Travel Assistant

     

**ValizGPT** is a high-performance, context-aware travel recommendation system built on a **Retrieval-Augmented Generation (RAG)** architecture.

Designed for the modern traveler, the system processes natural language queries to analyze complex variables like budget, duration, group size, and regional preferences. By integrating a **Milvus** vector database with local Large Language Models (LLMs) via **Ollama**, it provides accurate, grounded, and personalized hotel and itinerary suggestions without relying on external cloud-based AI APIs.

-----

## 🚀 Key Features

  * **Semantic Intelligence:** Utilizes Hugging Face `Sentence Transformers` to understand user intent beyond simple keyword matching.
  * **Contextual Grounding (RAG):** Prevents LLM hallucinations by retrieving real-world data from a local vector store before generation.
  * **Conversational Flow:** Orchestrated by **LangGraph**, enabling multi-turn dialogues and refined search capabilities.
  * **Privacy-First Design:** Runs LLMs locally via Ollama, ensuring user data never leaves the local environment.
  * **Production-Ready Infrastructure:** Fully containerized with Docker Compose for consistent cross-platform deployment.

-----

## 🏗️ System Architecture

1.  **Ingestion:** User input is received via the React/Vite frontend.
2.  **Vectorization:** The FastAPI backend generates high-dimensional embeddings using local NLP models.
3.  **Retrieval:** The system performs a similarity search in **Milvus** to fetch the most relevant hotel profiles from `hotels_big_db.json`.
4.  **Augmentation:** The retrieved context is injected into a specialized prompt template.
5.  **Generation:** The local LLM synthesizes a coherent, data-backed response.

### Tech Stack

| Layer | Technology |
| :--- | :--- |
| **Backend** | Python 3.11, FastAPI, LangGraph |
| **Frontend** | React, Vite, TailwindCSS |
| **Vector Engine** | Milvus (Standalone) |
| **LLM Inference** | Ollama |
| **Embeddings** | Hugging Face (all-MiniLM-L6-v2) |
| **Containerization** | Docker, Docker Compose |

-----

## ⚙️ Prerequisites

Ensure your environment meets the following requirements:

  * **Docker & Docker Compose**
  * **Ollama:** Running locally (The system routes to `http://host.docker.internal:11434`).
  * **Milvus:** Active instance (Standalone or via Docker).
  * **Hardware:** 16GB+ RAM recommended for local LLM inference.

-----

## 🛠️ Setup & Installation

### 1\. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/ValizGpt.git
cd ValizGpt
```

### 2\. Configure Environment

Create a `.env` file in the root directory:

```env
HF_TOKEN=your_huggingface_write_token

# Optional network overrides
# OLLAMA_HOST=http://host.docker.internal:11434
# MILVUS_HOST=host.docker.internal
```

### 3\. Launch Services

```bash
docker-compose up --build
```

Access the application:

  * **Frontend:** `http://localhost:5173`
  * **Backend Docs:** `http://localhost:9005/docs`

-----

## ⚠️ Engineering Considerations

  * **Embedding Persistence:** Do not remove `hotels_big_db.json`. It is the source-of-truth for the initial vector collection seeding.
  * **macOS Optimizations:** Specific configurations for Apple Silicon (M-series) are included in `docker-compose.yml` (e.g., `OMP_NUM_THREADS=1`) to prevent PyTorch deadlocks.
  * **Asynchronous Processing:** The backend leverages Python's `asyncio` for non-blocking I/O during vector search and LLM streaming.

-----

*Developed with a focus on RAG optimization and local AI scalability.*

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
