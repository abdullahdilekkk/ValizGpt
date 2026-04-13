FROM python:3.13.3

WORKDIR /Valizbackend

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 9005
# --preload: Embedding modeli TEK defa yüklenir, workers fork ile paylaşır → RAM dostu
# --workers 1: Tek worker (ChromaDB race condition önlenir)
# --worker-tmp-dir /dev/shm: Mac/Docker volume errno22 hatasını engeller
CMD ["gunicorn", "mainBack:app", "--workers", "1", "--worker-class", "uvicorn.workers.UvicornWorker", "--bind", "0.0.0.0:9005", "--timeout", "300", "--keep-alive", "5", "--preload", "--worker-tmp-dir", "/dev/shm"]
