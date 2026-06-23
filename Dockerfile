FROM python:3.12-slim

WORKDIR /app

# OpenCV headless precisa apenas destas libs de sistema
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 libgl1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY cortar_banners.py .
COPY deteccao_cv.py .
COPY app.py .
COPY index.html .

ENV PORT=8000
EXPOSE 8000

CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT}"]
