FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# A lógica de corte (módulo importado pelo app) e o serviço
COPY cortar_banners.py .
COPY app.py .

# A maioria das plataformas (Render, Railway, Fly) injeta a porta via $PORT
ENV PORT=8000
EXPOSE 8000

CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT}"]
