# Dockerfile for Spotify AI Agent API (production-ready)
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PYTHONUNBUFFERED=1
ENV ENV=production

CMD ["uvicorn", "v1.main:app", "--host", "0.0.0.0", "--port", "8000"]
