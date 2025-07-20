# Dockerfile optimisé IA/Audio/Backend (Python 3.10, sécurité, ML, Spleeter)
FROM python:3.10-slim

# Sécurité : utilisateur non-root
RUN useradd -m appuser
WORKDIR /app

# Dépendances système (audio, ML, sécurité, monitoring)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    build-essential \
    git \
    libsndfile1 \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Copie requirements
COPY requirements.txt ./

# Install Python dependencies (verrouillage sécurité, pas de cache, pas de pip upgrade)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copie code applicatif (sécurité: pas de .git, pas de fichiers secrets)
COPY . .

# Permissions sécurité
RUN chown -R appuser:appuser /app
USER appuser

# Expose port API
EXPOSE 8000

# Commande de démarrage (production, logs structurés)
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--proxy-headers", "--forwarded-allow-ips=*"]
