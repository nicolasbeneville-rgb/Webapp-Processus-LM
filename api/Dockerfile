FROM python:3.11-slim

WORKDIR /app

# Dépendances système
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc && rm -rf /var/lib/apt/lists/*

# Dépendances Python
COPY api/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Code de l'API
COPY api/ ./api/

# Dossier des modèles (monté ou copié au déploiement)
RUN mkdir -p api/models

# Port Cloud Run
ENV PORT=8080
EXPOSE 8080

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8080"]
