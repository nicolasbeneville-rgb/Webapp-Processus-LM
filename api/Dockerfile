FROM python:3.11-slim

WORKDIR /app

# Dépendances Python
COPY requirements_api.txt .
RUN pip install --no-cache-dir -r requirements_api.txt

# Serveur API
COPY api_server.py .

# Dossier des modèles (copier vos modèles exportés ici)
COPY models/ models/

# Port Cloud Run
ENV PORT=8080
ENV MODELS_DIR=models

EXPOSE 8080

CMD ["python", "api_server.py"]
