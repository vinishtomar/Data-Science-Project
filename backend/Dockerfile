# Image officielle Python
FROM python:3.11-slim

# Répertoire de travail dans le conteneur
WORKDIR /app

# Installer les dépendances Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copier tout le code dans le conteneur
COPY . .

# Port exposé (celui utilisé par Flask)
EXPOSE 5000

# Démarrer l'application Flask
CMD ["python", "app.py"]