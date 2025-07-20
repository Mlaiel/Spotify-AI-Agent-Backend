# Spleeter Microservice – Spotify AI Agent

## Présentation
Ce microservice expose la séparation de stems audio (voix, basse, batterie, etc.) via une API FastAPI sécurisée, orchestrée pour l'intégration industrielle dans l'écosystème Spotify AI Agent.

## Fonctionnalités principales
- Séparation de stems audio (2, 4, 5 stems)
- API REST sécurisée (clé API, JWT possible)
- Support des formats audio courants (WAV, MP3, FLAC)
- Traitement asynchrone et scalable (Docker, Uvicorn, Gunicorn)
- Monitoring Prometheus, logs structurés
- Prêt pour CI/CD, déploiement cloud, orchestration Kubernetes
- Sécurité avancée (limite de taille, scan antivirus possible, audit logs)

## Architecture
- **FastAPI** pour l'API REST
- **Spleeter** (Deezer) pour la séparation de stems
- **Docker** pour l'isolation et la portabilité
- **Prometheus** pour le monitoring
- **Logging structuré** (structlog)


## Déploiement avancé
### Docker Compose
```yaml
version: '3.8'
services:
  spleeter:
    build: .
    ports:
      - "8000:8000"
    environment:
      - SPLEETER_API_KEY=supersecret
      - SPLEETER_ENV=production
    volumes:
      - ./data:/data
    restart: unless-stopped
```

### Kubernetes (exemple)
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: spleeter-microservice
spec:
  replicas: 1
  selector:
    matchLabels:
      app: spleeter
  template:
    metadata:
      labels:
        app: spleeter
    spec:
      containers:
      - name: spleeter
        image: spleeter-microservice:latest
        ports:
        - containerPort: 8000
        env:
        - name: SPLEETER_API_KEY
          value: "supersecret"
        - name: SPLEETER_ENV
          value: "production"
        volumeMounts:
        - name: data
          mountPath: /data
      volumes:
      - name: data
        persistentVolumeClaim:
          claimName: spleeter-pvc
```

## Sécurité avancée
- Authentification par clé API (X-API-KEY)
- Prêt pour extension JWT/OAuth2
- Limitation stricte des extensions et taille de fichier
- Logs d'audit, monitoring Prometheus, alerting
- Placeholder pour scan antivirus (ClamAV, etc)

## Monitoring & Observabilité
- Endpoint `/metrics` Prometheus
- Logs JSON (structlog)
- Endpoint `/health` pour supervision

## FAQ
**Q: Comment changer la clé API ?**
A: Modifier la variable d'environnement `SPLEETER_API_KEY`.

**Q: Comment ajouter un scan antivirus ?**
A: Implémenter la fonction `scan_antivirus` dans `utils.py` (ex: ClamAV).

**Q: Comment déployer en production ?**
A: Utiliser Docker Compose ou Kubernetes, configurer les variables d'environnement, activer le monitoring.

**Q: Comment monitorer les erreurs ?**
A: Consulter les métriques Prometheus et les logs JSON.

**Q: Où sont les tests ?**
A: Les tests d'intégration sont réalisés côté backend principal, pas dans ce microservice.

## Bonnes pratiques
- Ne jamais exposer le service sans authentification
- Toujours limiter la taille et le type de fichiers
- Activer le monitoring et les logs
- Nettoyer les fichiers temporaires
- Sécuriser les accès réseau (firewall, VPC, etc)

## Rôles & Auteur
- **Lead Dev & Architecte IA** : Fahed Mlaiel
- **Développeur Backend Senior (Python/FastAPI/Django)** : Fahed Mlaiel
- **Ingénieur Machine Learning (TensorFlow/PyTorch/Hugging Face)** : Fahed Mlaiel
- **DBA & Data Engineer (PostgreSQL/Redis/MongoDB)** : Fahed Mlaiel
- **Spécialiste Sécurité Backend** : Fahed Mlaiel
- **Architecte Microservices** : Fahed Mlaiel

---
© 2025 Fahed Mlaiel – Tous droits réservés

## Sécurité
- Authentification par clé API (X-API-KEY)
- Limitation de taille et type de fichier
- Logs d'accès et d'erreur
- Prêt pour extension JWT, OAuth2, audit, antivirus

## Monitoring
- Endpoint `/metrics` compatible Prometheus
- Logs JSON pour SIEM

## Rôles & Auteur
- **Lead Dev & Architecte IA** : Fahed Mlaiel
- **Développeur Backend Senior (Python/FastAPI/Django)** : Fahed Mlaiel
- **Ingénieur Machine Learning (TensorFlow/PyTorch/Hugging Face)** : Fahed Mlaiel
- **DBA & Data Engineer (PostgreSQL/Redis/MongoDB)** : Fahed Mlaiel
- **Spécialiste Sécurité Backend** : Fahed Mlaiel
- **Architecte Microservices** : Fahed Mlaiel

---
© 2025 Fahed Mlaiel – Tous droits réservés
