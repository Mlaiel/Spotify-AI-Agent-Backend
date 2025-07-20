# Architecture – Backend IA Spotify

Cette section décrit l’architecture logicielle, data et ML du backend, avec schémas et patterns industriels.

## 1. Vue d’ensemble
- Microservices découplés (API, ML, tâches, data)
- Communication via REST, gRPC, events (RabbitMQ/Redis)
- Séparation stricte des responsabilités (SRP)

## 2. Schéma d’architecture
```
[Client] ⇄ [API Gateway] ⇄ [Services API/ML] ⇄ [DB/Cache/ML Models]
```
- API Gateway (auth, rate limit, audit)
- Services API (FastAPI), ML (TensorFlow/PyTorch), tâches (Celery)
- Bases de données (PostgreSQL, MongoDB), cache (Redis)

## 3. Patterns utilisés
- CQRS, Event Sourcing, Repository, Factory, Dependency Injection
- Sécurité Zero Trust, RBAC, audit trail
- Observabilité (tracing, metrics, logs)

## 4. Scalabilité & haute disponibilité
- Load balancing, auto-scaling, health checks
- Blue/Green deployment, rollback, canary releases

## 5. MLOps & DataOps
- Pipelines CI/CD ML (training, déploiement, monitoring)
- Versioning modèles, gestion des features, retraining automatisé

## 6. Sécurité by design
- Isolation réseau, secrets management, scans vulnérabilités

Pour chaque composant, voir détails dans les fichiers dédiés (services, ML, sécurité, etc.).
