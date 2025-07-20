# Architecture – Spotify AI Backend (EN)

This section describes the software, data, and ML architecture of the backend, with diagrams and industry patterns.

## 1. Overview
- Decoupled microservices (API, ML, tasks, data)
- Communication via REST, gRPC, events (RabbitMQ/Redis)
- Strict separation of concerns (SRP)

## 2. Architecture Diagram
```
[Client] ⇄ [API Gateway] ⇄ [API/ML Services] ⇄ [DB/Cache/ML Models]
```
- API Gateway (auth, rate limit, audit)
- API Services (FastAPI), ML (TensorFlow/PyTorch), tasks (Celery)
- Databases (PostgreSQL, MongoDB), cache (Redis)

## 3. Patterns Used
- CQRS, Event Sourcing, Repository, Factory, Dependency Injection
- Zero Trust security, RBAC, audit trail
- Observability (tracing, metrics, logs)

## 4. Scalability & High Availability
- Load balancing, auto-scaling, health checks
- Blue/Green deployment, rollback, canary releases

## 5. MLOps & DataOps
- CI/CD ML pipelines (training, deployment, monitoring)
- Model versioning, feature management, automated retraining

## 6. Security by Design
- Network isolation, secrets management, vulnerability scans

For each component, see details in the dedicated files (services, ML, security, etc.).
