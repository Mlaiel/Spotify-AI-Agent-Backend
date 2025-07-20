# Deployment – Spotify AI Backend (EN)

This section explains how to deploy, monitor, and maintain the backend in production.

## 1. Deployment Strategy
- Automated CI/CD (GitHub Actions, GitLab CI, Jenkins)
- Multi-stage Docker build, signed images
- Blue/Green, Canary, instant rollback

## 2. Deployment Procedure
```bash
# Build Docker image
make build
# Push image
make push
# Deploy (K8s, Docker Compose)
make deploy
```

## 3. Post-Deployment Monitoring
- Health checks, readiness/liveness probes
- Auto alerts (Prometheus, Grafana, Slack)
- Automated rollback on failure

## 4. Maintenance & Migrations
- DB migration scripts (`scripts/database/migrate.sh`)
- Automated backup/restore (`scripts/database/backup.sh`)
- Secret rotation, security patching

## 5. Example Provided Files
- `docker-compose.yml`, `Dockerfile`, `k8s/`, `scripts/`
- Config templates for each environment

> **Tip:** All scripts are ready to use in `scripts/` and `config/`.
