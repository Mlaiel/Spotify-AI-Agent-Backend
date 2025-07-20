# Déploiement – Backend IA Spotify (FR)

Cette section explique comment déployer, monitorer et maintenir le backend en production.

## 1. Stratégie de déploiement
- CI/CD automatisé (GitHub Actions, GitLab CI, Jenkins)
- Build multi-stage Docker, images signées
- Blue/Green, Canary, rollback instantané

## 2. Procédure de déploiement
```bash
# Build image Docker
make build
# Push image
make push
# Déploiement (K8s, Docker Compose)
make deploy
```

## 3. Monitoring post-déploiement
- Health checks, readiness/liveness probes
- Alertes auto (Prometheus, Grafana, Slack)
- Rollback automatisé en cas d’échec

## 4. Maintenance & migrations
- Scripts de migration DB (`scripts/database/migrate.sh`)
- Backup/restore automatisé (`scripts/database/backup.sh`)
- Rotation des secrets, patch sécurité

## 5. Exemples de fichiers fournis
- `docker-compose.yml`, `Dockerfile`, `k8s/`, `scripts/`
- Templates de config pour chaque environnement

> **Astuce** : Tous les scripts sont prêts à l’emploi dans `scripts/` et `config/`.
