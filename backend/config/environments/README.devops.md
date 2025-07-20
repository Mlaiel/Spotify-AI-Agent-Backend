# DevOps & CI/CD – Environments Guide

## Ziel
Dieses Dokument beschreibt Best Practices und konkrete Anleitungen für DevOps, CI/CD und Secrets-Management im Zusammenhang mit den Umgebungsdateien des Spotify AI Agent Backends.

---

## 1. Struktur & Versionierung
- **Alle .env-Dateien** sind im Verzeichnis `config/environments/` versioniert.
- **Keine echten Secrets** im Repository! Nur Beispielwerte oder Platzhalter.
- **.env.production** wird im Deployment durch echte Secrets ersetzt (z.B. via CI/CD-Secret-Store).

## 2. CI/CD-Integration
- **Empfohlene Tools:** GitHub Actions, GitLab CI, Azure DevOps, Jenkins, ArgoCD, Helm, Docker Compose, Kubernetes Secrets.
- **Beispiel-Workflow:**
  1. Secrets werden im CI/CD-System als sichere Variablen gespeichert.
  2. Beim Build/Deploy werden die Secrets in die Zielumgebung injiziert (z.B. als Umgebungsvariablen oder als .env-Datei).
  3. Die Anwendung liest die Variablen zur Laufzeit (12-Factor-Prinzip).

## 3. Docker & Kubernetes
- **Docker Compose:**
  - Nutze `env_file` in `docker-compose.yml` für lokale Entwicklung.
- **Kubernetes:**
  - Nutze `Secret`- und `ConfigMap`-Ressourcen für Umgebungsvariablen.
  - Beispiel:
    ```yaml
    apiVersion: v1
    kind: Secret
    metadata:
      name: spotify-ai-agent-secrets
    type: Opaque
    data:
      POSTGRES_PASSWORD: <base64-encoded>
    ```

## 4. Secrets-Management
- **Empfohlene Tools:** HashiCorp Vault, AWS Secrets Manager, Azure Key Vault, GCP Secret Manager.
- **Best Practice:**
  - Niemals Secrets im Code oder in Git speichern.
  - Zugriff auf Secrets nur für autorisierte Pipelines und Services.

## 5. Monitoring & Compliance
- **Sicherstellen, dass alle Monitoring- und Alerting-DSNs (Sentry, Prometheus, OTEL) korrekt gesetzt sind.**
- **GDPR- und Compliance-Variablen** (z.B. DATA_RETENTION_DAYS) müssen in Produktion aktiv sein.

## 6. Beispiel: GitHub Actions
```yaml
name: Deploy Backend
on: [push]
jobs:
  deploy:
    runs-on: ubuntu-latest
    env:
      SECRET_KEY: ${{ secrets.SECRET_KEY }}
      POSTGRES_PASSWORD: ${{ secrets.POSTGRES_PASSWORD }}
    steps:
      - uses: actions/checkout@v3
      - name: Build & Deploy
        run: |
          docker-compose --env-file config/environments/.env.production up -d --build
```

---

## Autoren & Rollen
- Lead Dev, Architecte IA, Backend Senior, ML Engineer, DBA/Data Engineer, Security Specialist, Microservices Architect

---
Für weitere Details siehe die Projekt-Checkliste und die einzelnen .env-Dateien.
