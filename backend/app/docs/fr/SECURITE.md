# Sécurité (FR)

Cette section détaille la politique de sécurité, les mécanismes de défense et les scripts de contrôle du backend.

## 1. Principes fondamentaux
- Zero Trust, défense en profondeur, RBAC strict
- Chiffrement (données au repos/en transit, TLS 1.3, AES-256)
- Gestion des secrets (Vault, variables d’env)

## 2. Authentification & autorisation
- OAuth2 (PKCE), JWT, scopes, refresh tokens
- Séparation des rôles (admin, artiste, IA, service)
- Rate limiting, brute-force protection

## 3. Audit & traçabilité
- Logging structuré (ELK, Graylog)
- Audit trail, alertes sécurité, SIEM ready

## 4. Sécurité API & microservices
- CORS restrictif, validation stricte des payloads
- Limitation des permissions inter-services
- Scans automatiques (SAST, DAST, dépendances)

## 5. Scripts & automatisation
- Scripts de scan vulnérabilités (`scripts/security/scan.sh`)
- Tests d’intrusion automatisés (`make pentest`)
- Monitoring sécurité (Prometheus, alertmanager)

## 6. Conformité & RGPD
- Droit à l’oubli, anonymisation, logs conformes
- Documentation des traitements de données

> **Astuce** : Tous les scripts et configurations sont fournis dans le dossier `security` et `scripts/security`.
