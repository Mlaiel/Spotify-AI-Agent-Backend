# Module Logging – Spotify AI Agent

## Présentation
Ce module fournit des configurations de logging centralisées pour tous les environnements (Développement, Production, Test). Optimisé pour la sécurité, l’audit, le ML/IA, l’observabilité et la conformité entreprise.

### Fonctionnalités
- Logs JSON (Production), logs structurés (Dev/Test)
- Rotation, handlers fichier/console
- Intégration Sentry (Prod), Prometheus/OTEL trace
- Masquage des données sensibles, audit, trace/user ID
- Loggers spécifiques ML/IA

### Bonnes pratiques
- Ne jamais logger de données sensibles en clair
- Rotation et rétention pour la conformité (GDPR, SOC2)
- Sentry/alerting activé uniquement en production
- Adapter la config à chaque environnement

### Équipe & Rôles
- Lead Dev, Architecte IA, Backend Senior, ML Engineer, DBA/Data Engineer, Security Specialist, Microservices Architect

---
Pour plus de détails, voir les fichiers de configuration et la checklist projet.
