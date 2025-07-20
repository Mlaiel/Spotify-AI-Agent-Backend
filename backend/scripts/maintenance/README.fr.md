# Scripts Maintenance – Spotify AI Agent

## Présentation
Ce dossier contient tous les scripts de maintenance prêts pour la production : cache, logs, optimisation DB, tuning performance, sécurité, conformité. Optimisé pour le business, le ML/IA, l'audit et l'observabilité.

### Fonctionnalités & Bonnes Pratiques
- **cache_warmup.py** : préchauffe tous les caches (Redis, modèles ML/IA, réponses API)
- **cleanup_logs.sh** : nettoyage automatisé et sécurisé des logs (rotation, archivage, RGPD, sécurité)
- **optimize_db.py** : optimisation de toutes les bases (Postgres, MongoDB, Redis)
- **performance_tuning.py** : tuning performance backend, ML/IA, DB, cache, observabilité
- Logging, gestion d'erreur, sécurité, conformité
- Extensible pour d'autres checks, audits, workflows ML/IA

### Recommandations
- Automatiser et tester régulièrement les scripts de maintenance
- Traiter logs et DB selon RGPD et standards sécurité
- Optimiser tuning et cache pour les workflows ML/IA
- Intégrer observabilité et conformité dans la CI/CD

### Équipe & Rôles
- Lead Dev, Architecte IA, Backend Senior, ML Engineer, DBA/Data Engineer, Security Specialist, Microservices Architect

---
**Voir les scripts et la checklist projet pour plus de détails.**
