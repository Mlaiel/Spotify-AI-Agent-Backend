# Scripts Deployment – Spotify AI Agent

## Présentation
Ce dossier contient tous les scripts de déploiement prêts pour la production : zero-downtime, health-check, rollback, backup, conformité. Optimisé pour la sécurité, l'observabilité, le ML/IA, l'audit et la logique métier.

### Fonctionnalités & Bonnes Pratiques
- **backup.sh** : backup complet et versionné avant chaque déploiement
- **deploy.sh** : déploiement zero-downtime avec health-check, logging, audit
- **health_check.sh** : health-check de tous les services clés (backend, celery, redis, postgres, nginx)
- **rollback.sh** : rollback automatisé et sécurisé sur le dernier backup
- Logging, gestion d'erreur, sécurité, conformité
- Extensible pour blue/green, canary, multi-région

### Recommandations
- Automatiser et tester backups et health-checks
- S'entraîner régulièrement aux rollbacks (disaster recovery)
- Intégrer les checks sécurité/conformité dans la CI/CD
- Tenir un audit-log pour tous les déploiements et rollbacks

### Équipe & Rôles
- Lead Dev, Architecte IA, Backend Senior, ML Engineer, DBA/Data Engineer, Security Specialist, Microservices Architect

---
**Voir les scripts et la checklist projet pour plus de détails.**
