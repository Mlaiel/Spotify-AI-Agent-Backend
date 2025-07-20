# Requirements – Spotify AI Agent

## Présentation
Ce dossier contient toutes les listes de dépendances pour le développement, la production, la sécurité, les tests et la conformité. La séparation par environnement et par rôle garantit sécurité, maintenabilité, scalabilité et conformité métier.

### Structure & Bonnes Pratiques
- **base.txt** : dépendances principales pour backend, ML/IA, sécurité, DB, API, microservices
- **development.txt** : outils dev, linter, debugger, test, mocking
- **production.txt** : uniquement des paquets stables et validés pour la prod
- **security.txt** : sécurité, audit, conformité, gestion des secrets, scan vulnérabilités
- **testing.txt** : test, mocking, coverage, intégration, charge, tests ML/IA

#### Recommandations
- Vérifier régulièrement les dépendances avec `safety` et `pip-audit`
- Séparer strictement dev/prod (aucun outil de debug/dev en prod)
- Modulariser ML/IA serving et data engineering selon les besoins
- Automatiser les scans sécurité et conformité (CI/CD)
- Versionner et relire tous les changements (code review, audit)

### Équipe & Rôles
- Lead Dev, Architecte IA, Backend Senior, ML Engineer, DBA/Data Engineer, Security Specialist, Microservices Architect

---
**Voir les fichiers requirements et la checklist projet pour plus de détails.**
