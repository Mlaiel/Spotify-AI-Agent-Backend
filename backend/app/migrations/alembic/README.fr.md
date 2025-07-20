# Documentation des migrations Alembic (FR)

**Créé par l’équipe : Spotify AI Agent Core Team**
- Lead Dev + Architecte IA
- Développeur Backend Senior (Python/FastAPI/Django)
- Ingénieur Machine Learning (TensorFlow/PyTorch/Hugging Face)
- DBA & Data Engineer (PostgreSQL/Redis/MongoDB)
- Spécialiste Sécurité Backend
- Architecte Microservices

---

## Objectif
Ce dossier contient tous les scripts de migration Alembic pour le schéma PostgreSQL. Il garantit des évolutions de schéma versionnées, auditables et réversibles pour le backend.

## Bonnes pratiques
- Toutes les migrations sont atomiques, idempotentes et réversibles.
- Chaque script est nommé avec timestamp et description claire.
- Utiliser `alembic revision --autogenerate -m "description"` pour chaque nouvelle migration.
- Tester chaque migration sur staging avant la production.
- Toutes les migrations sont relues et suivies en CI/CD.
- Sécurité, audit, analytics, partitionnement et conformité inclus pour l’entreprise.
- Utiliser les fonctionnalités avancées : partitionnement, triggers, contraintes, audit logging, rollback, masquage des données sensibles dans les logs.

## Utilisation
```bash
# Créer une nouvelle migration
alembic revision --autogenerate -m "ajout nouvelle table"
# Appliquer les migrations
alembic upgrade head
# Downgrade (si besoin)
alembic downgrade -1
```

## Structure du dossier
- `versions/` : Tous les scripts de migration (un fichier par évolution)
- `env.py`, `script.py.mako` : Environnement Alembic, templates, logging avancé, audit, sécurité

---

## Contact
Pour toute question ou modification, contactez le Core Team via Slack #spotify-ai-agent ou GitHub.

