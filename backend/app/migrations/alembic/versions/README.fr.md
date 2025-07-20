"""
README – Scripts de migration Alembic (FR)

Créé par l’équipe : Spotify AI Agent Core Team
- Lead Dev + Architecte IA
- Développeur Backend Senior (Python/FastAPI/Django)
- Ingénieur Machine Learning (TensorFlow/PyTorch/Hugging Face)
- DBA & Data Engineer (PostgreSQL/Redis/MongoDB)
- Spécialiste Sécurité Backend
- Architecte Microservices

Ce dossier contient tous les scripts de migration Alembic versionnés pour le schéma PostgreSQL. Chaque fichier représente une évolution de schéma atomique et réversible. Les scripts sont nommés avec un timestamp et une description claire.

## Bonnes pratiques
- Chaque migration est relue, suivie en CI/CD et testée sur staging avant la production.
- Utiliser le template fourni pour toute nouvelle migration (voir `template.py`).
- Toutes les migrations sont auditables et incluent la logique de rollback.
- Sécurité, audit, analytics et conformité inclus pour l’entreprise.
- Index, contraintes et partitionnement recommandés pour la performance et la conformité.

## Exemple d’utilisation
```bash
alembic upgrade head
alembic downgrade -1
```

## Sécurité & Gouvernance
- Toutes les migrations sont versionnées et relues
- Sécurité et conformité incluses pour audit et RGPD/DSGVO
- Toute modification nécessite une revue métier et sécurité

Pour toute question ou modification, contactez le Core Team via Slack #spotify-ai-agent ou GitHub.
"""
