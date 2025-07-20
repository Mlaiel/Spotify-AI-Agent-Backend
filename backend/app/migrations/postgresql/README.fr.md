# Documentation des migrations PostgreSQL (FR)

**Créé par l’équipe : Spotify AI Agent Core Team**
- Lead Dev + Architecte IA
- Développeur Backend Senior (Python/FastAPI/Django)
- Ingénieur Machine Learning (TensorFlow/PyTorch/Hugging Face)
- DBA & Data Engineer (PostgreSQL/Redis/MongoDB)
- Spécialiste Sécurité Backend
- Architecte Microservices

---

## Objectif
Ce module contient tous les scripts et migrations versionnées pour le schéma, les tables, les données avancées, la conformité et l’automatisation du backend Spotify AI Agent. Tous les scripts sont prêts pour la production, alignés métier et intégrés CI/CD.

## Structure
- `versions/` : Tous les scripts de migration (schéma, tables, logique métier, avancé, rollback, partitioning, bulk, health-check, RGPD, zero-downtime)
- `env.py`, `alembic.ini`, `script.py.mako` : Configuration Alembic et templates
- `__init__.py` : Auto-discovery, gouvernance, sécurité, helpers pour l’automatisation

## Fonctionnalités avancées & Bonnes pratiques
- Tous les scripts sont atomiques, idempotents, réversibles et CI/CD-ready
- Sécurité : audit, logs, gestion des accès, anonymisation RGPD, rollback, contrôles conformité
- Conformité : RGPD, SOC2, ISO 27001, audit trails, gestion du consentement, effacement automatisé
- Automatisation : auto-discovery, dry-run, audit, health-check, bulk import/export, partitioning, zero-downtime
- Multilingue, géo, IA/ML, analytics, versioning, logique métier supportés
- Tous les changements sont testés sur staging avant production

## Exemple d’utilisation
### Exécuter toutes les migrations (Core & Avancé)
```bash
alembic upgrade head
```

### Exécuter une migration manuellement
```bash
python versions/007_partitioning.py
```

### Import/Export massif
```python
from versions.008_bulk_import_export import bulk_import, bulk_export
bulk_import('analytics', 'analytics.csv')
bulk_export('users', 'users.csv')
```

### Health-Check
```python
from versions.009_health_check import check_health
check_health()
```

### Effacement RGPD
```python
from versions.010_gdpr_erasure import erase_user
erase_user(user_id=123)
```

### Migration sans interruption
```python
from versions.011_zero_downtime_migration import upgrade
upgrade()
```

### Rollback
```bash
python versions/006_rollback.py
```

## Exemples de requêtes
- Trouver tous les logs d’audit pour un utilisateur :
  ```sql
  SELECT * FROM audit_log WHERE entity_type = 'user' AND entity_id = 123;
  ```
- Lister tous les utilisateurs sans consentement :
  ```sql
  SELECT * FROM consent WHERE granted = false;
  ```
- Analytics par mois :
  ```sql
  SELECT date_trunc('month', created_at) AS month, COUNT(*) FROM analytics GROUP BY month;
  ```

## Gouvernance & Extension
- Tous les scripts doivent être relus et validés par le Core Team
- Les nouveaux scripts doivent respecter la convention de nommage/version et inclure des docstrings
- Les contrôles de sécurité et conformité sont obligatoires pour toute migration

## Contact
Pour toute modification, incident ou question, contactez le Core Team via Slack #spotify-ai-agent ou GitHub. Pour la sécurité/conformité, escalader au Security Officer.

---

*Cette documentation est générée et maintenue automatiquement dans le CI/CD. Dernière mise à jour : juillet 2025.*

