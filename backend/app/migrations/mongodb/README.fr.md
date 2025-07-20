# Documentation des migrations MongoDB (FR)

**Créé par l’équipe : Spotify AI Agent Core Team**
- Lead Dev + Architecte IA
- Développeur Backend Senior (Python/FastAPI/Django)
- Ingénieur Machine Learning (TensorFlow/PyTorch/Hugging Face)
- DBA & Data Engineer (PostgreSQL/Redis/MongoDB)
- Spécialiste Sécurité Backend
- Architecte Microservices

---

## Objectif
Ce module contient tous les scripts et migrations versionnées pour les collections, indexes, données avancées, conformité et automatisation du backend Spotify AI Agent. Tous les scripts sont prêts pour la production, alignés métier et intégrés CI/CD.

## Structure
- `migrations/` : Tous les scripts de migration (collections, indexes, data, updates, avancé, rollback, partitioning, bulk, health-check, RGPD, zero-downtime)
- `migrate.py` : Script d’automatisation pour exécuter toutes les migrations dans l’ordre (core & avancé)
- `__init__.py` : Auto-discovery, gouvernance, sécurité, helpers pour l’automatisation

## Fonctionnalités avancées & Bonnes pratiques
- Tous les scripts sont idempotents, versionnés et CI/CD-ready
- Sécurité : variables d’environnement pour credentials et endpoints, audit log, rollback, contrôles conformité
- Conformité : RGPD, SOC2, ISO 27001, audit trails, gestion du consentement, effacement automatisé
- Automatisation : auto-discovery, dry-run, audit, health-check, bulk import/export, partitioning, zero-downtime
- Multilingue, géo, IA/ML, analytics, versioning, logique métier supportés
- Tous les changements sont testés sur staging avant production

## Exemple d’utilisation
### Exécuter toutes les migrations (Core & Avancé)
```bash
python migrate.py --env=production
```

### Exécuter une migration manuellement
```bash
mongo < migrations/005_create_advanced_collections.js
```

### Import/Export massif
```bash
mongo < migrations/bulk_import_export.js --eval 'var mode="import"; var collection="users"; var file="users.json"'
```

### Health-Check
```bash
mongo < migrations/health_check.js
```

### Effacement RGPD
```bash
mongo < migrations/gdpr_erasure.js --eval 'var userId="..."'
```

### Migration sans interruption
```bash
mongo < migrations/zero_downtime_migration.js
```

### Rollback
```bash
mongo < migrations/rollback.js --eval 'var targetVersion="004"'
```

## Exemples de requêtes
- Trouver tous les logs d’audit pour un utilisateur :
  ```js
  db.audit_log.find({entity_type: "user", entity_id: ObjectId("...")})
  ```
- Lister tous les utilisateurs sans consentement :
  ```js
  db.consent.find({granted: false})
  ```
- Recherche géo pour les événements :
  ```js
  db.geo_events.find({location: {$near: {$geometry: {type: "Point", coordinates: [lng, lat]}, $maxDistance: 10000}}})
  ```
- Contenu multilingue par langue :
  ```js
  db.multilingual_content.find({lang: "fr"})
  ```

## Gouvernance & Extension
- Tous les scripts doivent être relus et validés par le Core Team
- Les nouveaux scripts doivent respecter la convention de nommage/version et inclure des docstrings
- Les contrôles de sécurité et conformité sont obligatoires pour toute migration

## Contact
Pour toute modification, incident ou question, contactez le Core Team via Slack #spotify-ai-agent ou GitHub. Pour la sécurité/conformité, escalader au Security Officer.

---

*Cette documentation est générée et maintenue automatiquement dans le CI/CD. Dernière mise à jour : juillet 2025.*

