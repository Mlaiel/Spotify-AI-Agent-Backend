# Documentation des scripts de migration MongoDB (FR)

**Créé par l’équipe : Spotify AI Agent Core Team**
- Lead Dev + Architecte IA
- Développeur Backend Senior (Python/FastAPI/Django)
- Ingénieur Machine Learning (TensorFlow/PyTorch/Hugging Face)
- DBA & Data Engineer (PostgreSQL/Redis/MongoDB)
- Spécialiste Sécurité Backend
- Architecte Microservices

---

## Objectif
Ce dossier contient tous les scripts de migration MongoDB versionnés pour les collections, indexes, transformations avancées et opérations de conformité. Tous les scripts sont conçus pour l’automatisation, le CI/CD, la logique métier, la sécurité et la conformité réglementaire (RGPD, SOC2, ISO 27001).

## Vue d’ensemble des scripts
- `001_create_collections.js` – Crée toutes les collections principales avec validation
- `002_add_indexes.js` – Ajoute des indexes pour la performance, la recherche, l’analytics
- `003_data_migration.js` – Transforme ou migre les données pour de nouvelles fonctionnalités
- `004_schema_updates.js` – Met à jour la validation du schéma ou ajoute des champs
- `005_create_advanced_collections.js` – Crée des collections avancées (IA, audit, sécurité, consentement, géo, multilingue, versioning)
- `006_add_advanced_indexes.js` – Ajoute des indexes avancés (IA, audit, sécurité, consentement, géo, multilingue, versioning)
- `rollback.js` – Rollback pour migrations sécurisées (voir ci-dessous)
- `partitioning.js` – Partitionnement des grandes collections pour la scalabilité
- `bulk_import_export.js` – Import/export massif pour migration et sauvegarde
- `health_check.js` – Health-check automatisé des collections et indexes
- `gdpr_erasure.js` – Effacement RGPD des données utilisateur
- `zero_downtime_migration.js` – Migrations sans interruption de service

> **Note :** Tous les scripts sont idempotents, prêts pour la production et supportent les modes dry-run et audit si applicable.

## Fonctionnalités avancées & Bonnes pratiques
- **Sécurité :**
  - Utilisation de variables d’environnement pour les credentials et endpoints
  - Toutes les actions sont loguées dans les collections `audit_log` et `security_events`
  - Validation, gestion d’erreur et rollback intégrés
- **Conformité :**
  - RGPD, SOC2, ISO 27001, audit trails, gestion du consentement
  - Scripts automatisés de rétention et d’effacement des données
- **Automatisation :**
  - CI/CD-ready, utilisables seuls ou en pipeline
  - Auto-discovery des scripts pour l’orchestration des migrations
- **Logique métier :**
  - Toutes les migrations sont alignées avec les besoins métier Spotify AI Agent
  - Support du multilingue, géo, IA/ML, analytics, versioning
- **Gouvernance :**
  - Tous les changements sont tracés, versionnés et auditables
  - Politiques d’usage et d’extension incluses

## Exemples d’utilisation
### Exécuter toutes les migrations (Core & Avancé)
```bash
mongo < 001_create_collections.js
mongo < 002_add_indexes.js
mongo < 003_data_migration.js
mongo < 004_schema_updates.js
mongo < 005_create_advanced_collections.js
mongo < 006_add_advanced_indexes.js
mongo < partitioning.js --eval 'var collection="analytics"; var shardKey="user_id"'
mongo < bulk_import_export.js --eval 'var mode="import"; var collection="users"; var file="users.json"'
mongo < health_check.js
mongo < gdpr_erasure.js --eval 'var userId="..."'
mongo < zero_downtime_migration.js
```

### Exemple de rollback
```bash
mongo < rollback.js --eval 'var targetVersion="004"'
```

### Import/Export massif
```bash
mongoimport --db spotify_ai --collection users --file users.json
mongoexport --db spotify_ai --collection audit_log --out audit_log.json
```

### Health-Check
```bash
mongo < health_check.js
```

### Effacement RGPD
```bash
mongo < gdpr_erasure.js --eval 'var userId="..."'
```

### Migration sans interruption
```bash
mongo < zero_downtime_migration.js
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

