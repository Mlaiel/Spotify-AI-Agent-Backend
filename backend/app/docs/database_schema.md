# Schéma de base de données – Backend IA Spotify

Cette section décrit la structure des bases de données, les modèles, les relations et les scripts associés.

## 1. Modèles principaux
- **User** : gestion des comptes, rôles, authentification
- **SpotifyData** : synchronisation des données Spotify (playlists, écoutes, followers)
- **AIContent** : contenus générés par l’IA (lyrics, musique, recommandations)
- **Collaboration** : gestion des collaborations, matching IA
- **Analytics** : statistiques avancées, logs, monitoring

## 2. Relations clés
- User 1:N SpotifyData
- User 1:N AIContent
- Collaboration N:N User
- Analytics 1:N User

## 3. Scripts de migration & backup
- `scripts/database/migrate.sh` : migration automatique
- `scripts/database/backup.sh` : backup/restauration
- `scripts/database/seed.sh` : initialisation des données

## 4. Bonnes pratiques
- Indexation, contraintes d’intégrité, anonymisation RGPD
- Monitoring des performances (requêtes lentes, locks)
- Sécurité : accès restreint, audit, logs DB

## 5. Exemples de fichiers fournis
- `models/`, `migrations/`, `scripts/database/`

> **Astuce** : Tous les scripts et modèles sont prêts à l’emploi pour industrialisation.
