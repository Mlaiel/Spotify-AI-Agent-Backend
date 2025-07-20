# Scripts PostgreSQL – Spotify AI Agent

Ce dossier contient des scripts industriels pour l’automatisation, l’audit, le monitoring, le backup et le seed de données métier.

## Équipe créatrice (rôles)
- Lead Dev & Architecte IA 
- Développeur Backend Senior (Python/FastAPI/Django) 
- Ingénieur Machine Learning (TensorFlow/PyTorch/Hugging Face) 
- DBA & Data Engineer (PostgreSQL/Redis/MongoDB) 
- Spécialiste Sécurité Backend 
- Architecte Microservices  

## Scripts inclus

- **seed_users.py** : Ajoute des utilisateurs de test (profils IA, rôles, emails)
- **seed_analytics.py** : Génère des données analytiques simulées pour tests IA
- **seed_music.py** : Ajoute des morceaux, artistes, genres, durées, popularité
- **audit_log.py** : Analyse les logs PostgreSQL, détecte les accès suspects, génère un rapport d’audit
- **monitoring.py** : Vérifie la santé, la taille, les connexions, les verrous et les requêtes lentes
- **auto_backup.py** : Backup automatique sécurisé avec rotation, logs, prêt pour cron/CI/CD
- **alert_slow_queries.py** : Détecte et alerte sur les requêtes lentes (>5min)

## Utilisation

```bash
python3 seed_users.py
python3 seed_analytics.py
python3 seed_music.py
python3 audit_log.py
python3 monitoring.py
python3 auto_backup.py
python3 alert_slow_queries.py
```

> Tous les scripts sont prêts pour l’automatisation (CI/CD, cron, etc.) et respectent la sécurité (pas de credentials en dur, logs, audit, rotation des backups).
