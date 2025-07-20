# Microservice Spleeter – Spotify AI Agent

## Description
Microservice avancé pour la séparation de stems audio, conçu pour l'intégration dans des architectures IA industrielles.

## Fonctionnalités
- Séparation de stems (2, 4, 5)
- API REST sécurisée (clé API)
- Monitoring Prometheus
- Logs structurés
- Prêt pour CI/CD et cloud

## Sécurité
- Authentification clé API
- Limitation taille/type fichier
- Logs d'audit


## Déploiement avancé
Voir README.md principal pour Docker Compose, Kubernetes, CI/CD.

## Sécurité avancée
- Authentification par clé API (X-API-KEY)
- Prêt pour extension JWT/OAuth2
- Limitation stricte des extensions et taille de fichier
- Logs d'audit, monitoring Prometheus, alerting
- Placeholder pour scan antivirus (ClamAV, etc)

## Monitoring & Observabilité
- Endpoint `/metrics` Prometheus
- Logs JSON (structlog)
- Endpoint `/health` pour supervision

## FAQ
**Q: Comment changer la clé API ?**
A: Modifier la variable d'environnement `SPLEETER_API_KEY`.

**Q: Comment ajouter un scan antivirus ?**
A: Implémenter la fonction `scan_antivirus` dans `utils.py` (ex: ClamAV).

**Q: Comment déployer en production ?**
A: Utiliser Docker Compose ou Kubernetes, configurer les variables d'environnement, activer le monitoring.

**Q: Comment monitorer les erreurs ?**
A: Consulter les métriques Prometheus et les logs JSON.

**Q: Où sont les tests ?**
A: Les tests d'intégration sont réalisés côté backend principal, pas dans ce microservice.

## Bonnes pratiques
- Ne jamais exposer le service sans authentification
- Toujours limiter la taille et le type de fichiers
- Activer le monitoring et les logs
- Nettoyer les fichiers temporaires
- Sécuriser les accès réseau (firewall, VPC, etc)

## Rôles & Auteur
- **Lead Dev & Architecte IA** : Fahed Mlaiel
- **Développeur Backend Senior (Python/FastAPI/Django)** : Fahed Mlaiel
- **Ingénieur Machine Learning (TensorFlow/PyTorch/Hugging Face)** : Fahed Mlaiel
- **DBA & Data Engineer (PostgreSQL/Redis/MongoDB)** : Fahed Mlaiel
- **Spécialiste Sécurité Backend** : Fahed Mlaiel
- **Architecte Microservices** : Fahed Mlaiel

---
© 2025 Fahed Mlaiel – Tous droits réservés

## Rôles & Auteur
- **Lead Dev & Architecte IA** : Fahed Mlaiel
- **Développeur Backend Senior (Python/FastAPI/Django)** : Fahed Mlaiel
- **Ingénieur Machine Learning (TensorFlow/PyTorch/Hugging Face)** : Fahed Mlaiel
- **DBA & Data Engineer (PostgreSQL/Redis/MongoDB)** : Fahed Mlaiel
- **Spécialiste Sécurité Backend** : Fahed Mlaiel
- **Architecte Microservices** : Fahed Mlaiel

---
© 2025 Fahed Mlaiel – Tous droits réservés
