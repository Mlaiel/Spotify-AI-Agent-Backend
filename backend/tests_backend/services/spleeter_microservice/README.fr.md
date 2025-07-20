# Tests avancés – Spleeter Microservice

Ce dossier contient tous les tests unitaires et d’intégration pour le microservice Spleeter du backend Spotify AI Agent.

## Structure
- `test_config.py` : tests de la configuration et des variables d’environnement
- `test_utils.py` : tests des utilitaires (validation, fichiers temporaires)
- `test_security.py` : tests de la sécurité (API key, JWT ready)
- `test_monitoring.py` : tests des métriques Prometheus
- `test_health.py` : tests de l’endpoint /health
- `__init__.py` : initialisation du module de tests

## Exécution
Lancer tous les tests avec :
```bash
pytest
```

## Bonnes pratiques
- Utiliser des fixtures audio libres de droits
- Ne jamais mocker la logique métier réelle (tests sur code réel)
- Documenter chaque cas de test métier
- Centraliser les tests d’intégration API ici

## Rôles & Auteur
- **Lead Dev & Architecte IA** : Fahed Mlaiel
- **Développeur Backend Senior (Python/FastAPI/Django)** : Fahed Mlaiel
- **Ingénieur Machine Learning (TensorFlow/PyTorch/Hugging Face)** : Fahed Mlaiel
- **DBA & Data Engineer (PostgreSQL/Redis/MongoDB)** : Fahed Mlaiel
- **Spécialiste Sécurité Backend** : Fahed Mlaiel
- **Architecte Microservices** : Fahed Mlaiel

© 2025 Fahed Mlaiel – Tous droits réservés
