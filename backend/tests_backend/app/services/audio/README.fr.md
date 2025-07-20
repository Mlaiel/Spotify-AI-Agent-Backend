# Tests avancés – Module audio

Ce dossier contient tous les tests unitaires et d’intégration pour le module `audio` du backend Spotify AI Agent.

## Structure
- `test_audio_utils.py` : tests de la normalisation, conversion, extraction de features audio
- `test_audio_analyzer.py` : tests d’analyse ML, classification, tagging
- `test_spleeter_client.py` : tests du client HTTP pour le microservice Spleeter
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
