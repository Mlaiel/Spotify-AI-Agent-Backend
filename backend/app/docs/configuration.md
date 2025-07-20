# Configuration – Backend IA Spotify

Cette section centralise la configuration des environnements, secrets, variables et bonnes pratiques DevOps.

## 1. Environnements supportés
- Développement, staging, production
- Fichiers `.env`, `config/environments/`, variables d’environnement Docker/K8s

## 2. Secrets & sécurité
- Gestion via Vault, Docker secrets, K8s secrets
- Jamais de secrets en dur dans le code
- Rotation régulière, audit des accès

## 3. Configuration des services
- API : ports, CORS, logging, rate limiting
- ML : modèles, ressources GPU, batch size
- DB : credentials, pool, timeout, backup
- Cache : Redis/Memcached, TTL, eviction

## 4. Bonnes pratiques
- Versionner les templates, jamais les secrets réels
- Scripts d’initialisation (`scripts/deployment/init_config.sh`)
- Validation automatique à chaque déploiement

## 5. Exemples de fichiers fournis
- `.env.example`, `config/environments/`, `config/logging/`, `config/security/`

> **Astuce** : Tous les templates et scripts sont prêts à l’emploi dans `config/` et `scripts/`.
