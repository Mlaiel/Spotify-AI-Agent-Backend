# Manifests Kubernetes pour l'Environnement de Développement
========================================================

## Vue d'ensemble

Ce module contient tous les manifests Kubernetes pour l'environnement de développement du système multi-tenant Spotify AI Agent.

**Développé par :** Fahed Mlaiel et son équipe d'experts

### Composition de l'équipe :
- **Lead Dev + Architecte IA :** Fahed Mlaiel - Architecture système et intégration IA
- **Développeur Backend Senior :** Développement Python/FastAPI/Django
- **Ingénieur Machine Learning :** Implémentation TensorFlow/PyTorch/Hugging Face
- **DBA & Data Engineer :** Optimisation PostgreSQL/Redis/MongoDB
- **Spécialiste Sécurité Backend :** Politiques de sécurité et conformité
- **Architecte Microservices :** Conception de services et orchestration

## Architecture

### Composants :
- **Deployments :** Déploiements d'applications
- **Services :** Services Kubernetes
- **ConfigMaps :** Gestion de configuration
- **Secrets :** Données sensibles
- **PersistentVolumes :** Solutions de stockage
- **NetworkPolicies :** Sécurité réseau
- **RBAC :** Contrôle d'accès basé sur les rôles
- **HPA :** Auto-scaling horizontal des pods
- **Ingress :** Exposition externe

### Structure des répertoires :
```
manifests/
├── deployments/          # Déploiements d'applications
├── services/            # Services Kubernetes
├── configs/            # ConfigMaps et configurations
├── secrets/            # Configurations secrètes
├── storage/            # Volumes persistants
├── networking/         # Politiques réseau et ingress
├── security/           # RBAC et politiques de sécurité
├── monitoring/         # Métriques et observabilité
├── autoscaling/        # Configurations d'auto-scaling
└── jobs/              # Jobs et CronJobs
```

## Utilisation

### Déploiement :
```bash
# Appliquer tous les manifests
kubectl apply -f manifests/

# Déployer un module spécifique
kubectl apply -f manifests/deployments/
```

### Surveillance :
```bash
# Vérifier le statut des pods
kubectl get pods -n spotify-ai-agent-dev

# Afficher les logs
kubectl logs -f deployment/spotify-ai-agent -n spotify-ai-agent-dev
```

## Configuration

### Variables d'environnement :
- `NAMESPACE` : Namespace Kubernetes (défaut : spotify-ai-agent-dev)
- `REPLICAS` : Nombre de répliques de pods
- `RESOURCES_LIMITS` : Limites de ressources pour les pods

### Labels :
Tous les manifests utilisent des labels standardisés pour une identification et gestion cohérentes.

## Sécurité

- Tous les manifests implémentent les meilleures pratiques de sécurité Kubernetes
- Politiques RBAC pour des permissions minimales
- NetworkPolicies pour l'isolation réseau
- Conformité aux Pod Security Standards

## Mise à l'échelle

Le système supporte l'auto-scaling horizontal basé sur :
- Utilisation CPU
- Consommation mémoire
- Métriques personnalisées

## Surveillance et journalisation

Intégration avec :
- Prometheus pour les métriques
- Grafana pour les tableaux de bord
- Stack ELK pour la journalisation
- Jaeger pour le tracing distribué

---

**Développé avec ❤️ par l'équipe d'experts de Fahed Mlaiel**
