# Tests du Système de Facturation

## Vue d'ensemble

Ce répertoire contient des tests complets pour le système de facturation avancé, couvrant tous les aspects de la gestion des abonnements, du traitement des paiements, de la facturation, de l'analytique et des tâches en arrière-plan.

## Structure des Tests

```
tests_backend/app/billing/
├── __init__.py              # Initialisation du package de tests
├── conftest.py              # Fixtures et configuration des tests
├── test_models.py           # Tests des modèles de base de données
├── test_core.py             # Tests du moteur de facturation principal
├── test_api.py              # Tests des endpoints FastAPI
├── test_invoices.py         # Tests de gestion des factures
├── test_webhooks.py         # Tests de traitement des webhooks
├── test_analytics.py        # Tests d'analytique et de rapports
└── test_tasks.py            # Tests des tâches en arrière-plan
```

## Catégories de Tests

### 🗄️ Tests de Modèles (`test_models.py`)
- **Modèle Client**: Création, validation, relations
- **Modèle Plan**: Tarification, intervalles, fonctionnalités, limites d'usage
- **Modèle Abonnement**: Cycle de vie, changements de statut, calculs
- **Modèle Paiement**: Traitement, échecs, remboursements, scoring de risque
- **Modèle Facture**: Génération, suivi des paiements, logique d'impayés
- **Modèle MéthoodePaiement**: Validation des cartes, expiration, sécurité
- **Relations**: Clés étrangères, cascades, intégrité des données

### ⚙️ Tests du Moteur Principal (`test_core.py`)
- **MoteurFacturation**: Gestion clients/abonnements, cycle de vie
- **ProcesseurPaiement**: Support multi-fournisseurs (Stripe, PayPal)
- **CalculateurTaxe**: TVA européenne, taxe de vente US, autoliquidation
- **DétectionFraude**: Scoring de risque, prédictions ML, vérifications de vélocité
- **Intégration**: Workflows de facturation de bout en bout

### 🌐 Tests API (`test_api.py`)
- **Endpoints Client**: Opérations CRUD, validation
- **Endpoints Plan**: Création, mises à jour, désactivation
- **Endpoints Abonnement**: Gestion du cycle de vie, mises à niveau
- **Endpoints Paiement**: Traitement, remboursements, méthodes
- **Endpoints Facture**: Génération, paiement, téléchargement PDF
- **Endpoints Webhook**: Gestion des événements Stripe/PayPal
- **Endpoints Analytique**: Rapports de revenus, métriques
- **Gestion d'Erreurs**: Validation, autorisation, limitation de taux

### 📄 Tests Factures (`test_invoices.py`)
- **ServiceFacture**: Génération, finalisation, suivi des paiements
- **GénérateurPDF**: PDFs multilingues, modèles, pièces jointes
- **ServiceEmail**: Livraison de factures, rappels, confirmations
- **GestionRelance**: Workflows de recouvrement automatisé
- **Intégration**: Workflows complets facture vers paiement

### 🔗 Tests Webhooks (`test_webhooks.py`)
- **ProcesseurWebhook**: Routage d'événements, logique de retry, déduplication
- **GestionnaireWebhookStripe**: Intents de paiement, abonnements, intents de configuration
- **GestionnaireWebhookPayPal**: Paiements, abonnements, notifications
- **Sécurité**: Vérification de signature, liste blanche IP, limitation de taux
- **Surveillance**: Logging, métriques, suivi d'erreurs

### 📊 Tests Analytique (`test_analytics.py`)
- **ServiceAnalytique**: Métriques de revenus, abonnements, clients
- **GénérateurRapport**: Rapports mensuels, segmentation, exports
- **MoteurPrévision**: Prédiction de revenus, analyse du churn, LTV
- **Performance**: Mise en cache, optimisation des requêtes, données temps réel
- **Intégration**: Données tableau de bord, rapports programmés

### 🔄 Tests Tâches (`test_tasks.py`)
- **GestionnaireTâchesFacturation**: Planification, surveillance, annulation
- **Renouvellements Abonnement**: Cycles de facturation automatisés, relance
- **Tentatives Paiement**: Logique de retry intelligente, escalade
- **Génération Factures**: Traitement par lots, gestion d'erreurs
- **Maintenance**: Nettoyage de données, synchronisation externe, traitement webhook

## Configuration des Tests

### Configuration Base de Données
```python
# Base de données de test isolée avec rollback automatique
@pytest.fixture
async def db_session():
    # Crée une session de base de données fraîche pour chaque test
    # Le rollback automatique assure l'isolation des tests
```

### Services Mockés
```python
# Mocking des services externes pour des tests fiables
@pytest.fixture
def mock_stripe():
    # Mock des appels API Stripe
    
@pytest.fixture  
def mock_paypal():
    # Mock des appels API PayPal

@pytest.fixture
def mock_service_email():
    # Mock de l'envoi d'emails
```

### Données de Test
```python
# Fixtures de test complètes
@pytest.fixture
def test_client():
    # Client exemple avec profil complet

@pytest.fixture
def test_abonnement_actif():
    # Abonnement actif avec méthode de paiement

@pytest.fixture
def test_facture_payee():
    # Facture terminée avec paiement
```

## Exécution des Tests

### Tous les Tests
```bash
# Exécuter la suite complète de tests de facturation
pytest tests_backend/app/billing/ -v

# Exécuter avec rapport de couverture
pytest tests_backend/app/billing/ --cov=billing --cov-report=html
```

### Catégories Spécifiques de Tests
```bash
# Tests de modèles uniquement
pytest tests_backend/app/billing/test_models.py -v

# Tests d'endpoints API
pytest tests_backend/app/billing/test_api.py -v

# Tests du moteur principal
pytest tests_backend/app/billing/test_core.py -v

# Tests de tâches en arrière-plan
pytest tests_backend/app/billing/test_tasks.py -v
```

### Motifs de Tests
```bash
# Exécuter les tests correspondant au motif
pytest tests_backend/app/billing/ -k "abonnement" -v

# Exécuter uniquement les tests échoués
pytest tests_backend/app/billing/ --lf

# Exécuter une classe de test spécifique
pytest tests_backend/app/billing/test_models.py::TestModeleClient -v
```

## Gestion des Données de Test

### Données de Test Client
```python
test_clients = [
    {
        "email": "test@exemple.com",
        "nom": "Client Test",
        "pays": "FR",
        "devise_preferee": "EUR"
    }
]
```

### Données de Test Plan
```python
test_plans = [
    {
        "nom": "Plan Basique",
        "montant": Decimal("29.99"),
        "intervalle": "MOIS",
        "fonctionnalites": ["acces_api", "support_basique"]
    }
]
```

### Données de Test Paiement
```python
test_paiements = [
    {
        "montant": Decimal("99.99"),
        "devise": "EUR",
        "statut": "REUSSI",
        "fournisseur": "STRIPE"
    }
]
```

## Tests de Performance

### Tests de Charge
```bash
# Test avec plusieurs utilisateurs simultanés
pytest tests_backend/app/billing/test_api.py -v --numprocesses=4

# Profilage mémoire
pytest tests_backend/app/billing/ --profile

# Benchmark d'opérations spécifiques
pytest tests_backend/app/billing/test_core.py::test_traitement_paiement --benchmark-only
```

### Performance Base de Données
```python
# Test d'optimisation des requêtes
def test_performance_requete_abonnement():
    # Vérifier la prévention des requêtes N+1
    # Vérifier l'utilisation d'index
    # Valider la performance de pagination
```

## Tests de Sécurité

### Tests d'Authentification
```python
def test_authentification_api():
    # Vérifier la validation des tokens JWT
    # Tester le contrôle d'accès basé sur les rôles
    # Vérifier la sécurité des clés API
```

### Tests de Protection des Données
```python
def test_chiffrement_donnees():
    # Vérifier le chiffrement des DPI
    # Tester la sécurité des données de paiement
    # Vérifier la journalisation d'audit
```

## Tests d'Intégration

### Intégration Fournisseurs de Paiement
```python
def test_integration_stripe():
    # Flux de paiement de bout en bout
    # Traitement des webhooks
    # Gestion d'erreurs
    
def test_integration_paypal():
    # Workflow PayPal complet
    # Gestion des abonnements
    # Gestion des litiges
```

### Intégration Services Externes
```python
def test_integration_email():
    # Rendu de modèles
    # Suivi de livraison
    # Gestion des rebonds

def test_generation_pdf():
    # Support multilingue
    # Personnalisation de modèles
    # Optimisation de performance
```

## Intégration Continue

### GitHub Actions
```yaml
# .github/workflows/billing-tests.yml
name: Tests Système Facturation
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Exécuter tests facturation
        run: pytest tests_backend/app/billing/ --cov=billing
```

### Exigences de Couverture de Tests
- **Couverture Minimale**: 95%
- **Chemins Critiques**: 100% (traitement paiements, sécurité)
- **Cas Limites**: Couverture complète des scénarios d'erreur
- **Performance**: Tests de charge pour opérations clés

## Meilleures Pratiques

### Écriture de Tests
1. **Noms Descriptifs**: Noms de méthodes de test clairs
2. **Responsabilité Unique**: Une assertion par test
3. **Indépendance des Tests**: Pas de dépendances entre tests
4. **Isolation des Données**: Données fraîches pour chaque test
5. **Mock Externe**: Mocker tous les appels de services externes

### Performance
1. **Optimisation Base de Données**: Utiliser les transactions pour rollback
2. **Exécution Parallèle**: Exécution de tests indépendante
3. **Nettoyage Ressources**: Nettoyage approprié des fixtures
4. **Mise en Cache**: Utilisation intelligente de la mise en cache des données de test

### Maintenance
1. **Mises à Jour Régulières**: Garder les tests à jour avec les changements de code
2. **Refactoring**: Éliminer la duplication du code de test
3. **Documentation**: Documentation claire des tests
4. **Surveillance**: Suivre le temps d'exécution et l'instabilité des tests

## Dépannage

### Problèmes Courants
```bash
# Problèmes de connexion base de données
export DATABASE_URL="postgresql://test:test@localhost/billing_test"

# Connexion Redis pour tests de cache
export REDIS_URL="redis://localhost:6379/1"

# Variables d'environnement de test
export STRIPE_TEST_SECRET_KEY="sk_test_..."
export PAYPAL_TEST_CLIENT_ID="test_client_id"
```

### Mode Debug
```bash
# Exécuter tests avec sortie détaillée
pytest tests_backend/app/billing/ -v -s --tb=long

# Déboguer un test spécifique
pytest tests_backend/app/billing/test_core.py::test_traitement_paiement -v -s --pdb
```

### Réinitialisation Données de Test
```bash
# Réinitialiser base de données de test
python -m billing.scripts.reset_test_db

# Régénérer fixtures de test
python -m billing.scripts.generate_test_data
```

## Contribution

### Ajout de Nouveaux Tests
1. Suivre la structure et les conventions de nommage existantes
2. Ajouter les fixtures appropriées dans `conftest.py`
3. Inclure les cas de test positifs et négatifs
4. Mettre à jour la documentation pour les nouvelles catégories de tests
5. Assurer une couverture minimale de 95% pour le nouveau code

### Liste de Vérification Révision Tests
- [ ] Les tests sont indépendants et isolés
- [ ] Les services externes sont correctement mockés
- [ ] Les scénarios d'erreur sont couverts
- [ ] Les implications de performance sont considérées
- [ ] La documentation est mise à jour
- [ ] Le pipeline CI/CD passe

---

**Note**: Cette suite de tests assure la fiabilité, la sécurité et la performance du système de facturation d'entreprise. Tous les tests doivent passer avant le déploiement en production.
