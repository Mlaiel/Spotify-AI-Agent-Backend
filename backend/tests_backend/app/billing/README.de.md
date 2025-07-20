# Abrechnungssystem Tests

## Überblick

Dieses Verzeichnis enthält umfassende Tests für das erweiterte Abrechnungssystem und deckt alle Aspekte des Abonnement-Managements, der Zahlungsverarbeitung, Rechnungsstellung, Analytik und Hintergrundaufgaben ab.

## Test-Struktur

```
tests_backend/app/billing/
├── __init__.py              # Test-Paket-Initialisierung
├── conftest.py              # Test-Fixtures und Konfiguration
├── test_models.py           # Datenbank-Modell-Tests
├── test_core.py             # Kern-Abrechnungsmotor-Tests
├── test_api.py              # FastAPI-Endpunkt-Tests
├── test_invoices.py         # Rechnungsmanagement-Tests
├── test_webhooks.py         # Webhook-Verarbeitungs-Tests
├── test_analytics.py        # Analytik- und Berichts-Tests
└── test_tasks.py            # Hintergrundaufgaben-Tests
```

## Test-Kategorien

### 🗄️ Modell-Tests (`test_models.py`)
- **Kunden-Modell**: Erstellung, Validierung, Beziehungen
- **Plan-Modell**: Preisgestaltung, Intervalle, Funktionen, Nutzungslimits
- **Abonnement-Modell**: Lebenszyklus, Statusänderungen, Berechnungen
- **Zahlung-Modell**: Verarbeitung, Fehlschläge, Rückerstattungen, Risikobewertung
- **Rechnung-Modell**: Generierung, Zahlungsverfolgung, Überfälligkeitslogik
- **Zahlungsmethode-Modell**: Kartenvalidierung, Ablauf, Sicherheit
- **Beziehungen**: Fremdschlüssel, Kaskadierung, Datenintegrität

### ⚙️ Kern-Motor-Tests (`test_core.py`)
- **AbrechnungsMotor**: Kunden-/Abonnement-Management, Lebenszyklus
- **ZahlungsProzessor**: Multi-Anbieter-Unterstützung (Stripe, PayPal)
- **SteuerRechner**: EU-MwSt., US-Verkaufssteuer, Reverse Charge
- **BetrugErkennung**: Risikobewertung, ML-Vorhersagen, Geschwindigkeitsprüfungen
- **Integration**: End-to-End-Abrechnungsworkflows

### 🌐 API-Tests (`test_api.py`)
- **Kunden-Endpunkte**: CRUD-Operationen, Validierung
- **Plan-Endpunkte**: Erstellung, Updates, Deaktivierung
- **Abonnement-Endpunkte**: Lebenszyklus-Management, Upgrades
- **Zahlungs-Endpunkte**: Verarbeitung, Rückerstattungen, Methoden
- **Rechnungs-Endpunkte**: Generierung, Zahlung, PDF-Download
- **Webhook-Endpunkte**: Stripe/PayPal-Ereignisbehandlung
- **Analytik-Endpunkte**: Umsatzberichte, Metriken
- **Fehlerbehandlung**: Validierung, Autorisierung, Rate Limiting

### 📄 Rechnungs-Tests (`test_invoices.py`)
- **RechnungsService**: Generierung, Finalisierung, Zahlungsverfolgung
- **PDF-Generator**: Mehrsprachige PDFs, Vorlagen, Anhänge
- **E-Mail-Service**: Rechnungsversand, Erinnerungen, Bestätigungen
- **Mahnwesen-Management**: Automatisierte Inkasso-Workflows
- **Integration**: Komplette Rechnung-zu-Zahlung-Workflows

### 🔗 Webhook-Tests (`test_webhooks.py`)
- **Webhook-Prozessor**: Ereignis-Routing, Retry-Logik, Deduplizierung
- **Stripe-Webhook-Handler**: Payment Intents, Abonnements, Setup Intents
- **PayPal-Webhook-Handler**: Zahlungen, Abonnements, Benachrichtigungen
- **Sicherheit**: Signaturverifikation, IP-Whitelisting, Rate Limiting
- **Überwachung**: Logging, Metriken, Fehler-Tracking

### 📊 Analytik-Tests (`test_analytics.py`)
- **Analytik-Service**: Umsatz-, Abonnement-, Kunden-Metriken
- **Berichts-Generator**: Monatsberichte, Segmentierung, Exporte
- **Prognose-Engine**: Umsatzvorhersage, Churn-Analyse, LTV
- **Performance**: Caching, Query-Optimierung, Echtzeitdaten
- **Integration**: Dashboard-Daten, geplante Berichte

### 🔄 Aufgaben-Tests (`test_tasks.py`)
- **Abrechnungsaufgaben-Manager**: Aufgabenplanung, Überwachung, Stornierung
- **Abonnement-Verlängerungen**: Automatisierte Abrechnungszyklen, Mahnung
- **Zahlungs-Wiederholungen**: Intelligente Retry-Logik, Eskalation
- **Rechnungs-Generierung**: Batch-Verarbeitung, Fehlerbehandlung
- **Wartung**: Datenbereinigung, externe Synchronisation, Webhook-Verarbeitung

## Test-Konfiguration

### Datenbank-Setup
```python
# Isolierte Test-Datenbank mit automatischem Rollback
@pytest.fixture
async def db_session():
    # Erstellt frische Datenbank-Session für jeden Test
    # Automatisches Rollback gewährleistet Test-Isolation
```

### Mock-Services
```python
# Mock externer Services für zuverlässige Tests
@pytest.fixture
def mock_stripe():
    # Mock von Stripe-API-Aufrufen
    
@pytest.fixture  
def mock_paypal():
    # Mock von PayPal-API-Aufrufen

@pytest.fixture
def mock_email_service():
    # Mock von E-Mail-Versand
```

### Test-Daten
```python
# Umfassende Test-Fixtures
@pytest.fixture
def test_kunde():
    # Beispiel-Kunde mit vollem Profil

@pytest.fixture
def test_abonnement_aktiv():
    # Aktives Abonnement mit Zahlungsmethode

@pytest.fixture
def test_rechnung_bezahlt():
    # Abgeschlossene Rechnung mit Zahlung
```

## Tests Ausführen

### Alle Tests
```bash
# Komplette Abrechnungs-Test-Suite ausführen
pytest tests_backend/app/billing/ -v

# Mit Coverage-Bericht ausführen
pytest tests_backend/app/billing/ --cov=billing --cov-report=html
```

### Spezifische Test-Kategorien
```bash
# Nur Modell-Tests
pytest tests_backend/app/billing/test_models.py -v

# API-Endpunkt-Tests
pytest tests_backend/app/billing/test_api.py -v

# Kern-Motor-Tests
pytest tests_backend/app/billing/test_core.py -v

# Hintergrundaufgaben-Tests
pytest tests_backend/app/billing/test_tasks.py -v
```

### Test-Muster
```bash
# Tests mit Muster ausführen
pytest tests_backend/app/billing/ -k "abonnement" -v

# Nur fehlgeschlagene Tests ausführen
pytest tests_backend/app/billing/ --lf

# Spezifische Test-Klasse ausführen
pytest tests_backend/app/billing/test_models.py::TestKundenModell -v
```

## Test-Daten-Management

### Kunden-Test-Daten
```python
test_kunden = [
    {
        "email": "test@beispiel.de",
        "name": "Test Kunde",
        "land": "DE",
        "bevorzugte_waehrung": "EUR"
    }
]
```

### Plan-Test-Daten
```python
test_plaene = [
    {
        "name": "Basis Plan",
        "betrag": Decimal("29.99"),
        "intervall": "MONAT",
        "funktionen": ["api_zugang", "basis_support"]
    }
]
```

### Zahlungs-Test-Daten
```python
test_zahlungen = [
    {
        "betrag": Decimal("99.99"),
        "waehrung": "EUR",
        "status": "ERFOLGREICH",
        "anbieter": "STRIPE"
    }
]
```

## Performance-Tests

### Last-Tests
```bash
# Test mit mehreren gleichzeitigen Benutzern
pytest tests_backend/app/billing/test_api.py -v --numprocesses=4

# Speicher-Profiling
pytest tests_backend/app/billing/ --profile

# Benchmark spezifischer Operationen
pytest tests_backend/app/billing/test_core.py::test_zahlungsverarbeitung --benchmark-only
```

### Datenbank-Performance
```python
# Test Query-Optimierung
def test_abonnement_query_performance():
    # N+1-Query-Prävention verifizieren
    # Index-Verwendung prüfen
    # Paginierungs-Performance validieren
```

## Sicherheits-Tests

### Authentifizierungs-Tests
```python
def test_api_authentifizierung():
    # JWT-Token-Validierung verifizieren
    # Rollenbasierte Zugriffskontrolle testen
    # API-Schlüssel-Sicherheit prüfen
```

### Datenschutz-Tests
```python
def test_datenverschluesselung():
    # PII-Verschlüsselung verifizieren
    # Zahlungsdaten-Sicherheit testen
    # Audit-Protokollierung prüfen
```

## Integrations-Tests

### Zahlungsanbieter-Integration
```python
def test_stripe_integration():
    # End-to-End-Zahlungsfluss
    # Webhook-Verarbeitung
    # Fehlerbehandlung
    
def test_paypal_integration():
    # Kompletter PayPal-Workflow
    # Abonnement-Management
    # Streitfall-Behandlung
```

### Externe Service-Integration
```python
def test_email_integration():
    # Vorlage-Rendering
    # Zustellungs-Tracking
    # Bounce-Behandlung

def test_pdf_generierung():
    # Mehrsprachige Unterstützung
    # Vorlage-Anpassung
    # Performance-Optimierung
```

## Kontinuierliche Integration

### GitHub Actions
```yaml
# .github/workflows/billing-tests.yml
name: Abrechnungssystem Tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Abrechnungs-Tests ausführen
        run: pytest tests_backend/app/billing/ --cov=billing
```

### Test-Coverage-Anforderungen
- **Minimale Abdeckung**: 95%
- **Kritische Pfade**: 100% (Zahlungsverarbeitung, Sicherheit)
- **Edge Cases**: Umfassende Fehler-Szenario-Abdeckung
- **Performance**: Last-Tests für Schlüssel-Operationen

## Best Practices

### Test-Erstellung
1. **Beschreibende Namen**: Klare Test-Methoden-Namen
2. **Einzelne Verantwortung**: Eine Assertion pro Test
3. **Test-Unabhängigkeit**: Keine Test-Abhängigkeiten
4. **Daten-Isolation**: Frische Daten für jeden Test
5. **Mock External**: Alle externen Service-Aufrufe mocken

### Performance
1. **Datenbank-Optimierung**: Transaktionen für Rollback verwenden
2. **Parallele Ausführung**: Unabhängige Test-Ausführung
3. **Ressourcen-Bereinigung**: Ordnungsgemäße Fixture-Bereinigung
4. **Caching**: Intelligente Nutzung von Test-Daten-Caching

### Wartung
1. **Regelmäßige Updates**: Tests aktuell mit Code-Änderungen halten
2. **Refactoring**: Test-Code-Duplikation eliminieren
3. **Dokumentation**: Klare Test-Dokumentation
4. **Überwachung**: Test-Ausführungszeit und Flakiness verfolgen

## Fehlerbehebung

### Häufige Probleme
```bash
# Datenbank-Verbindungsprobleme
export DATABASE_URL="postgresql://test:test@localhost/billing_test"

# Redis-Verbindung für Caching-Tests
export REDIS_URL="redis://localhost:6379/1"

# Test-Umgebungsvariablen
export STRIPE_TEST_SECRET_KEY="sk_test_..."
export PAYPAL_TEST_CLIENT_ID="test_client_id"
```

### Debug-Modus
```bash
# Tests mit detaillierter Ausgabe ausführen
pytest tests_backend/app/billing/ -v -s --tb=long

# Spezifischen Test debuggen
pytest tests_backend/app/billing/test_core.py::test_zahlungsverarbeitung -v -s --pdb
```

### Test-Daten-Reset
```bash
# Test-Datenbank zurücksetzen
python -m billing.scripts.reset_test_db

# Test-Fixtures regenerieren
python -m billing.scripts.generate_test_data
```

## Beitragen

### Neue Tests Hinzufügen
1. Bestehende Test-Struktur und Namenskonventionen befolgen
2. Angemessene Fixtures in `conftest.py` hinzufügen
3. Positive und negative Test-Fälle einschließen
4. Dokumentation für neue Test-Kategorien aktualisieren
5. Minimale 95% Abdeckung für neuen Code sicherstellen

### Test-Review-Checkliste
- [ ] Tests sind unabhängig und isoliert
- [ ] Externe Services sind ordnungsgemäß gemockt
- [ ] Fehler-Szenarien sind abgedeckt
- [ ] Performance-Auswirkungen berücksichtigt
- [ ] Dokumentation aktualisiert
- [ ] CI/CD-Pipeline läuft durch

---

**Hinweis**: Diese Test-Suite gewährleistet die Zuverlässigkeit, Sicherheit und Performance des Enterprise-Abrechnungssystems. Alle Tests müssen vor der Bereitstellung in der Produktion bestehen.
