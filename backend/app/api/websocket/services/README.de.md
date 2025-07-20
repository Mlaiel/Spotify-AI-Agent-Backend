# Erweiterte WebSocket-Services

Dieses Verzeichnis enthält industrielle Services für Skalierbarkeit, Audit, Analytics und KI im WebSocket-Modul:

- **redis_pubsub.py**: Redis Pub/Sub für Broadcast zwischen Instanzen (Skalierbarkeit, Hochverfügbarkeit)
- **postgres_audit.py**: PostgreSQL-Audit für alle sensiblen Aktionen (Verbindung, Nachricht, Fehler)
- **mongodb_events.py**: Speicherung von WebSocket-Events für Analytics, Logs, DSGVO-Konformität
- **ai_moderation.py**: KI-Moderationsservice (Hugging Face, eigene API) zum Filtern von Nachrichten

## Architektur-Empfehlungen
- Redis für Broadcast und Status-Synchronisation zwischen Instanzen nutzen (Cloud-native Skalierung)
- Audit zentral in PostgreSQL für Compliance und Nachvollziehbarkeit speichern
- Analytics-Events in MongoDB für Echtzeit-Analysen ablegen
- KI-Moderation an Hugging Face oder eigene Modelle (TensorFlow/PyTorch) anbinden

## Integrationsbeispiele
```python
from services import RedisPubSub, PostgresAuditService, MongoDBEventsService, AIModerationService

redis = RedisPubSub(redis_url="redis://localhost:6379/0")
audit = PostgresAuditService(dsn="postgresql://user:pass@localhost/db")
mongo = MongoDBEventsService(mongo_url="mongodb://localhost:27017")
ai = AIModerationService(api_url="https://api-inference.huggingface.co/models/xxx")
```

## Sicherheit & Compliance
- Zugriff auf Datenbanken einschränken (Firewall, VPN, starke Credentials)
- Alle kritischen Aktionen loggen und auditieren
- Sensible Daten für DSGVO-Konformität anonymisieren

## Erweiterbarkeit
- Weitere Services (Kafka, BigQuery, etc.) nach Bedarf ergänzen
- Audit an SIEM oder Data Lake anbinden
