// 006_add_advanced_indexes.js
// Created by: Spotify AI Agent Core Team
// Lead Dev + Architecte IA, Backend Senior, ML Engineer, DBA/Data Engineer, Sécurité, Microservices

// Add advanced indexes for AI, Audit, Security, Consent, Geo, Multilingual, Versioning

db.ai_content.createIndex({type: 1, status: 1}, {name: "ix_ai_content_type_status"});
db.audit_log.createIndex({entity_type: 1, entity_id: 1}, {name: "ix_audit_entity"});
db.security_events.createIndex({event_type: 1, detected_at: 1}, {name: "ix_security_event_type"});
db.consent.createIndex({user_id: 1, type: 1}, {name: "ix_consent_user_type"});
db.geo_events.createIndex({"location": "2dsphere"}, {name: "ix_geo_location"});
db.multilingual_content.createIndex({lang: 1, title: 1}, {name: "ix_multilingual_lang_title"});
db.versioning.createIndex({entity_type: 1, entity_id: 1, version: -1}, {name: "ix_versioning_entity_version"});
