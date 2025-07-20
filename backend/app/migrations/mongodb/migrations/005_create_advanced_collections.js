// 005_create_advanced_collections.js
// Created by: Spotify AI Agent Core Team
// Lead Dev + Architecte IA, Backend Senior, ML Engineer, DBA/Data Engineer, Sécurité, Microservices

// Create advanced collections: ai_content, audit_log, security_events, consent, geo_events, multilingual_content, versioning

db.createCollection("ai_content", {
  validator: {
    $jsonSchema: {
      bsonType: "object",
      required: ["type", "payload", "created_at"],
      properties: {
        type: {bsonType: "string"},
        payload: {bsonType: "object"},
        status: {bsonType: "string"},
        created_at: {bsonType: "date"},
        updated_at: {bsonType: "date"}
      }
    }
  }
});

db.createCollection("audit_log", {
  validator: {
    $jsonSchema: {
      bsonType: "object",
      required: ["entity_type", "entity_id", "action", "created_at"],
      properties: {
        entity_type: {bsonType: "string"},
        entity_id: {bsonType: "objectId"},
        action: {bsonType: "string"},
        details: {bsonType: "object"},
        performed_by: {bsonType: "objectId"},
        created_at: {bsonType: "date"}
      }
    }
  }
});

db.createCollection("security_events", {
  validator: {
    $jsonSchema: {
      bsonType: "object",
      required: ["event_type", "detected_at"],
      properties: {
        event_type: {bsonType: "string"},
        details: {bsonType: "object"},
        detected_at: {bsonType: "date"},
        resolved: {bsonType: "bool"}
      }
    }
  }
});

db.createCollection("consent", {
  validator: {
    $jsonSchema: {
      bsonType: "object",
      required: ["user_id", "type", "granted", "timestamp"],
      properties: {
        user_id: {bsonType: "objectId"},
        type: {bsonType: "string"},
        granted: {bsonType: "bool"},
        timestamp: {bsonType: "date"}
      }
    }
  }
});

db.createCollection("geo_events", {
  validator: {
    $jsonSchema: {
      bsonType: "object",
      required: ["location", "event_type", "created_at"],
      properties: {
        location: {bsonType: "object"}, // GeoJSON Point
        event_type: {bsonType: "string"},
        details: {bsonType: "object"},
        created_at: {bsonType: "date"}
      }
    }
  }
});

db.createCollection("multilingual_content", {
  validator: {
    $jsonSchema: {
      bsonType: "object",
      required: ["title", "lang", "created_at"],
      properties: {
        title: {bsonType: "string"},
        lang: {bsonType: "string"},
        content: {bsonType: "string"},
        tags: {bsonType: "array"},
        created_at: {bsonType: "date"},
        updated_at: {bsonType: "date"}
      }
    }
  }
});

db.createCollection("versioning", {
  validator: {
    $jsonSchema: {
      bsonType: "object",
      required: ["entity_type", "entity_id", "version", "created_at"],
      properties: {
        entity_type: {bsonType: "string"},
        entity_id: {bsonType: "objectId"},
        version: {bsonType: "int"},
        description: {bsonType: "string"},
        checksum: {bsonType: "string"},
        created_at: {bsonType: "date"}
      }
    }
  }
});
