// 001_create_collections.js
// Created by: Spotify AI Agent Core Team
// Lead Dev + Architecte IA, Backend Senior, ML Engineer, DBA/Data Engineer, Sécurité, Microservices

// Create user, artist, track, playlist, analytics collections with validation

db.createCollection("users", {
  validator: {
    $jsonSchema: {
      bsonType: "object",
      required: ["email", "role", "status", "created_at"],
      properties: {
        email: {bsonType: "string"},
        role: {bsonType: "string"},
        status: {bsonType: "string"},
        preferences: {bsonType: "object"},
        created_at: {bsonType: "date"},
        updated_at: {bsonType: "date"}
      }
    }
  }
});

db.createCollection("artists", {
  validator: {
    $jsonSchema: {
      bsonType: "object",
      required: ["user_id", "display_name", "created_at"],
      properties: {
        user_id: {bsonType: "objectId"},
        display_name: {bsonType: "string"},
        genres: {bsonType: "array"},
        bio: {bsonType: "string"},
        created_at: {bsonType: "date"},
        updated_at: {bsonType: "date"}
      }
    }
  }
});

db.createCollection("tracks", {
  validator: {
    $jsonSchema: {
      bsonType: "object",
      required: ["title", "artist_id", "created_at"],
      properties: {
        title: {bsonType: "string"},
        artist_id: {bsonType: "objectId"},
        album_id: {bsonType: "objectId"},
        duration_ms: {bsonType: "int"},
        audio_features: {bsonType: "object"},
        created_at: {bsonType: "date"},
        updated_at: {bsonType: "date"}
      }
    }
  }
});

db.createCollection("playlists", {
  validator: {
    $jsonSchema: {
      bsonType: "object",
      required: ["name", "owner_id", "created_at"],
      properties: {
        name: {bsonType: "string"},
        owner_id: {bsonType: "objectId"},
        track_ids: {bsonType: "array"},
        collaborative: {bsonType: "bool"},
        created_at: {bsonType: "date"},
        updated_at: {bsonType: "date"}
      }
    }
  }
});

db.createCollection("analytics", {
  validator: {
    $jsonSchema: {
      bsonType: "object",
      required: ["user_id", "event_type", "timestamp"],
      properties: {
        user_id: {bsonType: "objectId"},
        event_type: {bsonType: "string"},
        payload: {bsonType: "object"},
        timestamp: {bsonType: "date"}
      }
    }
  }
});
