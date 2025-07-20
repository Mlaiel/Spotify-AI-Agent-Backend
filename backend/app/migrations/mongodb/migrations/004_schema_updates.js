// 004_schema_updates.js
// Created by: Spotify AI Agent Core Team
// Lead Dev + Architecte IA, Backend Senior, ML Engineer, DBA/Data Engineer, Sécurité, Microservices

// Example: Add validation for new field 'subscription_type' in users

db.runCommand({
  collMod: "users",
  validator: {
    $jsonSchema: {
      bsonType: "object",
      required: ["email", "role", "status", "created_at", "subscription_type"],
      properties: {
        email: {bsonType: "string"},
        role: {bsonType: "string"},
        status: {bsonType: "string"},
        preferences: {bsonType: "object"},
        subscription_type: {bsonType: "string", enum: ["FREE", "PREMIUM", "FAMILY", "STUDENT", "TRIAL"]},
        created_at: {bsonType: "date"},
        updated_at: {bsonType: "date"}
      }
    }
  },
  validationLevel: "moderate"
});

// Example: Add new field to all existing users
db.users.updateMany(
  {subscription_type: {$exists: false}},
  {$set: {subscription_type: "FREE"}}
);
