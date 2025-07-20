// health_check.js
// Created by: Spotify AI Agent Core Team
// Automated health-check for MongoDB collections and indexes (advanced, auditable)

/*
Usage:
  mongo < health_check.js
- Checks existence, validation, and index health for all critical collections
- Logs results to audit_log
*/

const collections = [
  'users', 'artists', 'tracks', 'playlists', 'analytics',
  'ai_content', 'audit_log', 'security_events', 'consent',
  'geo_events', 'multilingual_content', 'versioning'
];

function logHealthCheck(result) {
  db.audit_log.insertOne({
    entity_type: 'health_check',
    action: 'check',
    details: result,
    performed_by: 'health_check_script',
    created_at: new Date()
  });
}

collections.forEach(function(coll) {
  let exists = db.getCollectionNames().indexOf(coll) !== -1;
  let indexes = exists ? db[coll].getIndexes() : [];
  let stats = exists ? db[coll].stats() : {};
  let result = {
    collection: coll,
    exists: exists,
    indexCount: indexes.length,
    documentCount: stats.count || 0,
    ok: exists && indexes.length > 0 && stats.ok === 1
  };
  printjson(result);
  logHealthCheck(result);
});
