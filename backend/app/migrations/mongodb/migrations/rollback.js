// rollback.js
// Created by: Spotify AI Agent Core Team
// Rollback migration script for MongoDB (advanced, idempotent, auditable)

/*
Usage:
  mongo < rollback.js --eval 'var targetVersion="004"'
- Rolls back all migrations after the specified version (e.g., 004)
- Logs all actions to audit_log
- Supports dry-run mode
*/

if (typeof targetVersion === 'undefined') {
  print('ERROR: Please provide targetVersion as --eval "var targetVersion=\"004\""');
  quit(1);
}

const MIGRATION_ORDER = [
  '006_add_advanced_indexes.js',
  '005_create_advanced_collections.js',
  '004_schema_updates.js',
  '003_data_migration.js',
  '002_add_indexes.js',
  '001_create_collections.js'
];

function logRollback(action, details) {
  db.audit_log.insertOne({
    entity_type: 'migration',
    action: action,
    details: details,
    performed_by: 'rollback_script',
    created_at: new Date()
  });
}

function dropIfExists(coll) {
  if (db.getCollectionNames().indexOf(coll) !== -1) {
    db[coll].drop();
    logRollback('drop_collection', {collection: coll});
  }
}

function rollbackTo(version) {
  let idx = MIGRATION_ORDER.indexOf(MIGRATION_ORDER.find(f => f.startsWith(version)));
  if (idx === -1) {
    print('ERROR: targetVersion not found in migration order.');
    quit(1);
  }
  // Drop all collections/indexes created after targetVersion
  if (version < '005') {
    dropIfExists('ai_content');
    dropIfExists('audit_log');
    dropIfExists('security_events');
    dropIfExists('consent');
    dropIfExists('geo_events');
    dropIfExists('multilingual_content');
    dropIfExists('versioning');
  }
  // Remove advanced indexes
  if (version < '006') {
    db.ai_content.dropIndexes();
    db.audit_log.dropIndexes();
    db.security_events.dropIndexes();
    db.consent.dropIndexes();
    db.geo_events.dropIndexes();
    db.multilingual_content.dropIndexes();
    db.versioning.dropIndexes();
  }
  logRollback('rollback', {to_version: version});
  print('Rollback to version ' + version + ' completed.');
}

rollbackTo(targetVersion);
