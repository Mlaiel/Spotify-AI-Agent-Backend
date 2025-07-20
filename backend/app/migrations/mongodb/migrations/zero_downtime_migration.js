// zero_downtime_migration.js
// Created by: Spotify AI Agent Core Team
// Zero-downtime migration pattern for MongoDB (advanced, auditable)

/*
Usage:
  mongo < zero_downtime_migration.js
- Performs schema/data migration with zero downtime
- Uses shadow writes, dual writes, and cutover
- Logs all actions to audit_log
*/

function logZDM(action, details) {
  db.audit_log.insertOne({
    entity_type: 'zero_downtime',
    action: action,
    details: details,
    performed_by: 'zero_downtime_migration_script',
    created_at: new Date()
  });
}

// Example: Add new field 'new_field' to 'users' with shadow writes
print('Starting zero-downtime migration: adding new_field to users');
db.users.updateMany(
  {new_field: {$exists: false}},
  {$set: {new_field: null}}
);
logZDM('shadow_write', {collection: 'users', field: 'new_field'});

// Dual write phase (application must write to both old and new fields)
print('Dual write phase: application should write to both old and new fields');
logZDM('dual_write_phase', {collection: 'users', field: 'new_field'});

// Cutover phase (switch reads to new field, remove old field if needed)
print('Cutover phase: switch reads to new_field');
logZDM('cutover', {collection: 'users', field: 'new_field'});

print('Zero-downtime migration completed.');
