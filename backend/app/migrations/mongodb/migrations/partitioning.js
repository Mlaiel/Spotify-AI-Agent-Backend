// partitioning.js
// Created by: Spotify AI Agent Core Team
// Partitioning script for large MongoDB collections (advanced, auditable)

/*
Usage:
  mongo < partitioning.js --eval 'var collection="analytics"; var shardKey="user_id"'
- Enables sharding/partitioning for the specified collection
- Logs all actions to audit_log
*/

if (typeof collection === 'undefined' || typeof shardKey === 'undefined') {
  print('ERROR: Please provide collection and shardKey as --eval');
  quit(1);
}

function logPartition(action, details) {
  db.audit_log.insertOne({
    entity_type: 'partitioning',
    action: action,
    details: details,
    performed_by: 'partitioning_script',
    created_at: new Date()
  });
}

print('Enabling sharding for ' + collection + ' on key ' + shardKey);
sh.enableSharding(db.getName());
sh.shardCollection(db.getName() + '.' + collection, { [shardKey]: 1 });
logPartition('enable_sharding', {collection: collection, shardKey: shardKey});
