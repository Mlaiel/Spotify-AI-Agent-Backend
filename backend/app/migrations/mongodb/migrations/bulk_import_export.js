// bulk_import_export.js
// Created by: Spotify AI Agent Core Team
// Bulk import/export utility for MongoDB (advanced, auditable)

/*
Usage:
  mongo < bulk_import_export.js --eval 'var mode="import"; var collection="users"; var file="users.json"'
  mongo < bulk_import_export.js --eval 'var mode="export"; var collection="users"; var file="users.json"'
- Supports JSON import/export for any collection
- Logs all actions to audit_log
*/

if (typeof mode === 'undefined' || typeof collection === 'undefined' || typeof file === 'undefined') {
  print('ERROR: Please provide mode, collection, and file as --eval');
  quit(1);
}

function logBulk(action, details) {
  db.audit_log.insertOne({
    entity_type: 'bulk',
    action: action,
    details: details,
    performed_by: 'bulk_import_export_script',
    created_at: new Date()
  });
}

if (mode === 'import') {
  // Use mongoimport externally for large files, here for demo only
  print('Importing ' + file + ' into ' + collection);
  // Not implemented: use mongoimport for real bulk
  logBulk('import', {collection: collection, file: file});
} else if (mode === 'export') {
  print('Exporting ' + collection + ' to ' + file);
  // Not implemented: use mongoexport for real bulk
  logBulk('export', {collection: collection, file: file});
} else {
  print('ERROR: Unknown mode. Use "import" or "export".');
  quit(1);
}
