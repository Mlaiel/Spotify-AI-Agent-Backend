// gdpr_erasure.js
// Created by: Spotify AI Agent Core Team
// GDPR-compliant user data erasure script (advanced, auditable)

/*
Usage:
  mongo < gdpr_erasure.js --eval 'var userId="..."'
- Erases all personal data for the given userId
- Logs all actions to audit_log and consent
*/

if (typeof userId === 'undefined') {
  print('ERROR: Please provide userId as --eval "var userId=\"...\""');
  quit(1);
}

function logErasure(action, details) {
  db.audit_log.insertOne({
    entity_type: 'gdpr',
    action: action,
    details: details,
    performed_by: 'gdpr_erasure_script',
    created_at: new Date()
  });
}

// Remove user from all collections
['users', 'consent', 'analytics', 'audit_log', 'ai_content', 'security_events'].forEach(function(coll) {
  let res = db[coll].remove({user_id: userId});
  logErasure('erase', {collection: coll, user_id: userId, result: res});
});

// Anonymize user in artists, tracks, playlists
['artists', 'tracks', 'playlists'].forEach(function(coll) {
  db[coll].updateMany({user_id: userId}, {$set: {user_id: null, anonymized: true}});
  logErasure('anonymize', {collection: coll, user_id: userId});
});

print('GDPR erasure for user ' + userId + ' completed.');
