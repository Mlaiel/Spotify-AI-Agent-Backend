// 002_add_indexes.js
// Created by: Spotify AI Agent Core Team
// Lead Dev + Architecte IA, Backend Senior, ML Engineer, DBA/Data Engineer, Sécurité, Microservices

// Add indexes for performance, search, and analytics

db.users.createIndex({email: 1}, {unique: true, name: "ix_user_email"});
db.artists.createIndex({user_id: 1}, {name: "ix_artist_user_id"});
db.tracks.createIndex({artist_id: 1}, {name: "ix_track_artist_id"});
db.tracks.createIndex({album_id: 1}, {name: "ix_track_album_id"});
db.playlists.createIndex({owner_id: 1}, {name: "ix_playlist_owner_id"});
db.analytics.createIndex({user_id: 1, event_type: 1}, {name: "ix_analytics_user_event"});
