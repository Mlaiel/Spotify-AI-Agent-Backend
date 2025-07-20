// 003_data_migration.js
// Created by: Spotify AI Agent Core Team
// Lead Dev + Architecte IA, Backend Senior, ML Engineer, DBA/Data Engineer, Sécurité, Microservices

// Example: Migrate all users with missing status to 'ACTIVE'
db.users.updateMany(
  {status: {$exists: false}},
  {$set: {status: "ACTIVE"}}
);

// Example: Normalize artist genres to lowercase arrays
db.artists.find({genres: {$exists: true}}).forEach(function(artist) {
  if (Array.isArray(artist.genres)) {
    db.artists.updateOne(
      {_id: artist._id},
      {$set: {genres: artist.genres.map(function(g) { return g.toLowerCase(); })}}
    );
  }
});

// Example: Add created_at to tracks if missing
db.tracks.updateMany(
  {created_at: {$exists: false}},
  {$set: {created_at: new Date()}}
);
