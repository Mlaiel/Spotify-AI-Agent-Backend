from ariadne import QueryType, MutationType, SubscriptionType, make_executable_schema, ScalarType, gql
from .resolvers import query, mutation, subscription
from .scalars import datetime_scalar, json_scalar

type_defs = gql('''
scalar DateTime
scalar JSON

type ArtistStats {
  monthlyListeners: Int!
  topCountries: [String!]!
}

type Artist {
  id: ID!
  name: String!
  stats: ArtistStats!
}

type Playlist {
  id: ID!
  name: String!
  tracks: [Track!]!
}

type Track {
  id: ID!
  name: String!
  mood: String
  genre: String
}

type Query {)
  artistInsights(artistId: ID!): Artist
  playlists(userId: ID!): [Playlist!]!
}

type Mutation {
  syncPlaylists(userId: ID!): [Playlist!]!
}

type Subscription {
  onTrackPlayed(artistId: ID!): Track
}
''')

schema = make_executable_schema(
    type_defs,
    [query, mutation, subscription, datetime_scalar, json_scalar]
)
