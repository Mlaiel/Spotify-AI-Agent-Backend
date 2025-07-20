from ariadne import QueryType, MutationType, SubscriptionType
import logging
import asyncio

query = QueryType()
mutation = MutationType()
subscription = SubscriptionType()
logger = logging.getLogger("GraphQLResolvers")

# Query resolvers
@query.field("artistInsights")
def resolve_artist_insights(_, info, artistId):
    # Simuler récupération stats artiste (en vrai, requête DB/Spotify)
    logger.info(f"GraphQL: artistInsights pour {artistId}")
    return {
        "id": artistId,
        "name": "Lofi Artist",
        "stats": {"monthlyListeners": 42000, "topCountries": ["FR", "US", "DE"]}
    }

@query.field("playlists")
def resolve_playlists(_, info, userId):
    logger.info(f"GraphQL: playlists pour {userId}")
    return [
        {"id": "p1", "name": "Chill Beats", "tracks": [{"id": "t1", "name": "Dreaming", "mood": "chill", "genre": "lofi"}]},
        {"id": "p2", "name": "EDM Party", "tracks": [{"id": "t2", "name": "Energy", "mood": "upbeat", "genre": "edm"}]}
    ]

# Mutation resolvers
@mutation.field("syncPlaylists")
def resolve_sync_playlists(_, info, userId):
    logger.info(f"GraphQL: syncPlaylists pour {userId}")
    # Simuler synchronisation
    return [
        {"id": "p1", "name": "Chill Beats", "tracks": []},
        {"id": "p2", "name": "EDM Party", "tracks": []}
    ]

# Subscription resolvers
@subscription.source("onTrackPlayed")
async def on_track_played_generator(_, info, artistId):
    # Simuler notifications temps réel (en vrai, websocket/event bus)
    for i in range(3):
        await asyncio.sleep(1)
        yield {"id": f"t{i}", "name": f"Track {i}", "mood": "chill", "genre": "lofi"}

@subscription.field("onTrackPlayed")
def on_track_played_resolver(event, info, artistId):
    logger.info(f"GraphQL: onTrackPlayed pour {artistId}")
    return event
