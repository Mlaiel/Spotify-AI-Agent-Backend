from ariadne import MutationType
import logging

mutation = MutationType()
logger = logging.getLogger("GraphQLMutations")

@mutation.field("createPlaylist")
def resolve_create_playlist(_, info, userId, name):
    logger.info(f"GraphQL: createPlaylist {name} pour {userId}")
    return {"id": "p3", "name": name, "tracks": []}

@mutation.field("addTrackToPlaylist")
def resolve_add_track(_, info, playlistId, trackId):
    logger.info(f"GraphQL: addTrackToPlaylist {trackId} dans {playlistId}")
    return {"id": playlistId, "name": "Chill Beats", "tracks": [{"id": trackId, "name": "New Track"}]}
