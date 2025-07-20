from enum import StrEnum, auto

class SpotifyEntityType(StrEnum):
    TRACK = auto()
    ALBUM = auto()
    ARTIST = auto()
    PLAYLIST = auto()
    SHOW = auto()
    EPISODE = auto()
    PODCAST = auto()

class PlaylistStatus(StrEnum):
    PUBLIC = auto()
    PRIVATE = auto()
    COLLABORATIVE = auto()
    ARCHIVED = auto()
    CURATED = auto()

class AudioFeature(StrEnum):
    DANCEABILITY = auto()
    ENERGY = auto()
    LOUDNESS = auto()
    SPEECHINESS = auto()
    ACOUSTICNESS = auto()
    INSTRUMENTALNESS = auto()
    LIVENESS = auto()
    VALENCE = auto()
    TEMPO = auto()
    MODE = auto()
    KEY = auto()
    TIME_SIGNATURE = auto()

class SpotifyMarket(StrEnum):
    US = auto()
    DE = auto()
    FR = auto()
    UK = auto()
    JP = auto()
    GLOBAL = auto()

class SpotifyReleaseType(StrEnum):
    SINGLE = auto()
    ALBUM = auto()
    EP = auto()
    COMPILATION = auto()
    REMIX = auto()

# Doc: All enums are ready for direct business use. Extend only via PR and with business justification.
