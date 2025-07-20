from enum import StrEnum, auto

class CollaborationStatus(StrEnum):
    PENDING = auto()
    ACCEPTED = auto()
    REJECTED = auto()
    IN_PROGRESS = auto()
    COMPLETED = auto()
    CANCELLED = auto()
    EXPIRED = auto()
    BLOCKED = auto()

class CollaborationRequestType(StrEnum):
    INVITE = auto()
    JOIN = auto()
    FEEDBACK = auto()
    AI_SUGGESTION = auto()
    EXTERNAL_REQUEST = auto()
    SYSTEM_INITIATED = auto()

class CollaborationRole(StrEnum):
    ARTIST = auto()
    PRODUCER = auto()
    MANAGER = auto()
    AI_AGENT = auto()
    LABEL = auto()
    GUEST = auto()
    EXTERNAL_PARTNER = auto()

class CollaborationFeedbackType(StrEnum):
    POSITIVE = auto()
    NEGATIVE = auto()
    NEUTRAL = auto()
    SUGGESTION = auto()
    REPORT = auto()

class CollaborationMatchingStrategy(StrEnum):
    GENRE = auto()
    AUDIENCE = auto()
    AI_SCORE = auto()
    MANUAL = auto()
    HYBRID = auto()

# Doc: All enums are ready for direct business use. Extend only via PR and with business justification.
