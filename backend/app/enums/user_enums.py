from enum import StrEnum, auto

class UserRole(StrEnum):
    ADMIN = auto()
    ARTIST = auto()
    PRODUCER = auto()
    LISTENER = auto()
    AI_AGENT = auto()
    LABEL = auto()
    AUDITOR = auto()
    EXTERNAL_USER = auto()

class AccountStatus(StrEnum):
    ACTIVE = auto()
    SUSPENDED = auto()
    DELETED = auto()
    PENDING = auto()
    BANNED = auto()
    LOCKED = auto()
    MFA_REQUIRED = auto()

class UserPermission(StrEnum):
    READ = auto()
    WRITE = auto()
    DELETE = auto()
    MANAGE_USERS = auto()
    MANAGE_CONTENT = auto()
    VIEW_ANALYTICS = auto()
    ADMIN_PANEL = auto()
    MANAGE_SECURITY = auto()
    EXPORT_DATA = auto()
    MANAGE_SUBSCRIPTION = auto()

class SubscriptionType(StrEnum):
    FREE = auto()
    PREMIUM = auto()
    FAMILY = auto()
    STUDENT = auto()
    TRIAL = auto()
    ENTERPRISE = auto()

class MFAStatus(StrEnum):
    ENABLED = auto()
    DISABLED = auto()
    REQUIRED = auto()
    PENDING = auto()

class ConsentType(StrEnum):
    TERMS = auto()
    PRIVACY = auto()
    MARKETING = auto()
    ANALYTICS = auto()
    THIRD_PARTY = auto()

class NotificationType(StrEnum):
    EMAIL = auto()
    SMS = auto()
    PUSH = auto()
    IN_APP = auto()

class DeviceType(StrEnum):
    DESKTOP = auto()
    MOBILE = auto()
    TABLET = auto()
    SMART_SPEAKER = auto()
    WEARABLE = auto()

# Doc: All enums are ready for direct business use. Extend only via PR and with business justification.
