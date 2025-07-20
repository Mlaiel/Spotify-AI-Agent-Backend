from enum import StrEnum, auto

class SystemStatus(StrEnum):
    OK = auto()
    WARNING = auto()
    ERROR = auto()
    DEGRADED = auto()
    MAINTENANCE = auto()
    OFFLINE = auto()

class Environment(StrEnum):
    DEVELOPMENT = auto()
    STAGING = auto()
    PRODUCTION = auto()
    TEST = auto()
    SANDBOX = auto()

class LogLevel(StrEnum):
    DEBUG = auto()
    INFO = auto()
    WARNING = auto()
    ERROR = auto()
    CRITICAL = auto()
    AUDIT = auto()
    SECURITY = auto()

class ErrorCode(StrEnum):
    UNKNOWN = auto()
    VALIDATION_ERROR = auto()
    AUTHENTICATION_FAILED = auto()
    PERMISSION_DENIED = auto()
    NOT_FOUND = auto()
    RATE_LIMITED = auto()
    INTERNAL_ERROR = auto()
    EXTERNAL_API_ERROR = auto()
    SECURITY_BREACH = auto()
    DATA_LEAK = auto()

class FeatureFlag(StrEnum):
    ENABLE_NEW_API = auto()
    ENABLE_ML_LOGGING = auto()
    ENABLE_AUDIT = auto()
    ENABLE_BETA = auto()

class MaintenanceReason(StrEnum):
    SCHEDULED = auto()
    EMERGENCY = auto()
    SECURITY_PATCH = auto()
    UPGRADE = auto()
    BACKUP = auto()

class APIVersion(StrEnum):
    V1 = auto()
    V2 = auto()
    V3 = auto()

# Doc: All enums are ready for direct business use. Extend only via PR and with business justification.
