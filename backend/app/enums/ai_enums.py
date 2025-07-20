from enum import StrEnum, auto

class AITaskType(StrEnum):
    RECOMMENDATION = auto()
    LYRICS_GENERATION = auto()
    AUDIO_ANALYSIS = auto()
    COLLABORATION_MATCHING = auto()
    CONTENT_OPTIMIZATION = auto()
    AUDIENCE_ANALYSIS = auto()
    PLAYLIST_GENERATION = auto()
    AB_TEST = auto()
    DATA_AUGMENTATION = auto()

class AIModelType(StrEnum):
    TENSORFLOW = "TensorFlow"
    PYTORCH = "PyTorch"
    HUGGINGFACE = "HuggingFace"
    CUSTOM = "Custom"
    ENSEMBLE = "Ensemble"
    DISTILLED = "Distilled"

class AIPipelineStage(StrEnum):
    DATA_PREPROCESSING = auto()
    TRAINING = auto()
    INFERENCE = auto()
    EVALUATION = auto()
    DEPLOYMENT = auto()
    MONITORING = auto()
    ROLLBACK = auto()

class AITrainingStatus(StrEnum):
    QUEUED = auto()
    RUNNING = auto()
    SUCCESS = auto()
    FAILED = auto()
    CANCELLED = auto()

class AIMLFeatureFlag(StrEnum):
    ENABLE_EXPLAINABILITY = auto()
    ENABLE_LOGGING = auto()
    ENABLE_AUDIT = auto()
    ENABLE_EXPERIMENTAL = auto()

# Doc: All enums are ready for direct business use. Extend only via PR and with business justification.
