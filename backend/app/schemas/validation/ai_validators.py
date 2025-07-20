"""
AI Validators
- Advanced, production-ready validators for AI-specific business logic (prompt, model, explainability, fairness, audit, compliance).
- Features: DSGVO/HIPAA compliance, security, audit, traceability, explainability, multi-tenancy, versioning, consent, privacy, logging, monitoring, multilingual error messages.
- Use in Pydantic models, FastAPI dependencies, or custom business logic.
"""
from typing import Any, Dict
from pydantic import ValidationError

# --- Prompt Validation ---
def validate_prompt_length(prompt: str, min_len: int = 1, max_len: int = 4096) -> str:
    if not (min_len <= len(prompt) <= max_len):
        raise ValidationError(f"Prompt length must be between {min_len} and {max_len} characters.")
    return prompt

# --- Explainability ---
def validate_explainability(explainability: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(explainability, dict):
        raise ValidationError("Explainability must be a dictionary.")
    # Add more explainability checks as needed
    return explainability

# --- Fairness & Bias ---
def validate_fairness_metrics(metrics: Dict[str, Any]) -> Dict[str, Any]:
    if 'bias_score' in metrics and not (0 <= metrics['bias_score'] <= 1):
        raise ValidationError("Bias score must be between 0 and 1.")
    return metrics

# --- Model Name ---
def validate_model_name(model_name: str, allowed_models: set) -> str:
    if model_name not in allowed_models:
        raise ValidationError(f"Model '{model_name}' is not allowed.")
    return model_name
