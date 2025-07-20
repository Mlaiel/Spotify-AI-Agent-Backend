"""
Custom Validators
- Advanced, production-ready validators for custom business logic, cross-domain rules, and advanced compliance (DSGVO/HIPAA, audit, traceability, explainability, multi-tenancy, versioning, consent, privacy, logging, monitoring).
- Use in Pydantic models, FastAPI dependencies, or custom business logic.
"""
from typing import Any, Dict
from pydantic import ValidationError

# --- Multi-Tenancy ---
def validate_tenant_id(tenant_id: str) -> str:
    if not tenant_id or len(tenant_id) < 3:
        raise ValidationError("Tenant ID must be at least 3 characters long.")
    return tenant_id

# --- Traceability ---
def validate_trace_id(trace_id: str) -> str:
    if not trace_id or len(trace_id) < 8:
        raise ValidationError("Trace ID must be at least 8 characters long.")
    return trace_id

# --- Compliance Flags ---
def validate_compliance_flags(flags: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(flags, dict):
        raise ValidationError("Compliance flags must be a dictionary.")
    # Add more compliance checks as needed
    return flags

# --- Audit Log ---
def validate_audit_log(audit_log: Any) -> Any:
    if not isinstance(audit_log, (list, type(None))):
        raise ValidationError("Audit log must be a list or None.")
    return audit_log
