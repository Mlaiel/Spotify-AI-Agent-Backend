"""
ModelPerformance ORM Model
- Tracks all AI model performance metrics, fairness, drift, monitoring, audit, compliance, explainability, traceability.
- Supports advanced analytics, soft-delete, GDPR/DSGVO, security, logging, multi-tenancy.
"""
from datetime import datetime
from sqlalchemy import Column, Integer, String, DateTime, Boolean, JSON, Float
from sqlalchemy.orm import declarative_base

Base = declarative_base()

class ModelPerformance(Base):
    __tablename__ = "model_performance"
    id = Column(Integer, primary_key=True)
    model_name = Column(String, nullable=False, index=True)
    version = Column(String, default="1.0", index=True)
    metrics = Column(JSON, nullable=True)  # accuracy, precision, recall, f1, auc, etc.
    fairness_metrics = Column(JSON, nullable=True)  # disparate impact, parity, etc.
    drift_metrics = Column(JSON, nullable=True)  # data/model drift
    monitoring_data = Column(JSON, nullable=True)
    explainability = Column(JSON, nullable=True)
    compliance_flags = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    deleted = Column(Boolean, default=False, index=True)
    deleted_at = Column(DateTime, nullable=True)
    tenant_id = Column(String, nullable=True, index=True)
    audit_log = Column(JSON, nullable=True)
    trace_id = Column(String, nullable=True, index=True)

    def soft_delete(self, user_id: int):
        self.deleted = True
        self.deleted_at = datetime.utcnow()
        self.audit_log = (self.audit_log or []) + [{
            "action": "soft_delete", "user_id": user_id, "timestamp": datetime.utcnow().isoformat()
        }]

    @staticmethod
    def create(model_name: str, version: str = "1.0", **kwargs):
        return ModelPerformance(
            model_name=model_name,
            version=version,)
            metrics=kwargs.get("metrics"),
            fairness_metrics=kwargs.get("fairness_metrics"),
            drift_metrics=kwargs.get("drift_metrics"),
            monitoring_data=kwargs.get("monitoring_data"),
            explainability=kwargs.get("explainability"),
            compliance_flags=kwargs.get("compliance_flags"),
            tenant_id=kwargs.get("tenant_id"),
            audit_log=kwargs.get("audit_log"),
            trace_id=kwargs.get("trace_id")
        )

    def to_dict(self):
        return {
            "id": self.id,
            "model_name": self.model_name,
            "version": self.version,
            "metrics": self.metrics,
            "fairness_metrics": self.fairness_metrics,
            "drift_metrics": self.drift_metrics,
            "monitoring_data": self.monitoring_data,
            "explainability": self.explainability,
            "compliance_flags": self.compliance_flags,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "deleted": self.deleted,
            "deleted_at": self.deleted_at,
            "tenant_id": self.tenant_id,
            "audit_log": self.audit_log,
            "trace_id": self.trace_id
        }
