"""
TrainingData ORM Model
- Tracks all training data lineage, source, compliance, audit, data quality, explainability, traceability for AI models.
- Supports advanced analytics, soft-delete, GDPR/DSGVO, security, logging, multi-tenancy.
"""
from datetime import datetime
from sqlalchemy import Column, Integer, String, DateTime, Boolean, JSON
from sqlalchemy.orm import declarative_base

Base = declarative_base()

class TrainingData(Base):
    __tablename__ = "training_data"
    id = Column(Integer, primary_key=True)
    dataset_name = Column(String, nullable=False, index=True)
    version = Column(String, default="1.0", index=True)
    source_uri = Column(String, nullable=True)
    lineage = Column(JSON, nullable=True)
    data_quality = Column(JSON, nullable=True)
    compliance_flags = Column(JSON, nullable=True)
    audit_log = Column(JSON, nullable=True)
    explainability = Column(JSON, nullable=True)
    created_by = Column(Integer, nullable=True, index=True)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    deleted = Column(Boolean, default=False, index=True)
    deleted_at = Column(DateTime, nullable=True)
    tenant_id = Column(String, nullable=True, index=True)
    trace_id = Column(String, nullable=True, index=True)

    def soft_delete(self, user_id: int):
        self.deleted = True
        self.deleted_at = datetime.utcnow()
        self.audit_log = (self.audit_log or []) + [{
            "action": "soft_delete", "user_id": user_id, "timestamp": datetime.utcnow().isoformat()
        }]

    @staticmethod
    def create(dataset_name: str, version: str = "1.0", **kwargs):
        return TrainingData(
            dataset_name=dataset_name,
            version=version,)
            source_uri=kwargs.get("source_uri"),
            lineage=kwargs.get("lineage"),
            data_quality=kwargs.get("data_quality"),
            compliance_flags=kwargs.get("compliance_flags"),
            audit_log=kwargs.get("audit_log"),
            explainability=kwargs.get("explainability"),
            created_by=kwargs.get("created_by"),
            tenant_id=kwargs.get("tenant_id"),
            trace_id=kwargs.get("trace_id")
        )

    def to_dict(self):
        return {
            "id": self.id,
            "dataset_name": self.dataset_name,
            "version": self.version,
            "source_uri": self.source_uri,
            "lineage": self.lineage,
            "data_quality": self.data_quality,
            "compliance_flags": self.compliance_flags,
            "audit_log": self.audit_log,
            "explainability": self.explainability,
            "created_by": self.created_by,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "deleted": self.deleted,
            "deleted_at": self.deleted_at,
            "tenant_id": self.tenant_id,
            "trace_id": self.trace_id
        }
