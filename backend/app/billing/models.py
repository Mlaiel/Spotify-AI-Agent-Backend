"""
Spotify AI Agent - Database Models for Billing System
====================================================

SQLAlchemy models for comprehensive billing system with relationships,
constraints, and business logic validation.

Features:
- Customer management with tiered subscriptions
- Payment processing with multiple providers
- Subscription lifecycle management
- Invoice generation and tracking
- Audit logging and compliance
- Multi-currency and tax support
"""

from datetime import datetime, timedelta
from decimal import Decimal
from typing import Optional, List, Dict, Any
from enum import Enum as PyEnum
import uuid
import json

from sqlalchemy import (
    Column, Integer, String, DateTime, Boolean, Numeric, Text, 
    ForeignKey, Enum, JSON, Index, CheckConstraint, UniqueConstraint,
    event, func, text
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, validates, Session
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.ext.hybrid import hybrid_property
from sqlalchemy.sql import case


Base = declarative_base()


class CustomerStatus(PyEnum):
    """Customer account status"""
    ACTIVE = "active"
    SUSPENDED = "suspended"
    CANCELLED = "cancelled"
    PENDING = "pending"


class SubscriptionStatus(PyEnum):
    """Subscription status enumeration"""
    TRIAL = "trial"
    ACTIVE = "active"
    PAST_DUE = "past_due"
    CANCELLED = "cancelled"
    UNPAID = "unpaid"
    INCOMPLETE = "incomplete"
    INCOMPLETE_EXPIRED = "incomplete_expired"
    PAUSED = "paused"


class PaymentStatus(PyEnum):
    """Payment status enumeration"""
    PENDING = "pending"
    PROCESSING = "processing"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"
    REFUNDED = "refunded"
    PARTIALLY_REFUNDED = "partially_refunded"


class InvoiceStatus(PyEnum):
    """Invoice status enumeration"""
    DRAFT = "draft"
    OPEN = "open"
    PAID = "paid"
    VOID = "void"
    UNCOLLECTIBLE = "uncollectible"


class PaymentProvider(PyEnum):
    """Payment provider enumeration"""
    STRIPE = "stripe"
    PAYPAL = "paypal"
    APPLE_PAY = "apple_pay"
    GOOGLE_PAY = "google_pay"
    BANK_TRANSFER = "bank_transfer"
    CRYPTOCURRENCY = "cryptocurrency"


class PlanInterval(PyEnum):
    """Billing interval enumeration"""
    DAY = "day"
    WEEK = "week"
    MONTH = "month"
    YEAR = "year"


class TaxType(PyEnum):
    """Tax type enumeration"""
    VAT = "vat"
    SALES_TAX = "sales_tax"
    GST = "gst"
    HST = "hst"
    NONE = "none"


class AuditAction(PyEnum):
    """Audit action types"""
    CREATE = "create"
    UPDATE = "update"
    DELETE = "delete"
    PAYMENT = "payment"
    REFUND = "refund"
    SUBSCRIPTION_CHANGE = "subscription_change"


class Customer(Base):
    """Customer entity with billing information"""
    __tablename__ = "customers"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    external_id = Column(String(100), unique=True, nullable=True, index=True)
    email = Column(String(255), unique=True, nullable=False, index=True)
    name = Column(String(255), nullable=False)
    company = Column(String(255), nullable=True)
    phone = Column(String(50), nullable=True)
    
    # Billing address
    address_line1 = Column(String(255), nullable=True)
    address_line2 = Column(String(255), nullable=True)
    city = Column(String(100), nullable=True)
    state = Column(String(100), nullable=True)
    postal_code = Column(String(20), nullable=True)
    country = Column(String(2), nullable=True)  # ISO country code
    
    # Tax information
    tax_id = Column(String(50), nullable=True)
    tax_exempt = Column(Boolean, default=False, nullable=False)
    
    # Account information
    status = Column(Enum(CustomerStatus), default=CustomerStatus.ACTIVE, nullable=False)
    preferred_currency = Column(String(3), default="EUR", nullable=False)
    preferred_language = Column(String(2), default="en", nullable=False)
    
    # Billing configuration
    payment_terms = Column(Integer, default=30, nullable=False)  # Net days
    credit_limit = Column(Numeric(10, 2), default=0, nullable=False)
    
    # Metadata and timestamps
    metadata = Column(JSONB, default=dict, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    
    # Relationships
    subscriptions = relationship("Subscription", back_populates="customer", cascade="all, delete-orphan")
    payments = relationship("Payment", back_populates="customer", cascade="all, delete-orphan")
    invoices = relationship("Invoice", back_populates="customer", cascade="all, delete-orphan")
    payment_methods = relationship("PaymentMethod", back_populates="customer", cascade="all, delete-orphan")
    
    # Indexes
    __table_args__ = (
        Index("idx_customer_email_status", "email", "status"),
        Index("idx_customer_external_id", "external_id"),
        CheckConstraint("credit_limit >= 0", name="check_credit_limit_positive"),
        CheckConstraint("payment_terms > 0", name="check_payment_terms_positive"),
    )
    
    @validates('email')
    def validate_email(self, key, email):
        """Validate email format"""
        import re
        if not re.match(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$', email):
            raise ValueError("Invalid email format")
        return email.lower()
    
    @validates('country')
    def validate_country(self, key, country):
        """Validate ISO country code"""
        if country and len(country) != 2:
            raise ValueError("Country must be a 2-letter ISO code")
        return country.upper() if country else None
    
    @hybrid_property
    def is_active(self):
        """Check if customer is active"""
        return self.status == CustomerStatus.ACTIVE
    
    @hybrid_property
    def full_address(self):
        """Get formatted full address"""
        parts = [
            self.address_line1,
            self.address_line2,
            self.city,
            self.state,
            self.postal_code,
            self.country
        ]
        return ", ".join(filter(None, parts))
    
    def __repr__(self):
        return f"<Customer(id={self.id}, email={self.email}, name={self.name})>"


class Plan(Base):
    """Subscription plan definition"""
    __tablename__ = "plans"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)
    
    # Pricing
    amount = Column(Numeric(10, 2), nullable=False)
    currency = Column(String(3), default="EUR", nullable=False)
    interval = Column(Enum(PlanInterval), default=PlanInterval.MONTH, nullable=False)
    interval_count = Column(Integer, default=1, nullable=False)
    
    # Trial configuration
    trial_period_days = Column(Integer, default=0, nullable=False)
    
    # Features and limits
    features = Column(JSONB, default=list, nullable=False)
    usage_limits = Column(JSONB, default=dict, nullable=False)
    
    # Plan metadata
    is_active = Column(Boolean, default=True, nullable=False)
    metadata = Column(JSONB, default=dict, nullable=False)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    
    # Relationships
    subscriptions = relationship("Subscription", back_populates="plan")
    
    # Constraints
    __table_args__ = (
        CheckConstraint("amount >= 0", name="check_plan_amount_positive"),
        CheckConstraint("interval_count > 0", name="check_interval_count_positive"),
        CheckConstraint("trial_period_days >= 0", name="check_trial_days_positive"),
        Index("idx_plan_active_currency", "is_active", "currency"),
    )
    
    @hybrid_property
    def monthly_amount(self):
        """Convert amount to monthly equivalent"""
        multipliers = {
            PlanInterval.DAY: Decimal('30'),
            PlanInterval.WEEK: Decimal('4.33'),
            PlanInterval.MONTH: Decimal('1'),
            PlanInterval.YEAR: Decimal('0.083')
        }
        return self.amount * multipliers[self.interval] / self.interval_count
    
    def __repr__(self):
        return f"<Plan(id={self.id}, name={self.name}, amount={self.amount})>"


class Subscription(Base):
    """Customer subscription to a plan"""
    __tablename__ = "subscriptions"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    customer_id = Column(UUID(as_uuid=True), ForeignKey("customers.id"), nullable=False)
    plan_id = Column(UUID(as_uuid=True), ForeignKey("plans.id"), nullable=False)
    
    # External provider IDs
    stripe_subscription_id = Column(String(255), nullable=True, unique=True)
    paypal_subscription_id = Column(String(255), nullable=True, unique=True)
    
    # Subscription state
    status = Column(Enum(SubscriptionStatus), default=SubscriptionStatus.TRIAL, nullable=False)
    current_period_start = Column(DateTime, nullable=False)
    current_period_end = Column(DateTime, nullable=False)
    
    # Trial information
    trial_start = Column(DateTime, nullable=True)
    trial_end = Column(DateTime, nullable=True)
    
    # Billing configuration
    billing_cycle_anchor = Column(DateTime, nullable=True)
    cancel_at_period_end = Column(Boolean, default=False, nullable=False)
    canceled_at = Column(DateTime, nullable=True)
    ended_at = Column(DateTime, nullable=True)
    
    # Pricing overrides
    custom_amount = Column(Numeric(10, 2), nullable=True)
    discount_percent = Column(Numeric(5, 2), default=0, nullable=False)
    
    # Usage tracking
    usage_data = Column(JSONB, default=dict, nullable=False)
    
    # Metadata
    metadata = Column(JSONB, default=dict, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    
    # Relationships
    customer = relationship("Customer", back_populates="subscriptions")
    plan = relationship("Plan", back_populates="subscriptions")
    invoices = relationship("Invoice", back_populates="subscription")
    
    # Constraints
    __table_args__ = (
        CheckConstraint("current_period_end > current_period_start", name="check_period_valid"),
        CheckConstraint("discount_percent >= 0 AND discount_percent <= 100", name="check_discount_valid"),
        CheckConstraint("custom_amount IS NULL OR custom_amount >= 0", name="check_custom_amount_positive"),
        Index("idx_subscription_customer_status", "customer_id", "status"),
        Index("idx_subscription_period", "current_period_start", "current_period_end"),
    )
    
    @hybrid_property
    def is_active(self):
        """Check if subscription is active"""
        return self.status in [SubscriptionStatus.ACTIVE, SubscriptionStatus.TRIAL]
    
    @hybrid_property
    def is_in_trial(self):
        """Check if subscription is in trial period"""
        now = datetime.utcnow()
        return (self.status == SubscriptionStatus.TRIAL and 
                self.trial_end and self.trial_end > now)
    
    @hybrid_property
    def effective_amount(self):
        """Get effective subscription amount considering overrides and discounts"""
        base_amount = self.custom_amount or self.plan.amount
        discount_amount = base_amount * (self.discount_percent / 100)
        return base_amount - discount_amount
    
    @hybrid_property
    def days_until_renewal(self):
        """Days until next renewal"""
        if not self.current_period_end:
            return None
        delta = self.current_period_end - datetime.utcnow()
        return max(0, delta.days)
    
    def __repr__(self):
        return f"<Subscription(id={self.id}, customer_id={self.customer_id}, status={self.status})>"


class PaymentMethod(Base):
    """Customer payment methods"""
    __tablename__ = "payment_methods"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    customer_id = Column(UUID(as_uuid=True), ForeignKey("customers.id"), nullable=False)
    
    # Provider information
    provider = Column(Enum(PaymentProvider), nullable=False)
    provider_payment_method_id = Column(String(255), nullable=False)
    
    # Card/Account details (tokenized)
    type = Column(String(50), nullable=False)  # card, bank_account, wallet, etc.
    last4 = Column(String(4), nullable=True)
    brand = Column(String(50), nullable=True)  # visa, mastercard, etc.
    exp_month = Column(Integer, nullable=True)
    exp_year = Column(Integer, nullable=True)
    
    # Status and configuration
    is_default = Column(Boolean, default=False, nullable=False)
    is_active = Column(Boolean, default=True, nullable=False)
    
    # Billing address
    billing_address = Column(JSONB, nullable=True)
    
    # Metadata
    metadata = Column(JSONB, default=dict, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    
    # Relationships
    customer = relationship("Customer", back_populates="payment_methods")
    payments = relationship("Payment", back_populates="payment_method")
    
    # Constraints
    __table_args__ = (
        UniqueConstraint("provider", "provider_payment_method_id", name="unique_provider_payment_method"),
        Index("idx_payment_method_customer", "customer_id", "is_active"),
        CheckConstraint("exp_month IS NULL OR (exp_month >= 1 AND exp_month <= 12)", name="check_exp_month_valid"),
        CheckConstraint("exp_year IS NULL OR exp_year >= 2020", name="check_exp_year_valid"),
    )
    
    @hybrid_property
    def is_expired(self):
        """Check if payment method is expired"""
        if not self.exp_month or not self.exp_year:
            return False
        
        now = datetime.utcnow()
        return (self.exp_year < now.year or 
                (self.exp_year == now.year and self.exp_month < now.month))
    
    def __repr__(self):
        return f"<PaymentMethod(id={self.id}, provider={self.provider}, type={self.type})>"


class Payment(Base):
    """Payment transactions"""
    __tablename__ = "payments"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    customer_id = Column(UUID(as_uuid=True), ForeignKey("customers.id"), nullable=False)
    payment_method_id = Column(UUID(as_uuid=True), ForeignKey("payment_methods.id"), nullable=True)
    subscription_id = Column(UUID(as_uuid=True), ForeignKey("subscriptions.id"), nullable=True)
    invoice_id = Column(UUID(as_uuid=True), ForeignKey("invoices.id"), nullable=True)
    
    # Provider transaction IDs
    provider = Column(Enum(PaymentProvider), nullable=False)
    provider_transaction_id = Column(String(255), nullable=False)
    provider_fee = Column(Numeric(10, 2), default=0, nullable=False)
    
    # Payment details
    amount = Column(Numeric(10, 2), nullable=False)
    currency = Column(String(3), nullable=False)
    status = Column(Enum(PaymentStatus), default=PaymentStatus.PENDING, nullable=False)
    
    # Processing information
    payment_date = Column(DateTime, nullable=True)
    failure_reason = Column(String(500), nullable=True)
    risk_score = Column(Numeric(3, 2), nullable=True)  # 0.00 to 1.00
    
    # Refund information
    refunded_amount = Column(Numeric(10, 2), default=0, nullable=False)
    refund_reason = Column(String(500), nullable=True)
    
    # Metadata
    metadata = Column(JSONB, default=dict, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    
    # Relationships
    customer = relationship("Customer", back_populates="payments")
    payment_method = relationship("PaymentMethod", back_populates="payments")
    invoice = relationship("Invoice", back_populates="payments")
    
    # Constraints
    __table_args__ = (
        UniqueConstraint("provider", "provider_transaction_id", name="unique_provider_transaction"),
        CheckConstraint("amount > 0", name="check_payment_amount_positive"),
        CheckConstraint("provider_fee >= 0", name="check_provider_fee_positive"),
        CheckConstraint("refunded_amount >= 0", name="check_refunded_amount_positive"),
        CheckConstraint("refunded_amount <= amount", name="check_refunded_amount_valid"),
        CheckConstraint("risk_score IS NULL OR (risk_score >= 0 AND risk_score <= 1)", name="check_risk_score_valid"),
        Index("idx_payment_customer_status", "customer_id", "status"),
        Index("idx_payment_date", "payment_date"),
        Index("idx_payment_provider_transaction", "provider", "provider_transaction_id"),
    )
    
    @hybrid_property
    def is_successful(self):
        """Check if payment was successful"""
        return self.status == PaymentStatus.SUCCEEDED
    
    @hybrid_property
    def net_amount(self):
        """Net amount after provider fees"""
        return self.amount - self.provider_fee
    
    @hybrid_property
    def available_for_refund(self):
        """Amount available for refund"""
        return self.amount - self.refunded_amount
    
    @hybrid_property
    def is_high_risk(self):
        """Check if payment is high risk"""
        return self.risk_score and self.risk_score > 0.7
    
    def __repr__(self):
        return f"<Payment(id={self.id}, amount={self.amount}, status={self.status})>"


class Invoice(Base):
    """Customer invoices"""
    __tablename__ = "invoices"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    number = Column(String(50), unique=True, nullable=False)
    customer_id = Column(UUID(as_uuid=True), ForeignKey("customers.id"), nullable=False)
    subscription_id = Column(UUID(as_uuid=True), ForeignKey("subscriptions.id"), nullable=True)
    
    # Invoice details
    status = Column(Enum(InvoiceStatus), default=InvoiceStatus.DRAFT, nullable=False)
    currency = Column(String(3), nullable=False)
    
    # Amounts
    subtotal = Column(Numeric(10, 2), nullable=False)
    tax_amount = Column(Numeric(10, 2), default=0, nullable=False)
    discount_amount = Column(Numeric(10, 2), default=0, nullable=False)
    total = Column(Numeric(10, 2), nullable=False)
    amount_paid = Column(Numeric(10, 2), default=0, nullable=False)
    amount_due = Column(Numeric(10, 2), nullable=False)
    
    # Line items (stored as JSONB for flexibility)
    line_items = Column(JSONB, nullable=False)
    
    # Tax information
    tax_rate = Column(Numeric(5, 4), default=0, nullable=False)  # Support up to 99.99%
    tax_type = Column(Enum(TaxType), default=TaxType.NONE, nullable=False)
    
    # Dates
    issue_date = Column(DateTime, nullable=False)
    due_date = Column(DateTime, nullable=False)
    paid_at = Column(DateTime, nullable=True)
    
    # Invoice configuration
    payment_terms = Column(String(100), nullable=True)
    notes = Column(Text, nullable=True)
    footer = Column(Text, nullable=True)
    
    # Document storage
    pdf_url = Column(String(500), nullable=True)
    
    # Metadata
    metadata = Column(JSONB, default=dict, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    
    # Relationships
    customer = relationship("Customer", back_populates="invoices")
    subscription = relationship("Subscription", back_populates="invoices")
    payments = relationship("Payment", back_populates="invoice")
    
    # Constraints
    __table_args__ = (
        CheckConstraint("subtotal >= 0", name="check_invoice_subtotal_positive"),
        CheckConstraint("tax_amount >= 0", name="check_invoice_tax_positive"),
        CheckConstraint("discount_amount >= 0", name="check_invoice_discount_positive"),
        CheckConstraint("total >= 0", name="check_invoice_total_positive"),
        CheckConstraint("amount_paid >= 0", name="check_invoice_paid_positive"),
        CheckConstraint("amount_due >= 0", name="check_invoice_due_positive"),
        CheckConstraint("amount_paid <= total", name="check_invoice_paid_valid"),
        CheckConstraint("due_date >= issue_date", name="check_invoice_dates_valid"),
        CheckConstraint("tax_rate >= 0 AND tax_rate <= 1", name="check_tax_rate_valid"),
        Index("idx_invoice_customer_status", "customer_id", "status"),
        Index("idx_invoice_number", "number"),
        Index("idx_invoice_due_date", "due_date"),
    )
    
    @hybrid_property
    def is_paid(self):
        """Check if invoice is fully paid"""
        return self.status == InvoiceStatus.PAID
    
    @hybrid_property
    def is_overdue(self):
        """Check if invoice is overdue"""
        return (self.status == InvoiceStatus.OPEN and 
                self.due_date < datetime.utcnow())
    
    @hybrid_property
    def days_overdue(self):
        """Number of days overdue"""
        if not self.is_overdue:
            return 0
        return (datetime.utcnow() - self.due_date).days
    
    def __repr__(self):
        return f"<Invoice(id={self.id}, number={self.number}, total={self.total})>"


class TaxRate(Base):
    """Tax rates by jurisdiction"""
    __tablename__ = "tax_rates"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(100), nullable=False)
    
    # Jurisdiction
    country = Column(String(2), nullable=False)  # ISO country code
    state = Column(String(100), nullable=True)
    city = Column(String(100), nullable=True)
    postal_code = Column(String(20), nullable=True)
    
    # Tax configuration
    tax_type = Column(Enum(TaxType), nullable=False)
    rate = Column(Numeric(5, 4), nullable=False)  # Support up to 99.99%
    
    # Applicability
    is_active = Column(Boolean, default=True, nullable=False)
    effective_from = Column(DateTime, nullable=False)
    effective_to = Column(DateTime, nullable=True)
    
    # Metadata
    metadata = Column(JSONB, default=dict, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    
    # Constraints
    __table_args__ = (
        CheckConstraint("rate >= 0 AND rate <= 1", name="check_tax_rate_valid"),
        CheckConstraint("effective_to IS NULL OR effective_to > effective_from", name="check_tax_dates_valid"),
        Index("idx_tax_rate_jurisdiction", "country", "state", "is_active"),
        Index("idx_tax_rate_effective", "effective_from", "effective_to"),
    )
    
    def __repr__(self):
        return f"<TaxRate(id={self.id}, name={self.name}, rate={self.rate})>"


class UsageRecord(Base):
    """Usage tracking for metered billing"""
    __tablename__ = "usage_records"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    subscription_id = Column(UUID(as_uuid=True), ForeignKey("subscriptions.id"), nullable=False)
    
    # Usage details
    metric_name = Column(String(100), nullable=False)
    quantity = Column(Numeric(15, 6), nullable=False)
    unit = Column(String(50), nullable=False)
    
    # Pricing
    unit_price = Column(Numeric(10, 6), nullable=True)
    total_amount = Column(Numeric(10, 2), nullable=True)
    
    # Timing
    usage_date = Column(DateTime, nullable=False)
    recorded_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    
    # Metadata
    metadata = Column(JSONB, default=dict, nullable=False)
    
    # Relationships
    subscription = relationship("Subscription")
    
    # Constraints
    __table_args__ = (
        CheckConstraint("quantity >= 0", name="check_usage_quantity_positive"),
        CheckConstraint("unit_price IS NULL OR unit_price >= 0", name="check_unit_price_positive"),
        CheckConstraint("total_amount IS NULL OR total_amount >= 0", name="check_total_amount_positive"),
        Index("idx_usage_subscription_metric", "subscription_id", "metric_name"),
        Index("idx_usage_date", "usage_date"),
    )
    
    def __repr__(self):
        return f"<UsageRecord(id={self.id}, metric={self.metric_name}, quantity={self.quantity})>"


class AuditLog(Base):
    """Audit trail for billing operations"""
    __tablename__ = "audit_logs"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    # Action details
    action = Column(Enum(AuditAction), nullable=False)
    entity_type = Column(String(50), nullable=False)
    entity_id = Column(UUID(as_uuid=True), nullable=False)
    
    # User information
    user_id = Column(UUID(as_uuid=True), nullable=True)
    user_email = Column(String(255), nullable=True)
    ip_address = Column(String(45), nullable=True)  # IPv6 compatible
    user_agent = Column(String(500), nullable=True)
    
    # Change details
    old_values = Column(JSONB, nullable=True)
    new_values = Column(JSONB, nullable=True)
    changes = Column(JSONB, nullable=True)
    
    # Context
    reason = Column(String(500), nullable=True)
    metadata = Column(JSONB, default=dict, nullable=False)
    
    # Timestamp
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    
    # Constraints
    __table_args__ = (
        Index("idx_audit_entity", "entity_type", "entity_id"),
        Index("idx_audit_action", "action", "created_at"),
        Index("idx_audit_user", "user_id", "created_at"),
    )
    
    def __repr__(self):
        return f"<AuditLog(id={self.id}, action={self.action}, entity_type={self.entity_type})>"


# Event listeners for audit logging
@event.listens_for(Customer, 'after_insert')
@event.listens_for(Customer, 'after_update')
@event.listens_for(Customer, 'after_delete')
def log_customer_changes(mapper, connection, target):
    """Log customer changes"""
    # Implementation would be added based on specific audit requirements
    pass


@event.listens_for(Payment, 'after_insert')
@event.listens_for(Payment, 'after_update')
def log_payment_changes(mapper, connection, target):
    """Log payment changes"""
    # Implementation would be added based on specific audit requirements
    pass


# Database utility functions
def create_all_tables(engine):
    """Create all tables"""
    Base.metadata.create_all(engine)


def get_customer_by_email(session: Session, email: str) -> Optional[Customer]:
    """Get customer by email"""
    return session.query(Customer).filter(Customer.email == email.lower()).first()


def get_active_subscriptions(session: Session, customer_id: uuid.UUID) -> List[Subscription]:
    """Get active subscriptions for customer"""
    return session.query(Subscription).filter(
        Subscription.customer_id == customer_id,
        Subscription.status.in_([SubscriptionStatus.ACTIVE, SubscriptionStatus.TRIAL])
    ).all()


def get_overdue_invoices(session: Session, limit: int = 100) -> List[Invoice]:
    """Get overdue invoices"""
    return session.query(Invoice).filter(
        Invoice.status == InvoiceStatus.OPEN,
        Invoice.due_date < datetime.utcnow()
    ).limit(limit).all()


def calculate_mrr(session: Session) -> Decimal:
    """Calculate Monthly Recurring Revenue"""
    # Complex query to calculate MRR from active subscriptions
    result = session.query(
        func.sum(
            case(
                (Subscription.status.in_([SubscriptionStatus.ACTIVE, SubscriptionStatus.TRIAL]), 
                 case(
                     (Plan.interval == PlanInterval.MONTH, Plan.amount / Plan.interval_count),
                     (Plan.interval == PlanInterval.YEAR, Plan.amount / (Plan.interval_count * 12)),
                     (Plan.interval == PlanInterval.WEEK, Plan.amount * 4.33 / Plan.interval_count),
                     (Plan.interval == PlanInterval.DAY, Plan.amount * 30 / Plan.interval_count),
                     else_=0
                 )
                ),
                else_=0
            )
        )
    ).join(Plan).filter(
        Subscription.status.in_([SubscriptionStatus.ACTIVE, SubscriptionStatus.TRIAL])
    ).scalar()
    
    return result or Decimal('0')


# Export all models
__all__ = [
    'Base',
    'Customer', 'Plan', 'Subscription', 'PaymentMethod', 'Payment', 
    'Invoice', 'TaxRate', 'UsageRecord', 'AuditLog',
    'CustomerStatus', 'SubscriptionStatus', 'PaymentStatus', 'InvoiceStatus',
    'PaymentProvider', 'PlanInterval', 'TaxType', 'AuditAction',
    'create_all_tables', 'get_customer_by_email', 'get_active_subscriptions',
    'get_overdue_invoices', 'calculate_mrr'
]
