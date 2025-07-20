"""
Spotify AI Agent - FastAPI Billing Routes
========================================

Comprehensive REST API endpoints for billing system with:
- Customer management and subscriptions
- Payment processing and webhooks
- Invoice generation and management
- Analytics and reporting
- Multi-provider integration
"""

from datetime import datetime, timedelta
from decimal import Decimal
from typing import List, Optional, Dict, Any
import uuid
import logging
from fastapi import APIRouter, Depends, HTTPException, status, BackgroundTasks, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field, validator
from sqlalchemy.orm import Session
from sqlalchemy import func, and_, or_

from ..core.database import get_db
from ..core.auth import get_current_user, require_permissions
from ..core.exceptions import BillingError, PaymentError, ValidationError
from .models import (
    Customer, Plan, Subscription, Payment, Invoice, PaymentMethod,
    CustomerStatus, SubscriptionStatus, PaymentStatus, InvoiceStatus,
    PaymentProvider, PlanInterval
)
from .core import billing_engine
from .invoices import invoice_manager


# Initialize router
router = APIRouter(prefix="/billing", tags=["billing"])
security = HTTPBearer()
logger = logging.getLogger(__name__)


# Pydantic schemas for request/response
class CustomerCreateRequest(BaseModel):
    email: str = Field(..., description="Customer email address")
    name: str = Field(..., description="Customer full name")
    company: Optional[str] = Field(None, description="Company name")
    phone: Optional[str] = Field(None, description="Phone number")
    address_line1: Optional[str] = Field(None, description="Address line 1")
    address_line2: Optional[str] = Field(None, description="Address line 2")
    city: Optional[str] = Field(None, description="City")
    state: Optional[str] = Field(None, description="State/Province")
    postal_code: Optional[str] = Field(None, description="Postal code")
    country: Optional[str] = Field(None, description="Country code (ISO 2-letter)")
    tax_id: Optional[str] = Field(None, description="Tax ID number")
    preferred_currency: str = Field("EUR", description="Preferred currency")
    preferred_language: str = Field("en", description="Preferred language")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    
    @validator('email')
    def validate_email(cls, v):
        import re
        if not re.match(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$', v):
            raise ValueError('Invalid email format')
        return v.lower()


class CustomerUpdateRequest(BaseModel):
    name: Optional[str] = None
    company: Optional[str] = None
    phone: Optional[str] = None
    address_line1: Optional[str] = None
    address_line2: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    postal_code: Optional[str] = None
    country: Optional[str] = None
    tax_id: Optional[str] = None
    preferred_currency: Optional[str] = None
    preferred_language: Optional[str] = None
    status: Optional[CustomerStatus] = None
    metadata: Optional[Dict[str, Any]] = None


class CustomerResponse(BaseModel):
    id: uuid.UUID
    email: str
    name: str
    company: Optional[str]
    status: CustomerStatus
    preferred_currency: str
    preferred_language: str
    created_at: datetime
    updated_at: datetime
    
    class Config:
        from_attributes = True


class PlanCreateRequest(BaseModel):
    name: str = Field(..., description="Plan name")
    description: Optional[str] = Field(None, description="Plan description")
    amount: Decimal = Field(..., description="Plan amount")
    currency: str = Field("EUR", description="Currency code")
    interval: PlanInterval = Field(PlanInterval.MONTH, description="Billing interval")
    interval_count: int = Field(1, description="Number of intervals between billings")
    trial_period_days: int = Field(0, description="Trial period in days")
    features: List[str] = Field(default_factory=list, description="Plan features")
    usage_limits: Dict[str, Any] = Field(default_factory=dict, description="Usage limits")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class PlanResponse(BaseModel):
    id: uuid.UUID
    name: str
    description: Optional[str]
    amount: Decimal
    currency: str
    interval: PlanInterval
    interval_count: int
    trial_period_days: int
    is_active: bool
    created_at: datetime
    
    class Config:
        from_attributes = True


class SubscriptionCreateRequest(BaseModel):
    customer_id: uuid.UUID = Field(..., description="Customer ID")
    plan_id: uuid.UUID = Field(..., description="Plan ID")
    payment_method_id: Optional[uuid.UUID] = Field(None, description="Payment method ID")
    trial_end: Optional[datetime] = Field(None, description="Trial end date")
    custom_amount: Optional[Decimal] = Field(None, description="Custom subscription amount")
    discount_percent: Decimal = Field(Decimal('0'), description="Discount percentage")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class SubscriptionResponse(BaseModel):
    id: uuid.UUID
    customer_id: uuid.UUID
    plan_id: uuid.UUID
    status: SubscriptionStatus
    current_period_start: datetime
    current_period_end: datetime
    trial_start: Optional[datetime]
    trial_end: Optional[datetime]
    cancel_at_period_end: bool
    effective_amount: Decimal
    created_at: datetime
    
    class Config:
        from_attributes = True


class PaymentMethodCreateRequest(BaseModel):
    customer_id: uuid.UUID = Field(..., description="Customer ID")
    provider: PaymentProvider = Field(..., description="Payment provider")
    provider_payment_method_id: str = Field(..., description="Provider payment method ID")
    type: str = Field(..., description="Payment method type")
    is_default: bool = Field(False, description="Set as default payment method")
    billing_address: Optional[Dict[str, str]] = Field(None, description="Billing address")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class PaymentMethodResponse(BaseModel):
    id: uuid.UUID
    customer_id: uuid.UUID
    provider: PaymentProvider
    type: str
    last4: Optional[str]
    brand: Optional[str]
    is_default: bool
    is_active: bool
    created_at: datetime
    
    class Config:
        from_attributes = True


class PaymentCreateRequest(BaseModel):
    customer_id: uuid.UUID = Field(..., description="Customer ID")
    amount: Decimal = Field(..., description="Payment amount")
    currency: str = Field("EUR", description="Currency code")
    payment_method_id: Optional[uuid.UUID] = Field(None, description="Payment method ID")
    subscription_id: Optional[uuid.UUID] = Field(None, description="Subscription ID")
    invoice_id: Optional[uuid.UUID] = Field(None, description="Invoice ID")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class PaymentResponse(BaseModel):
    id: uuid.UUID
    customer_id: uuid.UUID
    amount: Decimal
    currency: str
    status: PaymentStatus
    provider: PaymentProvider
    payment_date: Optional[datetime]
    failure_reason: Optional[str]
    created_at: datetime
    
    class Config:
        from_attributes = True


class InvoiceCreateRequest(BaseModel):
    customer_id: uuid.UUID = Field(..., description="Customer ID")
    subscription_id: Optional[uuid.UUID] = Field(None, description="Subscription ID")
    line_items: List[Dict[str, Any]] = Field(..., description="Invoice line items")
    currency: str = Field("EUR", description="Currency code")
    issue_date: Optional[datetime] = Field(None, description="Issue date")
    due_date: Optional[datetime] = Field(None, description="Due date")
    payment_terms: Optional[str] = Field(None, description="Payment terms")
    notes: Optional[str] = Field(None, description="Invoice notes")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class InvoiceResponse(BaseModel):
    id: uuid.UUID
    number: str
    customer_id: uuid.UUID
    status: InvoiceStatus
    currency: str
    subtotal: Decimal
    tax_amount: Decimal
    total: Decimal
    amount_due: Decimal
    issue_date: datetime
    due_date: datetime
    created_at: datetime
    
    class Config:
        from_attributes = True


class WebhookPayload(BaseModel):
    provider: PaymentProvider
    event_type: str
    data: Dict[str, Any]
    signature: Optional[str] = None
    timestamp: Optional[datetime] = None


class AnalyticsResponse(BaseModel):
    mrr: Decimal
    arr: Decimal
    active_customers: int
    active_subscriptions: int
    churn_rate: Decimal
    revenue_growth: Decimal
    payment_success_rate: Decimal


# Customer endpoints
@router.post("/customers", response_model=CustomerResponse, status_code=status.HTTP_201_CREATED)
async def create_customer(
    customer_data: CustomerCreateRequest,
    db: Session = Depends(get_db),
    current_user = Depends(get_current_user)
):
    """Create a new customer"""
    try:
        # Check if customer already exists
        existing = db.query(Customer).filter(Customer.email == customer_data.email).first()
        if existing:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Customer with this email already exists"
            )
        
        # Create customer
        customer = Customer(**customer_data.dict())
        db.add(customer)
        db.commit()
        db.refresh(customer)
        
        logger.info(f"Customer created: {customer.id} ({customer.email})")
        return customer
        
    except Exception as exc:
        logger.error(f"Customer creation failed: {exc}")
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc)
        )


@router.get("/customers/{customer_id}", response_model=CustomerResponse)
async def get_customer(
    customer_id: uuid.UUID,
    db: Session = Depends(get_db),
    current_user = Depends(get_current_user)
):
    """Get customer by ID"""
    customer = db.query(Customer).filter(Customer.id == customer_id).first()
    if not customer:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Customer not found"
        )
    
    return customer


@router.put("/customers/{customer_id}", response_model=CustomerResponse)
async def update_customer(
    customer_id: uuid.UUID,
    customer_data: CustomerUpdateRequest,
    db: Session = Depends(get_db),
    current_user = Depends(get_current_user)
):
    """Update customer information"""
    customer = db.query(Customer).filter(Customer.id == customer_id).first()
    if not customer:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Customer not found"
        )
    
    # Update fields
    update_data = customer_data.dict(exclude_unset=True)
    for field, value in update_data.items():
        setattr(customer, field, value)
    
    customer.updated_at = datetime.utcnow()
    db.commit()
    db.refresh(customer)
    
    return customer


@router.get("/customers", response_model=List[CustomerResponse])
async def list_customers(
    skip: int = 0,
    limit: int = 100,
    status_filter: Optional[CustomerStatus] = None,
    db: Session = Depends(get_db),
    current_user = Depends(get_current_user)
):
    """List customers with pagination and filtering"""
    query = db.query(Customer)
    
    if status_filter:
        query = query.filter(Customer.status == status_filter)
    
    customers = query.offset(skip).limit(limit).all()
    return customers


# Plan endpoints
@router.post("/plans", response_model=PlanResponse, status_code=status.HTTP_201_CREATED)
async def create_plan(
    plan_data: PlanCreateRequest,
    db: Session = Depends(get_db),
    current_user = Depends(require_permissions(["billing:write"]))
):
    """Create a new subscription plan"""
    try:
        plan = Plan(**plan_data.dict())
        db.add(plan)
        db.commit()
        db.refresh(plan)
        
        logger.info(f"Plan created: {plan.id} ({plan.name})")
        return plan
        
    except Exception as exc:
        logger.error(f"Plan creation failed: {exc}")
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc)
        )


@router.get("/plans", response_model=List[PlanResponse])
async def list_plans(
    active_only: bool = True,
    db: Session = Depends(get_db),
    current_user = Depends(get_current_user)
):
    """List available subscription plans"""
    query = db.query(Plan)
    
    if active_only:
        query = query.filter(Plan.is_active == True)
    
    plans = query.all()
    return plans


@router.get("/plans/{plan_id}", response_model=PlanResponse)
async def get_plan(
    plan_id: uuid.UUID,
    db: Session = Depends(get_db),
    current_user = Depends(get_current_user)
):
    """Get plan by ID"""
    plan = db.query(Plan).filter(Plan.id == plan_id).first()
    if not plan:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Plan not found"
        )
    
    return plan


# Subscription endpoints
@router.post("/subscriptions", response_model=SubscriptionResponse, status_code=status.HTTP_201_CREATED)
async def create_subscription(
    subscription_data: SubscriptionCreateRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    current_user = Depends(get_current_user)
):
    """Create a new subscription"""
    try:
        # Validate customer and plan exist
        customer = db.query(Customer).filter(Customer.id == subscription_data.customer_id).first()
        if not customer:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Customer not found"
            )
        
        plan = db.query(Plan).filter(Plan.id == subscription_data.plan_id).first()
        if not plan:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Plan not found"
            )
        
        # Create subscription using billing engine
        subscription = await billing_engine.create_subscription(
            customer_id=subscription_data.customer_id,
            plan_id=subscription_data.plan_id,
            payment_method_id=subscription_data.payment_method_id,
            trial_end=subscription_data.trial_end,
            custom_amount=subscription_data.custom_amount,
            discount_percent=subscription_data.discount_percent,
            metadata=subscription_data.metadata
        )
        
        # Schedule first billing if not in trial
        if not subscription.is_in_trial:
            background_tasks.add_task(
                billing_engine.process_subscription_billing,
                subscription.id
            )
        
        logger.info(f"Subscription created: {subscription.id}")
        return subscription
        
    except Exception as exc:
        logger.error(f"Subscription creation failed: {exc}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc)
        )


@router.get("/subscriptions/{subscription_id}", response_model=SubscriptionResponse)
async def get_subscription(
    subscription_id: uuid.UUID,
    db: Session = Depends(get_db),
    current_user = Depends(get_current_user)
):
    """Get subscription by ID"""
    subscription = db.query(Subscription).filter(Subscription.id == subscription_id).first()
    if not subscription:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Subscription not found"
        )
    
    return subscription


@router.put("/subscriptions/{subscription_id}/cancel")
async def cancel_subscription(
    subscription_id: uuid.UUID,
    at_period_end: bool = True,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    current_user = Depends(get_current_user)
):
    """Cancel a subscription"""
    try:
        subscription = db.query(Subscription).filter(Subscription.id == subscription_id).first()
        if not subscription:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Subscription not found"
            )
        
        # Cancel subscription using billing engine
        result = await billing_engine.cancel_subscription(
            subscription_id=subscription_id,
            at_period_end=at_period_end
        )
        
        if not result:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Failed to cancel subscription"
            )
        
        logger.info(f"Subscription cancelled: {subscription_id}")
        return {"message": "Subscription cancelled successfully"}
        
    except Exception as exc:
        logger.error(f"Subscription cancellation failed: {exc}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc)
        )


# Payment method endpoints
@router.post("/payment-methods", response_model=PaymentMethodResponse, status_code=status.HTTP_201_CREATED)
async def create_payment_method(
    payment_method_data: PaymentMethodCreateRequest,
    db: Session = Depends(get_db),
    current_user = Depends(get_current_user)
):
    """Add a payment method for a customer"""
    try:
        # Validate customer exists
        customer = db.query(Customer).filter(Customer.id == payment_method_data.customer_id).first()
        if not customer:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Customer not found"
            )
        
        # Create payment method using billing engine
        payment_method = await billing_engine.add_payment_method(
            customer_id=payment_method_data.customer_id,
            provider=payment_method_data.provider,
            provider_payment_method_id=payment_method_data.provider_payment_method_id,
            payment_method_type=payment_method_data.type,
            is_default=payment_method_data.is_default,
            billing_address=payment_method_data.billing_address,
            metadata=payment_method_data.metadata
        )
        
        logger.info(f"Payment method added: {payment_method.id}")
        return payment_method
        
    except Exception as exc:
        logger.error(f"Payment method creation failed: {exc}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc)
        )


@router.get("/customers/{customer_id}/payment-methods", response_model=List[PaymentMethodResponse])
async def list_customer_payment_methods(
    customer_id: uuid.UUID,
    db: Session = Depends(get_db),
    current_user = Depends(get_current_user)
):
    """List payment methods for a customer"""
    payment_methods = db.query(PaymentMethod).filter(
        PaymentMethod.customer_id == customer_id,
        PaymentMethod.is_active == True
    ).all()
    
    return payment_methods


# Payment endpoints
@router.post("/payments", response_model=PaymentResponse, status_code=status.HTTP_201_CREATED)
async def create_payment(
    payment_data: PaymentCreateRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    current_user = Depends(get_current_user)
):
    """Process a payment"""
    try:
        # Process payment using billing engine
        payment = await billing_engine.process_payment(
            customer_id=payment_data.customer_id,
            amount=payment_data.amount,
            currency=payment_data.currency,
            payment_method_id=payment_data.payment_method_id,
            subscription_id=payment_data.subscription_id,
            invoice_id=payment_data.invoice_id,
            metadata=payment_data.metadata
        )
        
        # Schedule fraud check and notification
        background_tasks.add_task(
            billing_engine.post_payment_processing,
            payment.id
        )
        
        logger.info(f"Payment processed: {payment.id}")
        return payment
        
    except Exception as exc:
        logger.error(f"Payment processing failed: {exc}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc)
        )


@router.get("/payments/{payment_id}", response_model=PaymentResponse)
async def get_payment(
    payment_id: uuid.UUID,
    db: Session = Depends(get_db),
    current_user = Depends(get_current_user)
):
    """Get payment by ID"""
    payment = db.query(Payment).filter(Payment.id == payment_id).first()
    if not payment:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Payment not found"
        )
    
    return payment


@router.post("/payments/{payment_id}/refund")
async def refund_payment(
    payment_id: uuid.UUID,
    amount: Optional[Decimal] = None,
    reason: Optional[str] = None,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    current_user = Depends(require_permissions(["billing:refund"]))
):
    """Refund a payment"""
    try:
        payment = db.query(Payment).filter(Payment.id == payment_id).first()
        if not payment:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Payment not found"
            )
        
        # Process refund using billing engine
        refund_result = await billing_engine.refund_payment(
            payment_id=payment_id,
            amount=amount,
            reason=reason
        )
        
        if not refund_result:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Failed to process refund"
            )
        
        logger.info(f"Payment refunded: {payment_id}")
        return {"message": "Refund processed successfully"}
        
    except Exception as exc:
        logger.error(f"Payment refund failed: {exc}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc)
        )


# Invoice endpoints
@router.post("/invoices", response_model=InvoiceResponse, status_code=status.HTTP_201_CREATED)
async def create_invoice(
    invoice_data: InvoiceCreateRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    current_user = Depends(get_current_user)
):
    """Create a new invoice"""
    try:
        # Validate customer exists
        customer = db.query(Customer).filter(Customer.id == invoice_data.customer_id).first()
        if not customer:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Customer not found"
            )
        
        # Create invoice using invoice manager
        await invoice_manager.initialize()
        
        # Convert line items to proper format
        from .invoices import InvoiceLineItem, InvoiceRecipient, InvoiceType
        
        recipient = InvoiceRecipient(
            name=customer.name,
            email=customer.email,
            company=customer.company,
            address_line1=customer.address_line1,
            address_line2=customer.address_line2,
            city=customer.city,
            state=customer.state,
            postal_code=customer.postal_code,
            country=customer.country,
            tax_id=customer.tax_id,
            language=customer.preferred_language
        )
        
        line_items = []
        for item_data in invoice_data.line_items:
            line_item = InvoiceLineItem(
                description=item_data.get('description', ''),
                quantity=Decimal(str(item_data.get('quantity', 1))),
                unit_price=Decimal(str(item_data.get('unit_price', 0))),
                tax_rate=Decimal(str(item_data.get('tax_rate', 0))),
                discount_rate=Decimal(str(item_data.get('discount_rate', 0))),
                product_code=item_data.get('product_code')
            )
            line_items.append(line_item)
        
        invoice_obj = await invoice_manager.create_invoice(
            recipient=recipient,
            line_items=line_items,
            invoice_type=InvoiceType.ONE_TIME,
            currency=invoice_data.currency,
            payment_terms=invoice_data.payment_terms,
            notes=invoice_data.notes,
            metadata=invoice_data.metadata
        )
        
        # Create database record
        invoice = Invoice(
            id=uuid.UUID(invoice_obj.id),
            number=invoice_obj.number,
            customer_id=invoice_data.customer_id,
            subscription_id=invoice_data.subscription_id,
            status=InvoiceStatus.DRAFT,
            currency=invoice_data.currency,
            subtotal=invoice_obj.subtotal,
            tax_amount=invoice_obj.tax_total,
            total=invoice_obj.total,
            amount_due=invoice_obj.total,
            line_items=[item_data for item_data in invoice_data.line_items],
            issue_date=invoice_data.issue_date or datetime.utcnow(),
            due_date=invoice_data.due_date or (datetime.utcnow() + timedelta(days=30)),
            payment_terms=invoice_data.payment_terms,
            notes=invoice_data.notes,
            metadata=invoice_data.metadata
        )
        
        db.add(invoice)
        db.commit()
        db.refresh(invoice)
        
        # Generate PDF in background
        background_tasks.add_task(
            invoice_manager.generate_pdf,
            invoice_obj.id
        )
        
        logger.info(f"Invoice created: {invoice.id} ({invoice.number})")
        return invoice
        
    except Exception as exc:
        logger.error(f"Invoice creation failed: {exc}")
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc)
        )


@router.get("/invoices/{invoice_id}", response_model=InvoiceResponse)
async def get_invoice(
    invoice_id: uuid.UUID,
    db: Session = Depends(get_db),
    current_user = Depends(get_current_user)
):
    """Get invoice by ID"""
    invoice = db.query(Invoice).filter(Invoice.id == invoice_id).first()
    if not invoice:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Invoice not found"
        )
    
    return invoice


@router.get("/invoices/{invoice_id}/pdf")
async def download_invoice_pdf(
    invoice_id: uuid.UUID,
    db: Session = Depends(get_db),
    current_user = Depends(get_current_user)
):
    """Download invoice PDF"""
    try:
        invoice = db.query(Invoice).filter(Invoice.id == invoice_id).first()
        if not invoice:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Invoice not found"
            )
        
        # Initialize invoice manager and generate PDF
        await invoice_manager.initialize()
        pdf_data = await invoice_manager.generate_pdf(str(invoice_id))
        
        # Return PDF as streaming response
        from io import BytesIO
        
        def iter_pdf():
            yield pdf_data
        
        return StreamingResponse(
            iter_pdf(),
            media_type="application/pdf",
            headers={"Content-Disposition": f"attachment; filename=invoice_{invoice.number}.pdf"}
        )
        
    except Exception as exc:
        logger.error(f"PDF download failed: {exc}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc)
        )


@router.post("/invoices/{invoice_id}/send")
async def send_invoice(
    invoice_id: uuid.UUID,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    current_user = Depends(get_current_user)
):
    """Send invoice to customer"""
    try:
        invoice = db.query(Invoice).filter(Invoice.id == invoice_id).first()
        if not invoice:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Invoice not found"
            )
        
        # Send invoice using invoice manager
        await invoice_manager.initialize()
        success = await invoice_manager.send_invoice(str(invoice_id))
        
        if success:
            # Update invoice status
            invoice.status = InvoiceStatus.OPEN
            invoice.updated_at = datetime.utcnow()
            db.commit()
            
            logger.info(f"Invoice sent: {invoice_id}")
            return {"message": "Invoice sent successfully"}
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Failed to send invoice"
            )
        
    except Exception as exc:
        logger.error(f"Invoice sending failed: {exc}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc)
        )


# Webhook endpoints
@router.post("/webhooks/{provider}")
async def handle_webhook(
    provider: PaymentProvider,
    request: Request,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """Handle payment provider webhooks"""
    try:
        # Get raw body for signature verification
        body = await request.body()
        headers = dict(request.headers)
        
        # Process webhook using billing engine
        result = await billing_engine.process_webhook(
            provider=provider,
            body=body,
            headers=headers
        )
        
        if result:
            logger.info(f"Webhook processed: {provider}")
            return {"status": "success"}
        else:
            logger.warning(f"Webhook processing failed: {provider}")
            return {"status": "failed"}
        
    except Exception as exc:
        logger.error(f"Webhook handling failed: {exc}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc)
        )


# Analytics endpoints
@router.get("/analytics/overview", response_model=AnalyticsResponse)
async def get_analytics_overview(
    db: Session = Depends(get_db),
    current_user = Depends(require_permissions(["billing:analytics"]))
):
    """Get billing analytics overview"""
    try:
        # Calculate MRR
        from .models import calculate_mrr
        mrr = calculate_mrr(db)
        arr = mrr * 12
        
        # Active customers
        active_customers = db.query(Customer).filter(
            Customer.status == CustomerStatus.ACTIVE
        ).count()
        
        # Active subscriptions
        active_subscriptions = db.query(Subscription).filter(
            Subscription.status.in_([SubscriptionStatus.ACTIVE, SubscriptionStatus.TRIAL])
        ).count()
        
        # Payment success rate (last 30 days)
        thirty_days_ago = datetime.utcnow() - timedelta(days=30)
        total_payments = db.query(Payment).filter(
            Payment.created_at >= thirty_days_ago
        ).count()
        
        successful_payments = db.query(Payment).filter(
            Payment.created_at >= thirty_days_ago,
            Payment.status == PaymentStatus.SUCCEEDED
        ).count()
        
        payment_success_rate = Decimal('0')
        if total_payments > 0:
            payment_success_rate = Decimal(successful_payments) / Decimal(total_payments) * 100
        
        return AnalyticsResponse(
            mrr=mrr,
            arr=arr,
            active_customers=active_customers,
            active_subscriptions=active_subscriptions,
            churn_rate=Decimal('0'),  # Would need more complex calculation
            revenue_growth=Decimal('0'),  # Would need historical data
            payment_success_rate=payment_success_rate
        )
        
    except Exception as exc:
        logger.error(f"Analytics calculation failed: {exc}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to calculate analytics"
        )


@router.get("/analytics/revenue")
async def get_revenue_analytics(
    start_date: datetime,
    end_date: datetime,
    db: Session = Depends(get_db),
    current_user = Depends(require_permissions(["billing:analytics"]))
):
    """Get revenue analytics for date range"""
    try:
        # Total revenue
        total_revenue = db.query(func.sum(Payment.amount)).filter(
            Payment.status == PaymentStatus.SUCCEEDED,
            Payment.payment_date >= start_date,
            Payment.payment_date <= end_date
        ).scalar() or Decimal('0')
        
        # Revenue by currency
        revenue_by_currency = db.query(
            Payment.currency,
            func.sum(Payment.amount).label('total')
        ).filter(
            Payment.status == PaymentStatus.SUCCEEDED,
            Payment.payment_date >= start_date,
            Payment.payment_date <= end_date
        ).group_by(Payment.currency).all()
        
        # Revenue by provider
        revenue_by_provider = db.query(
            Payment.provider,
            func.sum(Payment.amount).label('total')
        ).filter(
            Payment.status == PaymentStatus.SUCCEEDED,
            Payment.payment_date >= start_date,
            Payment.payment_date <= end_date
        ).group_by(Payment.provider).all()
        
        return {
            "total_revenue": total_revenue,
            "revenue_by_currency": [
                {"currency": row.currency, "amount": row.total}
                for row in revenue_by_currency
            ],
            "revenue_by_provider": [
                {"provider": row.provider.value, "amount": row.total}
                for row in revenue_by_provider
            ]
        }
        
    except Exception as exc:
        logger.error(f"Revenue analytics calculation failed: {exc}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to calculate revenue analytics"
        )


# Health check
@router.get("/health")
async def health_check():
    """Billing system health check"""
    try:
        # Check billing engine
        engine_status = await billing_engine.health_check()
        
        # Check invoice manager
        await invoice_manager.initialize()
        invoice_status = True
        
        return {
            "status": "healthy",
            "billing_engine": engine_status,
            "invoice_manager": invoice_status,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as exc:
        logger.error(f"Health check failed: {exc}")
        return {
            "status": "unhealthy",
            "error": str(exc),
            "timestamp": datetime.utcnow().isoformat()
        }


# Export router
__all__ = ['router']
