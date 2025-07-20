"""
Spotify AI Agent - Billing System Tasks & Background Jobs
=========================================================

Celery tasks for asynchronous billing operations:
- Subscription billing and renewal processing
- Failed payment retry mechanisms
- Invoice generation and delivery
- Dunning management for overdue accounts
- Analytics data aggregation
- Fraud detection and monitoring
"""

import asyncio
import logging
from datetime import datetime, timedelta
from decimal import Decimal
from typing import List, Optional, Dict, Any
import uuid
from celery import Celery, Task
from celery.schedules import crontab
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy import create_engine, and_, or_

from ..core.database import get_database_url
from .models import (
    Customer, Subscription, Payment, Invoice, PaymentMethod,
    SubscriptionStatus, PaymentStatus, InvoiceStatus,
    PaymentProvider
)
from .core import billing_engine
from .invoices import invoice_manager
from .webhooks import get_webhook_manager


# Initialize Celery
celery_app = Celery(
    'billing_tasks',
    broker='redis://localhost:6379/1',
    backend='redis://localhost:6379/1'
)

# Celery configuration
celery_app.conf.update(
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,
    task_track_started=True,
    task_time_limit=300,  # 5 minutes
    worker_prefetch_multiplier=1,
    task_acks_late=True,
    worker_disable_rate_limits=False,
    task_compression='gzip',
    result_compression='gzip',
)

# Database setup for tasks
engine = create_engine(get_database_url())
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

logger = logging.getLogger(__name__)


class DatabaseTask(Task):
    """Base task with database session management"""
    
    def __call__(self, *args, **kwargs):
        db_session = SessionLocal()
        try:
            return super().__call__(db_session, *args, **kwargs)
        finally:
            db_session.close()


# Subscription billing tasks
@celery_app.task(base=DatabaseTask, bind=True, max_retries=3)
def process_subscription_billing(self, db_session: Session, subscription_id: str):
    """Process billing for a subscription"""
    try:
        subscription = db_session.query(Subscription).filter(
            Subscription.id == subscription_id
        ).first()
        
        if not subscription:
            logger.error(f"Subscription not found: {subscription_id}")
            return {"status": "error", "message": "Subscription not found"}
        
        if subscription.status not in [SubscriptionStatus.ACTIVE, SubscriptionStatus.TRIAL]:
            logger.info(f"Subscription not active: {subscription_id}")
            return {"status": "skipped", "message": "Subscription not active"}
        
        # Check if billing is due
        now = datetime.utcnow()
        if subscription.current_period_end > now:
            logger.info(f"Billing not due yet: {subscription_id}")
            return {"status": "skipped", "message": "Billing not due"}
        
        # Process billing
        result = asyncio.run(billing_engine.process_subscription_billing(subscription_id))
        
        if result.get("success"):
            logger.info(f"Subscription billing processed: {subscription_id}")
            return {"status": "success", "payment_id": result.get("payment_id")}
        else:
            logger.error(f"Subscription billing failed: {subscription_id}")
            raise self.retry(countdown=300)  # Retry in 5 minutes
        
    except Exception as exc:
        logger.error(f"Subscription billing task failed: {exc}")
        raise self.retry(countdown=300, exc=exc)


@celery_app.task(base=DatabaseTask, bind=True)
def schedule_subscription_billings(self, db_session: Session):
    """Schedule billing for all due subscriptions"""
    try:
        # Find subscriptions due for billing
        now = datetime.utcnow()
        due_subscriptions = db_session.query(Subscription).filter(
            Subscription.status.in_([SubscriptionStatus.ACTIVE, SubscriptionStatus.TRIAL]),
            Subscription.current_period_end <= now + timedelta(hours=1)  # 1 hour buffer
        ).all()
        
        scheduled_count = 0
        for subscription in due_subscriptions:
            # Schedule billing task
            process_subscription_billing.delay(str(subscription.id))
            scheduled_count += 1
        
        logger.info(f"Scheduled billing for {scheduled_count} subscriptions")
        return {"scheduled_count": scheduled_count}
        
    except Exception as exc:
        logger.error(f"Subscription billing scheduling failed: {exc}")
        raise


# Payment retry tasks
@celery_app.task(base=DatabaseTask, bind=True, max_retries=5)
def retry_failed_payment(self, db_session: Session, payment_id: str, retry_count: int = 0):
    """Retry a failed payment"""
    try:
        payment = db_session.query(Payment).filter(
            Payment.id == payment_id
        ).first()
        
        if not payment:
            logger.error(f"Payment not found: {payment_id}")
            return {"status": "error", "message": "Payment not found"}
        
        if payment.status != PaymentStatus.FAILED:
            logger.info(f"Payment not in failed state: {payment_id}")
            return {"status": "skipped", "message": "Payment not failed"}
        
        # Calculate retry delay (exponential backoff)
        retry_delays = [1, 3, 6, 24, 72]  # hours
        if retry_count >= len(retry_delays):
            logger.error(f"Max retries exceeded for payment: {payment_id}")
            
            # Mark subscription as past due if applicable
            if payment.subscription_id:
                subscription = db_session.query(Subscription).filter(
                    Subscription.id == payment.subscription_id
                ).first()
                if subscription:
                    subscription.status = SubscriptionStatus.PAST_DUE
                    db_session.commit()
            
            return {"status": "failed", "message": "Max retries exceeded"}
        
        # Retry payment
        result = asyncio.run(billing_engine.retry_payment(payment_id))
        
        if result.get("success"):
            logger.info(f"Payment retry successful: {payment_id}")
            return {"status": "success"}
        else:
            # Schedule next retry
            delay_hours = retry_delays[retry_count]
            retry_failed_payment.apply_async(
                args=[payment_id, retry_count + 1],
                countdown=delay_hours * 3600
            )
            
            logger.info(f"Payment retry scheduled for {delay_hours}h: {payment_id}")
            return {"status": "retry_scheduled", "next_retry_hours": delay_hours}
        
    except Exception as exc:
        logger.error(f"Payment retry task failed: {exc}")
        raise self.retry(countdown=1800, exc=exc)  # Retry in 30 minutes


@celery_app.task(base=DatabaseTask, bind=True)
def process_failed_payments(self, db_session: Session):
    """Process all failed payments for retry"""
    try:
        # Find failed payments within retry window
        cutoff_date = datetime.utcnow() - timedelta(days=7)  # 7 day retry window
        failed_payments = db_session.query(Payment).filter(
            Payment.status == PaymentStatus.FAILED,
            Payment.created_at >= cutoff_date
        ).all()
        
        processed_count = 0
        for payment in failed_payments:
            # Check if already scheduled for retry
            if not payment.metadata.get('retry_scheduled'):
                retry_failed_payment.delay(str(payment.id))
                
                # Mark as scheduled
                payment.metadata = payment.metadata or {}
                payment.metadata['retry_scheduled'] = True
                processed_count += 1
        
        db_session.commit()
        logger.info(f"Processed {processed_count} failed payments for retry")
        return {"processed_count": processed_count}
        
    except Exception as exc:
        logger.error(f"Failed payment processing failed: {exc}")
        raise


# Invoice tasks
@celery_app.task(base=DatabaseTask, bind=True)
def generate_subscription_invoices(self, db_session: Session):
    """Generate invoices for subscription billing"""
    try:
        # Find subscriptions that need invoicing
        now = datetime.utcnow()
        invoice_date = now + timedelta(days=3)  # 3 days before billing
        
        subscriptions_to_invoice = db_session.query(Subscription).filter(
            Subscription.status.in_([SubscriptionStatus.ACTIVE, SubscriptionStatus.TRIAL]),
            Subscription.current_period_end <= invoice_date,
            Subscription.current_period_end > now
        ).all()
        
        generated_count = 0
        for subscription in subscriptions_to_invoice:
            # Check if invoice already exists for this period
            existing_invoice = db_session.query(Invoice).filter(
                Invoice.subscription_id == subscription.id,
                Invoice.issue_date >= subscription.current_period_start,
                Invoice.due_date <= subscription.current_period_end + timedelta(days=30)
            ).first()
            
            if not existing_invoice:
                generate_subscription_invoice.delay(str(subscription.id))
                generated_count += 1
        
        logger.info(f"Scheduled invoice generation for {generated_count} subscriptions")
        return {"generated_count": generated_count}
        
    except Exception as exc:
        logger.error(f"Invoice generation scheduling failed: {exc}")
        raise


@celery_app.task(base=DatabaseTask, bind=True, max_retries=3)
def generate_subscription_invoice(self, db_session: Session, subscription_id: str):
    """Generate invoice for a specific subscription"""
    try:
        subscription = db_session.query(Subscription).filter(
            Subscription.id == subscription_id
        ).first()
        
        if not subscription:
            logger.error(f"Subscription not found: {subscription_id}")
            return {"status": "error", "message": "Subscription not found"}
        
        customer = subscription.customer
        plan = subscription.plan
        
        # Initialize invoice manager
        asyncio.run(invoice_manager.initialize())
        
        # Create recipient
        from .invoices import InvoiceRecipient, InvoiceLineItem, InvoiceType
        
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
        
        # Create line item
        line_item = InvoiceLineItem(
            description=f"{plan.name} subscription",
            quantity=Decimal('1'),
            unit_price=subscription.effective_amount,
            tax_rate=Decimal('20'),  # Default VAT rate
            period_start=subscription.current_period_start,
            period_end=subscription.current_period_end
        )
        
        # Create invoice
        invoice_obj = asyncio.run(invoice_manager.create_invoice(
            recipient=recipient,
            line_items=[line_item],
            invoice_type=InvoiceType.SUBSCRIPTION,
            currency=plan.currency,
            metadata={'subscription_id': str(subscription_id)}
        ))
        
        # Create database record
        invoice = Invoice(
            id=uuid.UUID(invoice_obj.id),
            number=invoice_obj.number,
            customer_id=subscription.customer_id,
            subscription_id=subscription.id,
            status=InvoiceStatus.DRAFT,
            currency=plan.currency,
            subtotal=invoice_obj.subtotal,
            tax_amount=invoice_obj.tax_total,
            total=invoice_obj.total,
            amount_due=invoice_obj.total,
            line_items=[{
                'description': line_item.description,
                'quantity': str(line_item.quantity),
                'unit_price': str(line_item.unit_price),
                'total': str(line_item.total)
            }],
            issue_date=invoice_obj.issue_date,
            due_date=invoice_obj.due_date
        )
        
        db_session.add(invoice)
        db_session.commit()
        
        # Schedule PDF generation and sending
        generate_invoice_pdf.delay(str(invoice.id))
        
        logger.info(f"Invoice generated: {invoice.id} for subscription {subscription_id}")
        return {"status": "success", "invoice_id": str(invoice.id)}
        
    except Exception as exc:
        logger.error(f"Invoice generation failed: {exc}")
        raise self.retry(countdown=300, exc=exc)


@celery_app.task(base=DatabaseTask, bind=True, max_retries=3)
def generate_invoice_pdf(self, db_session: Session, invoice_id: str):
    """Generate PDF for an invoice"""
    try:
        invoice = db_session.query(Invoice).filter(
            Invoice.id == invoice_id
        ).first()
        
        if not invoice:
            logger.error(f"Invoice not found: {invoice_id}")
            return {"status": "error", "message": "Invoice not found"}
        
        # Initialize invoice manager
        asyncio.run(invoice_manager.initialize())
        
        # Generate PDF
        pdf_data = asyncio.run(invoice_manager.generate_pdf(invoice_id))
        
        logger.info(f"Invoice PDF generated: {invoice_id}")
        
        # Schedule sending if invoice is not draft
        if invoice.status != InvoiceStatus.DRAFT:
            send_invoice_email.delay(invoice_id)
        
        return {"status": "success", "pdf_size": len(pdf_data)}
        
    except Exception as exc:
        logger.error(f"Invoice PDF generation failed: {exc}")
        raise self.retry(countdown=300, exc=exc)


@celery_app.task(base=DatabaseTask, bind=True, max_retries=3)
def send_invoice_email(self, db_session: Session, invoice_id: str):
    """Send invoice via email"""
    try:
        invoice = db_session.query(Invoice).filter(
            Invoice.id == invoice_id
        ).first()
        
        if not invoice:
            logger.error(f"Invoice not found: {invoice_id}")
            return {"status": "error", "message": "Invoice not found"}
        
        # Initialize invoice manager
        asyncio.run(invoice_manager.initialize())
        
        # Send invoice
        success = asyncio.run(invoice_manager.send_invoice(invoice_id))
        
        if success:
            # Update invoice status
            invoice.status = InvoiceStatus.OPEN
            invoice.updated_at = datetime.utcnow()
            db_session.commit()
            
            logger.info(f"Invoice sent: {invoice_id}")
            return {"status": "success"}
        else:
            logger.error(f"Invoice sending failed: {invoice_id}")
            raise self.retry(countdown=600)  # Retry in 10 minutes
        
    except Exception as exc:
        logger.error(f"Invoice email task failed: {exc}")
        raise self.retry(countdown=600, exc=exc)


# Dunning management tasks
@celery_app.task(base=DatabaseTask, bind=True)
def process_overdue_invoices(self, db_session: Session):
    """Process overdue invoices for dunning"""
    try:
        # Find overdue invoices
        now = datetime.utcnow()
        overdue_invoices = db_session.query(Invoice).filter(
            Invoice.status == InvoiceStatus.OPEN,
            Invoice.due_date < now
        ).all()
        
        processed_count = 0
        for invoice in overdue_invoices:
            days_overdue = (now - invoice.due_date).days
            
            # Schedule dunning actions based on days overdue
            if days_overdue in [1, 7, 14, 30]:
                send_dunning_notice.delay(str(invoice.id), days_overdue)
                processed_count += 1
            elif days_overdue >= 60:
                # Mark as uncollectible
                mark_invoice_uncollectible.delay(str(invoice.id))
                processed_count += 1
        
        logger.info(f"Processed {processed_count} overdue invoices")
        return {"processed_count": processed_count}
        
    except Exception as exc:
        logger.error(f"Overdue invoice processing failed: {exc}")
        raise


@celery_app.task(base=DatabaseTask, bind=True, max_retries=3)
def send_dunning_notice(self, db_session: Session, invoice_id: str, days_overdue: int):
    """Send dunning notice for overdue invoice"""
    try:
        invoice = db_session.query(Invoice).filter(
            Invoice.id == invoice_id
        ).first()
        
        if not invoice:
            logger.error(f"Invoice not found: {invoice_id}")
            return {"status": "error", "message": "Invoice not found"}
        
        customer = invoice.customer
        
        # Create dunning notice content based on days overdue
        if days_overdue == 1:
            subject = f"Payment Reminder - Invoice {invoice.number}"
            urgency = "gentle"
        elif days_overdue <= 14:
            subject = f"Payment Overdue - Invoice {invoice.number}"
            urgency = "firm"
        else:
            subject = f"Final Notice - Invoice {invoice.number}"
            urgency = "final"
        
        # Send dunning email (implement email sending logic)
        email_sent = asyncio.run(send_dunning_email(
            customer.email,
            subject,
            invoice,
            urgency,
            days_overdue
        ))
        
        if email_sent:
            # Update invoice metadata
            invoice.metadata = invoice.metadata or {}
            invoice.metadata[f'dunning_notice_{days_overdue}'] = datetime.utcnow().isoformat()
            db_session.commit()
            
            logger.info(f"Dunning notice sent: {invoice_id} ({days_overdue} days)")
            return {"status": "success"}
        else:
            raise self.retry(countdown=1800)  # Retry in 30 minutes
        
    except Exception as exc:
        logger.error(f"Dunning notice task failed: {exc}")
        raise self.retry(countdown=1800, exc=exc)


@celery_app.task(base=DatabaseTask, bind=True)
def mark_invoice_uncollectible(self, db_session: Session, invoice_id: str):
    """Mark invoice as uncollectible"""
    try:
        invoice = db_session.query(Invoice).filter(
            Invoice.id == invoice_id
        ).first()
        
        if not invoice:
            logger.error(f"Invoice not found: {invoice_id}")
            return {"status": "error", "message": "Invoice not found"}
        
        # Mark as uncollectible
        invoice.status = InvoiceStatus.UNCOLLECTIBLE
        invoice.updated_at = datetime.utcnow()
        
        # Cancel related subscription if exists
        if invoice.subscription_id:
            subscription = db_session.query(Subscription).filter(
                Subscription.id == invoice.subscription_id
            ).first()
            if subscription and subscription.status != SubscriptionStatus.CANCELLED:
                subscription.status = SubscriptionStatus.CANCELLED
                subscription.canceled_at = datetime.utcnow()
                subscription.ended_at = datetime.utcnow()
        
        db_session.commit()
        
        logger.info(f"Invoice marked uncollectible: {invoice_id}")
        return {"status": "success"}
        
    except Exception as exc:
        logger.error(f"Mark uncollectible task failed: {exc}")
        raise


# Analytics tasks
@celery_app.task(base=DatabaseTask, bind=True)
def update_analytics_cache(self, db_session: Session):
    """Update analytics cache with latest data"""
    try:
        from .analytics import BillingAnalytics
        import json
        import redis
        
        analytics = BillingAnalytics(db_session)
        redis_client = redis.Redis.from_url('redis://localhost:6379/2')
        
        # Calculate current metrics
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=30)
        
        revenue_metrics = asyncio.run(analytics.get_revenue_metrics(start_date, end_date))
        customer_metrics = asyncio.run(analytics.get_customer_metrics(start_date, end_date))
        subscription_metrics = asyncio.run(analytics.get_subscription_metrics(start_date, end_date))
        payment_metrics = asyncio.run(analytics.get_payment_metrics(start_date, end_date))
        
        # Cache metrics
        cache_data = {
            'revenue': revenue_metrics.__dict__,
            'customers': customer_metrics.__dict__,
            'subscriptions': subscription_metrics.__dict__,
            'payments': payment_metrics.__dict__,
            'updated_at': datetime.utcnow().isoformat()
        }
        
        # Convert Decimal to string for JSON serialization
        cache_json = json.dumps(cache_data, default=str)
        redis_client.setex('billing_analytics_cache', 3600, cache_json)  # 1 hour TTL
        
        logger.info("Analytics cache updated successfully")
        return {"status": "success", "cached_at": datetime.utcnow().isoformat()}
        
    except Exception as exc:
        logger.error(f"Analytics cache update failed: {exc}")
        raise


# Fraud detection tasks
@celery_app.task(base=DatabaseTask, bind=True)
def detect_fraud_patterns(self, db_session: Session):
    """Detect potential fraud patterns in payments"""
    try:
        # Find recent high-risk payments
        cutoff_date = datetime.utcnow() - timedelta(hours=24)
        high_risk_payments = db_session.query(Payment).filter(
            Payment.created_at >= cutoff_date,
            Payment.risk_score > 0.7
        ).all()
        
        flagged_count = 0
        for payment in high_risk_payments:
            # Additional fraud checks
            if await is_fraud_pattern(payment, db_session):
                # Flag for manual review
                payment.metadata = payment.metadata or {}
                payment.metadata['fraud_review_required'] = True
                payment.metadata['fraud_detected_at'] = datetime.utcnow().isoformat()
                flagged_count += 1
        
        db_session.commit()
        
        logger.info(f"Fraud detection completed: {flagged_count} payments flagged")
        return {"flagged_count": flagged_count}
        
    except Exception as exc:
        logger.error(f"Fraud detection failed: {exc}")
        raise


# Utility functions
async def send_dunning_email(email: str, subject: str, invoice: Invoice, 
                           urgency: str, days_overdue: int) -> bool:
    """Send dunning email to customer"""
    try:
        # Implement email sending logic here
        # This is a placeholder implementation
        logger.info(f"Sending dunning email to {email}: {subject}")
        return True
    except Exception as exc:
        logger.error(f"Dunning email sending failed: {exc}")
        return False


async def is_fraud_pattern(payment: Payment, db_session: Session) -> bool:
    """Check if payment matches fraud patterns"""
    try:
        # Implement fraud pattern detection logic
        # Examples:
        # - Multiple failed attempts from same IP
        # - Unusual spending patterns
        # - Velocity checks
        # - Geolocation anomalies
        
        # Placeholder implementation
        return payment.risk_score > 0.8
    except Exception as exc:
        logger.error(f"Fraud pattern check failed: {exc}")
        return False


# Periodic task schedule
celery_app.conf.beat_schedule = {
    'schedule-subscription-billings': {
        'task': 'app.billing.tasks.schedule_subscription_billings',
        'schedule': crontab(minute=0),  # Every hour
    },
    'process-failed-payments': {
        'task': 'app.billing.tasks.process_failed_payments',
        'schedule': crontab(minute=30, hour='*/6'),  # Every 6 hours
    },
    'generate-subscription-invoices': {
        'task': 'app.billing.tasks.generate_subscription_invoices',
        'schedule': crontab(minute=0, hour=6),  # Daily at 6 AM
    },
    'process-overdue-invoices': {
        'task': 'app.billing.tasks.process_overdue_invoices',
        'schedule': crontab(minute=0, hour=10),  # Daily at 10 AM
    },
    'update-analytics-cache': {
        'task': 'app.billing.tasks.update_analytics_cache',
        'schedule': crontab(minute=0, hour='*/3'),  # Every 3 hours
    },
    'detect-fraud-patterns': {
        'task': 'app.billing.tasks.detect_fraud_patterns',
        'schedule': crontab(minute=15, hour='*/2'),  # Every 2 hours
    },
}

celery_app.conf.timezone = 'UTC'


# Export Celery app and main tasks
__all__ = [
    'celery_app',
    'process_subscription_billing',
    'retry_failed_payment',
    'generate_subscription_invoice',
    'send_invoice_email',
    'process_overdue_invoices',
    'update_analytics_cache',
    'detect_fraud_patterns'
]
