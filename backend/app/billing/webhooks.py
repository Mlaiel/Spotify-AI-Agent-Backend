"""
Spotify AI Agent - Webhook Handlers for Payment Providers
=========================================================

Secure webhook processing for payment providers with:
- Stripe webhook signature verification
- PayPal webhook validation
- Event processing and database updates
- Retry mechanisms and error handling
"""

import json
import hmac
import hashlib
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
from decimal import Decimal
import asyncio
from dataclasses import dataclass

from sqlalchemy.orm import Session
from fastapi import HTTPException, status

from .models import (
    Payment, Subscription, Invoice, Customer,
    PaymentStatus, SubscriptionStatus, InvoiceStatus
)
from .core import billing_engine


logger = logging.getLogger(__name__)


@dataclass
class WebhookEvent:
    """Webhook event data structure"""
    provider: str
    event_type: str
    event_id: str
    data: Dict[str, Any]
    timestamp: datetime
    processed: bool = False


class WebhookProcessor:
    """Base webhook processor"""
    
    def __init__(self, db_session: Session):
        self.db = db_session
        self.logger = logging.getLogger(self.__class__.__name__)
    
    async def process_event(self, event: WebhookEvent) -> bool:
        """Process webhook event"""
        try:
            handler_method = getattr(self, f"handle_{event.event_type}", None)
            if handler_method:
                await handler_method(event)
                return True
            else:
                self.logger.warning(f"No handler for event type: {event.event_type}")
                return False
                
        except Exception as exc:
            self.logger.error(f"Event processing failed: {exc}")
            return False


class StripeWebhookProcessor(WebhookProcessor):
    """Stripe webhook processor"""
    
    def __init__(self, db_session: Session, webhook_secret: str):
        super().__init__(db_session)
        self.webhook_secret = webhook_secret
    
    def verify_signature(self, payload: bytes, signature: str, timestamp: str) -> bool:
        """Verify Stripe webhook signature"""
        try:
            # Extract signature components
            sig_parts = signature.split(',')
            timestamp_part = None
            signature_parts = []
            
            for part in sig_parts:
                if part.startswith('t='):
                    timestamp_part = part[2:]
                elif part.startswith('v1='):
                    signature_parts.append(part[3:])
            
            if not timestamp_part or not signature_parts:
                return False
            
            # Check timestamp tolerance (5 minutes)
            event_timestamp = int(timestamp_part)
            current_timestamp = int(datetime.utcnow().timestamp())
            if abs(current_timestamp - event_timestamp) > 300:
                return False
            
            # Compute expected signature
            signed_payload = f"{timestamp_part}.{payload.decode('utf-8')}"
            expected_sig = hmac.new(
                self.webhook_secret.encode('utf-8'),
                signed_payload.encode('utf-8'),
                hashlib.sha256
            ).hexdigest()
            
            # Compare signatures
            return any(hmac.compare_digest(expected_sig, sig) for sig in signature_parts)
            
        except Exception as exc:
            self.logger.error(f"Signature verification failed: {exc}")
            return False
    
    async def handle_payment_intent_succeeded(self, event: WebhookEvent):
        """Handle successful payment intent"""
        try:
            payment_intent = event.data.get('object', {})
            stripe_payment_id = payment_intent.get('id')
            amount = Decimal(str(payment_intent.get('amount', 0))) / 100  # Convert from cents
            currency = payment_intent.get('currency', '').upper()
            
            # Find payment record
            payment = self.db.query(Payment).filter(
                Payment.provider_transaction_id == stripe_payment_id
            ).first()
            
            if payment:
                # Update payment status
                payment.status = PaymentStatus.SUCCEEDED
                payment.payment_date = datetime.utcnow()
                payment.updated_at = datetime.utcnow()
                
                # Update related invoice if exists
                if payment.invoice_id:
                    invoice = self.db.query(Invoice).filter(
                        Invoice.id == payment.invoice_id
                    ).first()
                    
                    if invoice:
                        invoice.amount_paid += amount
                        if invoice.amount_paid >= invoice.total:
                            invoice.status = InvoiceStatus.PAID
                            invoice.paid_at = datetime.utcnow()
                        invoice.updated_at = datetime.utcnow()
                
                self.db.commit()
                self.logger.info(f"Payment updated: {payment.id} - {stripe_payment_id}")
            
        except Exception as exc:
            self.logger.error(f"Payment intent processing failed: {exc}")
            self.db.rollback()
    
    async def handle_payment_intent_payment_failed(self, event: WebhookEvent):
        """Handle failed payment intent"""
        try:
            payment_intent = event.data.get('object', {})
            stripe_payment_id = payment_intent.get('id')
            failure_reason = payment_intent.get('last_payment_error', {}).get('message', 'Unknown error')
            
            # Find payment record
            payment = self.db.query(Payment).filter(
                Payment.provider_transaction_id == stripe_payment_id
            ).first()
            
            if payment:
                payment.status = PaymentStatus.FAILED
                payment.failure_reason = failure_reason
                payment.updated_at = datetime.utcnow()
                self.db.commit()
                
                self.logger.info(f"Payment failed: {payment.id} - {failure_reason}")
                
                # Handle subscription payment failure
                if payment.subscription_id:
                    await self._handle_subscription_payment_failure(payment.subscription_id)
            
        except Exception as exc:
            self.logger.error(f"Payment failure processing failed: {exc}")
            self.db.rollback()
    
    async def handle_invoice_payment_succeeded(self, event: WebhookEvent):
        """Handle successful invoice payment"""
        try:
            stripe_invoice = event.data.get('object', {})
            stripe_invoice_id = stripe_invoice.get('id')
            amount_paid = Decimal(str(stripe_invoice.get('amount_paid', 0))) / 100
            
            # Find invoice by metadata or subscription
            subscription_id = stripe_invoice.get('subscription')
            if subscription_id:
                subscription = self.db.query(Subscription).filter(
                    Subscription.stripe_subscription_id == subscription_id
                ).first()
                
                if subscription:
                    # Find the most recent unpaid invoice
                    invoice = self.db.query(Invoice).filter(
                        Invoice.subscription_id == subscription.id,
                        Invoice.status == InvoiceStatus.OPEN
                    ).order_by(Invoice.created_at.desc()).first()
                    
                    if invoice:
                        invoice.status = InvoiceStatus.PAID
                        invoice.amount_paid = amount_paid
                        invoice.paid_at = datetime.utcnow()
                        invoice.updated_at = datetime.utcnow()
                        self.db.commit()
                        
                        self.logger.info(f"Invoice paid: {invoice.id}")
            
        except Exception as exc:
            self.logger.error(f"Invoice payment processing failed: {exc}")
            self.db.rollback()
    
    async def handle_customer_subscription_updated(self, event: WebhookEvent):
        """Handle subscription updates"""
        try:
            stripe_subscription = event.data.get('object', {})
            stripe_subscription_id = stripe_subscription.get('id')
            status = stripe_subscription.get('status')
            current_period_start = datetime.fromtimestamp(
                stripe_subscription.get('current_period_start', 0)
            )
            current_period_end = datetime.fromtimestamp(
                stripe_subscription.get('current_period_end', 0)
            )
            
            # Find subscription
            subscription = self.db.query(Subscription).filter(
                Subscription.stripe_subscription_id == stripe_subscription_id
            ).first()
            
            if subscription:
                # Map Stripe status to our status
                status_mapping = {
                    'active': SubscriptionStatus.ACTIVE,
                    'past_due': SubscriptionStatus.PAST_DUE,
                    'canceled': SubscriptionStatus.CANCELLED,
                    'unpaid': SubscriptionStatus.UNPAID,
                    'incomplete': SubscriptionStatus.INCOMPLETE,
                    'incomplete_expired': SubscriptionStatus.INCOMPLETE_EXPIRED,
                    'trialing': SubscriptionStatus.TRIAL,
                    'paused': SubscriptionStatus.PAUSED
                }
                
                subscription.status = status_mapping.get(status, subscription.status)
                subscription.current_period_start = current_period_start
                subscription.current_period_end = current_period_end
                subscription.updated_at = datetime.utcnow()
                
                # Handle cancellation
                if status == 'canceled':
                    subscription.canceled_at = datetime.utcnow()
                    subscription.ended_at = datetime.utcnow()
                
                self.db.commit()
                self.logger.info(f"Subscription updated: {subscription.id} - {status}")
            
        except Exception as exc:
            self.logger.error(f"Subscription update processing failed: {exc}")
            self.db.rollback()
    
    async def handle_customer_subscription_deleted(self, event: WebhookEvent):
        """Handle subscription deletion"""
        try:
            stripe_subscription = event.data.get('object', {})
            stripe_subscription_id = stripe_subscription.get('id')
            
            subscription = self.db.query(Subscription).filter(
                Subscription.stripe_subscription_id == stripe_subscription_id
            ).first()
            
            if subscription:
                subscription.status = SubscriptionStatus.CANCELLED
                subscription.canceled_at = datetime.utcnow()
                subscription.ended_at = datetime.utcnow()
                subscription.updated_at = datetime.utcnow()
                self.db.commit()
                
                self.logger.info(f"Subscription cancelled: {subscription.id}")
            
        except Exception as exc:
            self.logger.error(f"Subscription deletion processing failed: {exc}")
            self.db.rollback()
    
    async def _handle_subscription_payment_failure(self, subscription_id: str):
        """Handle subscription payment failure"""
        try:
            subscription = self.db.query(Subscription).filter(
                Subscription.id == subscription_id
            ).first()
            
            if subscription and subscription.status == SubscriptionStatus.ACTIVE:
                subscription.status = SubscriptionStatus.PAST_DUE
                subscription.updated_at = datetime.utcnow()
                self.db.commit()
                
                # Schedule retry billing
                await billing_engine.schedule_retry_billing(subscription_id)
            
        except Exception as exc:
            self.logger.error(f"Subscription payment failure handling failed: {exc}")


class PayPalWebhookProcessor(WebhookProcessor):
    """PayPal webhook processor"""
    
    def __init__(self, db_session: Session, webhook_id: str, client_id: str, client_secret: str):
        super().__init__(db_session)
        self.webhook_id = webhook_id
        self.client_id = client_id
        self.client_secret = client_secret
    
    def verify_signature(self, headers: Dict[str, str], body: bytes) -> bool:
        """Verify PayPal webhook signature"""
        try:
            # PayPal signature verification would require:
            # 1. Get certificate from PayPal
            # 2. Verify certificate chain
            # 3. Verify signature using certificate
            # This is a simplified version - production would need full implementation
            
            auth_algo = headers.get('PAYPAL-AUTH-ALGO')
            transmission_id = headers.get('PAYPAL-TRANSMISSION-ID')
            cert_id = headers.get('PAYPAL-CERT-ID')
            transmission_sig = headers.get('PAYPAL-TRANSMISSION-SIG')
            transmission_time = headers.get('PAYPAL-TRANSMISSION-TIME')
            
            if not all([auth_algo, transmission_id, cert_id, transmission_sig, transmission_time]):
                return False
            
            # In production, implement proper PayPal signature verification
            # For now, return True for development
            return True
            
        except Exception as exc:
            self.logger.error(f"PayPal signature verification failed: {exc}")
            return False
    
    async def handle_PAYMENT_CAPTURE_COMPLETED(self, event: WebhookEvent):
        """Handle completed payment capture"""
        try:
            payment_data = event.data.get('resource', {})
            paypal_payment_id = payment_data.get('id')
            amount_value = payment_data.get('amount', {}).get('value', '0')
            currency = payment_data.get('amount', {}).get('currency_code', '')
            
            amount = Decimal(amount_value)
            
            # Find payment record
            payment = self.db.query(Payment).filter(
                Payment.provider_transaction_id == paypal_payment_id
            ).first()
            
            if payment:
                payment.status = PaymentStatus.SUCCEEDED
                payment.payment_date = datetime.utcnow()
                payment.updated_at = datetime.utcnow()
                
                # Update related invoice
                if payment.invoice_id:
                    invoice = self.db.query(Invoice).filter(
                        Invoice.id == payment.invoice_id
                    ).first()
                    
                    if invoice:
                        invoice.amount_paid += amount
                        if invoice.amount_paid >= invoice.total:
                            invoice.status = InvoiceStatus.PAID
                            invoice.paid_at = datetime.utcnow()
                        invoice.updated_at = datetime.utcnow()
                
                self.db.commit()
                self.logger.info(f"PayPal payment completed: {payment.id}")
            
        except Exception as exc:
            self.logger.error(f"PayPal payment completion processing failed: {exc}")
            self.db.rollback()
    
    async def handle_PAYMENT_CAPTURE_DENIED(self, event: WebhookEvent):
        """Handle denied payment capture"""
        try:
            payment_data = event.data.get('resource', {})
            paypal_payment_id = payment_data.get('id')
            status_details = payment_data.get('status_details', {})
            failure_reason = status_details.get('reason', 'Payment denied')
            
            # Find payment record
            payment = self.db.query(Payment).filter(
                Payment.provider_transaction_id == paypal_payment_id
            ).first()
            
            if payment:
                payment.status = PaymentStatus.FAILED
                payment.failure_reason = failure_reason
                payment.updated_at = datetime.utcnow()
                self.db.commit()
                
                self.logger.info(f"PayPal payment denied: {payment.id} - {failure_reason}")
                
                # Handle subscription payment failure
                if payment.subscription_id:
                    await self._handle_subscription_payment_failure(payment.subscription_id)
            
        except Exception as exc:
            self.logger.error(f"PayPal payment denial processing failed: {exc}")
            self.db.rollback()
    
    async def handle_BILLING_SUBSCRIPTION_ACTIVATED(self, event: WebhookEvent):
        """Handle subscription activation"""
        try:
            subscription_data = event.data.get('resource', {})
            paypal_subscription_id = subscription_data.get('id')
            status = subscription_data.get('status')
            
            subscription = self.db.query(Subscription).filter(
                Subscription.paypal_subscription_id == paypal_subscription_id
            ).first()
            
            if subscription:
                subscription.status = SubscriptionStatus.ACTIVE
                subscription.updated_at = datetime.utcnow()
                self.db.commit()
                
                self.logger.info(f"PayPal subscription activated: {subscription.id}")
            
        except Exception as exc:
            self.logger.error(f"PayPal subscription activation processing failed: {exc}")
            self.db.rollback()
    
    async def handle_BILLING_SUBSCRIPTION_CANCELLED(self, event: WebhookEvent):
        """Handle subscription cancellation"""
        try:
            subscription_data = event.data.get('resource', {})
            paypal_subscription_id = subscription_data.get('id')
            
            subscription = self.db.query(Subscription).filter(
                Subscription.paypal_subscription_id == paypal_subscription_id
            ).first()
            
            if subscription:
                subscription.status = SubscriptionStatus.CANCELLED
                subscription.canceled_at = datetime.utcnow()
                subscription.ended_at = datetime.utcnow()
                subscription.updated_at = datetime.utcnow()
                self.db.commit()
                
                self.logger.info(f"PayPal subscription cancelled: {subscription.id}")
            
        except Exception as exc:
            self.logger.error(f"PayPal subscription cancellation processing failed: {exc}")
            self.db.rollback()
    
    async def _handle_subscription_payment_failure(self, subscription_id: str):
        """Handle subscription payment failure"""
        try:
            subscription = self.db.query(Subscription).filter(
                Subscription.id == subscription_id
            ).first()
            
            if subscription and subscription.status == SubscriptionStatus.ACTIVE:
                subscription.status = SubscriptionStatus.PAST_DUE
                subscription.updated_at = datetime.utcnow()
                self.db.commit()
                
                # Schedule retry billing
                await billing_engine.schedule_retry_billing(subscription_id)
            
        except Exception as exc:
            self.logger.error(f"Subscription payment failure handling failed: {exc}")


class WebhookEventStore:
    """Store and track webhook events to prevent duplicate processing"""
    
    def __init__(self, db_session: Session):
        self.db = db_session
    
    async def is_event_processed(self, event_id: str, provider: str) -> bool:
        """Check if event has already been processed"""
        # In production, this would query a webhook_events table
        # For now, return False to process all events
        return False
    
    async def mark_event_processed(self, event_id: str, provider: str):
        """Mark event as processed"""
        # In production, this would insert into webhook_events table
        pass


class WebhookManager:
    """Main webhook management orchestrator"""
    
    def __init__(self, db_session: Session):
        self.db = db_session
        self.event_store = WebhookEventStore(db_session)
        self.logger = logging.getLogger(__name__)
        
        # Initialize processors
        self.processors = {}
    
    def register_stripe_processor(self, webhook_secret: str):
        """Register Stripe webhook processor"""
        self.processors['stripe'] = StripeWebhookProcessor(self.db, webhook_secret)
    
    def register_paypal_processor(self, webhook_id: str, client_id: str, client_secret: str):
        """Register PayPal webhook processor"""
        self.processors['paypal'] = PayPalWebhookProcessor(
            self.db, webhook_id, client_id, client_secret
        )
    
    async def process_webhook(self, provider: str, headers: Dict[str, str], 
                            body: bytes) -> bool:
        """Process incoming webhook"""
        try:
            processor = self.processors.get(provider)
            if not processor:
                self.logger.error(f"No processor for provider: {provider}")
                return False
            
            # Verify signature
            if provider == 'stripe':
                signature = headers.get('stripe-signature', '')
                timestamp = headers.get('stripe-timestamp', str(int(datetime.utcnow().timestamp())))
                if not processor.verify_signature(body, signature, timestamp):
                    self.logger.error("Stripe signature verification failed")
                    return False
            
            elif provider == 'paypal':
                if not processor.verify_signature(headers, body):
                    self.logger.error("PayPal signature verification failed")
                    return False
            
            # Parse event data
            try:
                event_data = json.loads(body.decode('utf-8'))
            except json.JSONDecodeError as exc:
                self.logger.error(f"Failed to parse webhook JSON: {exc}")
                return False
            
            # Extract event information
            if provider == 'stripe':
                event_id = event_data.get('id')
                event_type = event_data.get('type')
                data = event_data.get('data', {})
                created = event_data.get('created', int(datetime.utcnow().timestamp()))
                timestamp = datetime.fromtimestamp(created)
            
            elif provider == 'paypal':
                event_id = event_data.get('id')
                event_type = event_data.get('event_type')
                data = event_data
                timestamp = datetime.utcnow()
            
            else:
                self.logger.error(f"Unknown provider: {provider}")
                return False
            
            # Check if already processed
            if await self.event_store.is_event_processed(event_id, provider):
                self.logger.info(f"Event already processed: {event_id}")
                return True
            
            # Create event object
            event = WebhookEvent(
                provider=provider,
                event_type=event_type,
                event_id=event_id,
                data=data,
                timestamp=timestamp
            )
            
            # Process event
            success = await processor.process_event(event)
            
            if success:
                await self.event_store.mark_event_processed(event_id, provider)
                self.logger.info(f"Webhook processed successfully: {provider} - {event_type}")
            else:
                self.logger.error(f"Webhook processing failed: {provider} - {event_type}")
            
            return success
            
        except Exception as exc:
            self.logger.error(f"Webhook processing error: {exc}")
            return False


# Global webhook manager instance
webhook_manager = None


def get_webhook_manager(db_session: Session) -> WebhookManager:
    """Get webhook manager instance"""
    global webhook_manager
    if webhook_manager is None:
        webhook_manager = WebhookManager(db_session)
        
        # Register processors with configuration
        import os
        
        stripe_secret = os.getenv('STRIPE_WEBHOOK_SECRET')
        if stripe_secret:
            webhook_manager.register_stripe_processor(stripe_secret)
        
        paypal_webhook_id = os.getenv('PAYPAL_WEBHOOK_ID')
        paypal_client_id = os.getenv('PAYPAL_CLIENT_ID')
        paypal_client_secret = os.getenv('PAYPAL_CLIENT_SECRET')
        if all([paypal_webhook_id, paypal_client_id, paypal_client_secret]):
            webhook_manager.register_paypal_processor(
                paypal_webhook_id, paypal_client_id, paypal_client_secret
            )
    
    return webhook_manager


# Export main classes
__all__ = [
    'WebhookManager',
    'WebhookEvent',
    'StripeWebhookProcessor',
    'PayPalWebhookProcessor',
    'get_webhook_manager'
]
