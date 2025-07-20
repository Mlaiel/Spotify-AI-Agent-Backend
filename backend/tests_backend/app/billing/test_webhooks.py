"""
Tests for Webhook Processing System
==================================

Comprehensive tests for webhook handling from payment providers.
"""

import pytest
from datetime import datetime, timedelta
from decimal import Decimal
from unittest.mock import AsyncMock, Mock, patch, MagicMock
import json
import hmac
import hashlib
import base64

from billing.webhooks import WebhookProcessor, StripeWebhookHandler, PayPalWebhookHandler
from billing.models import (
    Payment, PaymentStatus, Invoice, InvoiceStatus, Subscription, 
    SubscriptionStatus, Customer, PaymentProvider
)


class TestWebhookProcessor:
    """Test main webhook processor functionality"""
    
    @pytest.mark.asyncio
    async def test_process_stripe_webhook(self, webhook_processor, mock_stripe_handler):
        """Test Stripe webhook processing"""
        webhook_data = {
            "id": "evt_stripe_test",
            "type": "payment_intent.succeeded",
            "data": {
                "object": {
                    "id": "pi_test_success",
                    "status": "succeeded",
                    "amount": 5000,
                    "currency": "eur"
                }
            }
        }
        
        signature = "test_stripe_signature"
        mock_stripe_handler.verify_webhook_signature.return_value = True
        mock_stripe_handler.process_event.return_value = {"processed": True, "payment_id": "pay_123"}
        
        result = await webhook_processor.process_stripe_webhook(
            payload=json.dumps(webhook_data),
            signature=signature
        )
        
        assert result["processed"] is True
        assert result["provider"] == "stripe"
        mock_stripe_handler.verify_webhook_signature.assert_called_once()
        mock_stripe_handler.process_event.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_process_paypal_webhook(self, webhook_processor, mock_paypal_handler):
        """Test PayPal webhook processing"""
        webhook_data = {
            "id": "WH-paypal-test",
            "event_type": "PAYMENT.CAPTURE.COMPLETED",
            "resource": {
                "id": "PAYID-test-success",
                "status": "COMPLETED",
                "amount": {
                    "value": "50.00",
                    "currency_code": "EUR"
                }
            }
        }
        
        mock_paypal_handler.verify_webhook_signature.return_value = True
        mock_paypal_handler.process_event.return_value = {"processed": True, "payment_id": "pay_456"}
        
        result = await webhook_processor.process_paypal_webhook(
            payload=json.dumps(webhook_data),
            headers={"paypal-transmission-id": "test-transmission-id"}
        )
        
        assert result["processed"] is True
        assert result["provider"] == "paypal"
        mock_paypal_handler.verify_webhook_signature.assert_called_once()
        mock_paypal_handler.process_event.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_webhook_signature_verification_failure(self, webhook_processor, mock_stripe_handler):
        """Test webhook processing with invalid signature"""
        webhook_data = {"id": "evt_invalid", "type": "test.event"}
        invalid_signature = "invalid_signature"
        
        mock_stripe_handler.verify_webhook_signature.return_value = False
        
        with pytest.raises(Exception) as exc_info:
            await webhook_processor.process_stripe_webhook(
                payload=json.dumps(webhook_data),
                signature=invalid_signature
            )
        
        assert "Invalid webhook signature" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_duplicate_webhook_handling(self, webhook_processor, mock_stripe_handler, db_session):
        """Test handling of duplicate webhook events"""
        webhook_data = {
            "id": "evt_duplicate_test",
            "type": "payment_intent.succeeded",
            "data": {"object": {"id": "pi_duplicate"}}
        }
        
        # First webhook processing
        mock_stripe_handler.verify_webhook_signature.return_value = True
        mock_stripe_handler.process_event.return_value = {"processed": True}
        
        result1 = await webhook_processor.process_stripe_webhook(
            payload=json.dumps(webhook_data),
            signature="signature1"
        )
        
        # Second webhook processing (duplicate)
        result2 = await webhook_processor.process_stripe_webhook(
            payload=json.dumps(webhook_data),
            signature="signature2"
        )
        
        assert result1["processed"] is True
        assert result2["processed"] is True
        assert result2["duplicate"] is True
        
        # Should only process once
        assert mock_stripe_handler.process_event.call_count == 1
    
    @pytest.mark.asyncio
    async def test_webhook_retry_mechanism(self, webhook_processor, mock_stripe_handler):
        """Test webhook retry mechanism for failed processing"""
        webhook_data = {
            "id": "evt_retry_test",
            "type": "payment_intent.failed",
            "data": {"object": {"id": "pi_retry_test"}}
        }
        
        mock_stripe_handler.verify_webhook_signature.return_value = True
        
        # First attempt fails
        mock_stripe_handler.process_event.side_effect = [
            Exception("Database connection failed"),
            {"processed": True}  # Second attempt succeeds
        ]
        
        result = await webhook_processor.process_stripe_webhook(
            payload=json.dumps(webhook_data),
            signature="signature",
            max_retries=1,
            retry_delay=0.1
        )
        
        assert result["processed"] is True
        assert result["retry_count"] == 1
        assert mock_stripe_handler.process_event.call_count == 2


class TestStripeWebhookHandler:
    """Test Stripe-specific webhook handling"""
    
    def test_verify_webhook_signature(self, stripe_webhook_handler):
        """Test Stripe webhook signature verification"""
        # Mock webhook data
        payload = '{"id":"evt_test","type":"test.event"}'
        secret = "whsec_test_secret"
        timestamp = str(int(datetime.utcnow().timestamp()))
        
        # Create valid signature
        signed_payload = f"{timestamp}.{payload}"
        signature = hmac.new(
            secret.encode('utf-8'),
            signed_payload.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        
        stripe_signature = f"t={timestamp},v1={signature}"
        
        with patch.object(stripe_webhook_handler, 'webhook_secret', secret):
            is_valid = stripe_webhook_handler.verify_webhook_signature(
                payload=payload,
                signature=stripe_signature
            )
        
        assert is_valid is True
    
    @pytest.mark.asyncio
    async def test_payment_intent_succeeded(self, stripe_webhook_handler, test_customer, db_session):
        """Test processing payment_intent.succeeded event"""
        event_data = {
            "id": "evt_payment_succeeded",
            "type": "payment_intent.succeeded",
            "data": {
                "object": {
                    "id": "pi_test_succeeded",
                    "status": "succeeded",
                    "amount": 2999,  # $29.99 in cents
                    "currency": "usd",
                    "customer": test_customer.id,
                    "charges": {
                        "data": [{
                            "balance_transaction": {
                                "fee": 117  # $1.17 in cents
                            }
                        }]
                    }
                }
            }
        }
        
        with patch.object(stripe_webhook_handler, 'payment_service') as mock_payment_service:
            mock_payment = Mock()
            mock_payment.id = "pay_123"
            mock_payment.status = PaymentStatus.SUCCEEDED
            mock_payment_service.update_payment_from_stripe.return_value = mock_payment
            
            result = await stripe_webhook_handler.process_event(event_data)
            
            assert result["processed"] is True
            assert result["event_type"] == "payment_intent.succeeded"
            mock_payment_service.update_payment_from_stripe.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_payment_intent_failed(self, stripe_webhook_handler, test_customer, db_session):
        """Test processing payment_intent.payment_failed event"""
        event_data = {
            "id": "evt_payment_failed",
            "type": "payment_intent.payment_failed",
            "data": {
                "object": {
                    "id": "pi_test_failed",
                    "status": "requires_payment_method",
                    "amount": 5000,
                    "currency": "eur",
                    "customer": test_customer.id,
                    "last_payment_error": {
                        "code": "card_declined",
                        "message": "Your card was declined."
                    }
                }
            }
        }
        
        with patch.object(stripe_webhook_handler, 'payment_service') as mock_payment_service:
            mock_payment = Mock()
            mock_payment.id = "pay_456"
            mock_payment.status = PaymentStatus.FAILED
            mock_payment_service.update_payment_from_stripe.return_value = mock_payment
            
            result = await stripe_webhook_handler.process_event(event_data)
            
            assert result["processed"] is True
            assert result["event_type"] == "payment_intent.payment_failed"
            mock_payment_service.update_payment_from_stripe.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_invoice_payment_succeeded(self, stripe_webhook_handler, test_subscription_active, test_invoice_draft, db_session):
        """Test processing invoice.payment_succeeded event"""
        event_data = {
            "id": "evt_invoice_paid",
            "type": "invoice.payment_succeeded",
            "data": {
                "object": {
                    "id": "in_stripe_test",
                    "status": "paid",
                    "subscription": test_subscription_active.id,
                    "amount_paid": 2999,
                    "currency": "usd",
                    "customer": test_subscription_active.customer_id
                }
            }
        }
        
        with patch.object(stripe_webhook_handler, 'invoice_service') as mock_invoice_service:
            mock_invoice = Mock()
            mock_invoice.id = test_invoice_draft.id
            mock_invoice.status = InvoiceStatus.PAID
            mock_invoice_service.mark_invoice_paid_from_stripe.return_value = mock_invoice
            
            result = await stripe_webhook_handler.process_event(event_data)
            
            assert result["processed"] is True
            assert result["event_type"] == "invoice.payment_succeeded"
            mock_invoice_service.mark_invoice_paid_from_stripe.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_customer_subscription_updated(self, stripe_webhook_handler, test_subscription_active, db_session):
        """Test processing customer.subscription.updated event"""
        event_data = {
            "id": "evt_subscription_updated",
            "type": "customer.subscription.updated",
            "data": {
                "object": {
                    "id": test_subscription_active.id,
                    "status": "active",
                    "current_period_start": int(datetime.utcnow().timestamp()),
                    "current_period_end": int((datetime.utcnow() + timedelta(days=30)).timestamp()),
                    "cancel_at_period_end": False
                }
            }
        }
        
        with patch.object(stripe_webhook_handler, 'subscription_service') as mock_subscription_service:
            mock_subscription = Mock()
            mock_subscription.id = test_subscription_active.id
            mock_subscription.status = SubscriptionStatus.ACTIVE
            mock_subscription_service.update_subscription_from_stripe.return_value = mock_subscription
            
            result = await stripe_webhook_handler.process_event(event_data)
            
            assert result["processed"] is True
            assert result["event_type"] == "customer.subscription.updated"
            mock_subscription_service.update_subscription_from_stripe.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_setup_intent_succeeded(self, stripe_webhook_handler, test_customer, db_session):
        """Test processing setup_intent.succeeded event"""
        event_data = {
            "id": "evt_setup_succeeded",
            "type": "setup_intent.succeeded",
            "data": {
                "object": {
                    "id": "seti_test_succeeded",
                    "status": "succeeded",
                    "customer": test_customer.id,
                    "payment_method": "pm_test_card"
                }
            }
        }
        
        with patch.object(stripe_webhook_handler, 'payment_method_service') as mock_pm_service:
            mock_payment_method = Mock()
            mock_payment_method.id = "pm_123"
            mock_payment_method.customer_id = test_customer.id
            mock_pm_service.activate_payment_method_from_stripe.return_value = mock_payment_method
            
            result = await stripe_webhook_handler.process_event(event_data)
            
            assert result["processed"] is True
            assert result["event_type"] == "setup_intent.succeeded"
            mock_pm_service.activate_payment_method_from_stripe.assert_called_once()


class TestPayPalWebhookHandler:
    """Test PayPal-specific webhook handling"""
    
    def test_verify_webhook_signature(self, paypal_webhook_handler):
        """Test PayPal webhook signature verification"""
        # Mock webhook data
        payload = '{"id":"WH-test","event_type":"PAYMENT.CAPTURE.COMPLETED"}'
        
        # Mock PayPal signature verification
        headers = {
            "paypal-transmission-id": "test-transmission-id",
            "paypal-cert-id": "test-cert-id",
            "paypal-transmission-sig": "test-signature",
            "paypal-transmission-time": "2025-01-27T10:00:00Z"
        }
        
        with patch('paypalrestsdk.WebhookEvent.verify') as mock_verify:
            mock_verify.return_value = True
            
            is_valid = paypal_webhook_handler.verify_webhook_signature(
                payload=payload,
                headers=headers
            )
        
        assert is_valid is True
        mock_verify.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_payment_capture_completed(self, paypal_webhook_handler, test_customer, db_session):
        """Test processing PAYMENT.CAPTURE.COMPLETED event"""
        event_data = {
            "id": "WH-payment-completed",
            "event_type": "PAYMENT.CAPTURE.COMPLETED",
            "resource": {
                "id": "PAYID-completed-test",
                "status": "COMPLETED",
                "amount": {
                    "value": "29.99",
                    "currency_code": "EUR"
                },
                "custom_id": test_customer.id,
                "seller_receivable_breakdown": {
                    "paypal_fee": {
                        "value": "1.17",
                        "currency_code": "EUR"
                    }
                }
            }
        }
        
        with patch.object(paypal_webhook_handler, 'payment_service') as mock_payment_service:
            mock_payment = Mock()
            mock_payment.id = "pay_paypal_123"
            mock_payment.status = PaymentStatus.SUCCEEDED
            mock_payment_service.update_payment_from_paypal.return_value = mock_payment
            
            result = await paypal_webhook_handler.process_event(event_data)
            
            assert result["processed"] is True
            assert result["event_type"] == "PAYMENT.CAPTURE.COMPLETED"
            mock_payment_service.update_payment_from_paypal.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_payment_capture_denied(self, paypal_webhook_handler, test_customer, db_session):
        """Test processing PAYMENT.CAPTURE.DENIED event"""
        event_data = {
            "id": "WH-payment-denied",
            "event_type": "PAYMENT.CAPTURE.DENIED",
            "resource": {
                "id": "PAYID-denied-test",
                "status": "DENIED",
                "amount": {
                    "value": "50.00",
                    "currency_code": "USD"
                },
                "custom_id": test_customer.id,
                "status_details": {
                    "reason": "DECLINED_BY_PROCESSOR"
                }
            }
        }
        
        with patch.object(paypal_webhook_handler, 'payment_service') as mock_payment_service:
            mock_payment = Mock()
            mock_payment.id = "pay_paypal_456"
            mock_payment.status = PaymentStatus.FAILED
            mock_payment_service.update_payment_from_paypal.return_value = mock_payment
            
            result = await paypal_webhook_handler.process_event(event_data)
            
            assert result["processed"] is True
            assert result["event_type"] == "PAYMENT.CAPTURE.DENIED"
            mock_payment_service.update_payment_from_paypal.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_billing_subscription_activated(self, paypal_webhook_handler, test_subscription_active, db_session):
        """Test processing BILLING.SUBSCRIPTION.ACTIVATED event"""
        event_data = {
            "id": "WH-subscription-activated",
            "event_type": "BILLING.SUBSCRIPTION.ACTIVATED",
            "resource": {
                "id": test_subscription_active.id,
                "status": "ACTIVE",
                "subscriber": {
                    "name": {
                        "given_name": "John",
                        "surname": "Doe"
                    },
                    "email_address": test_subscription_active.customer.email
                },
                "billing_info": {
                    "next_billing_time": "2025-02-27T10:00:00Z"
                }
            }
        }
        
        with patch.object(paypal_webhook_handler, 'subscription_service') as mock_subscription_service:
            mock_subscription = Mock()
            mock_subscription.id = test_subscription_active.id
            mock_subscription.status = SubscriptionStatus.ACTIVE
            mock_subscription_service.update_subscription_from_paypal.return_value = mock_subscription
            
            result = await paypal_webhook_handler.process_event(event_data)
            
            assert result["processed"] is True
            assert result["event_type"] == "BILLING.SUBSCRIPTION.ACTIVATED"
            mock_subscription_service.update_subscription_from_paypal.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_billing_subscription_cancelled(self, paypal_webhook_handler, test_subscription_active, db_session):
        """Test processing BILLING.SUBSCRIPTION.CANCELLED event"""
        event_data = {
            "id": "WH-subscription-cancelled",
            "event_type": "BILLING.SUBSCRIPTION.CANCELLED",
            "resource": {
                "id": test_subscription_active.id,
                "status": "CANCELLED",
                "status_update_time": "2025-01-27T10:00:00Z"
            }
        }
        
        with patch.object(paypal_webhook_handler, 'subscription_service') as mock_subscription_service:
            mock_subscription = Mock()
            mock_subscription.id = test_subscription_active.id
            mock_subscription.status = SubscriptionStatus.CANCELLED
            mock_subscription_service.update_subscription_from_paypal.return_value = mock_subscription
            
            result = await paypal_webhook_handler.process_event(event_data)
            
            assert result["processed"] is True
            assert result["event_type"] == "BILLING.SUBSCRIPTION.CANCELLED"
            mock_subscription_service.update_subscription_from_paypal.assert_called_once()


class TestWebhookSecurity:
    """Test webhook security features"""
    
    @pytest.mark.asyncio
    async def test_webhook_timestamp_validation(self, stripe_webhook_handler):
        """Test webhook timestamp validation to prevent replay attacks"""
        # Create payload with old timestamp (more than 5 minutes ago)
        old_timestamp = str(int((datetime.utcnow() - timedelta(minutes=10)).timestamp()))
        payload = '{"id":"evt_old","type":"test.event"}'
        secret = "whsec_test_secret"
        
        signed_payload = f"{old_timestamp}.{payload}"
        signature = hmac.new(
            secret.encode('utf-8'),
            signed_payload.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        
        stripe_signature = f"t={old_timestamp},v1={signature}"
        
        with patch.object(stripe_webhook_handler, 'webhook_secret', secret):
            is_valid = stripe_webhook_handler.verify_webhook_signature(
                payload=payload,
                signature=stripe_signature,
                tolerance=300  # 5 minutes
            )
        
        assert is_valid is False
    
    @pytest.mark.asyncio
    async def test_webhook_rate_limiting(self, webhook_processor):
        """Test webhook rate limiting protection"""
        webhook_data = {"id": "evt_rate_limit", "type": "test.event"}
        signature = "test_signature"
        
        # Mock rate limiter
        with patch.object(webhook_processor, 'rate_limiter') as mock_rate_limiter:
            mock_rate_limiter.is_allowed.return_value = False
            
            with pytest.raises(Exception) as exc_info:
                await webhook_processor.process_stripe_webhook(
                    payload=json.dumps(webhook_data),
                    signature=signature
                )
            
            assert "Rate limit exceeded" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_webhook_ip_whitelist(self, webhook_processor):
        """Test webhook IP address whitelisting"""
        webhook_data = {"id": "evt_ip_test", "type": "test.event"}
        
        # Test with allowed IP
        with patch.object(webhook_processor, 'is_ip_allowed') as mock_ip_check:
            mock_ip_check.return_value = True
            
            # Should proceed to signature verification
            with patch.object(webhook_processor.stripe_handler, 'verify_webhook_signature', return_value=False):
                with pytest.raises(Exception) as exc_info:
                    await webhook_processor.process_stripe_webhook(
                        payload=json.dumps(webhook_data),
                        signature="test_sig",
                        source_ip="54.187.174.169"  # Stripe IP
                    )
                
                assert "Invalid webhook signature" in str(exc_info.value)
        
        # Test with disallowed IP
        with patch.object(webhook_processor, 'is_ip_allowed') as mock_ip_check:
            mock_ip_check.return_value = False
            
            with pytest.raises(Exception) as exc_info:
                await webhook_processor.process_stripe_webhook(
                    payload=json.dumps(webhook_data),
                    signature="test_sig",
                    source_ip="192.168.1.1"  # Invalid IP
                )
            
            assert "IP address not allowed" in str(exc_info.value)


class TestWebhookLogging:
    """Test webhook logging and monitoring"""
    
    @pytest.mark.asyncio
    async def test_webhook_event_logging(self, webhook_processor, mock_stripe_handler, mock_logger):
        """Test webhook event logging"""
        webhook_data = {
            "id": "evt_logging_test",
            "type": "payment_intent.succeeded",
            "data": {"object": {"id": "pi_logging_test"}}
        }
        
        mock_stripe_handler.verify_webhook_signature.return_value = True
        mock_stripe_handler.process_event.return_value = {"processed": True}
        
        await webhook_processor.process_stripe_webhook(
            payload=json.dumps(webhook_data),
            signature="test_signature"
        )
        
        # Verify logging calls
        mock_logger.info.assert_called()
        log_calls = mock_logger.info.call_args_list
        assert any("webhook received" in str(call).lower() for call in log_calls)
        assert any("webhook processed" in str(call).lower() for call in log_calls)
    
    @pytest.mark.asyncio
    async def test_webhook_error_logging(self, webhook_processor, mock_stripe_handler, mock_logger):
        """Test webhook error logging"""
        webhook_data = {"id": "evt_error_test", "type": "test.event"}
        
        mock_stripe_handler.verify_webhook_signature.return_value = True
        mock_stripe_handler.process_event.side_effect = Exception("Processing failed")
        
        with pytest.raises(Exception):
            await webhook_processor.process_stripe_webhook(
                payload=json.dumps(webhook_data),
                signature="test_signature"
            )
        
        # Verify error logging
        mock_logger.error.assert_called()
        error_calls = mock_logger.error.call_args_list
        assert any("webhook processing failed" in str(call).lower() for call in error_calls)
    
    @pytest.mark.asyncio
    async def test_webhook_metrics_collection(self, webhook_processor, mock_metrics):
        """Test webhook metrics collection"""
        webhook_data = {"id": "evt_metrics_test", "type": "payment_intent.succeeded"}
        
        with patch.object(webhook_processor.stripe_handler, 'verify_webhook_signature', return_value=True):
            with patch.object(webhook_processor.stripe_handler, 'process_event', return_value={"processed": True}):
                await webhook_processor.process_stripe_webhook(
                    payload=json.dumps(webhook_data),
                    signature="test_signature"
                )
        
        # Verify metrics collection
        mock_metrics.increment.assert_called()
        metric_calls = mock_metrics.increment.call_args_list
        assert any("webhook.received" in str(call) for call in metric_calls)
        assert any("webhook.processed" in str(call) for call in metric_calls)
