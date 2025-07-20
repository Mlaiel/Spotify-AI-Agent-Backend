"""
Tests for Core Billing Engine
============================

Comprehensive tests for the core billing functionality.
"""

import pytest
from datetime import datetime, timedelta
from decimal import Decimal
from unittest.mock import AsyncMock, Mock, patch, MagicMock
import asyncio

from billing.core import BillingEngine, PaymentProcessor, TaxCalculator, FraudDetection
from billing.models import (
    Customer, Plan, Subscription, Payment, Invoice, PaymentMethod,
    SubscriptionStatus, PaymentStatus, InvoiceStatus, PaymentProvider,
    PlanInterval, CustomerStatus
)


class TestBillingEngine:
    """Test core billing engine functionality"""
    
    @pytest.mark.asyncio
    async def test_create_customer(self, billing_engine, db_session):
        """Test customer creation"""
        customer_data = {
            "email": "test@billingengine.com",
            "name": "Test Customer",
            "company": "Test Company",
            "preferred_currency": "EUR",
            "preferred_language": "en",
            "country": "FR"
        }
        
        customer = await billing_engine.create_customer(customer_data)
        
        assert customer.id is not None
        assert customer.email == "test@billingengine.com"
        assert customer.status == CustomerStatus.ACTIVE
        assert customer.preferred_currency == "EUR"
        assert customer.country == "FR"
    
    @pytest.mark.asyncio
    async def test_create_subscription(self, billing_engine, test_customer, test_plan_monthly):
        """Test subscription creation"""
        subscription_data = {
            "customer_id": test_customer.id,
            "plan_id": test_plan_monthly.id,
            "trial_period_days": 14,
            "proration_behavior": "create_prorations"
        }
        
        subscription = await billing_engine.create_subscription(subscription_data)
        
        assert subscription.id is not None
        assert subscription.customer_id == test_customer.id
        assert subscription.plan_id == test_plan_monthly.id
        assert subscription.status == SubscriptionStatus.TRIAL
        assert subscription.is_in_trial is True
    
    @pytest.mark.asyncio
    async def test_subscription_upgrade(self, billing_engine, test_subscription_active, test_plan_yearly):
        """Test subscription plan upgrade"""
        original_amount = test_subscription_active.effective_amount
        
        updated_subscription = await billing_engine.upgrade_subscription(
            test_subscription_active.id,
            test_plan_yearly.id,
            proration_behavior="immediate"
        )
        
        assert updated_subscription.plan_id == test_plan_yearly.id
        assert updated_subscription.effective_amount != original_amount
    
    @pytest.mark.asyncio
    async def test_subscription_cancellation(self, billing_engine, test_subscription_active):
        """Test subscription cancellation"""
        reason = "Customer requested cancellation"
        
        cancelled_subscription = await billing_engine.cancel_subscription(
            test_subscription_active.id,
            cancel_at_period_end=True,
            reason=reason
        )
        
        assert cancelled_subscription.status == SubscriptionStatus.CANCELING
        assert cancelled_subscription.cancel_at_period_end is True
        assert cancelled_subscription.cancellation_reason == reason
    
    @pytest.mark.asyncio
    async def test_subscription_reactivation(self, billing_engine, db_session):
        """Test subscription reactivation"""
        # First cancel a subscription
        cancelled_subscription = await billing_engine.cancel_subscription(
            test_subscription_active.id,
            cancel_at_period_end=False
        )
        
        # Then reactivate it
        reactivated_subscription = await billing_engine.reactivate_subscription(
            cancelled_subscription.id
        )
        
        assert reactivated_subscription.status == SubscriptionStatus.ACTIVE
        assert reactivated_subscription.cancel_at_period_end is False
        assert reactivated_subscription.cancelled_at is None
    
    @pytest.mark.asyncio
    async def test_invoice_generation(self, billing_engine, test_subscription_active):
        """Test automatic invoice generation"""
        invoice = await billing_engine.generate_invoice(
            subscription_id=test_subscription_active.id,
            period_start=datetime.utcnow(),
            period_end=datetime.utcnow() + timedelta(days=30)
        )
        
        assert invoice is not None
        assert invoice.subscription_id == test_subscription_active.id
        assert invoice.status == InvoiceStatus.OPEN
        assert invoice.total > Decimal('0')
        assert len(invoice.line_items) > 0
    
    @pytest.mark.asyncio
    async def test_invoice_payment_processing(self, billing_engine, test_invoice_draft, test_payment_method_stripe):
        """Test invoice payment processing"""
        # Finalize invoice first
        invoice = await billing_engine.finalize_invoice(test_invoice_draft.id)
        
        # Process payment
        payment = await billing_engine.pay_invoice(
            invoice.id,
            payment_method_id=test_payment_method_stripe.id
        )
        
        assert payment.status == PaymentStatus.SUCCEEDED
        assert invoice.status == InvoiceStatus.PAID
        assert invoice.amount_due == Decimal('0.00')
    
    @pytest.mark.asyncio
    async def test_usage_based_billing(self, billing_engine, test_subscription_active):
        """Test usage-based billing calculations"""
        usage_data = [
            {
                "metric_name": "api_calls",
                "quantity": 1500,
                "unit": "calls",
                "unit_price": 0.01
            },
            {
                "metric_name": "storage",
                "quantity": 50,
                "unit": "GB",
                "unit_price": 0.5
            }
        ]
        
        invoice = await billing_engine.bill_usage(
            test_subscription_active.id,
            usage_data,
            billing_period_start=datetime.utcnow() - timedelta(days=30),
            billing_period_end=datetime.utcnow()
        )
        
        assert invoice is not None
        assert len(invoice.line_items) == 2
        # API calls: 1500 * 0.01 = 15.00
        # Storage: 50 * 0.5 = 25.00
        expected_subtotal = Decimal('40.00')
        assert invoice.subtotal == expected_subtotal
    
    @pytest.mark.asyncio
    async def test_proration_calculations(self, billing_engine, test_subscription_active, test_plan_yearly):
        """Test proration calculations for plan changes"""
        # Change plan mid-cycle
        days_into_cycle = 15
        test_subscription_active.current_period_start = datetime.utcnow() - timedelta(days=days_into_cycle)
        test_subscription_active.current_period_end = datetime.utcnow() + timedelta(days=15)
        
        proration = await billing_engine.calculate_proration(
            test_subscription_active.id,
            test_plan_yearly.id,
            change_date=datetime.utcnow()
        )
        
        assert proration["credit_amount"] > Decimal('0')
        assert proration["charge_amount"] > Decimal('0')
        assert proration["net_amount"] != Decimal('0')
    
    @pytest.mark.asyncio
    async def test_dunning_management(self, billing_engine, test_customer):
        """Test dunning management for failed payments"""
        # Create failed payment
        failed_payment = Payment(
            customer_id=test_customer.id,
            provider=PaymentProvider.STRIPE,
            provider_transaction_id="pi_failed",
            amount=Decimal('29.99'),
            currency="EUR",
            status=PaymentStatus.FAILED,
            failure_reason="Insufficient funds"
        )
        
        # Process dunning
        dunning_result = await billing_engine.process_dunning(failed_payment.id)
        
        assert dunning_result["retry_scheduled"] is True
        assert dunning_result["next_retry_date"] is not None
        assert dunning_result["retry_count"] == 1


class TestPaymentProcessor:
    """Test payment processing functionality"""
    
    @pytest.mark.asyncio
    async def test_stripe_payment_processing(self, payment_processor, test_customer, test_payment_method_stripe):
        """Test Stripe payment processing"""
        payment_data = {
            "amount": Decimal('50.00'),
            "currency": "EUR",
            "customer_id": test_customer.id,
            "payment_method_id": test_payment_method_stripe.id,
            "description": "Test payment"
        }
        
        with patch('stripe.PaymentIntent.create') as mock_create:
            mock_create.return_value = Mock(
                id="pi_test_success",
                status="succeeded",
                amount=5000,  # Stripe uses cents
                currency="eur",
                charges=Mock(data=[Mock(
                    balance_transaction=Mock(fee=150)  # 1.50 EUR in cents
                )])
            )
            
            payment = await payment_processor.process_payment(payment_data)
            
            assert payment.status == PaymentStatus.SUCCEEDED
            assert payment.provider_transaction_id == "pi_test_success"
            assert payment.provider_fee == Decimal('1.50')
    
    @pytest.mark.asyncio
    async def test_paypal_payment_processing(self, payment_processor, test_customer, test_payment_method_paypal):
        """Test PayPal payment processing"""
        payment_data = {
            "amount": Decimal('75.00'),
            "currency": "EUR",
            "customer_id": test_customer.id,
            "payment_method_id": test_payment_method_paypal.id,
            "description": "PayPal test payment"
        }
        
        with patch('paypalrestsdk.Payment') as mock_payment:
            mock_instance = Mock()
            mock_instance.create.return_value = True
            mock_instance.id = "PAYID-TEST-SUCCESS"
            mock_instance.state = "approved"
            mock_payment.return_value = mock_instance
            
            payment = await payment_processor.process_payment(payment_data)
            
            assert payment.status == PaymentStatus.SUCCEEDED
            assert payment.provider_transaction_id == "PAYID-TEST-SUCCESS"
    
    @pytest.mark.asyncio
    async def test_payment_retry_logic(self, payment_processor, test_customer):
        """Test payment retry logic for failed payments"""
        payment_data = {
            "amount": Decimal('25.00'),
            "currency": "EUR",
            "customer_id": test_customer.id,
            "description": "Retry test payment"
        }
        
        with patch('stripe.PaymentIntent.create') as mock_create:
            # First attempt fails
            mock_create.side_effect = [
                Exception("card_declined"),
                Mock(id="pi_retry_success", status="succeeded", amount=2500, currency="eur")
            ]
            
            payment = await payment_processor.process_payment_with_retry(
                payment_data,
                max_retries=1,
                retry_delay=0.1
            )
            
            assert payment.status == PaymentStatus.SUCCEEDED
            assert mock_create.call_count == 2
    
    @pytest.mark.asyncio
    async def test_refund_processing(self, payment_processor, test_payment_successful):
        """Test payment refund processing"""
        refund_amount = Decimal('10.00')
        reason = "Customer request"
        
        with patch('stripe.Refund.create') as mock_refund:
            mock_refund.return_value = Mock(
                id="re_test_refund",
                status="succeeded",
                amount=1000  # 10.00 EUR in cents
            )
            
            refund = await payment_processor.process_refund(
                test_payment_successful.id,
                refund_amount,
                reason
            )
            
            assert refund["status"] == "succeeded"
            assert refund["amount"] == refund_amount
            assert test_payment_successful.refunded_amount == refund_amount
    
    @pytest.mark.asyncio
    async def test_payment_method_validation(self, payment_processor, test_customer):
        """Test payment method validation"""
        # Valid card
        valid_card_data = {
            "customer_id": test_customer.id,
            "provider": PaymentProvider.STRIPE,
            "type": "card",
            "card_number": "4242424242424242",
            "exp_month": 12,
            "exp_year": 2025,
            "cvc": "123"
        }
        
        with patch('stripe.PaymentMethod.create') as mock_create:
            mock_create.return_value = Mock(
                id="pm_valid_card",
                type="card",
                card=Mock(
                    last4="4242",
                    brand="visa",
                    exp_month=12,
                    exp_year=2025
                )
            )
            
            payment_method = await payment_processor.add_payment_method(valid_card_data)
            
            assert payment_method.provider_payment_method_id == "pm_valid_card"
            assert payment_method.last4 == "4242"
            assert payment_method.brand == "visa"


class TestTaxCalculator:
    """Test tax calculation functionality"""
    
    @pytest.mark.asyncio
    async def test_eu_vat_calculation(self, tax_calculator):
        """Test EU VAT calculation"""
        tax_data = {
            "amount": Decimal('100.00'),
            "customer_country": "FR",
            "business_country": "FR",
            "product_type": "digital_service",
            "customer_vat_number": None
        }
        
        tax_result = await tax_calculator.calculate_tax(tax_data)
        
        assert tax_result["tax_rate"] == Decimal('0.20')  # 20% French VAT
        assert tax_result["tax_amount"] == Decimal('20.00')
        assert tax_result["total_amount"] == Decimal('120.00')
        assert tax_result["tax_type"] == "VAT"
    
    @pytest.mark.asyncio
    async def test_us_sales_tax_calculation(self, tax_calculator):
        """Test US sales tax calculation"""
        tax_data = {
            "amount": Decimal('50.00'),
            "customer_country": "US",
            "customer_state": "CA",
            "business_country": "US",
            "business_state": "CA",
            "product_type": "digital_service"
        }
        
        tax_result = await tax_calculator.calculate_tax(tax_data)
        
        assert tax_result["tax_rate"] > Decimal('0')  # California has sales tax
        assert tax_result["tax_amount"] > Decimal('0')
        assert tax_result["tax_type"] == "SALES_TAX"
    
    @pytest.mark.asyncio
    async def test_reverse_charge_calculation(self, tax_calculator):
        """Test reverse charge for EU B2B transactions"""
        tax_data = {
            "amount": Decimal('200.00'),
            "customer_country": "DE",
            "business_country": "FR",
            "product_type": "digital_service",
            "customer_vat_number": "DE123456789",
            "customer_is_business": True
        }
        
        tax_result = await tax_calculator.calculate_tax(tax_data)
        
        assert tax_result["tax_rate"] == Decimal('0.00')  # Reverse charge
        assert tax_result["tax_amount"] == Decimal('0.00')
        assert tax_result["reverse_charge"] is True
        assert tax_result["tax_type"] == "VAT"
    
    @pytest.mark.asyncio
    async def test_tax_exemption(self, tax_calculator):
        """Test tax exemption handling"""
        tax_data = {
            "amount": Decimal('100.00'),
            "customer_country": "US",
            "customer_state": "DE",  # Delaware has no sales tax
            "business_country": "US",
            "product_type": "digital_service"
        }
        
        tax_result = await tax_calculator.calculate_tax(tax_data)
        
        assert tax_result["tax_rate"] == Decimal('0.00')
        assert tax_result["tax_amount"] == Decimal('0.00')
        assert tax_result["tax_exempt"] is True


class TestFraudDetection:
    """Test fraud detection system"""
    
    @pytest.mark.asyncio
    async def test_low_risk_transaction(self, fraud_detector, test_customer):
        """Test low-risk transaction scoring"""
        transaction_data = {
            "customer_id": test_customer.id,
            "amount": Decimal('29.99'),
            "currency": "EUR",
            "ip_address": "192.168.1.1",
            "user_agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "payment_method_type": "card",
            "country": "FR"
        }
        
        risk_score = await fraud_detector.calculate_risk_score(transaction_data)
        
        assert risk_score <= Decimal('0.3')  # Low risk threshold
        assert fraud_detector.is_high_risk(risk_score) is False
    
    @pytest.mark.asyncio
    async def test_high_risk_transaction(self, fraud_detector, test_customer):
        """Test high-risk transaction scoring"""
        transaction_data = {
            "customer_id": test_customer.id,
            "amount": Decimal('5000.00'),  # Large amount
            "currency": "EUR",
            "ip_address": "198.51.100.1",  # Different country IP
            "user_agent": "Unknown",
            "payment_method_type": "card",
            "country": "NG",  # High-risk country
            "velocity_check": {
                "transactions_last_hour": 10,
                "amount_last_hour": Decimal('10000.00')
            }
        }
        
        risk_score = await fraud_detector.calculate_risk_score(transaction_data)
        
        assert risk_score >= Decimal('0.7')  # High risk threshold
        assert fraud_detector.is_high_risk(risk_score) is True
    
    @pytest.mark.asyncio
    async def test_velocity_checks(self, fraud_detector, test_customer):
        """Test transaction velocity fraud checks"""
        # Simulate rapid transactions
        for i in range(5):
            transaction_data = {
                "customer_id": test_customer.id,
                "amount": Decimal('100.00'),
                "currency": "EUR",
                "timestamp": datetime.utcnow()
            }
            await fraud_detector.record_transaction(transaction_data)
        
        velocity_score = await fraud_detector.check_velocity(test_customer.id)
        
        assert velocity_score > Decimal('0.5')  # Should flag rapid transactions
    
    @pytest.mark.asyncio
    async def test_ml_fraud_prediction(self, fraud_detector):
        """Test ML-based fraud prediction"""
        features = {
            "amount": 250.0,
            "hour_of_day": 3,  # Unusual hour
            "country_risk_score": 0.8,
            "customer_age_days": 1,  # New customer
            "payment_method_age_days": 0,  # New payment method
            "previous_failed_payments": 2
        }
        
        with patch.object(fraud_detector, 'ml_model') as mock_model:
            mock_model.predict_proba.return_value = [[0.1, 0.9]]  # High fraud probability
            
            fraud_probability = await fraud_detector.predict_fraud(features)
            
            assert fraud_probability >= 0.8
            mock_model.predict_proba.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_device_fingerprinting(self, fraud_detector):
        """Test device fingerprinting for fraud detection"""
        device_data = {
            "user_agent": "Mozilla/5.0 (iPhone; CPU iPhone OS 14_0 like Mac OS X)",
            "screen_resolution": "375x812",
            "timezone": "Europe/Paris",
            "language": "fr-FR",
            "plugins": ["PDF Viewer", "Chrome PDF Plugin"]
        }
        
        device_fingerprint = await fraud_detector.generate_device_fingerprint(device_data)
        
        assert device_fingerprint is not None
        assert len(device_fingerprint) == 64  # SHA-256 hash
        
        # Same device data should generate same fingerprint
        duplicate_fingerprint = await fraud_detector.generate_device_fingerprint(device_data)
        assert device_fingerprint == duplicate_fingerprint


class TestBillingEngineIntegration:
    """Integration tests for billing engine components"""
    
    @pytest.mark.asyncio
    async def test_end_to_end_subscription_billing(self, billing_engine, test_customer, test_plan_monthly, test_payment_method_stripe):
        """Test complete subscription billing flow"""
        # 1. Create subscription
        subscription_data = {
            "customer_id": test_customer.id,
            "plan_id": test_plan_monthly.id,
            "payment_method_id": test_payment_method_stripe.id,
            "trial_period_days": 0  # No trial
        }
        
        subscription = await billing_engine.create_subscription(subscription_data)
        assert subscription.status == SubscriptionStatus.ACTIVE
        
        # 2. Generate invoice
        invoice = await billing_engine.generate_invoice(
            subscription_id=subscription.id,
            period_start=datetime.utcnow(),
            period_end=datetime.utcnow() + timedelta(days=30)
        )
        assert invoice.status == InvoiceStatus.OPEN
        
        # 3. Process payment
        with patch('stripe.PaymentIntent.create') as mock_payment:
            mock_payment.return_value = Mock(
                id="pi_integration_test",
                status="succeeded",
                amount=int(invoice.total * 100),
                currency="eur"
            )
            
            payment = await billing_engine.pay_invoice(
                invoice.id,
                payment_method_id=test_payment_method_stripe.id
            )
            
            assert payment.status == PaymentStatus.SUCCEEDED
            assert invoice.status == InvoiceStatus.PAID
    
    @pytest.mark.asyncio
    async def test_subscription_lifecycle_management(self, billing_engine, test_customer, test_plan_monthly, test_plan_yearly):
        """Test complete subscription lifecycle"""
        # 1. Create trial subscription
        subscription = await billing_engine.create_subscription({
            "customer_id": test_customer.id,
            "plan_id": test_plan_monthly.id,
            "trial_period_days": 14
        })
        assert subscription.status == SubscriptionStatus.TRIAL
        
        # 2. Convert trial to active
        subscription = await billing_engine.convert_trial_to_active(subscription.id)
        assert subscription.status == SubscriptionStatus.ACTIVE
        
        # 3. Upgrade plan
        subscription = await billing_engine.upgrade_subscription(
            subscription.id,
            test_plan_yearly.id
        )
        assert subscription.plan_id == test_plan_yearly.id
        
        # 4. Cancel subscription
        subscription = await billing_engine.cancel_subscription(
            subscription.id,
            cancel_at_period_end=True
        )
        assert subscription.status == SubscriptionStatus.CANCELING
        
        # 5. Reactivate subscription
        subscription = await billing_engine.reactivate_subscription(subscription.id)
        assert subscription.status == SubscriptionStatus.ACTIVE
