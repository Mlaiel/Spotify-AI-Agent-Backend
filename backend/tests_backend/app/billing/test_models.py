"""
Tests for Billing Models
========================

Comprehensive tests for SQLAlchemy models in the billing system.
"""

import pytest
from datetime import datetime, timedelta
from decimal import Decimal
import uuid

from billing.models import (
    Customer, Plan, Subscription, Payment, Invoice, PaymentMethod, TaxRate, UsageRecord,
    CustomerStatus, SubscriptionStatus, PaymentStatus, InvoiceStatus,
    PaymentProvider, PlanInterval, TaxType
)


class TestCustomerModel:
    """Test Customer model functionality"""
    
    def test_create_customer(self, db_session):
        """Test basic customer creation"""
        customer = Customer(
            email="create@example.com",
            name="Create Test Customer",
            company="Create Test Company",
            preferred_currency="EUR",
            preferred_language="en"
        )
        db_session.add(customer)
        db_session.commit()
        
        assert customer.id is not None
        assert customer.email == "create@example.com"
        assert customer.status == CustomerStatus.ACTIVE
        assert customer.is_active is True
        assert customer.credit_limit == Decimal('0')
        assert customer.payment_terms == 30
    
    def test_customer_email_validation(self, db_session):
        """Test email format validation"""
        customer = Customer(
            email="INVALID@EXAMPLE.COM",
            name="Test Customer"
        )
        db_session.add(customer)
        db_session.commit()
        
        # Email should be stored in lowercase
        assert customer.email == "invalid@example.com"
    
    def test_customer_email_uniqueness(self, db_session, test_customer):
        """Test email uniqueness constraint"""
        duplicate_customer = Customer(
            email=test_customer.email,
            name="Duplicate Customer"
        )
        db_session.add(duplicate_customer)
        
        with pytest.raises(Exception):  # Should raise IntegrityError
            db_session.commit()
    
    def test_customer_country_validation(self, db_session):
        """Test country code validation"""
        customer = Customer(
            email="country@example.com",
            name="Country Test",
            country="fr"
        )
        db_session.add(customer)
        db_session.commit()
        
        # Country should be uppercase
        assert customer.country == "FR"
    
    def test_customer_full_address_property(self, test_customer):
        """Test full address formatted property"""
        test_customer.address_line1 = "123 Main St"
        test_customer.address_line2 = "Apt 4B"
        test_customer.city = "Paris"
        test_customer.state = "Île-de-France"
        test_customer.postal_code = "75001"
        test_customer.country = "FR"
        
        expected = "123 Main St, Apt 4B, Paris, Île-de-France, 75001, FR"
        assert test_customer.full_address == expected
    
    def test_customer_relationships(self, db_session, test_customer, test_plan_monthly):
        """Test customer model relationships"""
        # Create subscription
        subscription = Subscription(
            customer_id=test_customer.id,
            plan_id=test_plan_monthly.id,
            status=SubscriptionStatus.ACTIVE,
            current_period_start=datetime.utcnow(),
            current_period_end=datetime.utcnow() + timedelta(days=30)
        )
        db_session.add(subscription)
        db_session.commit()
        
        # Test relationship
        assert len(test_customer.subscriptions) == 1
        assert test_customer.subscriptions[0].plan_id == test_plan_monthly.id


class TestPlanModel:
    """Test Plan model functionality"""
    
    def test_create_plan(self, db_session):
        """Test basic plan creation"""
        plan = Plan(
            name="Premium Plan",
            description="Premium subscription plan",
            amount=Decimal('99.99'),
            currency="USD",
            interval=PlanInterval.YEAR,
            interval_count=1,
            trial_period_days=30
        )
        db_session.add(plan)
        db_session.commit()
        
        assert plan.id is not None
        assert plan.amount == Decimal('99.99')
        assert plan.is_active is True
        assert plan.features == []
        assert plan.usage_limits == {}
    
    def test_plan_amount_constraints(self, db_session):
        """Test plan amount constraints"""
        # Negative amount should fail
        plan = Plan(
            name="Invalid Plan",
            amount=Decimal('-10.00'),
            currency="EUR"
        )
        db_session.add(plan)
        
        with pytest.raises(Exception):  # Should raise CheckConstraint violation
            db_session.commit()
    
    def test_plan_monthly_amount_calculation(self, test_plan_monthly, test_plan_yearly):
        """Test monthly amount conversion"""
        # Monthly plan
        assert test_plan_monthly.monthly_amount == test_plan_monthly.amount
        
        # Yearly plan - should convert to monthly
        expected_monthly = test_plan_yearly.amount * Decimal('0.083')
        assert abs(test_plan_yearly.monthly_amount - expected_monthly) < Decimal('0.01')
    
    def test_plan_features_and_limits(self, db_session):
        """Test plan features and usage limits"""
        plan = Plan(
            name="Feature Plan",
            amount=Decimal('49.99'),
            currency="EUR",
            features=["api_access", "premium_support", "analytics"],
            usage_limits={
                "api_calls_per_month": 10000,
                "storage_gb": 100,
                "users": 5
            }
        )
        db_session.add(plan)
        db_session.commit()
        
        assert "api_access" in plan.features
        assert plan.usage_limits["api_calls_per_month"] == 10000


class TestSubscriptionModel:
    """Test Subscription model functionality"""
    
    def test_create_subscription(self, db_session, test_customer, test_plan_monthly):
        """Test basic subscription creation"""
        now = datetime.utcnow()
        subscription = Subscription(
            customer_id=test_customer.id,
            plan_id=test_plan_monthly.id,
            status=SubscriptionStatus.TRIAL,
            current_period_start=now,
            current_period_end=now + timedelta(days=30),
            trial_start=now,
            trial_end=now + timedelta(days=14)
        )
        db_session.add(subscription)
        db_session.commit()
        
        assert subscription.id is not None
        assert subscription.customer_id == test_customer.id
        assert subscription.plan_id == test_plan_monthly.id
        assert subscription.is_active is True
        assert subscription.is_in_trial is True
    
    def test_subscription_effective_amount(self, test_subscription_active):
        """Test effective amount calculation with discounts"""
        # No discount
        expected = test_subscription_active.plan.amount
        assert test_subscription_active.effective_amount == expected
        
        # With 20% discount
        test_subscription_active.discount_percent = Decimal('20')
        expected_with_discount = expected * Decimal('0.8')
        assert test_subscription_active.effective_amount == expected_with_discount
        
        # With custom amount
        test_subscription_active.custom_amount = Decimal('19.99')
        test_subscription_active.discount_percent = Decimal('10')
        expected_custom = Decimal('19.99') * Decimal('0.9')
        assert test_subscription_active.effective_amount == expected_custom
    
    def test_subscription_days_until_renewal(self, test_subscription_active):
        """Test days until renewal calculation"""
        # Set period end to 7 days from now
        test_subscription_active.current_period_end = datetime.utcnow() + timedelta(days=7)
        assert test_subscription_active.days_until_renewal == 7
        
        # Past due subscription
        test_subscription_active.current_period_end = datetime.utcnow() - timedelta(days=2)
        assert test_subscription_active.days_until_renewal == 0
    
    def test_subscription_trial_properties(self, test_subscription_trial):
        """Test trial-related properties"""
        assert test_subscription_trial.is_in_trial is True
        assert test_subscription_trial.status == SubscriptionStatus.TRIAL
        
        # Expire trial
        test_subscription_trial.trial_end = datetime.utcnow() - timedelta(days=1)
        assert test_subscription_trial.is_in_trial is False


class TestPaymentMethodModel:
    """Test PaymentMethod model functionality"""
    
    def test_create_payment_method_card(self, db_session, test_customer):
        """Test credit card payment method creation"""
        payment_method = PaymentMethod(
            customer_id=test_customer.id,
            provider=PaymentProvider.STRIPE,
            provider_payment_method_id="pm_card_visa",
            type="card",
            last4="4242",
            brand="visa",
            exp_month=12,
            exp_year=2025,
            is_default=True
        )
        db_session.add(payment_method)
        db_session.commit()
        
        assert payment_method.id is not None
        assert payment_method.provider == PaymentProvider.STRIPE
        assert payment_method.last4 == "4242"
        assert payment_method.is_active is True
    
    def test_payment_method_expiration(self, db_session, test_customer):
        """Test payment method expiration logic"""
        # Expired card
        expired_method = PaymentMethod(
            customer_id=test_customer.id,
            provider=PaymentProvider.STRIPE,
            provider_payment_method_id="pm_expired",
            type="card",
            exp_month=1,
            exp_year=2020
        )
        db_session.add(expired_method)
        db_session.commit()
        
        assert expired_method.is_expired is True
        
        # Valid card
        valid_method = PaymentMethod(
            customer_id=test_customer.id,
            provider=PaymentProvider.STRIPE,
            provider_payment_method_id="pm_valid",
            type="card",
            exp_month=12,
            exp_year=2030
        )
        db_session.add(valid_method)
        db_session.commit()
        
        assert valid_method.is_expired is False
    
    def test_payment_method_uniqueness(self, db_session, test_payment_method_stripe):
        """Test provider payment method ID uniqueness"""
        duplicate_method = PaymentMethod(
            customer_id=test_payment_method_stripe.customer_id,
            provider=test_payment_method_stripe.provider,
            provider_payment_method_id=test_payment_method_stripe.provider_payment_method_id,
            type="card"
        )
        db_session.add(duplicate_method)
        
        with pytest.raises(Exception):  # Should raise UniqueConstraint violation
            db_session.commit()


class TestPaymentModel:
    """Test Payment model functionality"""
    
    def test_create_successful_payment(self, db_session, test_customer, test_payment_method_stripe):
        """Test successful payment creation"""
        payment = Payment(
            customer_id=test_customer.id,
            payment_method_id=test_payment_method_stripe.id,
            provider=PaymentProvider.STRIPE,
            provider_transaction_id="pi_successful_test",
            amount=Decimal('50.00'),
            currency="EUR",
            status=PaymentStatus.SUCCEEDED,
            payment_date=datetime.utcnow(),
            provider_fee=Decimal('1.50'),
            risk_score=Decimal('0.1')
        )
        db_session.add(payment)
        db_session.commit()
        
        assert payment.id is not None
        assert payment.is_successful is True
        assert payment.net_amount == Decimal('48.50')  # 50.00 - 1.50
        assert payment.available_for_refund == Decimal('50.00')
        assert payment.is_high_risk is False
    
    def test_create_failed_payment(self, db_session, test_customer):
        """Test failed payment creation"""
        payment = Payment(
            customer_id=test_customer.id,
            provider=PaymentProvider.STRIPE,
            provider_transaction_id="pi_failed_test",
            amount=Decimal('25.99'),
            currency="EUR",
            status=PaymentStatus.FAILED,
            failure_reason="Insufficient funds"
        )
        db_session.add(payment)
        db_session.commit()
        
        assert payment.is_successful is False
        assert payment.failure_reason == "Insufficient funds"
    
    def test_payment_refund_calculations(self, test_payment_successful):
        """Test payment refund amount calculations"""
        # Initial state
        assert test_payment_successful.available_for_refund == test_payment_successful.amount
        
        # Partial refund
        test_payment_successful.refunded_amount = Decimal('10.00')
        expected_available = test_payment_successful.amount - Decimal('10.00')
        assert test_payment_successful.available_for_refund == expected_available
        
        # Full refund
        test_payment_successful.refunded_amount = test_payment_successful.amount
        assert test_payment_successful.available_for_refund == Decimal('0.00')
    
    def test_payment_risk_scoring(self, db_session, test_customer):
        """Test payment risk scoring"""
        # High risk payment
        high_risk_payment = Payment(
            customer_id=test_customer.id,
            provider=PaymentProvider.STRIPE,
            provider_transaction_id="pi_high_risk",
            amount=Decimal('1000.00'),
            currency="EUR",
            status=PaymentStatus.SUCCEEDED,
            risk_score=Decimal('0.85')
        )
        db_session.add(high_risk_payment)
        db_session.commit()
        
        assert high_risk_payment.is_high_risk is True
        
        # Low risk payment
        low_risk_payment = Payment(
            customer_id=test_customer.id,
            provider=PaymentProvider.STRIPE,
            provider_transaction_id="pi_low_risk",
            amount=Decimal('29.99'),
            currency="EUR",
            status=PaymentStatus.SUCCEEDED,
            risk_score=Decimal('0.2')
        )
        db_session.add(low_risk_payment)
        db_session.commit()
        
        assert low_risk_payment.is_high_risk is False


class TestInvoiceModel:
    """Test Invoice model functionality"""
    
    def test_create_invoice(self, db_session, test_customer, test_subscription_active):
        """Test basic invoice creation"""
        invoice = Invoice(
            number="INV-2025-001",
            customer_id=test_customer.id,
            subscription_id=test_subscription_active.id,
            status=InvoiceStatus.DRAFT,
            currency="EUR",
            subtotal=Decimal('100.00'),
            tax_amount=Decimal('20.00'),
            total=Decimal('120.00'),
            amount_due=Decimal('120.00'),
            line_items=[
                {
                    "description": "Service subscription",
                    "quantity": "1",
                    "unit_price": "100.00",
                    "total": "100.00"
                }
            ],
            issue_date=datetime.utcnow(),
            due_date=datetime.utcnow() + timedelta(days=30)
        )
        db_session.add(invoice)
        db_session.commit()
        
        assert invoice.id is not None
        assert invoice.number == "INV-2025-001"
        assert invoice.is_paid is False
        assert invoice.is_overdue is False
        assert len(invoice.line_items) == 1
    
    def test_invoice_payment_tracking(self, test_invoice_paid):
        """Test invoice payment status tracking"""
        assert test_invoice_paid.is_paid is True
        assert test_invoice_paid.amount_paid == test_invoice_paid.total
        assert test_invoice_paid.amount_due == Decimal('0.00')
        assert test_invoice_paid.paid_at is not None
    
    def test_invoice_overdue_calculation(self, db_session, test_customer):
        """Test overdue invoice calculation"""
        # Create overdue invoice
        overdue_invoice = Invoice(
            number="INV-OVERDUE-001",
            customer_id=test_customer.id,
            status=InvoiceStatus.OPEN,
            currency="EUR",
            subtotal=Decimal('50.00'),
            tax_amount=Decimal('10.00'),
            total=Decimal('60.00'),
            amount_due=Decimal('60.00'),
            line_items=[{"description": "Overdue service", "total": "50.00"}],
            issue_date=datetime.utcnow() - timedelta(days=40),
            due_date=datetime.utcnow() - timedelta(days=10)
        )
        db_session.add(overdue_invoice)
        db_session.commit()
        
        assert overdue_invoice.is_overdue is True
        assert overdue_invoice.days_overdue == 10
    
    def test_invoice_number_uniqueness(self, db_session, test_invoice_draft):
        """Test invoice number uniqueness"""
        duplicate_invoice = Invoice(
            number=test_invoice_draft.number,
            customer_id=test_invoice_draft.customer_id,
            status=InvoiceStatus.DRAFT,
            currency="EUR",
            subtotal=Decimal('10.00'),
            total=Decimal('10.00'),
            amount_due=Decimal('10.00'),
            line_items=[],
            issue_date=datetime.utcnow(),
            due_date=datetime.utcnow() + timedelta(days=30)
        )
        db_session.add(duplicate_invoice)
        
        with pytest.raises(Exception):  # Should raise UniqueConstraint violation
            db_session.commit()


class TestTaxRateModel:
    """Test TaxRate model functionality"""
    
    def test_create_tax_rate(self, db_session):
        """Test tax rate creation"""
        tax_rate = TaxRate(
            name="French VAT",
            country="FR",
            tax_type=TaxType.VAT,
            rate=Decimal('0.2000'),  # 20%
            effective_from=datetime.utcnow(),
            is_active=True
        )
        db_session.add(tax_rate)
        db_session.commit()
        
        assert tax_rate.id is not None
        assert tax_rate.rate == Decimal('0.2000')
        assert tax_rate.is_active is True
    
    def test_tax_rate_constraints(self, db_session):
        """Test tax rate validation constraints"""
        # Invalid rate (over 100%)
        invalid_rate = TaxRate(
            name="Invalid Tax",
            country="US",
            tax_type=TaxType.SALES_TAX,
            rate=Decimal('1.5'),  # 150%
            effective_from=datetime.utcnow()
        )
        db_session.add(invalid_rate)
        
        with pytest.raises(Exception):  # Should raise CheckConstraint violation
            db_session.commit()


class TestUsageRecordModel:
    """Test UsageRecord model functionality"""
    
    def test_create_usage_record(self, db_session, test_subscription_active):
        """Test usage record creation"""
        usage_record = UsageRecord(
            subscription_id=test_subscription_active.id,
            metric_name="api_calls",
            quantity=Decimal('150'),
            unit="calls",
            unit_price=Decimal('0.01'),
            total_amount=Decimal('1.50'),
            usage_date=datetime.utcnow()
        )
        db_session.add(usage_record)
        db_session.commit()
        
        assert usage_record.id is not None
        assert usage_record.metric_name == "api_calls"
        assert usage_record.quantity == Decimal('150')
        assert usage_record.total_amount == Decimal('1.50')
    
    def test_usage_record_constraints(self, db_session, test_subscription_active):
        """Test usage record validation constraints"""
        # Negative quantity should fail
        invalid_usage = UsageRecord(
            subscription_id=test_subscription_active.id,
            metric_name="invalid_metric",
            quantity=Decimal('-10'),
            unit="units",
            usage_date=datetime.utcnow()
        )
        db_session.add(invalid_usage)
        
        with pytest.raises(Exception):  # Should raise CheckConstraint violation
            db_session.commit()


class TestModelRelationships:
    """Test model relationships and cascading"""
    
    def test_customer_subscription_cascade(self, db_session, test_customer, test_plan_monthly):
        """Test customer deletion cascades to subscriptions"""
        # Create subscription
        subscription = Subscription(
            customer_id=test_customer.id,
            plan_id=test_plan_monthly.id,
            status=SubscriptionStatus.ACTIVE,
            current_period_start=datetime.utcnow(),
            current_period_end=datetime.utcnow() + timedelta(days=30)
        )
        db_session.add(subscription)
        db_session.commit()
        
        subscription_id = subscription.id
        
        # Delete customer
        db_session.delete(test_customer)
        db_session.commit()
        
        # Subscription should be deleted too
        deleted_subscription = db_session.query(Subscription).filter(
            Subscription.id == subscription_id
        ).first()
        assert deleted_subscription is None
    
    def test_subscription_payment_relationship(self, test_subscription_active, test_payment_successful):
        """Test subscription-payment relationship"""
        # Payment should be linked to subscription
        assert test_payment_successful.subscription_id == test_subscription_active.id
        
        # Subscription should have the payment
        payments = [p for p in test_subscription_active.customer.payments 
                   if p.subscription_id == test_subscription_active.id]
        assert len(payments) == 1
        assert payments[0].id == test_payment_successful.id
