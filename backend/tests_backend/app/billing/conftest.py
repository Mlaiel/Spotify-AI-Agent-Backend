"""
Test Configuration and Fixtures for Billing System
==================================================

Shared test configuration, fixtures, and utilities for billing tests.
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from decimal import Decimal
from unittest.mock import Mock, patch, AsyncMock
import uuid
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from fastapi.testclient import TestClient
import tempfile
import os

from billing.models import (
    Base, Customer, Plan, Subscription, Payment, Invoice, PaymentMethod,
    CustomerStatus, SubscriptionStatus, PaymentStatus, InvoiceStatus,
    PaymentProvider, PlanInterval, TaxType
)


# Test database configuration
def get_test_database_url():
    """Get test database URL"""
    return "sqlite:///./test_billing.db"


@pytest.fixture(scope="session")
def test_engine():
    """Create test database engine"""
    engine = create_engine(
        get_test_database_url(),
        connect_args={"check_same_thread": False}
    )
    Base.metadata.create_all(bind=engine)
    yield engine
    Base.metadata.drop_all(bind=engine)


@pytest.fixture(scope="session")
def TestSessionLocal(test_engine):
    """Create test session factory"""
    return sessionmaker(autocommit=False, autoflush=False, bind=test_engine)


@pytest.fixture
def db_session(TestSessionLocal):
    """Create test database session"""
    session = TestSessionLocal()
    try:
        yield session
    finally:
        session.rollback()
        session.close()


@pytest.fixture
def test_customer(db_session):
    """Create test customer"""
    customer = Customer(
        email="test@example.com",
        name="Test Customer",
        company="Test Company",
        phone="+33123456789",
        address_line1="123 Test Street",
        address_line2="Apt 456",
        city="Paris",
        state="Île-de-France",
        postal_code="75001",
        country="FR",
        tax_id="FR12345678901",
        preferred_currency="EUR",
        preferred_language="fr",
        status=CustomerStatus.ACTIVE,
        payment_terms=30,
        credit_limit=Decimal('5000.00'),
        metadata={"test": True}
    )
    db_session.add(customer)
    db_session.commit()
    db_session.refresh(customer)
    return customer


@pytest.fixture
def test_plan_monthly(db_session):
    """Create test monthly plan"""
    plan = Plan(
        name="Plan Mensuel Test",
        description="Plan de test mensuel",
        amount=Decimal('29.99'),
        currency="EUR",
        interval=PlanInterval.MONTH,
        interval_count=1,
        trial_period_days=14,
        features=["feature1", "feature2"],
        usage_limits={"api_calls": 1000, "storage_gb": 10},
        is_active=True,
        metadata={"test": True}
    )
    db_session.add(plan)
    db_session.commit()
    db_session.refresh(plan)
    return plan


@pytest.fixture
def test_plan_yearly(db_session):
    """Create test yearly plan"""
    plan = Plan(
        name="Plan Annuel Test",
        description="Plan de test annuel",
        amount=Decimal('299.99'),
        currency="EUR",
        interval=PlanInterval.YEAR,
        interval_count=1,
        trial_period_days=30,
        features=["feature1", "feature2", "feature3"],
        usage_limits={"api_calls": 12000, "storage_gb": 100},
        is_active=True,
        metadata={"test": True}
    )
    db_session.add(plan)
    db_session.commit()
    db_session.refresh(plan)
    return plan


@pytest.fixture
def test_subscription_active(db_session, test_customer, test_plan_monthly):
    """Create active test subscription"""
    subscription = Subscription(
        customer_id=test_customer.id,
        plan_id=test_plan_monthly.id,
        status=SubscriptionStatus.ACTIVE,
        current_period_start=datetime.utcnow(),
        current_period_end=datetime.utcnow() + timedelta(days=30),
        trial_start=None,
        trial_end=None,
        billing_cycle_anchor=datetime.utcnow(),
        cancel_at_period_end=False,
        custom_amount=None,
        discount_percent=Decimal('0'),
        usage_data={"api_calls": 500},
        metadata={"test": True}
    )
    db_session.add(subscription)
    db_session.commit()
    db_session.refresh(subscription)
    return subscription


@pytest.fixture
def test_subscription_trial(db_session, test_customer, test_plan_monthly):
    """Create trial test subscription"""
    now = datetime.utcnow()
    subscription = Subscription(
        customer_id=test_customer.id,
        plan_id=test_plan_monthly.id,
        status=SubscriptionStatus.TRIAL,
        current_period_start=now,
        current_period_end=now + timedelta(days=30),
        trial_start=now,
        trial_end=now + timedelta(days=14),
        billing_cycle_anchor=now,
        cancel_at_period_end=False,
        metadata={"test": True}
    )
    db_session.add(subscription)
    db_session.commit()
    db_session.refresh(subscription)
    return subscription


@pytest.fixture
def test_payment_method_stripe(db_session, test_customer):
    """Create Stripe test payment method"""
    payment_method = PaymentMethod(
        customer_id=test_customer.id,
        provider=PaymentProvider.STRIPE,
        provider_payment_method_id="pm_test_visa",
        type="card",
        last4="4242",
        brand="visa",
        exp_month=12,
        exp_year=2027,
        is_default=True,
        is_active=True,
        billing_address={
            "line1": "123 Test Street",
            "city": "Paris",
            "postal_code": "75001",
            "country": "FR"
        },
        metadata={"test": True}
    )
    db_session.add(payment_method)
    db_session.commit()
    db_session.refresh(payment_method)
    return payment_method


@pytest.fixture
def test_payment_method_paypal(db_session, test_customer):
    """Create PayPal test payment method"""
    payment_method = PaymentMethod(
        customer_id=test_customer.id,
        provider=PaymentProvider.PAYPAL,
        provider_payment_method_id="paypal_test_account",
        type="paypal",
        is_default=False,
        is_active=True,
        metadata={"test": True}
    )
    db_session.add(payment_method)
    db_session.commit()
    db_session.refresh(payment_method)
    return payment_method


@pytest.fixture
def test_payment_successful(db_session, test_customer, test_payment_method_stripe, test_subscription_active):
    """Create successful test payment"""
    payment = Payment(
        customer_id=test_customer.id,
        payment_method_id=test_payment_method_stripe.id,
        subscription_id=test_subscription_active.id,
        provider=PaymentProvider.STRIPE,
        provider_transaction_id="pi_test_successful",
        provider_fee=Decimal('0.90'),
        amount=Decimal('29.99'),
        currency="EUR",
        status=PaymentStatus.SUCCEEDED,
        payment_date=datetime.utcnow(),
        risk_score=Decimal('0.1'),
        metadata={"test": True}
    )
    db_session.add(payment)
    db_session.commit()
    db_session.refresh(payment)
    return payment


@pytest.fixture
def test_payment_failed(db_session, test_customer, test_payment_method_stripe):
    """Create failed test payment"""
    payment = Payment(
        customer_id=test_customer.id,
        payment_method_id=test_payment_method_stripe.id,
        provider=PaymentProvider.STRIPE,
        provider_transaction_id="pi_test_failed",
        amount=Decimal('29.99'),
        currency="EUR",
        status=PaymentStatus.FAILED,
        failure_reason="Your card was declined.",
        risk_score=Decimal('0.3'),
        metadata={"test": True}
    )
    db_session.add(payment)
    db_session.commit()
    db_session.refresh(payment)
    return payment


@pytest.fixture
def test_invoice_draft(db_session, test_customer, test_subscription_active):
    """Create draft test invoice"""
    invoice = Invoice(
        number="INV-TEST-001",
        customer_id=test_customer.id,
        subscription_id=test_subscription_active.id,
        status=InvoiceStatus.DRAFT,
        currency="EUR",
        subtotal=Decimal('29.99'),
        tax_amount=Decimal('6.00'),
        discount_amount=Decimal('0.00'),
        total=Decimal('35.99'),
        amount_paid=Decimal('0.00'),
        amount_due=Decimal('35.99'),
        line_items=[
            {
                "description": "Plan Mensuel Test",
                "quantity": "1",
                "unit_price": "29.99",
                "tax_rate": "20.00",
                "total": "35.99"
            }
        ],
        tax_rate=Decimal('0.2000'),
        tax_type=TaxType.VAT,
        issue_date=datetime.utcnow(),
        due_date=datetime.utcnow() + timedelta(days=30),
        payment_terms="Net 30",
        notes="Facture de test",
        metadata={"test": True}
    )
    db_session.add(invoice)
    db_session.commit()
    db_session.refresh(invoice)
    return invoice


@pytest.fixture
def test_invoice_paid(db_session, test_customer, test_subscription_active, test_payment_successful):
    """Create paid test invoice"""
    invoice = Invoice(
        number="INV-TEST-002",
        customer_id=test_customer.id,
        subscription_id=test_subscription_active.id,
        status=InvoiceStatus.PAID,
        currency="EUR",
        subtotal=Decimal('29.99'),
        tax_amount=Decimal('6.00'),
        total=Decimal('35.99'),
        amount_paid=Decimal('35.99'),
        amount_due=Decimal('0.00'),
        line_items=[
            {
                "description": "Plan Mensuel Test",
                "quantity": "1",
                "unit_price": "29.99",
                "tax_rate": "20.00",
                "total": "35.99"
            }
        ],
        tax_rate=Decimal('0.2000'),
        tax_type=TaxType.VAT,
        issue_date=datetime.utcnow() - timedelta(days=5),
        due_date=datetime.utcnow() + timedelta(days=25),
        paid_at=datetime.utcnow() - timedelta(days=2),
        payment_terms="Net 30",
        metadata={"test": True}
    )
    db_session.add(invoice)
    db_session.commit()
    db_session.refresh(invoice)
    
    # Link payment to invoice
    test_payment_successful.invoice_id = invoice.id
    db_session.commit()
    
    return invoice


# Mock fixtures
@pytest.fixture
def mock_redis():
    """Mock Redis client"""
    mock_client = AsyncMock()
    mock_client.get.return_value = None
    mock_client.set.return_value = True
    mock_client.setex.return_value = True
    mock_client.incr.return_value = 1
    mock_client.lpush.return_value = 1
    mock_client.lrange.return_value = []
    return mock_client


@pytest.fixture
def mock_stripe():
    """Mock Stripe API"""
    with patch('stripe.PaymentIntent') as mock_payment_intent, \
         patch('stripe.Customer') as mock_customer, \
         patch('stripe.Subscription') as mock_subscription, \
         patch('stripe.PaymentMethod') as mock_payment_method:
        
        # Configure mocks
        mock_payment_intent.create.return_value = Mock(
            id="pi_test_123",
            status="succeeded",
            amount=2999,
            currency="eur",
            charges=Mock(data=[Mock(outcome=Mock(risk_level="normal"))])
        )
        
        mock_customer.create.return_value = Mock(
            id="cus_test_123",
            email="test@example.com"
        )
        
        mock_subscription.create.return_value = Mock(
            id="sub_test_123",
            status="active",
            current_period_start=1640995200,
            current_period_end=1643673600
        )
        
        yield {
            'payment_intent': mock_payment_intent,
            'customer': mock_customer,
            'subscription': mock_subscription,
            'payment_method': mock_payment_method
        }


@pytest.fixture
def mock_paypal():
    """Mock PayPal API"""
    with patch('paypalrestsdk.Payment') as mock_payment, \
         patch('paypalrestsdk.BillingPlan') as mock_plan, \
         patch('paypalrestsdk.BillingAgreement') as mock_agreement:
        
        # Configure mocks
        mock_payment.return_value.create.return_value = True
        mock_payment.return_value.id = "PAYID-TEST-123"
        mock_payment.return_value.state = "approved"
        
        yield {
            'payment': mock_payment,
            'plan': mock_plan,
            'agreement': mock_agreement
        }


@pytest.fixture
def mock_email():
    """Mock email sending"""
    with patch('smtplib.SMTP') as mock_smtp:
        mock_server = Mock()
        mock_smtp.return_value.__enter__.return_value = mock_server
        mock_server.send_message.return_value = {}
        yield mock_server


@pytest.fixture
def temp_invoice_storage():
    """Create temporary directory for invoice storage"""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield temp_dir


@pytest.fixture
def mock_s3():
    """Mock AWS S3 client"""
    with patch('boto3.client') as mock_boto3:
        mock_client = Mock()
        mock_boto3.return_value = mock_client
        
        mock_client.put_object.return_value = {'ETag': '"test-etag"'}
        mock_client.get_object.return_value = {
            'Body': Mock(read=lambda: b'test pdf content')
        }
        
        yield mock_client


# Utility functions
def create_test_customers(db_session, count=5):
    """Create multiple test customers"""
    customers = []
    for i in range(count):
        customer = Customer(
            email=f"test{i}@example.com",
            name=f"Test Customer {i}",
            company=f"Test Company {i}",
            preferred_currency="EUR",
            status=CustomerStatus.ACTIVE
        )
        db_session.add(customer)
        customers.append(customer)
    
    db_session.commit()
    return customers


def create_test_subscriptions(db_session, customers, plan, status=SubscriptionStatus.ACTIVE):
    """Create test subscriptions for customers"""
    subscriptions = []
    now = datetime.utcnow()
    
    for customer in customers:
        subscription = Subscription(
            customer_id=customer.id,
            plan_id=plan.id,
            status=status,
            current_period_start=now,
            current_period_end=now + timedelta(days=30)
        )
        db_session.add(subscription)
        subscriptions.append(subscription)
    
    db_session.commit()
    return subscriptions


async def assert_payment_processed(payment, expected_status=PaymentStatus.SUCCEEDED):
    """Assert that payment was processed correctly"""
    assert payment is not None
    assert payment.status == expected_status
    if expected_status == PaymentStatus.SUCCEEDED:
        assert payment.payment_date is not None
        assert payment.risk_score is not None


async def assert_subscription_active(subscription):
    """Assert that subscription is in active state"""
    assert subscription.status in [SubscriptionStatus.ACTIVE, SubscriptionStatus.TRIAL]
    assert subscription.current_period_end > datetime.utcnow()


def generate_test_webhook_payload(provider, event_type, data):
    """Generate test webhook payload"""
    if provider == "stripe":
        return {
            "id": f"evt_test_{uuid.uuid4().hex[:8]}",
            "type": event_type,
            "data": {"object": data},
            "created": int(datetime.utcnow().timestamp())
        }
    elif provider == "paypal":
        return {
            "id": f"WH-{uuid.uuid4().hex[:8]}",
            "event_type": event_type,
            "resource": data,
            "create_time": datetime.utcnow().isoformat()
        }


# Test markers
pytestmark = [
    pytest.mark.asyncio,
    pytest.mark.billing
]


# Configuration
pytest_plugins = ['pytest_asyncio']
