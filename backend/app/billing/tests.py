"""
Spotify AI Agent - Billing System Tests
======================================

Comprehensive test suite for billing system functionality:
- Unit tests for core billing logic
- Integration tests for payment processing
- API endpoint testing
- Database model validation
- Mock external service dependencies
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

from ..models import (
    Base, Customer, Plan, Subscription, Payment, Invoice, PaymentMethod,
    CustomerStatus, SubscriptionStatus, PaymentStatus, InvoiceStatus,
    PaymentProvider, PlanInterval
)
from ..core import BillingEngine
from ..invoices import InvoiceManager
from ..analytics import BillingAnalytics
from ..api import router
from ...main import app


# Test database setup
SQLITE_URL = "sqlite:///./test_billing.db"
engine = create_engine(SQLITE_URL, connect_args={"check_same_thread": False})
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


@pytest.fixture(scope="module")
def test_db():
    """Create test database"""
    Base.metadata.create_all(bind=engine)
    yield
    Base.metadata.drop_all(bind=engine)


@pytest.fixture
def db_session(test_db):
    """Create test database session"""
    session = TestingSessionLocal()
    try:
        yield session
    finally:
        session.close()


@pytest.fixture
def test_customer(db_session):
    """Create test customer"""
    customer = Customer(
        email="test@example.com",
        name="Test Customer",
        company="Test Company",
        preferred_currency="EUR",
        preferred_language="en"
    )
    db_session.add(customer)
    db_session.commit()
    db_session.refresh(customer)
    return customer


@pytest.fixture
def test_plan(db_session):
    """Create test plan"""
    plan = Plan(
        name="Basic Plan",
        description="Basic subscription plan",
        amount=Decimal('29.99'),
        currency="EUR",
        interval=PlanInterval.MONTH,
        interval_count=1,
        trial_period_days=14
    )
    db_session.add(plan)
    db_session.commit()
    db_session.refresh(plan)
    return plan


@pytest.fixture
def test_subscription(db_session, test_customer, test_plan):
    """Create test subscription"""
    subscription = Subscription(
        customer_id=test_customer.id,
        plan_id=test_plan.id,
        status=SubscriptionStatus.ACTIVE,
        current_period_start=datetime.utcnow(),
        current_period_end=datetime.utcnow() + timedelta(days=30)
    )
    db_session.add(subscription)
    db_session.commit()
    db_session.refresh(subscription)
    return subscription


@pytest.fixture
def test_payment_method(db_session, test_customer):
    """Create test payment method"""
    payment_method = PaymentMethod(
        customer_id=test_customer.id,
        provider=PaymentProvider.STRIPE,
        provider_payment_method_id="pm_test_123",
        type="card",
        last4="4242",
        brand="visa",
        is_default=True
    )
    db_session.add(payment_method)
    db_session.commit()
    db_session.refresh(payment_method)
    return payment_method


@pytest.fixture
def billing_engine():
    """Create billing engine instance"""
    return BillingEngine()


@pytest.fixture
def client():
    """Create test client"""
    return TestClient(app)


# Model tests
class TestCustomerModel:
    """Test Customer model"""
    
    def test_create_customer(self, db_session):
        """Test customer creation"""
        customer = Customer(
            email="new@example.com",
            name="New Customer",
            preferred_currency="USD"
        )
        db_session.add(customer)
        db_session.commit()
        
        assert customer.id is not None
        assert customer.email == "new@example.com"
        assert customer.status == CustomerStatus.ACTIVE
        assert customer.is_active is True
    
    def test_customer_email_validation(self, db_session):
        """Test email validation"""
        with pytest.raises(ValueError):
            customer = Customer(
                email="invalid-email",
                name="Test Customer"
            )
            db_session.add(customer)
            db_session.commit()
    
    def test_customer_full_address(self, test_customer):
        """Test full address property"""
        test_customer.address_line1 = "123 Main St"
        test_customer.city = "Paris"
        test_customer.country = "FR"
        
        expected = "123 Main St, Paris, FR"
        assert test_customer.full_address == expected


class TestPlanModel:
    """Test Plan model"""
    
    def test_create_plan(self, db_session):
        """Test plan creation"""
        plan = Plan(
            name="Premium Plan",
            amount=Decimal('99.99'),
            currency="EUR",
            interval=PlanInterval.YEAR
        )
        db_session.add(plan)
        db_session.commit()
        
        assert plan.id is not None
        assert plan.amount == Decimal('99.99')
        assert plan.is_active is True
    
    def test_monthly_amount_calculation(self, test_plan):
        """Test monthly amount conversion"""
        # Monthly plan should return same amount
        assert test_plan.monthly_amount == test_plan.amount
        
        # Yearly plan should return amount/12
        yearly_plan = Plan(
            name="Yearly Plan",
            amount=Decimal('120.00'),
            currency="EUR",
            interval=PlanInterval.YEAR,
            interval_count=1
        )
        expected_monthly = Decimal('120.00') * Decimal('0.083')
        assert abs(yearly_plan.monthly_amount - expected_monthly) < Decimal('0.01')


class TestSubscriptionModel:
    """Test Subscription model"""
    
    def test_create_subscription(self, test_subscription):
        """Test subscription creation"""
        assert test_subscription.id is not None
        assert test_subscription.status == SubscriptionStatus.ACTIVE
        assert test_subscription.is_active is True
    
    def test_effective_amount(self, test_subscription):
        """Test effective amount calculation"""
        # No custom amount or discount
        expected = test_subscription.plan.amount
        assert test_subscription.effective_amount == expected
        
        # With discount
        test_subscription.discount_percent = Decimal('10')
        expected_with_discount = expected * Decimal('0.9')
        assert test_subscription.effective_amount == expected_with_discount
    
    def test_days_until_renewal(self, test_subscription):
        """Test days until renewal calculation"""
        # Set period end to 5 days from now
        test_subscription.current_period_end = datetime.utcnow() + timedelta(days=5)
        assert test_subscription.days_until_renewal == 5


class TestPaymentModel:
    """Test Payment model"""
    
    def test_create_payment(self, db_session, test_customer):
        """Test payment creation"""
        payment = Payment(
            customer_id=test_customer.id,
            provider=PaymentProvider.STRIPE,
            provider_transaction_id="pi_test_123",
            amount=Decimal('29.99'),
            currency="EUR",
            status=PaymentStatus.SUCCEEDED
        )
        db_session.add(payment)
        db_session.commit()
        
        assert payment.id is not None
        assert payment.is_successful is True
        assert payment.net_amount == payment.amount - payment.provider_fee
    
    def test_available_for_refund(self, db_session, test_customer):
        """Test refund amount calculation"""
        payment = Payment(
            customer_id=test_customer.id,
            provider=PaymentProvider.STRIPE,
            provider_transaction_id="pi_test_456",
            amount=Decimal('100.00'),
            currency="EUR",
            status=PaymentStatus.SUCCEEDED,
            refunded_amount=Decimal('30.00')
        )
        db_session.add(payment)
        db_session.commit()
        
        assert payment.available_for_refund == Decimal('70.00')


# Core billing engine tests
class TestBillingEngine:
    """Test BillingEngine class"""
    
    @pytest.mark.asyncio
    async def test_initialize(self, billing_engine):
        """Test billing engine initialization"""
        with patch('aioredis.from_url') as mock_redis:
            mock_redis.return_value = Mock()
            await billing_engine.initialize()
            assert billing_engine.redis_client is not None
    
    @pytest.mark.asyncio
    async def test_create_subscription(self, billing_engine, db_session, test_customer, test_plan):
        """Test subscription creation"""
        with patch.object(billing_engine, 'redis_client', Mock()):
            subscription = await billing_engine.create_subscription(
                customer_id=test_customer.id,
                plan_id=test_plan.id
            )
            
            assert subscription.customer_id == test_customer.id
            assert subscription.plan_id == test_plan.id
            assert subscription.status in [SubscriptionStatus.ACTIVE, SubscriptionStatus.TRIAL]
    
    @pytest.mark.asyncio
    async def test_process_payment_stripe(self, billing_engine, test_customer, test_payment_method):
        """Test Stripe payment processing"""
        with patch('stripe.PaymentIntent.create') as mock_stripe:
            mock_stripe.return_value = Mock(
                id="pi_test_123",
                status="succeeded",
                amount=2999,
                currency="eur"
            )
            
            payment = await billing_engine.process_payment(
                customer_id=test_customer.id,
                amount=Decimal('29.99'),
                currency="EUR",
                payment_method_id=test_payment_method.id
            )
            
            assert payment.provider == PaymentProvider.STRIPE
            assert payment.amount == Decimal('29.99')
            assert payment.status == PaymentStatus.SUCCEEDED
    
    @pytest.mark.asyncio
    async def test_fraud_detection(self, billing_engine):
        """Test fraud detection"""
        payment_data = {
            'amount': Decimal('10000.00'),  # High amount
            'customer_email': 'suspicious@example.com',
            'ip_address': '192.168.1.1',
            'user_agent': 'suspicious_agent'
        }
        
        risk_score = await billing_engine.calculate_fraud_risk(payment_data)
        assert 0 <= risk_score <= 1
        assert isinstance(risk_score, float)


# API endpoint tests
class TestBillingAPI:
    """Test billing API endpoints"""
    
    def test_create_customer(self, client):
        """Test customer creation endpoint"""
        customer_data = {
            "email": "api@example.com",
            "name": "API Test Customer",
            "preferred_currency": "EUR"
        }
        
        with patch('app.core.auth.get_current_user', return_value=Mock()):
            response = client.post("/billing/customers", json=customer_data)
            
        assert response.status_code == 201
        data = response.json()
        assert data["email"] == customer_data["email"]
        assert data["name"] == customer_data["name"]
    
    def test_get_customer(self, client, test_customer):
        """Test get customer endpoint"""
        with patch('app.core.auth.get_current_user', return_value=Mock()):
            response = client.get(f"/billing/customers/{test_customer.id}")
        
        assert response.status_code == 200
        data = response.json()
        assert data["email"] == test_customer.email
    
    def test_create_subscription(self, client, test_customer, test_plan):
        """Test subscription creation endpoint"""
        subscription_data = {
            "customer_id": str(test_customer.id),
            "plan_id": str(test_plan.id)
        }
        
        with patch('app.core.auth.get_current_user', return_value=Mock()):
            with patch('app.billing.core.billing_engine.create_subscription') as mock_create:
                mock_create.return_value = Mock(
                    id=str(uuid.uuid4()),
                    customer_id=test_customer.id,
                    plan_id=test_plan.id,
                    status=SubscriptionStatus.ACTIVE
                )
                
                response = client.post("/billing/subscriptions", json=subscription_data)
        
        assert response.status_code == 201
    
    def test_process_payment(self, client, test_customer):
        """Test payment processing endpoint"""
        payment_data = {
            "customer_id": str(test_customer.id),
            "amount": "29.99",
            "currency": "EUR"
        }
        
        with patch('app.core.auth.get_current_user', return_value=Mock()):
            with patch('app.billing.core.billing_engine.process_payment') as mock_process:
                mock_process.return_value = Mock(
                    id=str(uuid.uuid4()),
                    customer_id=test_customer.id,
                    amount=Decimal('29.99'),
                    status=PaymentStatus.SUCCEEDED
                )
                
                response = client.post("/billing/payments", json=payment_data)
        
        assert response.status_code == 201


# Invoice tests
class TestInvoiceManager:
    """Test InvoiceManager class"""
    
    @pytest.mark.asyncio
    async def test_create_invoice(self):
        """Test invoice creation"""
        from ..invoices import InvoiceRecipient, InvoiceLineItem, InvoiceType
        
        recipient = InvoiceRecipient(
            name="Test Customer",
            email="test@example.com"
        )
        
        line_item = InvoiceLineItem(
            description="Test Service",
            quantity=Decimal('1'),
            unit_price=Decimal('29.99'),
            tax_rate=Decimal('20')
        )
        
        invoice_manager = InvoiceManager()
        
        with patch.object(invoice_manager, 'redis_client', Mock()):
            with patch.object(invoice_manager, 'number_generator') as mock_generator:
                mock_generator.generate_number.return_value = "INV-2024-000001"
                
                invoice = await invoice_manager.create_invoice(
                    recipient=recipient,
                    line_items=[line_item],
                    invoice_type=InvoiceType.ONE_TIME
                )
        
        assert invoice.number == "INV-2024-000001"
        assert invoice.total == Decimal('35.99')  # 29.99 + 20% tax
        assert len(invoice.line_items) == 1
    
    @pytest.mark.asyncio
    async def test_generate_pdf(self):
        """Test PDF generation"""
        invoice_manager = InvoiceManager()
        
        with patch.object(invoice_manager, '_get_invoice_data') as mock_get:
            mock_invoice = Mock()
            mock_invoice.recipient.language = "en"
            mock_get.return_value = mock_invoice
            
            with patch.object(invoice_manager.pdf_generator, 'generate_pdf') as mock_pdf:
                mock_pdf.return_value = b"PDF content"
                
                pdf_data = await invoice_manager.generate_pdf("test-invoice-id")
        
        assert pdf_data == b"PDF content"


# Analytics tests
class TestBillingAnalytics:
    """Test BillingAnalytics class"""
    
    @pytest.mark.asyncio
    async def test_get_revenue_metrics(self, db_session):
        """Test revenue metrics calculation"""
        analytics = BillingAnalytics(db_session)
        
        start_date = datetime.utcnow() - timedelta(days=30)
        end_date = datetime.utcnow()
        
        metrics = await analytics.get_revenue_metrics(start_date, end_date)
        
        assert metrics.currency == "EUR"
        assert metrics.total_revenue >= 0
        assert metrics.mrr >= 0
        assert metrics.arr == metrics.mrr * 12
    
    @pytest.mark.asyncio
    async def test_get_customer_metrics(self, db_session, test_customer):
        """Test customer metrics calculation"""
        analytics = BillingAnalytics(db_session)
        
        start_date = datetime.utcnow() - timedelta(days=30)
        end_date = datetime.utcnow()
        
        metrics = await analytics.get_customer_metrics(start_date, end_date)
        
        assert metrics.total_customers >= 1  # At least our test customer
        assert metrics.active_customers >= 0
        assert metrics.churn_rate >= 0
        assert metrics.retention_rate <= 100


# Integration tests
class TestBillingIntegration:
    """Test billing system integration"""
    
    @pytest.mark.asyncio
    async def test_complete_subscription_flow(self, db_session, billing_engine):
        """Test complete subscription creation and billing flow"""
        # Create customer
        customer = Customer(
            email="integration@example.com",
            name="Integration Test Customer"
        )
        db_session.add(customer)
        db_session.commit()
        
        # Create plan
        plan = Plan(
            name="Integration Plan",
            amount=Decimal('19.99'),
            currency="EUR",
            interval=PlanInterval.MONTH
        )
        db_session.add(plan)
        db_session.commit()
        
        # Create subscription
        with patch.object(billing_engine, 'redis_client', Mock()):
            subscription = await billing_engine.create_subscription(
                customer_id=customer.id,
                plan_id=plan.id
            )
        
        assert subscription.customer_id == customer.id
        assert subscription.plan_id == plan.id
        
        # Process billing
        with patch('stripe.PaymentIntent.create') as mock_stripe:
            mock_stripe.return_value = Mock(
                id="pi_integration_test",
                status="succeeded",
                amount=1999,
                currency="eur"
            )
            
            payment = await billing_engine.process_subscription_billing(subscription.id)
        
        assert payment["success"] is True
    
    @pytest.mark.asyncio
    async def test_webhook_processing(self, db_session):
        """Test webhook event processing"""
        from ..webhooks import WebhookManager
        
        webhook_manager = WebhookManager(db_session)
        
        # Mock Stripe webhook event
        stripe_event = {
            "id": "evt_test_123",
            "type": "payment_intent.succeeded",
            "data": {
                "object": {
                    "id": "pi_test_123",
                    "amount": 2999,
                    "currency": "eur",
                    "status": "succeeded"
                }
            },
            "created": int(datetime.utcnow().timestamp())
        }
        
        headers = {
            "stripe-signature": "t=123,v1=test_signature"
        }
        
        with patch.object(webhook_manager.processors['stripe'], 'verify_signature', return_value=True):
            with patch.object(webhook_manager.event_store, 'is_event_processed', return_value=False):
                result = await webhook_manager.process_webhook(
                    "stripe",
                    headers,
                    json.dumps(stripe_event).encode()
                )
        
        assert result is True


# Performance tests
class TestBillingPerformance:
    """Test billing system performance"""
    
    @pytest.mark.asyncio
    async def test_bulk_subscription_processing(self, db_session, billing_engine):
        """Test processing multiple subscriptions"""
        import time
        
        # Create multiple customers and subscriptions
        customers = []
        subscriptions = []
        
        for i in range(10):
            customer = Customer(
                email=f"perf{i}@example.com",
                name=f"Performance Customer {i}"
            )
            db_session.add(customer)
            customers.append(customer)
        
        plan = Plan(
            name="Performance Plan",
            amount=Decimal('9.99'),
            currency="EUR"
        )
        db_session.add(plan)
        db_session.commit()
        
        for customer in customers:
            subscription = Subscription(
                customer_id=customer.id,
                plan_id=plan.id,
                status=SubscriptionStatus.ACTIVE,
                current_period_start=datetime.utcnow(),
                current_period_end=datetime.utcnow() + timedelta(days=30)
            )
            db_session.add(subscription)
            subscriptions.append(subscription)
        
        db_session.commit()
        
        # Time bulk processing
        start_time = time.time()
        
        with patch.object(billing_engine, 'redis_client', Mock()):
            tasks = [
                billing_engine.process_subscription_billing(str(sub.id))
                for sub in subscriptions
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        # Should process 10 subscriptions in under 5 seconds
        assert processing_time < 5.0
        assert len(results) == 10


# Fixtures for mocking external services
@pytest.fixture
def mock_stripe():
    """Mock Stripe API calls"""
    with patch('stripe.PaymentIntent') as mock_payment_intent:
        with patch('stripe.Customer') as mock_customer:
            with patch('stripe.Subscription') as mock_subscription:
                yield {
                    'payment_intent': mock_payment_intent,
                    'customer': mock_customer,
                    'subscription': mock_subscription
                }


@pytest.fixture
def mock_paypal():
    """Mock PayPal API calls"""
    with patch('paypalrestsdk.Payment') as mock_payment:
        with patch('paypalrestsdk.BillingPlan') as mock_plan:
            yield {
                'payment': mock_payment,
                'plan': mock_plan
            }


@pytest.fixture
def mock_redis():
    """Mock Redis client"""
    with patch('aioredis.from_url') as mock_from_url:
        mock_client = AsyncMock()
        mock_from_url.return_value = mock_client
        yield mock_client


# Test configuration
pytest_plugins = ['pytest_asyncio']


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--cov=app.billing", "--cov-report=html"])
