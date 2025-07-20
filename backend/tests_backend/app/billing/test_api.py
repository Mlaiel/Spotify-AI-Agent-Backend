"""
Tests for Billing API Endpoints
==============================

Comprehensive tests for FastAPI billing endpoints.
"""

import pytest
from datetime import datetime, timedelta
from decimal import Decimal
from unittest.mock import AsyncMock, Mock, patch
import json

from fastapi.testclient import TestClient
from fastapi import status

from billing.models import (
    Customer, Plan, Subscription, Payment, Invoice, PaymentMethod,
    SubscriptionStatus, PaymentStatus, InvoiceStatus, PaymentProvider,
    CustomerStatus, PlanInterval
)


class TestCustomerEndpoints:
    """Test customer management endpoints"""
    
    def test_create_customer(self, client, mock_billing_engine):
        """Test POST /api/v1/billing/customers"""
        customer_data = {
            "email": "api@example.com",
            "name": "API Test Customer",
            "company": "API Test Company",
            "preferred_currency": "EUR",
            "preferred_language": "en",
            "country": "FR",
            "address": {
                "line1": "123 API Street",
                "city": "Paris",
                "postal_code": "75001",
                "country": "FR"
            }
        }
        
        # Mock billing engine response
        mock_customer = Mock()
        mock_customer.id = "cust_123"
        mock_customer.email = customer_data["email"]
        mock_customer.name = customer_data["name"]
        mock_customer.status = CustomerStatus.ACTIVE
        mock_billing_engine.create_customer.return_value = mock_customer
        
        response = client.post("/api/v1/billing/customers", json=customer_data)
        
        assert response.status_code == status.HTTP_201_CREATED
        data = response.json()
        assert data["email"] == customer_data["email"]
        assert data["status"] == "ACTIVE"
        mock_billing_engine.create_customer.assert_called_once()
    
    def test_get_customer(self, client, test_customer):
        """Test GET /api/v1/billing/customers/{customer_id}"""
        response = client.get(f"/api/v1/billing/customers/{test_customer.id}")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["id"] == test_customer.id
        assert data["email"] == test_customer.email
        assert data["name"] == test_customer.name
    
    def test_get_customer_not_found(self, client):
        """Test GET /api/v1/billing/customers/{customer_id} - not found"""
        response = client.get("/api/v1/billing/customers/nonexistent")
        
        assert response.status_code == status.HTTP_404_NOT_FOUND
        assert "Customer not found" in response.json()["detail"]
    
    def test_update_customer(self, client, test_customer):
        """Test PUT /api/v1/billing/customers/{customer_id}"""
        update_data = {
            "name": "Updated Customer Name",
            "company": "Updated Company",
            "preferred_currency": "USD"
        }
        
        response = client.put(
            f"/api/v1/billing/customers/{test_customer.id}",
            json=update_data
        )
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["name"] == update_data["name"]
        assert data["company"] == update_data["company"]
        assert data["preferred_currency"] == update_data["preferred_currency"]
    
    def test_list_customers(self, client, test_customer):
        """Test GET /api/v1/billing/customers"""
        response = client.get("/api/v1/billing/customers")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "items" in data
        assert "total" in data
        assert "page" in data
        assert "per_page" in data
        assert len(data["items"]) >= 1
    
    def test_list_customers_with_filters(self, client, test_customer):
        """Test GET /api/v1/billing/customers with filters"""
        response = client.get(
            "/api/v1/billing/customers",
            params={
                "status": "ACTIVE",
                "country": "FR",
                "page": 1,
                "per_page": 10
            }
        )
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert all(customer["status"] == "ACTIVE" for customer in data["items"])
    
    def test_delete_customer(self, client, test_customer, mock_billing_engine):
        """Test DELETE /api/v1/billing/customers/{customer_id}"""
        mock_billing_engine.delete_customer.return_value = True
        
        response = client.delete(f"/api/v1/billing/customers/{test_customer.id}")
        
        assert response.status_code == status.HTTP_204_NO_CONTENT
        mock_billing_engine.delete_customer.assert_called_once_with(test_customer.id)


class TestPlanEndpoints:
    """Test plan management endpoints"""
    
    def test_create_plan(self, client, mock_billing_engine):
        """Test POST /api/v1/billing/plans"""
        plan_data = {
            "name": "Premium Plan",
            "description": "Premium subscription plan",
            "amount": "99.99",
            "currency": "EUR",
            "interval": "MONTH",
            "interval_count": 1,
            "trial_period_days": 14,
            "features": ["api_access", "premium_support"],
            "usage_limits": {
                "api_calls_per_month": 10000,
                "storage_gb": 100
            }
        }
        
        mock_plan = Mock()
        mock_plan.id = "plan_123"
        mock_plan.name = plan_data["name"]
        mock_plan.amount = Decimal(plan_data["amount"])
        mock_plan.is_active = True
        mock_billing_engine.create_plan.return_value = mock_plan
        
        response = client.post("/api/v1/billing/plans", json=plan_data)
        
        assert response.status_code == status.HTTP_201_CREATED
        data = response.json()
        assert data["name"] == plan_data["name"]
        assert data["amount"] == plan_data["amount"]
        mock_billing_engine.create_plan.assert_called_once()
    
    def test_get_plan(self, client, test_plan_monthly):
        """Test GET /api/v1/billing/plans/{plan_id}"""
        response = client.get(f"/api/v1/billing/plans/{test_plan_monthly.id}")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["id"] == test_plan_monthly.id
        assert data["name"] == test_plan_monthly.name
    
    def test_list_plans(self, client, test_plan_monthly, test_plan_yearly):
        """Test GET /api/v1/billing/plans"""
        response = client.get("/api/v1/billing/plans")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "items" in data
        assert len(data["items"]) >= 2
    
    def test_update_plan(self, client, test_plan_monthly):
        """Test PUT /api/v1/billing/plans/{plan_id}"""
        update_data = {
            "name": "Updated Premium Plan",
            "description": "Updated description",
            "amount": "39.99"
        }
        
        response = client.put(
            f"/api/v1/billing/plans/{test_plan_monthly.id}",
            json=update_data
        )
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["name"] == update_data["name"]
        assert data["amount"] == update_data["amount"]
    
    def test_deactivate_plan(self, client, test_plan_monthly, mock_billing_engine):
        """Test DELETE /api/v1/billing/plans/{plan_id}"""
        mock_billing_engine.deactivate_plan.return_value = True
        
        response = client.delete(f"/api/v1/billing/plans/{test_plan_monthly.id}")
        
        assert response.status_code == status.HTTP_204_NO_CONTENT
        mock_billing_engine.deactivate_plan.assert_called_once_with(test_plan_monthly.id)


class TestSubscriptionEndpoints:
    """Test subscription management endpoints"""
    
    def test_create_subscription(self, client, test_customer, test_plan_monthly, mock_billing_engine):
        """Test POST /api/v1/billing/subscriptions"""
        subscription_data = {
            "customer_id": test_customer.id,
            "plan_id": test_plan_monthly.id,
            "trial_period_days": 14,
            "payment_method_id": "pm_test_card"
        }
        
        mock_subscription = Mock()
        mock_subscription.id = "sub_123"
        mock_subscription.customer_id = test_customer.id
        mock_subscription.plan_id = test_plan_monthly.id
        mock_subscription.status = SubscriptionStatus.TRIAL
        mock_billing_engine.create_subscription.return_value = mock_subscription
        
        response = client.post("/api/v1/billing/subscriptions", json=subscription_data)
        
        assert response.status_code == status.HTTP_201_CREATED
        data = response.json()
        assert data["customer_id"] == test_customer.id
        assert data["plan_id"] == test_plan_monthly.id
        assert data["status"] == "TRIAL"
    
    def test_get_subscription(self, client, test_subscription_active):
        """Test GET /api/v1/billing/subscriptions/{subscription_id}"""
        response = client.get(f"/api/v1/billing/subscriptions/{test_subscription_active.id}")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["id"] == test_subscription_active.id
        assert data["status"] == "ACTIVE"
    
    def test_list_subscriptions(self, client, test_subscription_active):
        """Test GET /api/v1/billing/subscriptions"""
        response = client.get("/api/v1/billing/subscriptions")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "items" in data
        assert len(data["items"]) >= 1
    
    def test_list_customer_subscriptions(self, client, test_customer, test_subscription_active):
        """Test GET /api/v1/billing/customers/{customer_id}/subscriptions"""
        response = client.get(f"/api/v1/billing/customers/{test_customer.id}/subscriptions")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "items" in data
        assert all(sub["customer_id"] == test_customer.id for sub in data["items"])
    
    def test_upgrade_subscription(self, client, test_subscription_active, test_plan_yearly, mock_billing_engine):
        """Test PUT /api/v1/billing/subscriptions/{subscription_id}/upgrade"""
        upgrade_data = {
            "new_plan_id": test_plan_yearly.id,
            "proration_behavior": "immediate"
        }
        
        mock_upgraded_subscription = Mock()
        mock_upgraded_subscription.id = test_subscription_active.id
        mock_upgraded_subscription.plan_id = test_plan_yearly.id
        mock_upgraded_subscription.status = SubscriptionStatus.ACTIVE
        mock_billing_engine.upgrade_subscription.return_value = mock_upgraded_subscription
        
        response = client.put(
            f"/api/v1/billing/subscriptions/{test_subscription_active.id}/upgrade",
            json=upgrade_data
        )
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["plan_id"] == test_plan_yearly.id
        mock_billing_engine.upgrade_subscription.assert_called_once()
    
    def test_cancel_subscription(self, client, test_subscription_active, mock_billing_engine):
        """Test POST /api/v1/billing/subscriptions/{subscription_id}/cancel"""
        cancel_data = {
            "cancel_at_period_end": True,
            "reason": "Customer request"
        }
        
        mock_cancelled_subscription = Mock()
        mock_cancelled_subscription.id = test_subscription_active.id
        mock_cancelled_subscription.status = SubscriptionStatus.CANCELING
        mock_cancelled_subscription.cancel_at_period_end = True
        mock_billing_engine.cancel_subscription.return_value = mock_cancelled_subscription
        
        response = client.post(
            f"/api/v1/billing/subscriptions/{test_subscription_active.id}/cancel",
            json=cancel_data
        )
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["status"] == "CANCELING"
        mock_billing_engine.cancel_subscription.assert_called_once()
    
    def test_reactivate_subscription(self, client, test_subscription_active, mock_billing_engine):
        """Test POST /api/v1/billing/subscriptions/{subscription_id}/reactivate"""
        mock_reactivated_subscription = Mock()
        mock_reactivated_subscription.id = test_subscription_active.id
        mock_reactivated_subscription.status = SubscriptionStatus.ACTIVE
        mock_billing_engine.reactivate_subscription.return_value = mock_reactivated_subscription
        
        response = client.post(f"/api/v1/billing/subscriptions/{test_subscription_active.id}/reactivate")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["status"] == "ACTIVE"
        mock_billing_engine.reactivate_subscription.assert_called_once()


class TestPaymentMethodEndpoints:
    """Test payment method management endpoints"""
    
    def test_add_payment_method(self, client, test_customer, mock_payment_processor):
        """Test POST /api/v1/billing/customers/{customer_id}/payment-methods"""
        payment_method_data = {
            "provider": "STRIPE",
            "type": "card",
            "card_number": "4242424242424242",
            "exp_month": 12,
            "exp_year": 2025,
            "cvc": "123",
            "is_default": True
        }
        
        mock_payment_method = Mock()
        mock_payment_method.id = "pm_123"
        mock_payment_method.customer_id = test_customer.id
        mock_payment_method.provider = PaymentProvider.STRIPE
        mock_payment_method.last4 = "4242"
        mock_payment_method.brand = "visa"
        mock_payment_processor.add_payment_method.return_value = mock_payment_method
        
        response = client.post(
            f"/api/v1/billing/customers/{test_customer.id}/payment-methods",
            json=payment_method_data
        )
        
        assert response.status_code == status.HTTP_201_CREATED
        data = response.json()
        assert data["customer_id"] == test_customer.id
        assert data["last4"] == "4242"
        assert data["brand"] == "visa"
    
    def test_list_payment_methods(self, client, test_customer, test_payment_method_stripe):
        """Test GET /api/v1/billing/customers/{customer_id}/payment-methods"""
        response = client.get(f"/api/v1/billing/customers/{test_customer.id}/payment-methods")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "items" in data
        assert len(data["items"]) >= 1
        assert all(pm["customer_id"] == test_customer.id for pm in data["items"])
    
    def test_get_payment_method(self, client, test_payment_method_stripe):
        """Test GET /api/v1/billing/payment-methods/{payment_method_id}"""
        response = client.get(f"/api/v1/billing/payment-methods/{test_payment_method_stripe.id}")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["id"] == test_payment_method_stripe.id
        assert data["type"] == "card"
    
    def test_set_default_payment_method(self, client, test_payment_method_stripe, mock_payment_processor):
        """Test PUT /api/v1/billing/payment-methods/{payment_method_id}/default"""
        mock_payment_processor.set_default_payment_method.return_value = test_payment_method_stripe
        
        response = client.put(f"/api/v1/billing/payment-methods/{test_payment_method_stripe.id}/default")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["is_default"] is True
    
    def test_delete_payment_method(self, client, test_payment_method_stripe, mock_payment_processor):
        """Test DELETE /api/v1/billing/payment-methods/{payment_method_id}"""
        mock_payment_processor.delete_payment_method.return_value = True
        
        response = client.delete(f"/api/v1/billing/payment-methods/{test_payment_method_stripe.id}")
        
        assert response.status_code == status.HTTP_204_NO_CONTENT
        mock_payment_processor.delete_payment_method.assert_called_once()


class TestInvoiceEndpoints:
    """Test invoice management endpoints"""
    
    def test_get_invoice(self, client, test_invoice_draft):
        """Test GET /api/v1/billing/invoices/{invoice_id}"""
        response = client.get(f"/api/v1/billing/invoices/{test_invoice_draft.id}")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["id"] == test_invoice_draft.id
        assert data["number"] == test_invoice_draft.number
    
    def test_list_invoices(self, client, test_invoice_draft):
        """Test GET /api/v1/billing/invoices"""
        response = client.get("/api/v1/billing/invoices")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "items" in data
        assert len(data["items"]) >= 1
    
    def test_list_customer_invoices(self, client, test_customer, test_invoice_draft):
        """Test GET /api/v1/billing/customers/{customer_id}/invoices"""
        response = client.get(f"/api/v1/billing/customers/{test_customer.id}/invoices")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "items" in data
        assert all(inv["customer_id"] == test_customer.id for inv in data["items"])
    
    def test_pay_invoice(self, client, test_invoice_draft, test_payment_method_stripe, mock_billing_engine):
        """Test POST /api/v1/billing/invoices/{invoice_id}/pay"""
        payment_data = {
            "payment_method_id": test_payment_method_stripe.id
        }
        
        mock_payment = Mock()
        mock_payment.id = "pay_123"
        mock_payment.status = PaymentStatus.SUCCEEDED
        mock_payment.amount = test_invoice_draft.total
        mock_billing_engine.pay_invoice.return_value = mock_payment
        
        response = client.post(
            f"/api/v1/billing/invoices/{test_invoice_draft.id}/pay",
            json=payment_data
        )
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["status"] == "SUCCEEDED"
        mock_billing_engine.pay_invoice.assert_called_once()
    
    def test_download_invoice_pdf(self, client, test_invoice_paid, mock_invoice_service):
        """Test GET /api/v1/billing/invoices/{invoice_id}/pdf"""
        mock_pdf_data = b"PDF content"
        mock_invoice_service.generate_pdf.return_value = mock_pdf_data
        
        response = client.get(f"/api/v1/billing/invoices/{test_invoice_paid.id}/pdf")
        
        assert response.status_code == status.HTTP_200_OK
        assert response.headers["content-type"] == "application/pdf"
        assert response.content == mock_pdf_data
    
    def test_send_invoice_email(self, client, test_invoice_paid, mock_email_service):
        """Test POST /api/v1/billing/invoices/{invoice_id}/send"""
        send_data = {
            "recipient_email": "custom@example.com",
            "message": "Here is your invoice"
        }
        
        mock_email_service.send_invoice_email.return_value = True
        
        response = client.post(
            f"/api/v1/billing/invoices/{test_invoice_paid.id}/send",
            json=send_data
        )
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["sent"] is True
        mock_email_service.send_invoice_email.assert_called_once()


class TestPaymentEndpoints:
    """Test payment processing endpoints"""
    
    def test_create_payment(self, client, test_customer, test_payment_method_stripe, mock_payment_processor):
        """Test POST /api/v1/billing/payments"""
        payment_data = {
            "amount": "50.00",
            "currency": "EUR",
            "customer_id": test_customer.id,
            "payment_method_id": test_payment_method_stripe.id,
            "description": "Test payment"
        }
        
        mock_payment = Mock()
        mock_payment.id = "pay_123"
        mock_payment.status = PaymentStatus.SUCCEEDED
        mock_payment.amount = Decimal("50.00")
        mock_payment_processor.process_payment.return_value = mock_payment
        
        response = client.post("/api/v1/billing/payments", json=payment_data)
        
        assert response.status_code == status.HTTP_201_CREATED
        data = response.json()
        assert data["status"] == "SUCCEEDED"
        assert data["amount"] == "50.00"
    
    def test_get_payment(self, client, test_payment_successful):
        """Test GET /api/v1/billing/payments/{payment_id}"""
        response = client.get(f"/api/v1/billing/payments/{test_payment_successful.id}")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["id"] == test_payment_successful.id
        assert data["status"] == "SUCCEEDED"
    
    def test_list_payments(self, client, test_payment_successful):
        """Test GET /api/v1/billing/payments"""
        response = client.get("/api/v1/billing/payments")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "items" in data
        assert len(data["items"]) >= 1
    
    def test_refund_payment(self, client, test_payment_successful, mock_payment_processor):
        """Test POST /api/v1/billing/payments/{payment_id}/refund"""
        refund_data = {
            "amount": "10.00",
            "reason": "Customer request"
        }
        
        mock_refund_result = {
            "id": "re_123",
            "status": "succeeded",
            "amount": Decimal("10.00")
        }
        mock_payment_processor.process_refund.return_value = mock_refund_result
        
        response = client.post(
            f"/api/v1/billing/payments/{test_payment_successful.id}/refund",
            json=refund_data
        )
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["status"] == "succeeded"
        assert data["amount"] == "10.00"


class TestWebhookEndpoints:
    """Test webhook handling endpoints"""
    
    def test_stripe_webhook(self, client, mock_webhook_processor):
        """Test POST /api/v1/billing/webhooks/stripe"""
        webhook_payload = {
            "id": "evt_123",
            "type": "payment_intent.succeeded",
            "data": {
                "object": {
                    "id": "pi_123",
                    "status": "succeeded",
                    "amount": 5000,
                    "currency": "eur"
                }
            }
        }
        
        mock_webhook_processor.process_stripe_webhook.return_value = {"processed": True}
        
        headers = {
            "stripe-signature": "test_signature"
        }
        
        response = client.post(
            "/api/v1/billing/webhooks/stripe",
            json=webhook_payload,
            headers=headers
        )
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["processed"] is True
    
    def test_paypal_webhook(self, client, mock_webhook_processor):
        """Test POST /api/v1/billing/webhooks/paypal"""
        webhook_payload = {
            "id": "WH-123",
            "event_type": "PAYMENT.CAPTURE.COMPLETED",
            "resource": {
                "id": "PAYID-123",
                "status": "COMPLETED",
                "amount": {
                    "value": "50.00",
                    "currency_code": "EUR"
                }
            }
        }
        
        mock_webhook_processor.process_paypal_webhook.return_value = {"processed": True}
        
        response = client.post("/api/v1/billing/webhooks/paypal", json=webhook_payload)
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["processed"] is True


class TestAnalyticsEndpoints:
    """Test analytics and reporting endpoints"""
    
    def test_revenue_analytics(self, client, mock_analytics_service):
        """Test GET /api/v1/billing/analytics/revenue"""
        mock_analytics_service.get_revenue_analytics.return_value = {
            "total_revenue": "10000.00",
            "monthly_recurring_revenue": "2500.00",
            "annual_recurring_revenue": "30000.00",
            "revenue_by_month": [
                {"month": "2025-01", "revenue": "2500.00"},
                {"month": "2025-02", "revenue": "2750.00"}
            ]
        }
        
        response = client.get(
            "/api/v1/billing/analytics/revenue",
            params={
                "start_date": "2025-01-01",
                "end_date": "2025-02-28"
            }
        )
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "total_revenue" in data
        assert "monthly_recurring_revenue" in data
        assert "revenue_by_month" in data
    
    def test_subscription_analytics(self, client, mock_analytics_service):
        """Test GET /api/v1/billing/analytics/subscriptions"""
        mock_analytics_service.get_subscription_analytics.return_value = {
            "total_subscriptions": 150,
            "active_subscriptions": 120,
            "churned_subscriptions": 10,
            "trial_subscriptions": 20,
            "churn_rate": "6.67",
            "growth_rate": "15.38"
        }
        
        response = client.get("/api/v1/billing/analytics/subscriptions")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "total_subscriptions" in data
        assert "churn_rate" in data
        assert "growth_rate" in data


class TestErrorHandling:
    """Test API error handling"""
    
    def test_validation_error(self, client):
        """Test validation error response"""
        invalid_customer_data = {
            "email": "invalid-email",  # Invalid email format
            "name": "",  # Empty name
            "preferred_currency": "INVALID"  # Invalid currency
        }
        
        response = client.post("/api/v1/billing/customers", json=invalid_customer_data)
        
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
        data = response.json()
        assert "detail" in data
        assert len(data["detail"]) > 0  # Should have validation errors
    
    def test_unauthorized_access(self, client_no_auth):
        """Test unauthorized access response"""
        response = client_no_auth.get("/api/v1/billing/customers")
        
        assert response.status_code == status.HTTP_401_UNAUTHORIZED
        data = response.json()
        assert "detail" in data
    
    def test_rate_limiting(self, client):
        """Test rate limiting response"""
        # Simulate rate limit exceeded
        with patch('app.middleware.rate_limiter.is_allowed', return_value=False):
            response = client.get("/api/v1/billing/customers")
            
            assert response.status_code == status.HTTP_429_TOO_MANY_REQUESTS
            data = response.json()
            assert "rate limit" in data["detail"].lower()
    
    def test_internal_server_error(self, client, mock_billing_engine):
        """Test internal server error handling"""
        mock_billing_engine.create_customer.side_effect = Exception("Database connection failed")
        
        customer_data = {
            "email": "error@example.com",
            "name": "Error Test"
        }
        
        response = client.post("/api/v1/billing/customers", json=customer_data)
        
        assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
        data = response.json()
        assert "detail" in data
        # Should not expose internal error details
        assert "Database connection failed" not in data["detail"]
