"""
Tests for Background Tasks System
================================

Comprehensive tests for Celery background tasks and job processing.
"""

import pytest
from datetime import datetime, timedelta
from decimal import Decimal
from unittest.mock import AsyncMock, Mock, patch, MagicMock
from celery import Celery
import asyncio

from billing.tasks import (
    BillingTaskManager, process_subscription_renewal, process_payment_retry,
    generate_monthly_invoices, send_payment_reminders, calculate_revenue_metrics,
    process_dunning_workflow, cleanup_expired_data, sync_external_payments,
    process_webhook_queue, generate_scheduled_reports
)
from billing.models import (
    Subscription, Payment, Invoice, Customer, Plan,
    SubscriptionStatus, PaymentStatus, InvoiceStatus, PaymentProvider
)


class TestBillingTaskManager:
    """Test billing task manager functionality"""
    
    @pytest.mark.asyncio
    async def test_schedule_subscription_renewal(self, task_manager, test_subscription_active):
        """Test scheduling subscription renewal task"""
        renewal_date = datetime.utcnow() + timedelta(days=1)
        
        task_id = await task_manager.schedule_subscription_renewal(
            subscription_id=test_subscription_active.id,
            renewal_date=renewal_date
        )
        
        assert task_id is not None
        assert isinstance(task_id, str)
        
        # Verify task is scheduled
        task_info = await task_manager.get_task_status(task_id)
        assert task_info["status"] in ["PENDING", "SCHEDULED"]
        assert task_info["task_name"] == "process_subscription_renewal"
    
    @pytest.mark.asyncio
    async def test_schedule_payment_retry(self, task_manager, test_payment_failed):
        """Test scheduling payment retry task"""
        retry_date = datetime.utcnow() + timedelta(hours=2)
        
        task_id = await task_manager.schedule_payment_retry(
            payment_id=test_payment_failed.id,
            retry_date=retry_date,
            retry_count=1
        )
        
        assert task_id is not None
        
        # Verify task parameters
        task_info = await task_manager.get_task_status(task_id)
        assert task_info["args"][0] == test_payment_failed.id
        assert task_info["kwargs"]["retry_count"] == 1
    
    @pytest.mark.asyncio
    async def test_schedule_recurring_invoices(self, task_manager):
        """Test scheduling recurring invoice generation"""
        schedule_date = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
        
        task_id = await task_manager.schedule_recurring_invoices(
            schedule_date=schedule_date,
            batch_size=100
        )
        
        assert task_id is not None
        
        # Verify task configuration
        task_info = await task_manager.get_task_status(task_id)
        assert task_info["task_name"] == "generate_monthly_invoices"
        assert task_info["kwargs"]["batch_size"] == 100
    
    @pytest.mark.asyncio
    async def test_bulk_task_scheduling(self, task_manager, test_subscriptions_data):
        """Test bulk task scheduling for multiple subscriptions"""
        subscription_ids = [sub.id for sub in test_subscriptions_data[:5]]
        renewal_date = datetime.utcnow() + timedelta(days=1)
        
        task_ids = await task_manager.bulk_schedule_renewals(
            subscription_ids=subscription_ids,
            renewal_date=renewal_date
        )
        
        assert len(task_ids) == 5
        assert all(isinstance(task_id, str) for task_id in task_ids)
        
        # Verify all tasks are scheduled
        for task_id in task_ids:
            task_info = await task_manager.get_task_status(task_id)
            assert task_info["status"] in ["PENDING", "SCHEDULED"]
    
    @pytest.mark.asyncio
    async def test_task_cancellation(self, task_manager, test_subscription_active):
        """Test cancelling scheduled tasks"""
        # Schedule a task
        task_id = await task_manager.schedule_subscription_renewal(
            subscription_id=test_subscription_active.id,
            renewal_date=datetime.utcnow() + timedelta(days=1)
        )
        
        # Cancel the task
        cancelled = await task_manager.cancel_task(task_id)
        assert cancelled is True
        
        # Verify task is cancelled
        task_info = await task_manager.get_task_status(task_id)
        assert task_info["status"] == "REVOKED"
    
    @pytest.mark.asyncio
    async def test_task_retry_logic(self, task_manager):
        """Test task retry logic and failure handling"""
        # Mock a task that fails multiple times
        with patch('billing.tasks.process_payment_retry') as mock_task:
            mock_task.side_effect = [
                Exception("Temporary failure"),
                Exception("Another failure"),
                {"status": "success", "payment_id": "pay_123"}  # Success on 3rd try
            ]
            
            result = await task_manager.execute_with_retry(
                task_func=mock_task,
                args=("payment_123",),
                max_retries=3,
                retry_delay=0.1
            )
            
            assert result["status"] == "success"
            assert mock_task.call_count == 3
    
    @pytest.mark.asyncio
    async def test_task_monitoring(self, task_manager):
        """Test task monitoring and health checks"""
        # Get active tasks
        active_tasks = await task_manager.get_active_tasks()
        assert isinstance(active_tasks, list)
        
        # Get failed tasks
        failed_tasks = await task_manager.get_failed_tasks(limit=10)
        assert isinstance(failed_tasks, list)
        
        # Get task statistics
        stats = await task_manager.get_task_statistics()
        expected_keys = ["total_tasks", "pending_tasks", "running_tasks", "failed_tasks", "success_rate"]
        for key in expected_keys:
            assert key in stats


class TestSubscriptionRenewalTasks:
    """Test subscription renewal task processing"""
    
    @pytest.mark.asyncio
    async def test_process_subscription_renewal_success(self, mock_billing_engine, test_subscription_active, test_payment_method_stripe):
        """Test successful subscription renewal processing"""
        # Mock successful payment processing
        mock_payment = Mock()
        mock_payment.status = PaymentStatus.SUCCEEDED
        mock_payment.amount = test_subscription_active.effective_amount
        mock_billing_engine.process_subscription_payment.return_value = mock_payment
        
        # Mock invoice generation
        mock_invoice = Mock()
        mock_invoice.status = InvoiceStatus.PAID
        mock_billing_engine.generate_subscription_invoice.return_value = mock_invoice
        
        result = await process_subscription_renewal(
            subscription_id=test_subscription_active.id,
            billing_engine=mock_billing_engine
        )
        
        assert result["status"] == "success"
        assert result["payment_id"] is not None
        assert result["invoice_id"] is not None
        
        # Verify billing engine calls
        mock_billing_engine.generate_subscription_invoice.assert_called_once()
        mock_billing_engine.process_subscription_payment.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_process_subscription_renewal_payment_failure(self, mock_billing_engine, test_subscription_active):
        """Test subscription renewal with payment failure"""
        # Mock failed payment
        mock_billing_engine.process_subscription_payment.side_effect = Exception("Payment failed")
        
        result = await process_subscription_renewal(
            subscription_id=test_subscription_active.id,
            billing_engine=mock_billing_engine
        )
        
        assert result["status"] == "payment_failed"
        assert "error" in result
        
        # Should schedule retry
        assert result["retry_scheduled"] is True
        assert result["next_retry"] is not None
    
    @pytest.mark.asyncio
    async def test_process_subscription_renewal_dunning(self, mock_billing_engine, test_subscription_active):
        """Test subscription renewal with dunning management"""
        # Mock multiple payment failures to trigger dunning
        test_subscription_active.failed_payment_count = 2
        
        mock_billing_engine.process_subscription_payment.side_effect = Exception("Card declined")
        mock_billing_engine.process_dunning.return_value = {
            "action": "suspend_subscription",
            "notification_sent": True
        }
        
        result = await process_subscription_renewal(
            subscription_id=test_subscription_active.id,
            billing_engine=mock_billing_engine,
            enable_dunning=True
        )
        
        assert result["status"] == "dunning_triggered"
        assert result["dunning_action"] == "suspend_subscription"
        
        # Verify dunning process was called
        mock_billing_engine.process_dunning.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_batch_subscription_renewals(self, mock_billing_engine, test_subscriptions_data):
        """Test batch processing of subscription renewals"""
        subscription_ids = [sub.id for sub in test_subscriptions_data[:3]]
        
        # Mock successful renewals
        mock_billing_engine.process_subscription_payment.return_value = Mock(status=PaymentStatus.SUCCEEDED)
        mock_billing_engine.generate_subscription_invoice.return_value = Mock(status=InvoiceStatus.PAID)
        
        results = await asyncio.gather(*[
            process_subscription_renewal(sub_id, mock_billing_engine)
            for sub_id in subscription_ids
        ])
        
        assert len(results) == 3
        assert all(result["status"] == "success" for result in results)
        
        # Verify batch processing efficiency
        assert mock_billing_engine.generate_subscription_invoice.call_count == 3
        assert mock_billing_engine.process_subscription_payment.call_count == 3


class TestPaymentRetryTasks:
    """Test payment retry task processing"""
    
    @pytest.mark.asyncio
    async def test_process_payment_retry_success(self, mock_payment_processor, test_payment_failed):
        """Test successful payment retry"""
        # Mock successful retry
        mock_payment_processor.retry_payment.return_value = Mock(
            status=PaymentStatus.SUCCEEDED,
            amount=test_payment_failed.amount
        )
        
        result = await process_payment_retry(
            payment_id=test_payment_failed.id,
            payment_processor=mock_payment_processor,
            retry_count=1
        )
        
        assert result["status"] == "success"
        assert result["retry_count"] == 1
        assert result["final_status"] == "SUCCEEDED"
        
        mock_payment_processor.retry_payment.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_process_payment_retry_failure(self, mock_payment_processor, test_payment_failed):
        """Test payment retry failure with escalation"""
        # Mock continued failure
        mock_payment_processor.retry_payment.side_effect = Exception("Card still declined")
        
        result = await process_payment_retry(
            payment_id=test_payment_failed.id,
            payment_processor=mock_payment_processor,
            retry_count=3,
            max_retries=3
        )
        
        assert result["status"] == "failed"
        assert result["retry_count"] == 3
        assert result["max_retries_reached"] is True
        
        # Should trigger escalation
        assert result["escalation_triggered"] is True
    
    @pytest.mark.asyncio
    async def test_intelligent_retry_scheduling(self, mock_payment_processor, test_payment_failed):
        """Test intelligent retry scheduling based on failure reason"""
        # Mock specific failure reasons
        failure_scenarios = [
            ("insufficient_funds", timedelta(hours=24)),
            ("card_declined", timedelta(hours=2)),
            ("expired_card", timedelta(minutes=5)),
            ("network_error", timedelta(minutes=15))
        ]
        
        for failure_reason, expected_delay in failure_scenarios:
            test_payment_failed.failure_reason = failure_reason
            
            result = await process_payment_retry(
                payment_id=test_payment_failed.id,
                payment_processor=mock_payment_processor,
                retry_count=1,
                intelligent_scheduling=True
            )
            
            # Verify retry delay is appropriate for failure reason
            if result["next_retry_scheduled"]:
                next_retry = datetime.fromisoformat(result["next_retry_date"])
                actual_delay = next_retry - datetime.utcnow()
                
                # Allow some tolerance in timing
                assert abs(actual_delay - expected_delay) < timedelta(minutes=5)


class TestInvoiceGenerationTasks:
    """Test invoice generation task processing"""
    
    @pytest.mark.asyncio
    async def test_generate_monthly_invoices(self, mock_invoice_service, test_subscriptions_data):
        """Test monthly invoice generation task"""
        # Filter active subscriptions
        active_subs = [sub for sub in test_subscriptions_data if sub.status == SubscriptionStatus.ACTIVE]
        
        # Mock invoice generation
        mock_invoice_service.generate_subscription_invoice.return_value = Mock(
            id="inv_123",
            status=InvoiceStatus.OPEN,
            total=Decimal("99.99")
        )
        
        result = await generate_monthly_invoices(
            invoice_service=mock_invoice_service,
            billing_date=datetime.utcnow().date(),
            batch_size=50
        )
        
        assert result["status"] == "completed"
        assert result["invoices_generated"] >= len(active_subs)
        assert result["total_amount"] > Decimal("0")
        
        # Verify invoice service calls
        assert mock_invoice_service.generate_subscription_invoice.call_count >= len(active_subs)
    
    @pytest.mark.asyncio
    async def test_generate_usage_invoices(self, mock_invoice_service, test_subscriptions_data):
        """Test usage-based invoice generation"""
        # Mock usage data
        usage_records = [
            {"subscription_id": "sub_1", "metric": "api_calls", "quantity": 1500, "unit_price": 0.01},
            {"subscription_id": "sub_2", "metric": "storage", "quantity": 100, "unit_price": 0.50}
        ]
        
        mock_invoice_service.generate_usage_invoice.return_value = Mock(
            id="inv_usage_123",
            status=InvoiceStatus.OPEN,
            total=Decimal("65.00")
        )
        
        result = await generate_monthly_invoices(
            invoice_service=mock_invoice_service,
            billing_date=datetime.utcnow().date(),
            include_usage=True,
            usage_data=usage_records
        )
        
        assert result["usage_invoices_generated"] == len(usage_records)
        assert mock_invoice_service.generate_usage_invoice.call_count == len(usage_records)
    
    @pytest.mark.asyncio
    async def test_invoice_generation_error_handling(self, mock_invoice_service, test_subscription_active):
        """Test invoice generation with error handling"""
        # Mock partial failures
        mock_invoice_service.generate_subscription_invoice.side_effect = [
            Mock(id="inv_1", status=InvoiceStatus.OPEN),  # Success
            Exception("Database error"),  # Failure
            Mock(id="inv_3", status=InvoiceStatus.OPEN)   # Success
        ]
        
        result = await generate_monthly_invoices(
            invoice_service=mock_invoice_service,
            billing_date=datetime.utcnow().date(),
            batch_size=3,
            continue_on_error=True
        )
        
        assert result["status"] == "completed_with_errors"
        assert result["invoices_generated"] == 2
        assert result["failed_invoices"] == 1
        assert len(result["errors"]) == 1


class TestReminderTasks:
    """Test payment reminder task processing"""
    
    @pytest.mark.asyncio
    async def test_send_payment_reminders(self, mock_email_service, mock_invoice_service):
        """Test sending payment reminders for overdue invoices"""
        # Mock overdue invoices
        overdue_invoices = [
            Mock(id="inv_1", customer=Mock(email="customer1@example.com"), 
                 days_overdue=5, amount_due=Decimal("99.99")),
            Mock(id="inv_2", customer=Mock(email="customer2@example.com"), 
                 days_overdue=15, amount_due=Decimal("199.99")),
            Mock(id="inv_3", customer=Mock(email="customer3@example.com"), 
                 days_overdue=30, amount_due=Decimal("299.99"))
        ]
        
        mock_invoice_service.get_overdue_invoices.return_value = overdue_invoices
        mock_email_service.send_payment_reminder.return_value = True
        
        result = await send_payment_reminders(
            email_service=mock_email_service,
            invoice_service=mock_invoice_service
        )
        
        assert result["status"] == "completed"
        assert result["reminders_sent"] == 3
        assert result["total_amount_reminded"] == Decimal("599.97")
        
        # Verify reminder types based on days overdue
        email_calls = mock_email_service.send_payment_reminder.call_args_list
        assert len(email_calls) == 3
    
    @pytest.mark.asyncio
    async def test_escalated_reminders(self, mock_email_service, mock_invoice_service):
        """Test escalated reminders for severely overdue invoices"""
        # Mock severely overdue invoice
        overdue_invoice = Mock(
            id="inv_overdue",
            customer=Mock(email="overdue@example.com", name="Overdue Customer"),
            days_overdue=45,
            amount_due=Decimal("500.00")
        )
        
        mock_invoice_service.get_overdue_invoices.return_value = [overdue_invoice]
        mock_email_service.send_escalated_reminder.return_value = True
        
        result = await send_payment_reminders(
            email_service=mock_email_service,
            invoice_service=mock_invoice_service,
            escalation_threshold=30
        )
        
        assert result["escalated_reminders"] == 1
        mock_email_service.send_escalated_reminder.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_reminder_frequency_control(self, mock_email_service, mock_invoice_service):
        """Test reminder frequency control to avoid spam"""
        # Mock invoice with recent reminder
        recent_reminder_invoice = Mock(
            id="inv_recent",
            customer=Mock(email="recent@example.com"),
            days_overdue=10,
            last_reminder_sent=datetime.utcnow() - timedelta(days=2)  # Sent 2 days ago
        )
        
        mock_invoice_service.get_overdue_invoices.return_value = [recent_reminder_invoice]
        
        result = await send_payment_reminders(
            email_service=mock_email_service,
            invoice_service=mock_invoice_service,
            min_reminder_interval_days=7  # Don't send more than once per week
        )
        
        assert result["reminders_skipped"] == 1
        mock_email_service.send_payment_reminder.assert_not_called()


class TestMetricsCalculationTasks:
    """Test metrics calculation task processing"""
    
    @pytest.mark.asyncio
    async def test_calculate_revenue_metrics(self, mock_analytics_service):
        """Test revenue metrics calculation task"""
        # Mock analytics data
        mock_analytics_service.get_revenue_analytics.return_value = {
            "total_revenue": Decimal("50000.00"),
            "mrr": Decimal("12500.00"),
            "arr": Decimal("150000.00"),
            "growth_rate": 15.5
        }
        
        mock_analytics_service.get_subscription_analytics.return_value = {
            "total_subscriptions": 500,
            "churn_rate": 5.2,
            "ltv": Decimal("2400.00")
        }
        
        result = await calculate_revenue_metrics(
            analytics_service=mock_analytics_service,
            period="monthly"
        )
        
        assert result["status"] == "completed"
        assert "revenue_metrics" in result
        assert "subscription_metrics" in result
        assert result["calculated_at"] is not None
        
        # Verify metrics are calculated
        revenue_metrics = result["revenue_metrics"]
        assert revenue_metrics["total_revenue"] == Decimal("50000.00")
        assert revenue_metrics["mrr"] == Decimal("12500.00")
    
    @pytest.mark.asyncio
    async def test_metrics_caching(self, mock_analytics_service, mock_redis):
        """Test metrics caching for performance"""
        cache_key = "revenue_metrics_monthly"
        
        # First calculation - should cache result
        mock_redis.get.return_value = None  # Cache miss
        mock_analytics_service.get_revenue_analytics.return_value = {"mrr": Decimal("10000.00")}
        
        result = await calculate_revenue_metrics(
            analytics_service=mock_analytics_service,
            period="monthly",
            cache_ttl=3600
        )
        
        assert result["status"] == "completed"
        mock_redis.set.assert_called_once()
        
        # Second calculation - should use cache
        mock_redis.get.return_value = '{"mrr": "10000.00"}'  # Cache hit
        
        result2 = await calculate_revenue_metrics(
            analytics_service=mock_analytics_service,
            period="monthly",
            cache_ttl=3600
        )
        
        # Should not call analytics service again
        assert mock_analytics_service.get_revenue_analytics.call_count == 1


class TestMaintenanceTasks:
    """Test maintenance and cleanup task processing"""
    
    @pytest.mark.asyncio
    async def test_cleanup_expired_data(self, db_session):
        """Test cleanup of expired data"""
        # Mock expired data
        cutoff_date = datetime.utcnow() - timedelta(days=365)
        
        result = await cleanup_expired_data(
            cutoff_date=cutoff_date,
            data_types=["payment_logs", "webhook_events", "failed_tasks"]
        )
        
        assert result["status"] == "completed"
        assert "records_deleted" in result
        assert "data_types_processed" in result
        
        # Verify cleanup targets
        assert "payment_logs" in result["data_types_processed"]
        assert "webhook_events" in result["data_types_processed"]
    
    @pytest.mark.asyncio
    async def test_sync_external_payments(self, mock_payment_processor):
        """Test syncing payments with external providers"""
        # Mock external payment data
        external_payments = [
            {"provider_id": "pi_stripe_1", "status": "succeeded", "amount": "99.99"},
            {"provider_id": "PAYID_paypal_1", "status": "completed", "amount": "149.99"}
        ]
        
        mock_payment_processor.sync_stripe_payments.return_value = external_payments[:1]
        mock_payment_processor.sync_paypal_payments.return_value = external_payments[1:]
        
        result = await sync_external_payments(
            payment_processor=mock_payment_processor,
            providers=["stripe", "paypal"],
            sync_hours=24
        )
        
        assert result["status"] == "completed"
        assert result["payments_synced"] == 2
        assert result["providers_synced"] == ["stripe", "paypal"]
    
    @pytest.mark.asyncio
    async def test_process_webhook_queue(self, mock_webhook_processor):
        """Test processing queued webhook events"""
        # Mock queued webhooks
        queued_webhooks = [
            {"id": "wh_1", "provider": "stripe", "event_type": "payment_intent.succeeded"},
            {"id": "wh_2", "provider": "paypal", "event_type": "PAYMENT.CAPTURE.COMPLETED"},
            {"id": "wh_3", "provider": "stripe", "event_type": "invoice.payment_succeeded"}
        ]
        
        mock_webhook_processor.get_queued_webhooks.return_value = queued_webhooks
        mock_webhook_processor.process_webhook.return_value = {"processed": True}
        
        result = await process_webhook_queue(
            webhook_processor=mock_webhook_processor,
            batch_size=10
        )
        
        assert result["status"] == "completed"
        assert result["webhooks_processed"] == 3
        assert result["success_count"] == 3
        assert result["failure_count"] == 0


class TestTaskIntegration:
    """Integration tests for task system"""
    
    @pytest.mark.asyncio
    async def test_end_to_end_billing_cycle(self, task_manager, mock_billing_engine, test_subscription_active):
        """Test complete billing cycle task orchestration"""
        # 1. Schedule invoice generation
        invoice_task_id = await task_manager.schedule_invoice_generation(
            subscription_id=test_subscription_active.id,
            due_date=datetime.utcnow() + timedelta(days=1)
        )
        
        # 2. Mock invoice generation success
        mock_billing_engine.generate_subscription_invoice.return_value = Mock(
            id="inv_123",
            status=InvoiceStatus.OPEN,
            total=Decimal("99.99")
        )
        
        # 3. Execute invoice generation
        invoice_result = await task_manager.execute_task(invoice_task_id)
        assert invoice_result["status"] == "completed"
        
        # 4. Schedule payment processing
        payment_task_id = await task_manager.schedule_payment_processing(
            invoice_id="inv_123",
            payment_date=datetime.utcnow()
        )
        
        # 5. Mock payment success
        mock_billing_engine.process_invoice_payment.return_value = Mock(
            status=PaymentStatus.SUCCEEDED,
            amount=Decimal("99.99")
        )
        
        # 6. Execute payment processing
        payment_result = await task_manager.execute_task(payment_task_id)
        assert payment_result["status"] == "completed"
        
        # 7. Verify billing cycle completion
        cycle_summary = await task_manager.get_billing_cycle_summary(test_subscription_active.id)
        assert cycle_summary["invoice_generated"] is True
        assert cycle_summary["payment_processed"] is True
        assert cycle_summary["cycle_status"] == "completed"
