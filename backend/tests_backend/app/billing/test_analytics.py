"""
Tests for Analytics and Reporting System
=======================================

Comprehensive tests for billing analytics, reporting, and forecasting.
"""

import pytest
from datetime import datetime, timedelta
from decimal import Decimal
from unittest.mock import AsyncMock, Mock, patch, MagicMock
import pandas as pd
import numpy as np

from billing.analytics import AnalyticsService, ReportGenerator, ForecastingEngine
from billing.models import (
    Customer, Plan, Subscription, Payment, Invoice, 
    SubscriptionStatus, PaymentStatus, InvoiceStatus,
    PlanInterval, PaymentProvider
)


class TestAnalyticsService:
    """Test analytics service functionality"""
    
    @pytest.mark.asyncio
    async def test_revenue_analytics(self, analytics_service, db_session, test_payments_data):
        """Test revenue analytics calculation"""
        start_date = datetime.utcnow() - timedelta(days=90)
        end_date = datetime.utcnow()
        
        revenue_data = await analytics_service.get_revenue_analytics(
            start_date=start_date,
            end_date=end_date
        )
        
        assert "total_revenue" in revenue_data
        assert "monthly_recurring_revenue" in revenue_data
        assert "annual_recurring_revenue" in revenue_data
        assert "revenue_by_period" in revenue_data
        assert "revenue_growth_rate" in revenue_data
        
        # Verify calculations
        assert isinstance(revenue_data["total_revenue"], Decimal)
        assert revenue_data["total_revenue"] > Decimal('0')
        assert isinstance(revenue_data["monthly_recurring_revenue"], Decimal)
    
    @pytest.mark.asyncio
    async def test_subscription_analytics(self, analytics_service, db_session, test_subscriptions_data):
        """Test subscription analytics calculation"""
        analytics_data = await analytics_service.get_subscription_analytics()
        
        expected_keys = [
            "total_subscriptions",
            "active_subscriptions", 
            "trial_subscriptions",
            "cancelled_subscriptions",
            "churned_subscriptions",
            "churn_rate",
            "growth_rate",
            "ltv",  # Lifetime Value
            "avg_subscription_length"
        ]
        
        for key in expected_keys:
            assert key in analytics_data
        
        # Verify data types and ranges
        assert isinstance(analytics_data["total_subscriptions"], int)
        assert analytics_data["total_subscriptions"] >= 0
        assert isinstance(analytics_data["churn_rate"], float)
        assert 0 <= analytics_data["churn_rate"] <= 100
    
    @pytest.mark.asyncio
    async def test_customer_analytics(self, analytics_service, db_session, test_customers_data):
        """Test customer analytics calculation"""
        customer_data = await analytics_service.get_customer_analytics()
        
        expected_keys = [
            "total_customers",
            "new_customers_this_month",
            "customer_growth_rate",
            "customers_by_country",
            "customers_by_plan",
            "avg_customer_value",
            "customer_lifetime_value"
        ]
        
        for key in expected_keys:
            assert key in customer_data
        
        # Verify geographic distribution
        assert isinstance(customer_data["customers_by_country"], dict)
        assert len(customer_data["customers_by_country"]) > 0
        
        # Verify plan distribution
        assert isinstance(customer_data["customers_by_plan"], dict)
    
    @pytest.mark.asyncio
    async def test_payment_analytics(self, analytics_service, db_session, test_payments_data):
        """Test payment analytics calculation"""
        payment_data = await analytics_service.get_payment_analytics(
            start_date=datetime.utcnow() - timedelta(days=30),
            end_date=datetime.utcnow()
        )
        
        expected_keys = [
            "total_payments",
            "successful_payments",
            "failed_payments",
            "success_rate",
            "payment_volume",
            "average_payment_amount",
            "payments_by_provider",
            "payment_methods_distribution",
            "decline_reasons"
        ]
        
        for key in expected_keys:
            assert key in payment_data
        
        # Verify success rate calculation
        total = payment_data["total_payments"]
        successful = payment_data["successful_payments"]
        if total > 0:
            expected_rate = (successful / total) * 100
            assert abs(payment_data["success_rate"] - expected_rate) < 0.01
    
    @pytest.mark.asyncio
    async def test_cohort_analysis(self, analytics_service, db_session):
        """Test customer cohort analysis"""
        cohort_data = await analytics_service.get_cohort_analysis(
            cohort_type="monthly",
            start_date=datetime.utcnow() - timedelta(days=365),
            end_date=datetime.utcnow()
        )
        
        assert "cohort_sizes" in cohort_data
        assert "retention_rates" in cohort_data
        assert "revenue_cohorts" in cohort_data
        
        # Verify cohort structure
        retention_rates = cohort_data["retention_rates"]
        assert isinstance(retention_rates, dict)
        
        # Each cohort should have retention rates for subsequent months
        for cohort_month, rates in retention_rates.items():
            assert isinstance(rates, list)
            assert all(0 <= rate <= 100 for rate in rates)
    
    @pytest.mark.asyncio
    async def test_mrr_movement_analysis(self, analytics_service, db_session):
        """Test Monthly Recurring Revenue movement analysis"""
        mrr_data = await analytics_service.get_mrr_movement(
            start_date=datetime.utcnow() - timedelta(days=90),
            end_date=datetime.utcnow()
        )
        
        expected_keys = [
            "starting_mrr",
            "ending_mrr",
            "new_business",
            "expansion", 
            "contraction",
            "churn",
            "net_movement",
            "growth_rate"
        ]
        
        for key in expected_keys:
            assert key in mrr_data
        
        # Verify MRR movement equation: 
        # Ending MRR = Starting MRR + New Business + Expansion - Contraction - Churn
        starting = mrr_data["starting_mrr"]
        ending = mrr_data["ending_mrr"]
        new_business = mrr_data["new_business"]
        expansion = mrr_data["expansion"]
        contraction = mrr_data["contraction"]
        churn = mrr_data["churn"]
        
        calculated_ending = starting + new_business + expansion - contraction - churn
        assert abs(ending - calculated_ending) < Decimal('0.01')
    
    @pytest.mark.asyncio
    async def test_funnel_analysis(self, analytics_service, db_session):
        """Test conversion funnel analysis"""
        funnel_data = await analytics_service.get_conversion_funnel(
            start_date=datetime.utcnow() - timedelta(days=30),
            end_date=datetime.utcnow()
        )
        
        expected_stages = [
            "visitors",
            "signups", 
            "trial_starts",
            "trial_conversions",
            "paying_customers",
            "retained_customers"
        ]
        
        for stage in expected_stages:
            assert stage in funnel_data
            assert isinstance(funnel_data[stage], int)
            assert funnel_data[stage] >= 0
        
        # Verify funnel logic (each stage should be <= previous stage)
        assert funnel_data["signups"] <= funnel_data["visitors"]
        assert funnel_data["trial_starts"] <= funnel_data["signups"]
        assert funnel_data["trial_conversions"] <= funnel_data["trial_starts"]
        assert funnel_data["paying_customers"] <= funnel_data["trial_conversions"]
    
    @pytest.mark.asyncio
    async def test_geographic_analytics(self, analytics_service, db_session):
        """Test geographic revenue and customer distribution"""
        geo_data = await analytics_service.get_geographic_analytics()
        
        assert "revenue_by_country" in geo_data
        assert "customers_by_country" in geo_data
        assert "top_countries" in geo_data
        assert "country_growth_rates" in geo_data
        
        # Verify data structure
        revenue_by_country = geo_data["revenue_by_country"]
        assert isinstance(revenue_by_country, dict)
        
        for country_code, revenue in revenue_by_country.items():
            assert len(country_code) == 2  # ISO country code
            assert isinstance(revenue, Decimal)
            assert revenue >= Decimal('0')


class TestReportGenerator:
    """Test report generation functionality"""
    
    @pytest.mark.asyncio
    async def test_monthly_revenue_report(self, report_generator, db_session):
        """Test monthly revenue report generation"""
        report_date = datetime.utcnow().replace(day=1)  # First day of current month
        
        report = await report_generator.generate_monthly_revenue_report(report_date)
        
        assert "report_period" in report
        assert "summary" in report
        assert "revenue_breakdown" in report
        assert "year_over_year_comparison" in report
        assert "top_customers" in report
        assert "generated_at" in report
        
        # Verify summary data
        summary = report["summary"]
        assert "total_revenue" in summary
        assert "mrr" in summary
        assert "new_customers" in summary
        assert "churn_rate" in summary
    
    @pytest.mark.asyncio
    async def test_subscription_health_report(self, report_generator, db_session):
        """Test subscription health report generation"""
        report = await report_generator.generate_subscription_health_report()
        
        assert "subscription_metrics" in report
        assert "churn_analysis" in report
        assert "upgrade_downgrade_analysis" in report
        assert "trial_conversion_rates" in report
        assert "recommendations" in report
        
        # Verify recommendations are provided
        recommendations = report["recommendations"]
        assert isinstance(recommendations, list)
        assert len(recommendations) > 0
    
    @pytest.mark.asyncio
    async def test_financial_summary_report(self, report_generator, db_session):
        """Test financial summary report generation"""
        start_date = datetime.utcnow() - timedelta(days=90)
        end_date = datetime.utcnow()
        
        report = await report_generator.generate_financial_summary(
            start_date=start_date,
            end_date=end_date
        )
        
        expected_sections = [
            "revenue_summary",
            "payment_summary", 
            "outstanding_invoices",
            "tax_summary",
            "refund_summary",
            "key_metrics"
        ]
        
        for section in expected_sections:
            assert section in report
        
        # Verify outstanding invoices section
        outstanding = report["outstanding_invoices"]
        assert "total_amount" in outstanding
        assert "count" in outstanding
        assert "overdue_amount" in outstanding
    
    @pytest.mark.asyncio
    async def test_customer_segmentation_report(self, report_generator, db_session):
        """Test customer segmentation report generation"""
        report = await report_generator.generate_customer_segmentation_report()
        
        assert "segments" in report
        assert "segment_analysis" in report
        assert "revenue_by_segment" in report
        assert "churn_by_segment" in report
        
        # Verify segment definitions
        segments = report["segments"]
        expected_segments = ["high_value", "medium_value", "low_value", "at_risk"]
        
        for segment in expected_segments:
            assert segment in segments
            assert "customer_count" in segments[segment]
            assert "revenue_contribution" in segments[segment]
    
    @pytest.mark.asyncio
    async def test_export_report_csv(self, report_generator, db_session):
        """Test exporting report data to CSV"""
        # Generate sample revenue data
        revenue_data = [
            {"month": "2025-01", "revenue": Decimal("10000.00"), "customers": 100},
            {"month": "2025-02", "revenue": Decimal("12000.00"), "customers": 120},
            {"month": "2025-03", "revenue": Decimal("14000.00"), "customers": 140},
        ]
        
        csv_data = await report_generator.export_to_csv(
            data=revenue_data,
            filename="revenue_report.csv"
        )
        
        assert csv_data is not None
        assert "month,revenue,customers" in csv_data  # CSV header
        assert "2025-01,10000.00,100" in csv_data
        assert "2025-02,12000.00,120" in csv_data
    
    @pytest.mark.asyncio
    async def test_export_report_pdf(self, report_generator, db_session):
        """Test exporting report to PDF"""
        report_data = {
            "title": "Monthly Revenue Report",
            "period": "January 2025",
            "summary": {
                "total_revenue": Decimal("50000.00"),
                "new_customers": 25,
                "churn_rate": 5.2
            }
        }
        
        pdf_data = await report_generator.export_to_pdf(
            report_data=report_data,
            template="monthly_revenue_template.html"
        )
        
        assert pdf_data is not None
        assert len(pdf_data) > 0
        assert pdf_data.startswith(b'%PDF')  # PDF header
    
    @pytest.mark.asyncio
    async def test_scheduled_report_generation(self, report_generator, mock_scheduler):
        """Test scheduled report generation"""
        report_config = {
            "report_type": "monthly_revenue",
            "schedule": "monthly",
            "recipients": ["finance@company.com", "ceo@company.com"],
            "format": "pdf"
        }
        
        with patch.object(report_generator, 'email_service') as mock_email:
            mock_email.send_report_email.return_value = True
            
            result = await report_generator.schedule_report(report_config)
            
            assert result["scheduled"] is True
            assert result["next_run"] is not None
            mock_scheduler.add_job.assert_called_once()


class TestForecastingEngine:
    """Test forecasting and predictive analytics"""
    
    @pytest.mark.asyncio
    async def test_revenue_forecasting(self, forecasting_engine, db_session):
        """Test revenue forecasting"""
        # Historical revenue data
        historical_data = [
            {"month": "2024-01", "revenue": 8000.00},
            {"month": "2024-02", "revenue": 8500.00},
            {"month": "2024-03", "revenue": 9200.00},
            {"month": "2024-04", "revenue": 9800.00},
            {"month": "2024-05", "revenue": 10500.00},
            {"month": "2024-06", "revenue": 11200.00},
        ]
        
        forecast = await forecasting_engine.forecast_revenue(
            historical_data=historical_data,
            forecast_periods=6,  # 6 months ahead
            model_type="linear_regression"
        )
        
        assert "forecast" in forecast
        assert "confidence_intervals" in forecast
        assert "model_accuracy" in forecast
        assert "trend" in forecast
        
        # Verify forecast structure
        forecast_data = forecast["forecast"]
        assert len(forecast_data) == 6  # 6 months
        
        for month_forecast in forecast_data:
            assert "month" in month_forecast
            assert "predicted_revenue" in month_forecast
            assert isinstance(month_forecast["predicted_revenue"], float)
            assert month_forecast["predicted_revenue"] > 0
    
    @pytest.mark.asyncio
    async def test_churn_prediction(self, forecasting_engine, db_session):
        """Test customer churn prediction"""
        # Customer feature data
        customer_features = [
            {
                "customer_id": "cust_1",
                "subscription_length_days": 365,
                "last_payment_date": "2024-12-15",
                "payment_failures": 0,
                "support_tickets": 2,
                "feature_usage_score": 8.5,
                "plan_value": 99.99
            },
            {
                "customer_id": "cust_2", 
                "subscription_length_days": 45,
                "last_payment_date": "2024-11-20",
                "payment_failures": 2,
                "support_tickets": 8,
                "feature_usage_score": 3.2,
                "plan_value": 29.99
            }
        ]
        
        predictions = await forecasting_engine.predict_churn(
            customer_features=customer_features,
            model_type="random_forest"
        )
        
        assert len(predictions) == 2
        
        for prediction in predictions:
            assert "customer_id" in prediction
            assert "churn_probability" in prediction
            assert "risk_level" in prediction
            assert "factors" in prediction
            
            # Verify probability is between 0 and 1
            prob = prediction["churn_probability"]
            assert 0 <= prob <= 1
            
            # Verify risk level categorization
            risk = prediction["risk_level"]
            assert risk in ["low", "medium", "high"]
    
    @pytest.mark.asyncio
    async def test_ltv_prediction(self, forecasting_engine, db_session):
        """Test customer lifetime value prediction"""
        customer_data = [
            {
                "customer_id": "cust_1",
                "monthly_revenue": 99.99,
                "subscription_start": "2024-01-15",
                "payment_history": [99.99, 99.99, 99.99, 99.99],
                "churn_probability": 0.1
            },
            {
                "customer_id": "cust_2",
                "monthly_revenue": 29.99,
                "subscription_start": "2024-06-01",
                "payment_history": [29.99, 29.99],
                "churn_probability": 0.7
            }
        ]
        
        ltv_predictions = await forecasting_engine.predict_ltv(
            customer_data=customer_data,
            time_horizon_months=24
        )
        
        assert len(ltv_predictions) == 2
        
        for prediction in ltv_predictions:
            assert "customer_id" in prediction
            assert "predicted_ltv" in prediction
            assert "confidence_score" in prediction
            assert "expected_lifetime_months" in prediction
            
            # LTV should be positive
            assert prediction["predicted_ltv"] > 0
            
            # Confidence should be between 0 and 1
            assert 0 <= prediction["confidence_score"] <= 1
    
    @pytest.mark.asyncio
    async def test_demand_forecasting(self, forecasting_engine, db_session):
        """Test demand forecasting for subscription plans"""
        plan_demand_data = [
            {"month": "2024-01", "plan_id": "plan_basic", "new_subscriptions": 50},
            {"month": "2024-02", "plan_id": "plan_basic", "new_subscriptions": 55},
            {"month": "2024-03", "plan_id": "plan_basic", "new_subscriptions": 62},
            {"month": "2024-01", "plan_id": "plan_premium", "new_subscriptions": 20},
            {"month": "2024-02", "plan_id": "plan_premium", "new_subscriptions": 25},
            {"month": "2024-03", "plan_id": "plan_premium", "new_subscriptions": 30},
        ]
        
        demand_forecast = await forecasting_engine.forecast_demand(
            historical_data=plan_demand_data,
            forecast_periods=3,
            seasonality=True
        )
        
        assert "forecasts_by_plan" in demand_forecast
        assert "total_demand_forecast" in demand_forecast
        assert "seasonality_factors" in demand_forecast
        
        # Verify plan-specific forecasts
        plan_forecasts = demand_forecast["forecasts_by_plan"]
        assert "plan_basic" in plan_forecasts
        assert "plan_premium" in plan_forecasts
        
        for plan_id, forecast_data in plan_forecasts.items():
            assert len(forecast_data) == 3  # 3 months forecast
            for month_data in forecast_data:
                assert "month" in month_data
                assert "predicted_subscriptions" in month_data
    
    @pytest.mark.asyncio
    async def test_anomaly_detection(self, forecasting_engine, db_session):
        """Test anomaly detection in revenue patterns"""
        revenue_data = [
            {"date": "2024-01-01", "revenue": 1000.00},
            {"date": "2024-01-02", "revenue": 1050.00},
            {"date": "2024-01-03", "revenue": 980.00},
            {"date": "2024-01-04", "revenue": 1200.00},  # Potential anomaly
            {"date": "2024-01-05", "revenue": 5000.00},  # Clear anomaly
            {"date": "2024-01-06", "revenue": 1100.00},
            {"date": "2024-01-07", "revenue": 950.00},
        ]
        
        anomalies = await forecasting_engine.detect_anomalies(
            data=revenue_data,
            metric="revenue",
            sensitivity=0.05  # 95% confidence
        )
        
        assert "anomalies" in anomalies
        assert "anomaly_dates" in anomalies
        assert "anomaly_scores" in anomalies
        
        # Should detect the high revenue day
        anomaly_dates = anomalies["anomaly_dates"]
        assert "2024-01-05" in anomaly_dates  # 5000.00 revenue day
        
        # Verify anomaly scores
        for score in anomalies["anomaly_scores"]:
            assert 0 <= score <= 1


class TestAnalyticsIntegration:
    """Integration tests for analytics system"""
    
    @pytest.mark.asyncio
    async def test_real_time_dashboard_data(self, analytics_service, report_generator, db_session):
        """Test real-time dashboard data aggregation"""
        dashboard_data = await analytics_service.get_dashboard_data()
        
        expected_widgets = [
            "revenue_today",
            "revenue_this_month", 
            "active_subscriptions",
            "new_customers_today",
            "churn_rate_this_month",
            "payment_success_rate",
            "mrr",
            "arr"
        ]
        
        for widget in expected_widgets:
            assert widget in dashboard_data
        
        # Verify data freshness
        assert "last_updated" in dashboard_data
        last_updated = datetime.fromisoformat(dashboard_data["last_updated"])
        time_diff = datetime.utcnow() - last_updated
        assert time_diff.total_seconds() < 300  # Updated within 5 minutes
    
    @pytest.mark.asyncio
    async def test_analytics_caching(self, analytics_service, mock_redis):
        """Test analytics data caching"""
        cache_key = "revenue_analytics_30d"
        
        # First call should hit database
        with patch.object(analytics_service, '_calculate_revenue_from_db') as mock_db_call:
            mock_db_call.return_value = {"total_revenue": Decimal("10000.00")}
            mock_redis.get.return_value = None  # Cache miss
            
            result1 = await analytics_service.get_revenue_analytics(cache_ttl=3600)
            
            mock_db_call.assert_called_once()
            mock_redis.set.assert_called_once()
        
        # Second call should hit cache
        with patch.object(analytics_service, '_calculate_revenue_from_db') as mock_db_call:
            mock_redis.get.return_value = '{"total_revenue": "10000.00"}'  # Cache hit
            
            result2 = await analytics_service.get_revenue_analytics(cache_ttl=3600)
            
            mock_db_call.assert_not_called()  # Should not hit database
            assert result2["total_revenue"] == Decimal("10000.00")
    
    @pytest.mark.asyncio
    async def test_analytics_performance(self, analytics_service, db_session):
        """Test analytics query performance"""
        import time
        
        start_time = time.time()
        
        # Run multiple analytics queries
        tasks = [
            analytics_service.get_revenue_analytics(),
            analytics_service.get_subscription_analytics(),
            analytics_service.get_customer_analytics(),
            analytics_service.get_payment_analytics()
        ]
        
        results = await asyncio.gather(*tasks)
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        # All queries should complete within reasonable time
        assert execution_time < 5.0  # 5 seconds max
        assert len(results) == 4
        assert all(result is not None for result in results)
