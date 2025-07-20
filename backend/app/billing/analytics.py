"""
Spotify AI Agent - Billing Analytics Dashboard
=============================================

Comprehensive analytics and reporting system for billing operations:
- Revenue tracking and forecasting
- Customer lifetime value analysis
- Subscription metrics and churn analysis
- Payment provider performance
- Real-time dashboard with charts and KPIs
"""

import json
import asyncio
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import pandas as pd
import numpy as np
from sqlalchemy.orm import Session
from sqlalchemy import func, case, and_, or_, extract, text
from fastapi import APIRouter, Depends, Query
from fastapi.responses import JSONResponse

from ..core.database import get_db
from ..core.auth import get_current_user, require_permissions
from .models import (
    Customer, Subscription, Payment, Invoice, Plan,
    CustomerStatus, SubscriptionStatus, PaymentStatus, InvoiceStatus,
    PaymentProvider, PlanInterval
)


class MetricPeriod(Enum):
    """Time period for metrics calculation"""
    DAY = "day"
    WEEK = "week"
    MONTH = "month"
    QUARTER = "quarter"
    YEAR = "year"


@dataclass
class RevenueMetrics:
    """Revenue metrics data structure"""
    total_revenue: Decimal
    recurring_revenue: Decimal
    one_time_revenue: Decimal
    mrr: Decimal  # Monthly Recurring Revenue
    arr: Decimal  # Annual Recurring Revenue
    growth_rate: Decimal
    currency: str
    period_start: datetime
    period_end: datetime


@dataclass
class CustomerMetrics:
    """Customer metrics data structure"""
    total_customers: int
    active_customers: int
    new_customers: int
    churned_customers: int
    customer_acquisition_cost: Decimal
    customer_lifetime_value: Decimal
    churn_rate: Decimal
    retention_rate: Decimal


@dataclass
class SubscriptionMetrics:
    """Subscription metrics data structure"""
    total_subscriptions: int
    active_subscriptions: int
    trial_subscriptions: int
    canceled_subscriptions: int
    subscription_growth_rate: Decimal
    average_revenue_per_user: Decimal
    plan_distribution: Dict[str, int]


@dataclass
class PaymentMetrics:
    """Payment processing metrics"""
    total_payments: int
    successful_payments: int
    failed_payments: int
    success_rate: Decimal
    total_volume: Decimal
    average_transaction_size: Decimal
    provider_performance: Dict[str, Dict[str, Any]]
    fraud_rate: Decimal


class BillingAnalytics:
    """Main analytics engine for billing system"""
    
    def __init__(self, db_session: Session):
        self.db = db_session
    
    async def get_revenue_metrics(self, 
                                start_date: datetime,
                                end_date: datetime,
                                currency: str = "EUR") -> RevenueMetrics:
        """Calculate revenue metrics for period"""
        
        # Total revenue from successful payments
        total_revenue_query = self.db.query(func.sum(Payment.amount)).filter(
            Payment.status == PaymentStatus.SUCCEEDED,
            Payment.payment_date >= start_date,
            Payment.payment_date <= end_date,
            Payment.currency == currency
        )
        total_revenue = total_revenue_query.scalar() or Decimal('0')
        
        # Recurring revenue (from subscriptions)
        recurring_revenue_query = self.db.query(func.sum(Payment.amount)).filter(
            Payment.status == PaymentStatus.SUCCEEDED,
            Payment.payment_date >= start_date,
            Payment.payment_date <= end_date,
            Payment.currency == currency,
            Payment.subscription_id.isnot(None)
        )
        recurring_revenue = recurring_revenue_query.scalar() or Decimal('0')
        
        # One-time revenue
        one_time_revenue = total_revenue - recurring_revenue
        
        # Calculate MRR (Monthly Recurring Revenue)
        mrr = await self._calculate_mrr(currency)
        
        # ARR (Annual Recurring Revenue)
        arr = mrr * 12
        
        # Growth rate calculation (compared to previous period)
        period_length = (end_date - start_date).days
        previous_start = start_date - timedelta(days=period_length)
        previous_end = start_date
        
        previous_revenue_query = self.db.query(func.sum(Payment.amount)).filter(
            Payment.status == PaymentStatus.SUCCEEDED,
            Payment.payment_date >= previous_start,
            Payment.payment_date < previous_end,
            Payment.currency == currency
        )
        previous_revenue = previous_revenue_query.scalar() or Decimal('0')
        
        if previous_revenue > 0:
            growth_rate = ((total_revenue - previous_revenue) / previous_revenue) * 100
        else:
            growth_rate = Decimal('0')
        
        return RevenueMetrics(
            total_revenue=total_revenue,
            recurring_revenue=recurring_revenue,
            one_time_revenue=one_time_revenue,
            mrr=mrr,
            arr=arr,
            growth_rate=growth_rate,
            currency=currency,
            period_start=start_date,
            period_end=end_date
        )
    
    async def get_customer_metrics(self,
                                 start_date: datetime,
                                 end_date: datetime) -> CustomerMetrics:
        """Calculate customer metrics for period"""
        
        # Total customers
        total_customers = self.db.query(Customer).count()
        
        # Active customers
        active_customers = self.db.query(Customer).filter(
            Customer.status == CustomerStatus.ACTIVE
        ).count()
        
        # New customers in period
        new_customers = self.db.query(Customer).filter(
            Customer.created_at >= start_date,
            Customer.created_at <= end_date
        ).count()
        
        # Churned customers (cancelled subscriptions in period)
        churned_customers = self.db.query(Customer).join(Subscription).filter(
            Subscription.canceled_at >= start_date,
            Subscription.canceled_at <= end_date
        ).distinct().count()
        
        # Customer Acquisition Cost (CAC)
        # This would need marketing spend data - simplified calculation
        total_marketing_spend = Decimal('10000')  # Placeholder
        cac = total_marketing_spend / max(new_customers, 1)
        
        # Customer Lifetime Value (CLV)
        clv = await self._calculate_customer_lifetime_value()
        
        # Churn rate
        if active_customers > 0:
            churn_rate = (churned_customers / active_customers) * 100
        else:
            churn_rate = Decimal('0')
        
        # Retention rate
        retention_rate = 100 - churn_rate
        
        return CustomerMetrics(
            total_customers=total_customers,
            active_customers=active_customers,
            new_customers=new_customers,
            churned_customers=churned_customers,
            customer_acquisition_cost=cac,
            customer_lifetime_value=clv,
            churn_rate=churn_rate,
            retention_rate=retention_rate
        )
    
    async def get_subscription_metrics(self,
                                     start_date: datetime,
                                     end_date: datetime) -> SubscriptionMetrics:
        """Calculate subscription metrics for period"""
        
        # Total subscriptions
        total_subscriptions = self.db.query(Subscription).count()
        
        # Active subscriptions
        active_subscriptions = self.db.query(Subscription).filter(
            Subscription.status.in_([SubscriptionStatus.ACTIVE, SubscriptionStatus.TRIAL])
        ).count()
        
        # Trial subscriptions
        trial_subscriptions = self.db.query(Subscription).filter(
            Subscription.status == SubscriptionStatus.TRIAL
        ).count()
        
        # Canceled subscriptions in period
        canceled_subscriptions = self.db.query(Subscription).filter(
            Subscription.canceled_at >= start_date,
            Subscription.canceled_at <= end_date
        ).count()
        
        # Subscription growth rate
        previous_start = start_date - (end_date - start_date)
        previous_active = self.db.query(Subscription).filter(
            Subscription.created_at < start_date,
            or_(
                Subscription.canceled_at.is_(None),
                Subscription.canceled_at >= previous_start
            )
        ).count()
        
        if previous_active > 0:
            growth_rate = ((active_subscriptions - previous_active) / previous_active) * 100
        else:
            growth_rate = Decimal('0')
        
        # Average Revenue Per User (ARPU)
        mrr = await self._calculate_mrr()
        arpu = mrr / max(active_subscriptions, 1)
        
        # Plan distribution
        plan_distribution = {}
        plan_counts = self.db.query(
            Plan.name,
            func.count(Subscription.id)
        ).join(Subscription).filter(
            Subscription.status.in_([SubscriptionStatus.ACTIVE, SubscriptionStatus.TRIAL])
        ).group_by(Plan.name).all()
        
        for plan_name, count in plan_counts:
            plan_distribution[plan_name] = count
        
        return SubscriptionMetrics(
            total_subscriptions=total_subscriptions,
            active_subscriptions=active_subscriptions,
            trial_subscriptions=trial_subscriptions,
            canceled_subscriptions=canceled_subscriptions,
            subscription_growth_rate=growth_rate,
            average_revenue_per_user=arpu,
            plan_distribution=plan_distribution
        )
    
    async def get_payment_metrics(self,
                                start_date: datetime,
                                end_date: datetime) -> PaymentMetrics:
        """Calculate payment processing metrics"""
        
        # Total payments in period
        total_payments = self.db.query(Payment).filter(
            Payment.created_at >= start_date,
            Payment.created_at <= end_date
        ).count()
        
        # Successful payments
        successful_payments = self.db.query(Payment).filter(
            Payment.created_at >= start_date,
            Payment.created_at <= end_date,
            Payment.status == PaymentStatus.SUCCEEDED
        ).count()
        
        # Failed payments
        failed_payments = total_payments - successful_payments
        
        # Success rate
        if total_payments > 0:
            success_rate = (successful_payments / total_payments) * 100
        else:
            success_rate = Decimal('0')
        
        # Total volume
        total_volume = self.db.query(func.sum(Payment.amount)).filter(
            Payment.created_at >= start_date,
            Payment.created_at <= end_date,
            Payment.status == PaymentStatus.SUCCEEDED
        ).scalar() or Decimal('0')
        
        # Average transaction size
        if successful_payments > 0:
            avg_transaction_size = total_volume / successful_payments
        else:
            avg_transaction_size = Decimal('0')
        
        # Provider performance
        provider_performance = {}
        provider_stats = self.db.query(
            Payment.provider,
            func.count(Payment.id).label('total'),
            func.sum(case(
                (Payment.status == PaymentStatus.SUCCEEDED, 1),
                else_=0
            )).label('successful'),
            func.sum(case(
                (Payment.status == PaymentStatus.SUCCEEDED, Payment.amount),
                else_=0
            )).label('volume')
        ).filter(
            Payment.created_at >= start_date,
            Payment.created_at <= end_date
        ).group_by(Payment.provider).all()
        
        for provider, total, successful, volume in provider_stats:
            success_rate_provider = (successful / total * 100) if total > 0 else 0
            provider_performance[provider.value] = {
                'total_payments': total,
                'successful_payments': successful,
                'success_rate': success_rate_provider,
                'volume': volume or Decimal('0')
            }
        
        # Fraud rate (based on high risk score)
        high_risk_payments = self.db.query(Payment).filter(
            Payment.created_at >= start_date,
            Payment.created_at <= end_date,
            Payment.risk_score > 0.7
        ).count()
        
        if total_payments > 0:
            fraud_rate = (high_risk_payments / total_payments) * 100
        else:
            fraud_rate = Decimal('0')
        
        return PaymentMetrics(
            total_payments=total_payments,
            successful_payments=successful_payments,
            failed_payments=failed_payments,
            success_rate=success_rate,
            total_volume=total_volume,
            average_transaction_size=avg_transaction_size,
            provider_performance=provider_performance,
            fraud_rate=fraud_rate
        )
    
    async def get_revenue_forecast(self,
                                 months_ahead: int = 12,
                                 currency: str = "EUR") -> List[Dict[str, Any]]:
        """Generate revenue forecast based on current trends"""
        
        # Get historical MRR data for trend analysis
        historical_data = []
        current_date = datetime.utcnow().replace(day=1)
        
        for i in range(12):  # Last 12 months
            month_start = current_date - timedelta(days=30 * i)
            month_end = month_start + timedelta(days=30)
            
            mrr = await self._calculate_mrr_for_period(month_start, month_end, currency)
            historical_data.append({
                'date': month_start,
                'mrr': float(mrr)
            })
        
        historical_data.reverse()  # Chronological order
        
        # Simple linear regression for trend
        if len(historical_data) >= 3:
            # Convert to pandas for easier calculation
            df = pd.DataFrame(historical_data)
            df['month_num'] = range(len(df))
            
            # Calculate trend
            correlation = df['month_num'].corr(df['mrr'])
            growth_rate = correlation * 0.1  # Simplified growth rate
        else:
            growth_rate = 0.05  # Default 5% growth
        
        # Generate forecast
        forecast = []
        current_mrr = await self._calculate_mrr(currency)
        
        for i in range(months_ahead):
            forecast_date = current_date + timedelta(days=30 * (i + 1))
            forecasted_mrr = float(current_mrr) * (1 + growth_rate) ** (i + 1)
            
            forecast.append({
                'date': forecast_date.isoformat(),
                'forecasted_mrr': round(forecasted_mrr, 2),
                'forecasted_arr': round(forecasted_mrr * 12, 2),
                'confidence': max(0.5, 1.0 - (i * 0.05))  # Decreasing confidence
            })
        
        return forecast
    
    async def get_cohort_analysis(self,
                                cohort_months: int = 12) -> Dict[str, Any]:
        """Generate customer cohort analysis"""
        
        cohorts = {}
        current_date = datetime.utcnow()
        
        for i in range(cohort_months):
            cohort_start = current_date - timedelta(days=30 * (i + 1))
            cohort_end = cohort_start + timedelta(days=30)
            
            # Get customers who started in this cohort
            cohort_customers = self.db.query(Customer.id).filter(
                Customer.created_at >= cohort_start,
                Customer.created_at < cohort_end
            ).subquery()
            
            # Calculate retention for each subsequent month
            retention_data = []
            for month in range(min(i + 1, 12)):  # Up to 12 months retention
                retention_start = cohort_start + timedelta(days=30 * month)
                retention_end = retention_start + timedelta(days=30)
                
                # Count active subscriptions in retention month
                active_in_month = self.db.query(func.count(Subscription.id)).filter(
                    Subscription.customer_id.in_(cohort_customers),
                    Subscription.status.in_([SubscriptionStatus.ACTIVE, SubscriptionStatus.TRIAL]),
                    Subscription.created_at <= retention_end,
                    or_(
                        Subscription.canceled_at.is_(None),
                        Subscription.canceled_at > retention_start
                    )
                ).scalar() or 0
                
                # Calculate retention rate
                total_cohort_size = self.db.query(func.count(Customer.id)).filter(
                    Customer.created_at >= cohort_start,
                    Customer.created_at < cohort_end
                ).scalar() or 1
                
                retention_rate = (active_in_month / total_cohort_size) * 100 if total_cohort_size > 0 else 0
                
                retention_data.append({
                    'month': month,
                    'retention_rate': round(retention_rate, 2),
                    'active_customers': active_in_month
                })
            
            cohorts[cohort_start.strftime('%Y-%m')] = {
                'cohort_size': total_cohort_size,
                'retention_data': retention_data
            }
        
        return cohorts
    
    async def _calculate_mrr(self, currency: str = "EUR") -> Decimal:
        """Calculate Monthly Recurring Revenue"""
        
        # Get all active subscriptions
        active_subscriptions = self.db.query(Subscription, Plan).join(Plan).filter(
            Subscription.status.in_([SubscriptionStatus.ACTIVE, SubscriptionStatus.TRIAL])
        ).all()
        
        total_mrr = Decimal('0')
        
        for subscription, plan in active_subscriptions:
            if plan.currency == currency:
                # Convert plan amount to monthly
                monthly_amount = self._convert_to_monthly(plan.amount, plan.interval, plan.interval_count)
                
                # Apply any custom amount or discount
                if subscription.custom_amount:
                    monthly_amount = self._convert_to_monthly(
                        subscription.custom_amount, plan.interval, plan.interval_count
                    )
                
                # Apply discount
                discount_amount = monthly_amount * (subscription.discount_percent / 100)
                effective_amount = monthly_amount - discount_amount
                
                total_mrr += effective_amount
        
        return total_mrr
    
    async def _calculate_mrr_for_period(self, start_date: datetime, end_date: datetime, 
                                      currency: str = "EUR") -> Decimal:
        """Calculate MRR for a specific period"""
        
        # This is a simplified version - in practice, you'd need to track
        # subscription states at specific points in time
        subscriptions = self.db.query(Subscription, Plan).join(Plan).filter(
            Subscription.created_at <= end_date,
            or_(
                Subscription.canceled_at.is_(None),
                Subscription.canceled_at > start_date
            ),
            Plan.currency == currency
        ).all()
        
        total_mrr = Decimal('0')
        
        for subscription, plan in subscriptions:
            monthly_amount = self._convert_to_monthly(plan.amount, plan.interval, plan.interval_count)
            
            if subscription.custom_amount:
                monthly_amount = self._convert_to_monthly(
                    subscription.custom_amount, plan.interval, plan.interval_count
                )
            
            discount_amount = monthly_amount * (subscription.discount_percent / 100)
            effective_amount = monthly_amount - discount_amount
            
            total_mrr += effective_amount
        
        return total_mrr
    
    def _convert_to_monthly(self, amount: Decimal, interval: PlanInterval, 
                          interval_count: int) -> Decimal:
        """Convert plan amount to monthly equivalent"""
        
        multipliers = {
            PlanInterval.DAY: Decimal('30'),
            PlanInterval.WEEK: Decimal('4.33'),
            PlanInterval.MONTH: Decimal('1'),
            PlanInterval.YEAR: Decimal('0.083')
        }
        
        multiplier = multipliers[interval]
        return (amount * multiplier) / interval_count
    
    async def _calculate_customer_lifetime_value(self) -> Decimal:
        """Calculate average Customer Lifetime Value"""
        
        # Get average revenue per customer per month
        mrr = await self._calculate_mrr()
        active_customers = self.db.query(Customer).filter(
            Customer.status == CustomerStatus.ACTIVE
        ).count()
        
        if active_customers == 0:
            return Decimal('0')
        
        avg_revenue_per_customer = mrr / active_customers
        
        # Calculate average customer lifespan
        completed_subscriptions = self.db.query(Subscription).filter(
            Subscription.status == SubscriptionStatus.CANCELLED,
            Subscription.canceled_at.isnot(None)
        ).all()
        
        if not completed_subscriptions:
            # Default to 24 months if no churn data
            avg_lifespan_months = 24
        else:
            total_months = 0
            for subscription in completed_subscriptions:
                months = (subscription.canceled_at - subscription.created_at).days / 30
                total_months += months
            
            avg_lifespan_months = total_months / len(completed_subscriptions)
        
        clv = avg_revenue_per_customer * Decimal(str(avg_lifespan_months))
        return clv


# FastAPI router for analytics endpoints
router = APIRouter(prefix="/analytics", tags=["analytics"])


@router.get("/overview")
async def get_analytics_overview(
    start_date: datetime = Query(..., description="Start date for analysis"),
    end_date: datetime = Query(..., description="End date for analysis"),
    currency: str = Query("EUR", description="Currency code"),
    db: Session = Depends(get_db),
    current_user = Depends(require_permissions(["billing:analytics"]))
):
    """Get comprehensive analytics overview"""
    
    analytics = BillingAnalytics(db)
    
    # Fetch all metrics concurrently
    revenue_metrics, customer_metrics, subscription_metrics, payment_metrics = await asyncio.gather(
        analytics.get_revenue_metrics(start_date, end_date, currency),
        analytics.get_customer_metrics(start_date, end_date),
        analytics.get_subscription_metrics(start_date, end_date),
        analytics.get_payment_metrics(start_date, end_date)
    )
    
    return {
        "period": {
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
            "currency": currency
        },
        "revenue": asdict(revenue_metrics),
        "customers": asdict(customer_metrics),
        "subscriptions": asdict(subscription_metrics),
        "payments": asdict(payment_metrics)
    }


@router.get("/revenue/forecast")
async def get_revenue_forecast(
    months_ahead: int = Query(12, ge=1, le=24, description="Months to forecast"),
    currency: str = Query("EUR", description="Currency code"),
    db: Session = Depends(get_db),
    current_user = Depends(require_permissions(["billing:analytics"]))
):
    """Get revenue forecast"""
    
    analytics = BillingAnalytics(db)
    forecast = await analytics.get_revenue_forecast(months_ahead, currency)
    
    return {
        "forecast_period": f"{months_ahead} months",
        "currency": currency,
        "forecast": forecast
    }


@router.get("/cohorts")
async def get_cohort_analysis(
    cohort_months: int = Query(12, ge=1, le=24, description="Number of cohort months"),
    db: Session = Depends(get_db),
    current_user = Depends(require_permissions(["billing:analytics"]))
):
    """Get customer cohort analysis"""
    
    analytics = BillingAnalytics(db)
    cohorts = await analytics.get_cohort_analysis(cohort_months)
    
    return {
        "cohort_months": cohort_months,
        "cohorts": cohorts
    }


@router.get("/revenue/trends")
async def get_revenue_trends(
    period: MetricPeriod = Query(MetricPeriod.MONTH, description="Period granularity"),
    periods_back: int = Query(12, ge=1, le=36, description="Number of periods to analyze"),
    currency: str = Query("EUR", description="Currency code"),
    db: Session = Depends(get_db),
    current_user = Depends(require_permissions(["billing:analytics"]))
):
    """Get revenue trends over time"""
    
    analytics = BillingAnalytics(db)
    trends = []
    
    current_date = datetime.utcnow()
    
    for i in range(periods_back):
        if period == MetricPeriod.MONTH:
            period_start = current_date.replace(day=1) - timedelta(days=30 * i)
            period_end = period_start + timedelta(days=30)
        elif period == MetricPeriod.WEEK:
            period_start = current_date - timedelta(weeks=i, days=current_date.weekday())
            period_end = period_start + timedelta(days=7)
        else:  # Daily
            period_start = current_date - timedelta(days=i)
            period_end = period_start + timedelta(days=1)
        
        revenue_metrics = await analytics.get_revenue_metrics(period_start, period_end, currency)
        
        trends.append({
            "period_start": period_start.isoformat(),
            "period_end": period_end.isoformat(),
            "total_revenue": float(revenue_metrics.total_revenue),
            "recurring_revenue": float(revenue_metrics.recurring_revenue),
            "mrr": float(revenue_metrics.mrr),
            "growth_rate": float(revenue_metrics.growth_rate)
        })
    
    trends.reverse()  # Chronological order
    
    return {
        "period": period.value,
        "periods_analyzed": periods_back,
        "currency": currency,
        "trends": trends
    }


@router.get("/export")
async def export_analytics_data(
    start_date: datetime = Query(..., description="Start date for export"),
    end_date: datetime = Query(..., description="End date for export"),
    format: str = Query("json", regex="^(json|csv)$", description="Export format"),
    db: Session = Depends(get_db),
    current_user = Depends(require_permissions(["billing:analytics"]))
):
    """Export analytics data"""
    
    analytics = BillingAnalytics(db)
    
    # Get comprehensive data
    revenue_metrics = await analytics.get_revenue_metrics(start_date, end_date)
    customer_metrics = await analytics.get_customer_metrics(start_date, end_date)
    subscription_metrics = await analytics.get_subscription_metrics(start_date, end_date)
    payment_metrics = await analytics.get_payment_metrics(start_date, end_date)
    
    data = {
        "export_date": datetime.utcnow().isoformat(),
        "period": {
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat()
        },
        "revenue": asdict(revenue_metrics),
        "customers": asdict(customer_metrics),
        "subscriptions": asdict(subscription_metrics),
        "payments": asdict(payment_metrics)
    }
    
    if format == "json":
        return JSONResponse(content=data)
    else:  # CSV format
        # Convert to CSV format (simplified)
        import io
        import csv
        
        output = io.StringIO()
        writer = csv.writer(output)
        
        # Write headers and data
        writer.writerow(["Metric", "Value"])
        writer.writerow(["Total Revenue", str(revenue_metrics.total_revenue)])
        writer.writerow(["MRR", str(revenue_metrics.mrr)])
        writer.writerow(["Active Customers", str(customer_metrics.active_customers)])
        writer.writerow(["Churn Rate", str(customer_metrics.churn_rate)])
        
        from fastapi.responses import StreamingResponse
        
        output.seek(0)
        return StreamingResponse(
            io.StringIO(output.getvalue()),
            media_type="text/csv",
            headers={"Content-Disposition": "attachment; filename=analytics_export.csv"}
        )


# Export router
__all__ = ['router', 'BillingAnalytics', 'RevenueMetrics', 'CustomerMetrics', 
           'SubscriptionMetrics', 'PaymentMetrics']
