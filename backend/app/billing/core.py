"""
Spotify AI Agent - Enterprise Billing Core Module
=================================================

Core billing functionality and business logic orchestration.
Centralizes payment processing, subscription management, and financial operations.

Architecture:
- Payment Processing Engine
- Subscription Lifecycle Management  
- Revenue Recognition & Accounting
- Fraud Detection & Risk Management
- Multi-Provider Payment Gateway Integration
"""

import os
import asyncio
import logging
from datetime import datetime, timedelta, timezone
from decimal import Decimal, ROUND_HALF_UP
from typing import Dict, Any, Optional, List, Union, Tuple, Set
from enum import Enum, IntEnum
from dataclasses import dataclass, field
import uuid
import json
import hashlib
import hmac
import base64
from contextlib import asynccontextmanager
import aioredis
from sqlalchemy import Column, String, Float, Boolean, DateTime, ForeignKey, Integer, Text, DECIMAL
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, sessionmaker
from sqlalchemy.dialects.postgresql import UUID, JSONB
import stripe
import paypalrestsdk
from fastapi import HTTPException, status
import pydantic
from pydantic import BaseModel, validator, Field
import asyncpg


class PaymentProvider(Enum):
    """Supported payment providers"""
    STRIPE = "stripe"
    PAYPAL = "paypal"
    APPLE_PAY = "apple_pay"
    GOOGLE_PAY = "google_pay"
    CRYPTO = "crypto"
    BANK_TRANSFER = "bank_transfer"


class PaymentStatus(Enum):
    """Payment transaction status"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    REFUNDED = "refunded"
    DISPUTED = "disputed"


class SubscriptionStatus(Enum):
    """Subscription lifecycle status"""
    TRIAL = "trial"
    ACTIVE = "active"
    PAST_DUE = "past_due"
    CANCELLED = "cancelled"
    EXPIRED = "expired"
    SUSPENDED = "suspended"


class PlanType(Enum):
    """Subscription plan types"""
    FREE = "free"
    BASIC = "basic"
    PREMIUM = "premium"
    ENTERPRISE = "enterprise"
    CUSTOM = "custom"


class CurrencyCode(Enum):
    """Supported currencies ISO 4217"""
    USD = "USD"
    EUR = "EUR"
    GBP = "GBP"
    JPY = "JPY"
    CAD = "CAD"
    AUD = "AUD"
    CHF = "CHF"
    CNY = "CNY"


class FraudRiskLevel(IntEnum):
    """Fraud risk assessment levels"""
    VERY_LOW = 1
    LOW = 2
    MEDIUM = 3
    HIGH = 4
    CRITICAL = 5


@dataclass
class PaymentMethodInfo:
    """Payment method details"""
    id: str
    type: str
    last4: Optional[str] = None
    brand: Optional[str] = None
    exp_month: Optional[int] = None
    exp_year: Optional[int] = None
    country: Optional[str] = None
    is_default: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BillingAddress:
    """Customer billing address"""
    line1: str
    city: str
    country: str
    postal_code: str
    line2: Optional[str] = None
    state: Optional[str] = None


@dataclass
class TaxCalculation:
    """Tax calculation result"""
    amount: Decimal
    rate: Decimal
    jurisdiction: str
    tax_type: str
    exemption_applied: bool = False


@dataclass
class PricingRule:
    """Dynamic pricing rule"""
    rule_id: str
    condition: Dict[str, Any]
    adjustment_type: str  # percentage, fixed_amount
    adjustment_value: Decimal
    priority: int
    active: bool = True


class BillingConfig:
    """Central billing configuration"""
    
    def __init__(self):
        # Payment providers configuration
        self.stripe_config = {
            'secret_key': os.getenv('STRIPE_SECRET_KEY'),
            'publishable_key': os.getenv('STRIPE_PUBLISHABLE_KEY'),
            'webhook_secret': os.getenv('STRIPE_WEBHOOK_SECRET'),
            'api_version': '2023-10-16'
        }
        
        self.paypal_config = {
            'client_id': os.getenv('PAYPAL_CLIENT_ID'),
            'client_secret': os.getenv('PAYPAL_CLIENT_SECRET'),
            'mode': os.getenv('PAYPAL_MODE', 'sandbox'),  # live or sandbox
            'webhook_id': os.getenv('PAYPAL_WEBHOOK_ID')
        }
        
        # Database configuration
        self.database_url = os.getenv('BILLING_DATABASE_URL', 'postgresql://localhost/billing')
        self.redis_url = os.getenv('REDIS_BILLING_URL', 'redis://localhost:6379/1')
        
        # Business rules
        self.default_currency = CurrencyCode.EUR
        self.trial_period_days = 14
        self.grace_period_days = 3
        self.max_retry_attempts = 3
        
        # Fraud detection
        self.fraud_detection_enabled = True
        self.fraud_threshold = 0.75
        self.auto_block_threshold = 0.95
        
        # Compliance
        self.pci_dss_level = 1
        self.gdpr_compliance = True
        self.data_retention_years = 7


class PaymentProcessor:
    """Core payment processing engine"""
    
    def __init__(self, config: BillingConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.redis_client = None
        self.fraud_detector = FraudDetector()
        self.tax_calculator = TaxCalculator()
        
        # Initialize payment providers
        self._setup_stripe()
        self._setup_paypal()
        
    async def initialize(self):
        """Initialize async components"""
        self.redis_client = await aioredis.from_url(self.config.redis_url)
        await self.fraud_detector.initialize()
        
    def _setup_stripe(self):
        """Initialize Stripe SDK"""
        if self.config.stripe_config['secret_key']:
            stripe.api_key = self.config.stripe_config['secret_key']
            stripe.api_version = self.config.stripe_config['api_version']
            
    def _setup_paypal(self):
        """Initialize PayPal SDK"""
        if all([self.config.paypal_config['client_id'], self.config.paypal_config['client_secret']]):
            paypalrestsdk.configure({
                'mode': self.config.paypal_config['mode'],
                'client_id': self.config.paypal_config['client_id'],
                'client_secret': self.config.paypal_config['client_secret']
            })
    
    async def process_payment(self, 
                            amount: Decimal,
                            currency: str,
                            customer_id: str,
                            payment_method_id: str,
                            description: str = "",
                            metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """Process a payment transaction"""
        try:
            # Validate input parameters
            if amount <= 0:
                raise ValueError("Amount must be greater than 0")
            
            # Fraud detection
            fraud_assessment = await self.fraud_detector.assess_transaction(
                customer_id=customer_id,
                amount=amount,
                currency=currency,
                payment_method_id=payment_method_id
            )
            
            if fraud_assessment['risk_level'] >= FraudRiskLevel.HIGH:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Transaction blocked due to high fraud risk"
                )
            
            # Tax calculation
            tax_amount = await self.tax_calculator.calculate_tax(
                amount=amount,
                currency=currency,
                customer_id=customer_id
            )
            
            total_amount = amount + tax_amount.amount
            
            # Determine payment provider
            payment_method = await self._get_payment_method(payment_method_id)
            provider = self._determine_provider(payment_method)
            
            # Process payment with appropriate provider
            if provider == PaymentProvider.STRIPE:
                result = await self._process_stripe_payment(
                    amount=total_amount,
                    currency=currency,
                    customer_id=customer_id,
                    payment_method_id=payment_method_id,
                    description=description,
                    metadata=metadata
                )
            elif provider == PaymentProvider.PAYPAL:
                result = await self._process_paypal_payment(
                    amount=total_amount,
                    currency=currency,
                    customer_id=customer_id,
                    description=description,
                    metadata=metadata
                )
            else:
                raise ValueError(f"Unsupported payment provider: {provider}")
            
            # Store transaction record
            transaction_id = await self._store_transaction(
                amount=amount,
                tax_amount=tax_amount.amount,
                total_amount=total_amount,
                currency=currency,
                customer_id=customer_id,
                payment_method_id=payment_method_id,
                provider=provider,
                provider_transaction_id=result['transaction_id'],
                status=result['status'],
                fraud_score=fraud_assessment['score'],
                metadata=metadata
            )
            
            # Update cache
            await self._update_payment_cache(customer_id, transaction_id, result)
            
            return {
                'transaction_id': transaction_id,
                'status': result['status'],
                'amount': float(amount),
                'tax_amount': float(tax_amount.amount),
                'total_amount': float(total_amount),
                'currency': currency,
                'provider': provider.value,
                'provider_transaction_id': result['transaction_id'],
                'fraud_score': fraud_assessment['score'],
                'created_at': datetime.utcnow().isoformat()
            }
            
        except Exception as exc:
            self.logger.error(f"Payment processing failed: {exc}")
            # Store failed transaction for analysis
            await self._store_failed_transaction(
                customer_id=customer_id,
                amount=amount,
                currency=currency,
                error=str(exc),
                metadata=metadata
            )
            raise
    
    async def _process_stripe_payment(self, **kwargs) -> Dict[str, Any]:
        """Process payment through Stripe"""
        try:
            intent = stripe.PaymentIntent.create(
                amount=int(kwargs['amount'] * 100),  # Convert to cents
                currency=kwargs['currency'].lower(),
                customer=kwargs['customer_id'],
                payment_method=kwargs['payment_method_id'],
                confirmation_method='manual',
                confirm=True,
                description=kwargs.get('description', ''),
                metadata=kwargs.get('metadata', {})
            )
            
            return {
                'transaction_id': intent.id,
                'status': PaymentStatus.COMPLETED if intent.status == 'succeeded' else PaymentStatus.PENDING,
                'provider_response': intent
            }
            
        except stripe.error.CardError as e:
            # Card was declined
            self.logger.warning(f"Stripe card declined: {e.user_message}")
            return {
                'transaction_id': e.payment_intent.id if e.payment_intent else None,
                'status': PaymentStatus.FAILED,
                'error': e.user_message
            }
            
        except stripe.error.StripeError as e:
            # Other Stripe errors
            self.logger.error(f"Stripe error: {e}")
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail="Payment processor error"
            )
    
    async def _process_paypal_payment(self, **kwargs) -> Dict[str, Any]:
        """Process payment through PayPal"""
        try:
            payment = paypalrestsdk.Payment({
                'intent': 'sale',
                'payer': {
                    'payment_method': 'paypal'
                },
                'transactions': [{
                    'amount': {
                        'total': str(kwargs['amount']),
                        'currency': kwargs['currency']
                    },
                    'description': kwargs.get('description', '')
                }],
                'redirect_urls': {
                    'return_url': 'https://api.spotify-ai.com/billing/paypal/success',
                    'cancel_url': 'https://api.spotify-ai.com/billing/paypal/cancel'
                }
            })
            
            if payment.create():
                return {
                    'transaction_id': payment.id,
                    'status': PaymentStatus.PENDING,
                    'approval_url': next(link.href for link in payment.links if link.rel == 'approval_url')
                }
            else:
                raise Exception(f"PayPal payment creation failed: {payment.error}")
                
        except Exception as e:
            self.logger.error(f"PayPal error: {e}")
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail="Payment processor error"
            )
    
    async def _get_payment_method(self, payment_method_id: str) -> PaymentMethodInfo:
        """Retrieve payment method details"""
        # Try cache first
        cached = await self.redis_client.get(f"payment_method:{payment_method_id}")
        if cached:
            return PaymentMethodInfo(**json.loads(cached))
        
        # Fetch from Stripe
        try:
            pm = stripe.PaymentMethod.retrieve(payment_method_id)
            payment_method = PaymentMethodInfo(
                id=pm.id,
                type=pm.type,
                last4=pm.card.last4 if pm.card else None,
                brand=pm.card.brand if pm.card else None,
                exp_month=pm.card.exp_month if pm.card else None,
                exp_year=pm.card.exp_year if pm.card else None,
                country=pm.card.country if pm.card else None
            )
            
            # Cache for 1 hour
            await self.redis_client.setex(
                f"payment_method:{payment_method_id}",
                3600,
                json.dumps(payment_method.__dict__)
            )
            
            return payment_method
            
        except stripe.error.StripeError:
            # Fallback or alternative provider lookup
            pass
        
        raise ValueError(f"Payment method not found: {payment_method_id}")
    
    def _determine_provider(self, payment_method: PaymentMethodInfo) -> PaymentProvider:
        """Determine which payment provider to use"""
        if payment_method.type in ['card', 'sepa_debit']:
            return PaymentProvider.STRIPE
        elif payment_method.type == 'paypal':
            return PaymentProvider.PAYPAL
        else:
            return PaymentProvider.STRIPE  # Default fallback
    
    async def _store_transaction(self, **kwargs) -> str:
        """Store transaction record in database"""
        transaction_id = str(uuid.uuid4())
        
        # Database storage logic would go here
        # For now, store in Redis as well
        transaction_data = {
            'id': transaction_id,
            'created_at': datetime.utcnow().isoformat(),
            **kwargs
        }
        
        await self.redis_client.setex(
            f"transaction:{transaction_id}",
            86400 * 30,  # 30 days
            json.dumps(transaction_data, default=str)
        )
        
        return transaction_id
    
    async def _update_payment_cache(self, customer_id: str, transaction_id: str, result: Dict):
        """Update payment-related cache entries"""
        # Update customer's recent payments
        await self.redis_client.lpush(
            f"customer_payments:{customer_id}",
            transaction_id
        )
        await self.redis_client.ltrim(f"customer_payments:{customer_id}", 0, 99)  # Keep last 100
        
        # Update payment stats
        stats_key = f"payment_stats:{datetime.utcnow().strftime('%Y-%m-%d')}"
        await self.redis_client.hincrby(stats_key, 'total_count', 1)
        await self.redis_client.expire(stats_key, 86400 * 7)  # Keep for 7 days
    
    async def _store_failed_transaction(self, **kwargs):
        """Store failed transaction for analysis"""
        failed_transaction = {
            'id': str(uuid.uuid4()),
            'timestamp': datetime.utcnow().isoformat(),
            'type': 'payment_failure',
            **kwargs
        }
        
        await self.redis_client.lpush(
            'failed_transactions',
            json.dumps(failed_transaction, default=str)
        )


class SubscriptionManager:
    """Subscription lifecycle management"""
    
    def __init__(self, config: BillingConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.redis_client = None
        self.payment_processor = PaymentProcessor(config)
        
    async def initialize(self):
        """Initialize async components"""
        self.redis_client = await aioredis.from_url(self.config.redis_url)
        await self.payment_processor.initialize()
    
    async def create_subscription(self,
                                customer_id: str,
                                plan_id: str,
                                payment_method_id: str,
                                trial_days: Optional[int] = None,
                                coupon_code: Optional[str] = None) -> Dict[str, Any]:
        """Create a new subscription"""
        try:
            # Validate plan exists
            plan = await self._get_plan(plan_id)
            if not plan:
                raise ValueError(f"Plan not found: {plan_id}")
            
            # Apply coupon if provided
            discount_amount = Decimal('0')
            if coupon_code:
                discount = await self._apply_coupon(coupon_code, plan['amount'])
                discount_amount = discount['amount']
            
            # Calculate trial end date
            trial_end = None
            if trial_days or self.config.trial_period_days:
                trial_days = trial_days or self.config.trial_period_days
                trial_end = datetime.utcnow() + timedelta(days=trial_days)
            
            # Create subscription record
            subscription_id = str(uuid.uuid4())
            subscription_data = {
                'id': subscription_id,
                'customer_id': customer_id,
                'plan_id': plan_id,
                'status': SubscriptionStatus.TRIAL if trial_end else SubscriptionStatus.ACTIVE,
                'current_period_start': datetime.utcnow(),
                'current_period_end': self._calculate_period_end(plan['interval']),
                'trial_end': trial_end,
                'payment_method_id': payment_method_id,
                'amount': plan['amount'] - discount_amount,
                'currency': plan['currency'],
                'discount_amount': discount_amount,
                'coupon_code': coupon_code,
                'created_at': datetime.utcnow(),
                'metadata': {}
            }
            
            # Store subscription
            await self._store_subscription(subscription_data)
            
            # Schedule first payment if not in trial
            if not trial_end:
                await self._schedule_subscription_payment(subscription_id)
            else:
                # Schedule trial end reminder
                await self._schedule_trial_end_notification(subscription_id, trial_end)
            
            # Update customer cache
            await self._update_customer_subscription_cache(customer_id, subscription_id)
            
            self.logger.info(f"Subscription created: {subscription_id} for customer: {customer_id}")
            
            return {
                'subscription_id': subscription_id,
                'status': subscription_data['status'].value,
                'plan': plan,
                'trial_end': trial_end.isoformat() if trial_end else None,
                'next_billing_date': subscription_data['current_period_end'].isoformat(),
                'amount': float(subscription_data['amount']),
                'currency': subscription_data['currency']
            }
            
        except Exception as exc:
            self.logger.error(f"Subscription creation failed: {exc}")
            raise
    
    async def cancel_subscription(self, subscription_id: str, 
                                immediate: bool = False,
                                reason: str = "") -> Dict[str, Any]:
        """Cancel a subscription"""
        try:
            subscription = await self._get_subscription(subscription_id)
            if not subscription:
                raise ValueError(f"Subscription not found: {subscription_id}")
            
            if subscription['status'] in [SubscriptionStatus.CANCELLED, SubscriptionStatus.EXPIRED]:
                raise ValueError("Subscription already cancelled or expired")
            
            # Determine cancellation date
            cancellation_date = datetime.utcnow()
            effective_date = cancellation_date if immediate else subscription['current_period_end']
            
            # Update subscription status
            subscription['status'] = SubscriptionStatus.CANCELLED
            subscription['cancelled_at'] = cancellation_date
            subscription['cancel_at'] = effective_date
            subscription['cancellation_reason'] = reason
            
            # Store updated subscription
            await self._store_subscription(subscription)
            
            # Handle immediate cancellation
            if immediate:
                await self._process_immediate_cancellation(subscription_id)
            
            # Cancel scheduled payments
            await self._cancel_scheduled_payments(subscription_id)
            
            # Send cancellation notification
            await self._send_cancellation_notification(subscription['customer_id'], subscription)
            
            self.logger.info(f"Subscription cancelled: {subscription_id}")
            
            return {
                'subscription_id': subscription_id,
                'status': 'cancelled',
                'cancelled_at': cancellation_date.isoformat(),
                'effective_date': effective_date.isoformat(),
                'refund_amount': 0.0  # Calculate if immediate cancellation with prorations
            }
            
        except Exception as exc:
            self.logger.error(f"Subscription cancellation failed: {exc}")
            raise
    
    async def upgrade_subscription(self, subscription_id: str, new_plan_id: str) -> Dict[str, Any]:
        """Upgrade subscription to a different plan"""
        try:
            subscription = await self._get_subscription(subscription_id)
            old_plan = await self._get_plan(subscription['plan_id'])
            new_plan = await self._get_plan(new_plan_id)
            
            # Calculate proration
            proration = await self._calculate_upgrade_proration(
                subscription, old_plan, new_plan
            )
            
            # Process immediate payment for difference
            if proration['amount_due'] > 0:
                payment_result = await self.payment_processor.process_payment(
                    amount=proration['amount_due'],
                    currency=new_plan['currency'],
                    customer_id=subscription['customer_id'],
                    payment_method_id=subscription['payment_method_id'],
                    description=f"Upgrade to {new_plan['name']}",
                    metadata={'subscription_id': subscription_id, 'upgrade': True}
                )
                
                if payment_result['status'] != PaymentStatus.COMPLETED:
                    raise Exception("Upgrade payment failed")
            
            # Update subscription
            subscription['plan_id'] = new_plan_id
            subscription['amount'] = new_plan['amount']
            subscription['upgraded_at'] = datetime.utcnow()
            
            await self._store_subscription(subscription)
            
            return {
                'subscription_id': subscription_id,
                'new_plan': new_plan,
                'proration_amount': float(proration['amount_due']),
                'effective_immediately': True
            }
            
        except Exception as exc:
            self.logger.error(f"Subscription upgrade failed: {exc}")
            raise
    
    async def _get_plan(self, plan_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve subscription plan details"""
        # Try cache first
        cached = await self.redis_client.get(f"plan:{plan_id}")
        if cached:
            return json.loads(cached)
        
        # Default plans (in production, this would come from database)
        plans = {
            'free': {
                'id': 'free',
                'name': 'Free',
                'amount': Decimal('0'),
                'currency': 'EUR',
                'interval': 'month',
                'features': ['basic_streaming', 'ads']
            },
            'premium_monthly': {
                'id': 'premium_monthly',
                'name': 'Premium Monthly',
                'amount': Decimal('9.99'),
                'currency': 'EUR',
                'interval': 'month',
                'features': ['ad_free_streaming', 'offline_download', 'hq_audio']
            },
            'premium_yearly': {
                'id': 'premium_yearly',
                'name': 'Premium Yearly',
                'amount': Decimal('99.99'),
                'currency': 'EUR',
                'interval': 'year',
                'features': ['ad_free_streaming', 'offline_download', 'hq_audio', 'exclusive_content']
            }
        }
        
        plan = plans.get(plan_id)
        if plan:
            # Cache for 1 hour
            await self.redis_client.setex(f"plan:{plan_id}", 3600, json.dumps(plan, default=str))
        
        return plan
    
    def _calculate_period_end(self, interval: str) -> datetime:
        """Calculate subscription period end date"""
        now = datetime.utcnow()
        if interval == 'month':
            return now + timedelta(days=30)
        elif interval == 'year':
            return now + timedelta(days=365)
        else:
            return now + timedelta(days=30)  # Default to monthly
    
    async def _store_subscription(self, subscription_data: Dict[str, Any]):
        """Store subscription in database/cache"""
        subscription_id = subscription_data['id']
        
        # Store in Redis
        await self.redis_client.setex(
            f"subscription:{subscription_id}",
            86400 * 365,  # 1 year
            json.dumps(subscription_data, default=str)
        )
        
        # Database storage would go here
    
    async def _get_subscription(self, subscription_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve subscription details"""
        cached = await self.redis_client.get(f"subscription:{subscription_id}")
        if cached:
            return json.loads(cached)
        return None
    
    async def _schedule_subscription_payment(self, subscription_id: str):
        """Schedule recurring payment for subscription"""
        # Implementation would use Celery or similar for scheduling
        pass
    
    async def _schedule_trial_end_notification(self, subscription_id: str, trial_end: datetime):
        """Schedule trial end notification"""
        # Implementation would schedule email/notification
        pass
    
    async def _update_customer_subscription_cache(self, customer_id: str, subscription_id: str):
        """Update customer's subscription cache"""
        await self.redis_client.set(f"customer_subscription:{customer_id}", subscription_id)


class FraudDetector:
    """Advanced fraud detection system"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.redis_client = None
        
    async def initialize(self):
        """Initialize fraud detection components"""
        self.redis_client = await aioredis.from_url(os.getenv('REDIS_URL', 'redis://localhost:6379'))
    
    async def assess_transaction(self, 
                               customer_id: str,
                               amount: Decimal,
                               currency: str,
                               payment_method_id: str) -> Dict[str, Any]:
        """Assess fraud risk for a transaction"""
        try:
            risk_factors = []
            risk_score = 0.0
            
            # Velocity checks
            velocity_risk = await self._check_velocity(customer_id, amount)
            risk_score += velocity_risk['score']
            if velocity_risk['flagged']:
                risk_factors.append('high_velocity')
            
            # Amount analysis
            amount_risk = await self._analyze_amount(customer_id, amount)
            risk_score += amount_risk['score']
            if amount_risk['flagged']:
                risk_factors.append('unusual_amount')
            
            # Payment method analysis
            pm_risk = await self._analyze_payment_method(payment_method_id)
            risk_score += pm_risk['score']
            if pm_risk['flagged']:
                risk_factors.append('risky_payment_method')
            
            # Geolocation check
            geo_risk = await self._check_geolocation(customer_id)
            risk_score += geo_risk['score']
            if geo_risk['flagged']:
                risk_factors.append('unusual_location')
            
            # Determine risk level
            risk_level = self._calculate_risk_level(risk_score)
            
            # Store assessment
            await self._store_fraud_assessment(
                customer_id, amount, currency, risk_score, risk_factors
            )
            
            return {
                'score': risk_score,
                'risk_level': risk_level,
                'factors': risk_factors,
                'recommendation': self._get_recommendation(risk_level)
            }
            
        except Exception as exc:
            self.logger.error(f"Fraud assessment failed: {exc}")
            # Default to low risk if assessment fails
            return {
                'score': 0.1,
                'risk_level': FraudRiskLevel.LOW,
                'factors': [],
                'recommendation': 'proceed'
            }
    
    async def _check_velocity(self, customer_id: str, amount: Decimal) -> Dict[str, Any]:
        """Check transaction velocity patterns"""
        # Get recent transactions
        recent_key = f"recent_transactions:{customer_id}"
        recent_count = await self.redis_client.llen(recent_key)
        
        # Check transaction frequency
        if recent_count > 10:  # More than 10 transactions recently
            return {'score': 0.3, 'flagged': True}
        elif recent_count > 5:
            return {'score': 0.1, 'flagged': False}
        
        return {'score': 0.0, 'flagged': False}
    
    async def _analyze_amount(self, customer_id: str, amount: Decimal) -> Dict[str, Any]:
        """Analyze transaction amount patterns"""
        # Get customer's typical transaction amounts
        avg_amount_key = f"avg_amount:{customer_id}"
        avg_amount_str = await self.redis_client.get(avg_amount_key)
        
        if avg_amount_str:
            avg_amount = Decimal(avg_amount_str)
            ratio = float(amount / avg_amount)
            
            if ratio > 10:  # 10x larger than typical
                return {'score': 0.4, 'flagged': True}
            elif ratio > 5:  # 5x larger than typical
                return {'score': 0.2, 'flagged': False}
        
        # Check absolute amount thresholds
        if amount > Decimal('10000'):  # Very large transaction
            return {'score': 0.3, 'flagged': True}
        elif amount > Decimal('1000'):
            return {'score': 0.1, 'flagged': False}
        
        return {'score': 0.0, 'flagged': False}
    
    async def _analyze_payment_method(self, payment_method_id: str) -> Dict[str, Any]:
        """Analyze payment method risk"""
        # Check if payment method is blacklisted
        blacklist_key = f"blacklisted_pm:{payment_method_id}"
        is_blacklisted = await self.redis_client.exists(blacklist_key)
        
        if is_blacklisted:
            return {'score': 0.9, 'flagged': True}
        
        # Check payment method failure rate
        failure_key = f"pm_failures:{payment_method_id}"
        failure_count = await self.redis_client.get(failure_key)
        
        if failure_count and int(failure_count) > 3:
            return {'score': 0.2, 'flagged': True}
        
        return {'score': 0.0, 'flagged': False}
    
    async def _check_geolocation(self, customer_id: str) -> Dict[str, Any]:
        """Check for unusual geolocation patterns"""
        # Implementation would check customer's typical location vs current
        # For now, return low risk
        return {'score': 0.0, 'flagged': False}
    
    def _calculate_risk_level(self, score: float) -> FraudRiskLevel:
        """Convert risk score to risk level"""
        if score >= 0.8:
            return FraudRiskLevel.CRITICAL
        elif score >= 0.6:
            return FraudRiskLevel.HIGH
        elif score >= 0.4:
            return FraudRiskLevel.MEDIUM
        elif score >= 0.2:
            return FraudRiskLevel.LOW
        else:
            return FraudRiskLevel.VERY_LOW
    
    def _get_recommendation(self, risk_level: FraudRiskLevel) -> str:
        """Get recommendation based on risk level"""
        recommendations = {
            FraudRiskLevel.VERY_LOW: 'proceed',
            FraudRiskLevel.LOW: 'proceed',
            FraudRiskLevel.MEDIUM: 'review',
            FraudRiskLevel.HIGH: 'challenge',
            FraudRiskLevel.CRITICAL: 'block'
        }
        return recommendations[risk_level]
    
    async def _store_fraud_assessment(self, customer_id: str, amount: Decimal, 
                                    currency: str, score: float, factors: List[str]):
        """Store fraud assessment for analysis"""
        assessment = {
            'customer_id': customer_id,
            'amount': str(amount),
            'currency': currency,
            'score': score,
            'factors': factors,
            'timestamp': datetime.utcnow().isoformat()
        }
        
        await self.redis_client.lpush(
            'fraud_assessments',
            json.dumps(assessment)
        )
        await self.redis_client.ltrim('fraud_assessments', 0, 9999)  # Keep last 10k


class TaxCalculator:
    """Tax calculation engine"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
    async def calculate_tax(self, amount: Decimal, currency: str, customer_id: str) -> TaxCalculation:
        """Calculate applicable taxes"""
        try:
            # Get customer's tax jurisdiction
            jurisdiction = await self._get_customer_jurisdiction(customer_id)
            
            # Get tax rate for jurisdiction and amount
            tax_rate = await self._get_tax_rate(jurisdiction, amount, currency)
            
            # Calculate tax amount
            tax_amount = amount * tax_rate / 100
            tax_amount = tax_amount.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)
            
            return TaxCalculation(
                amount=tax_amount,
                rate=tax_rate,
                jurisdiction=jurisdiction,
                tax_type='VAT' if jurisdiction.startswith('EU_') else 'SALES_TAX'
            )
            
        except Exception as exc:
            self.logger.error(f"Tax calculation failed: {exc}")
            # Return zero tax as fallback
            return TaxCalculation(
                amount=Decimal('0'),
                rate=Decimal('0'),
                jurisdiction='UNKNOWN',
                tax_type='NONE'
            )
    
    async def _get_customer_jurisdiction(self, customer_id: str) -> str:
        """Determine customer's tax jurisdiction"""
        # Implementation would look up customer's billing address
        # For now, return EU default
        return 'EU_FR'  # France
    
    async def _get_tax_rate(self, jurisdiction: str, amount: Decimal, currency: str) -> Decimal:
        """Get tax rate for jurisdiction"""
        # Tax rates by jurisdiction
        tax_rates = {
            'EU_FR': Decimal('20.0'),  # 20% VAT in France
            'EU_DE': Decimal('19.0'),  # 19% VAT in Germany
            'EU_IT': Decimal('22.0'),  # 22% VAT in Italy
            'US_CA': Decimal('7.25'),  # California sales tax
            'US_NY': Decimal('8.0'),   # New York sales tax
            'UK': Decimal('20.0'),     # 20% VAT in UK
            'CA': Decimal('5.0'),      # 5% GST in Canada
        }
        
        return tax_rates.get(jurisdiction, Decimal('0'))


# Global instances
billing_config = BillingConfig()
payment_processor = PaymentProcessor(billing_config)
subscription_manager = SubscriptionManager(billing_config)


# Async initialization function
async def initialize_billing_system():
    """Initialize all billing system components"""
    await payment_processor.initialize()
    await subscription_manager.initialize()


# Export main classes and functions
__all__ = [
    'BillingConfig',
    'PaymentProcessor', 
    'SubscriptionManager',
    'FraudDetector',
    'TaxCalculator',
    'PaymentProvider',
    'PaymentStatus',
    'SubscriptionStatus',
    'PlanType',
    'CurrencyCode',
    'FraudRiskLevel',
    'PaymentMethodInfo',
    'BillingAddress',
    'TaxCalculation',
    'PricingRule',
    'billing_config',
    'payment_processor',
    'subscription_manager',
    'initialize_billing_system'
]
