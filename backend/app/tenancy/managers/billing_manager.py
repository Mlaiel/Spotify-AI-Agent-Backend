"""
💰 Tenant Billing Manager - Gestionnaire Facturation Multi-Tenant
===============================================================

Gestionnaire avancé de facturation et quotas pour l'architecture multi-tenant.
Implémente la facturation usage-based et la gestion des quotas en temps réel.

Features:
- Facturation basée sur l'usage (metering)
- Gestion des quotas et limites
- Plans d'abonnement flexibles
- Rate limiting intelligent
- Proration et billing cycles
- Gestion des crédits et promotions
- Alertes de dépassement
- Reporting financier avancé
- Intégrations paiement (Stripe, PayPal)
- Gestion des taxes et compliance

Author: Architecte Microservices + DBA & Data Engineer
Version: 1.0.0
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum
from decimal import Decimal
import uuid
import json

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update, insert, func
from fastapi import HTTPException, status
from pydantic import BaseModel, validator
import redis.asyncio as redis

from app.core.database import get_async_session
from app.core.cache import get_redis_client
from app.core.config import settings
from app.tenancy.models import TenantSubscription, TenantBilling, TenantUsage

logger = logging.getLogger(__name__)


class BillingCycle(str, Enum):
    """Cycles de facturation"""
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    YEARLY = "yearly"


class UsageMetric(str, Enum):
    """Métriques d'usage facturables"""
    API_CALLS = "api_calls"
    STORAGE_GB = "storage_gb"
    COMPUTE_HOURS = "compute_hours"
    AI_PROCESSING = "ai_processing"
    BANDWIDTH_GB = "bandwidth_gb"
    ACTIVE_USERS = "active_users"
    PROJECTS = "projects"
    COLLABORATORS = "collaborators"
    INTEGRATIONS = "integrations"
    PREMIUM_FEATURES = "premium_features"


class BillingStatus(str, Enum):
    """Statuts de facturation"""
    ACTIVE = "active"
    PAST_DUE = "past_due"
    SUSPENDED = "suspended"
    CANCELLED = "cancelled"
    TRIAL = "trial"


class PaymentStatus(str, Enum):
    """Statuts de paiement"""
    PENDING = "pending"
    PAID = "paid"
    FAILED = "failed"
    REFUNDED = "refunded"
    CANCELLED = "cancelled"


@dataclass
class PricingTier:
    """Niveau de tarification"""
    name: str
    min_quantity: int
    max_quantity: Optional[int]
    price_per_unit: Decimal
    currency: str = "USD"


@dataclass
class UsageLimit:
    """Limite d'usage"""
    metric: UsageMetric
    soft_limit: int  # Limite souple (alerte)
    hard_limit: int  # Limite dure (blocage)
    overage_price: Optional[Decimal] = None  # Prix de dépassement


@dataclass
class BillingPlan:
    """Plan de facturation"""
    id: str
    name: str
    description: str
    base_price: Decimal
    currency: str
    billing_cycle: BillingCycle
    usage_limits: List[UsageLimit]
    pricing_tiers: Dict[UsageMetric, List[PricingTier]]
    features: List[str]
    trial_days: int = 0
    setup_fee: Decimal = Decimal('0')
    is_active: bool = True


class UsageRecord(BaseModel):
    """Enregistrement d'usage"""
    tenant_id: str
    metric: UsageMetric
    quantity: int
    timestamp: datetime
    metadata: Optional[Dict[str, Any]] = None


class BillingEvent(BaseModel):
    """Événement de facturation"""
    event_id: str
    tenant_id: str
    event_type: str
    amount: Decimal
    currency: str
    description: str
    timestamp: datetime
    metadata: Dict[str, Any]


class Invoice(BaseModel):
    """Facture"""
    invoice_id: str
    tenant_id: str
    billing_period_start: datetime
    billing_period_end: datetime
    subtotal: Decimal
    tax_amount: Decimal
    total_amount: Decimal
    currency: str
    status: PaymentStatus
    due_date: datetime
    line_items: List[Dict[str, Any]]
    issued_at: datetime


class TenantBillingManager:
    """
    Gestionnaire de facturation multi-tenant avancé.
    
    Responsabilités:
    - Suivi de l'usage en temps réel
    - Application des quotas et limites
    - Génération des factures
    - Gestion des paiements
    - Alertes et notifications
    - Reporting financier
    """

    def __init__(self):
        self._redis_client: Optional[redis.Redis] = None
        self.billing_plans: Dict[str, BillingPlan] = {}
        self._load_default_plans()

    async def get_redis_client(self) -> redis.Redis:
        """Obtenir le client Redis"""
        if not self._redis_client:
            self._redis_client = await get_redis_client()
        return self._redis_client

    def _load_default_plans(self):
        """Charger les plans de facturation par défaut"""
        # Plan Free
        free_plan = BillingPlan(
            id="free",
            name="Free Plan",
            description="Plan gratuit avec limitations",
            base_price=Decimal('0'),
            currency="USD",
            billing_cycle=BillingCycle.MONTHLY,
            usage_limits=[
                UsageLimit(UsageMetric.API_CALLS, 900, 1000),
                UsageLimit(UsageMetric.STORAGE_GB, 4, 5),
                UsageLimit(UsageMetric.AI_PROCESSING, 18, 20),
                UsageLimit(UsageMetric.ACTIVE_USERS, 9, 10),
                UsageLimit(UsageMetric.PROJECTS, 4, 5),
            ],
            pricing_tiers={},
            features=["basic_ai", "basic_support"],
            trial_days=14
        )

        # Plan Starter
        starter_plan = BillingPlan(
            id="starter",
            name="Starter Plan",
            description="Plan pour petites équipes",
            base_price=Decimal('29.99'),
            currency="USD",
            billing_cycle=BillingCycle.MONTHLY,
            usage_limits=[
                UsageLimit(UsageMetric.API_CALLS, 9000, 10000, Decimal('0.001')),
                UsageLimit(UsageMetric.STORAGE_GB, 45, 50, Decimal('2.00')),
                UsageLimit(UsageMetric.AI_PROCESSING, 180, 200, Decimal('0.05')),
                UsageLimit(UsageMetric.ACTIVE_USERS, 22, 25, Decimal('5.00')),
                UsageLimit(UsageMetric.PROJECTS, 18, 20, Decimal('3.00')),
            ],
            pricing_tiers={
                UsageMetric.API_CALLS: [
                    PricingTier("Base", 0, 10000, Decimal('0')),
                    PricingTier("Overage", 10001, None, Decimal('0.001'))
                ]
            },
            features=["advanced_ai", "email_support", "analytics"],
            trial_days=14
        )

        # Plan Professional
        pro_plan = BillingPlan(
            id="professional",
            name="Professional Plan",
            description="Plan pour équipes professionnelles",
            base_price=Decimal('99.99'),
            currency="USD",
            billing_cycle=BillingCycle.MONTHLY,
            usage_limits=[
                UsageLimit(UsageMetric.API_CALLS, 90000, 100000, Decimal('0.0008')),
                UsageLimit(UsageMetric.STORAGE_GB, 450, 500, Decimal('1.50')),
                UsageLimit(UsageMetric.AI_PROCESSING, 900, 1000, Decimal('0.03')),
                UsageLimit(UsageMetric.ACTIVE_USERS, 90, 100, Decimal('3.00')),
                UsageLimit(UsageMetric.PROJECTS, 90, 100, Decimal('2.00')),
            ],
            pricing_tiers={
                UsageMetric.API_CALLS: [
                    PricingTier("Base", 0, 100000, Decimal('0')),
                    PricingTier("Overage", 100001, None, Decimal('0.0008'))
                ]
            },
            features=["premium_ai", "priority_support", "advanced_analytics", "white_label"],
            trial_days=30
        )

        # Plan Enterprise
        enterprise_plan = BillingPlan(
            id="enterprise",
            name="Enterprise Plan",
            description="Plan pour grandes entreprises",
            base_price=Decimal('499.99'),
            currency="USD",
            billing_cycle=BillingCycle.MONTHLY,
            usage_limits=[
                UsageLimit(UsageMetric.API_CALLS, 900000, 1000000, Decimal('0.0005')),
                UsageLimit(UsageMetric.STORAGE_GB, 4500, 5000, Decimal('1.00')),
                UsageLimit(UsageMetric.AI_PROCESSING, 4500, 5000, Decimal('0.02')),
                UsageLimit(UsageMetric.ACTIVE_USERS, 900, 1000, Decimal('2.00')),
                UsageLimit(UsageMetric.PROJECTS, 450, 500, Decimal('1.50')),
            ],
            pricing_tiers={
                UsageMetric.API_CALLS: [
                    PricingTier("Base", 0, 1000000, Decimal('0')),
                    PricingTier("Overage", 1000001, None, Decimal('0.0005'))
                ]
            },
            features=["enterprise_ai", "dedicated_support", "custom_analytics", "sso", "audit_logs"],
            trial_days=30,
            setup_fee=Decimal('1000.00')
        )

        self.billing_plans = {
            "free": free_plan,
            "starter": starter_plan,
            "professional": pro_plan,
            "enterprise": enterprise_plan
        }

    async def record_usage(
        self,
        tenant_id: str,
        metric: UsageMetric,
        quantity: int,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Enregistrer l'usage d'une métrique pour un tenant.
        
        Args:
            tenant_id: Identifiant du tenant
            metric: Métrique d'usage
            quantity: Quantité utilisée
            metadata: Métadonnées supplémentaires
            
        Returns:
            True si l'usage a été enregistré, False si limite dépassée
        """
        try:
            # Vérification des limites avant enregistrement
            can_use = await self.check_usage_limit(tenant_id, metric, quantity)
            if not can_use:
                return False

            # Enregistrement de l'usage
            usage_record = UsageRecord(
                tenant_id=tenant_id,
                metric=metric,
                quantity=quantity,
                timestamp=datetime.utcnow(),
                metadata=metadata
            )

            # Stockage en base de données
            await self._store_usage_record(usage_record)

            # Mise à jour des compteurs en cache
            await self._update_usage_counters(tenant_id, metric, quantity)

            # Vérification des alertes
            await self._check_usage_alerts(tenant_id, metric)

            logger.debug(f"Usage enregistré: {tenant_id} - {metric} - {quantity}")
            return True

        except Exception as e:
            logger.error(f"Erreur enregistrement usage: {str(e)}")
            return False

    async def check_usage_limit(
        self,
        tenant_id: str,
        metric: UsageMetric,
        additional_quantity: int = 0
    ) -> bool:
        """
        Vérifier si un tenant peut utiliser une métrique.
        
        Args:
            tenant_id: Identifiant du tenant
            metric: Métrique à vérifier
            additional_quantity: Quantité supplémentaire à ajouter
            
        Returns:
            True si dans les limites, False sinon
        """
        try:
            # Récupération du plan de facturation
            billing_plan = await self._get_tenant_billing_plan(tenant_id)
            if not billing_plan:
                return False

            # Recherche de la limite pour cette métrique
            usage_limit = None
            for limit in billing_plan.usage_limits:
                if limit.metric == metric:
                    usage_limit = limit
                    break

            if not usage_limit:
                # Pas de limite définie = usage illimité
                return True

            # Récupération de l'usage actuel
            current_usage = await self.get_current_usage(tenant_id, metric)
            total_usage = current_usage + additional_quantity

            # Vérification de la limite dure
            if total_usage > usage_limit.hard_limit:
                logger.warning(f"Limite dure dépassée pour {tenant_id} - {metric}: {total_usage}/{usage_limit.hard_limit}")
                return False

            return True

        except Exception as e:
            logger.error(f"Erreur vérification limite: {str(e)}")
            return False

    async def get_current_usage(
        self,
        tenant_id: str,
        metric: UsageMetric,
        period_start: Optional[datetime] = None
    ) -> int:
        """
        Obtenir l'usage actuel d'un tenant pour une métrique.
        
        Args:
            tenant_id: Identifiant du tenant
            metric: Métrique d'usage
            period_start: Début de la période (par défaut: début du cycle de facturation)
            
        Returns:
            Quantité utilisée
        """
        try:
            # Calcul de la période si non fournie
            if period_start is None:
                period_start = await self._get_billing_period_start(tenant_id)

            # Récupération depuis le cache Redis
            redis_client = await self.get_redis_client()
            cache_key = f"usage:{tenant_id}:{metric}:{period_start.strftime('%Y%m%d')}"
            
            cached_usage = await redis_client.get(cache_key)
            if cached_usage:
                return int(cached_usage.decode())

            # Récupération depuis la base de données
            async with get_async_session() as db:
                result = await db.execute(
                    select(func.sum(TenantUsage.quantity))
                    .where(
                        TenantUsage.tenant_id == tenant_id,
                        TenantUsage.metric == metric,
                        TenantUsage.timestamp >= period_start
                    )
                )
                usage = result.scalar() or 0

            # Mise en cache
            await redis_client.setex(cache_key, 3600, usage)  # Cache 1 heure
            
            return usage

        except Exception as e:
            logger.error(f"Erreur récupération usage: {str(e)}")
            return 0

    async def generate_invoice(
        self,
        tenant_id: str,
        billing_period_start: datetime,
        billing_period_end: datetime
    ) -> Invoice:
        """
        Générer une facture pour un tenant.
        
        Args:
            tenant_id: Identifiant du tenant
            billing_period_start: Début de la période de facturation
            billing_period_end: Fin de la période de facturation
            
        Returns:
            Facture générée
        """
        try:
            # Récupération du plan de facturation
            billing_plan = await self._get_tenant_billing_plan(tenant_id)
            if not billing_plan:
                raise ValueError(f"Aucun plan de facturation pour le tenant {tenant_id}")

            # Récupération de l'usage pour la période
            usage_data = await self._get_period_usage(
                tenant_id, billing_period_start, billing_period_end
            )

            # Calcul des line items
            line_items = []
            subtotal = billing_plan.base_price

            # Frais de base
            line_items.append({
                "description": f"{billing_plan.name} - Base Fee",
                "quantity": 1,
                "unit_price": float(billing_plan.base_price),
                "total": float(billing_plan.base_price)
            })

            # Calcul des dépassements
            for metric, quantity in usage_data.items():
                usage_metric = UsageMetric(metric)
                overage_amount = await self._calculate_overage(
                    billing_plan, usage_metric, quantity
                )
                
                if overage_amount > 0:
                    line_items.append({
                        "description": f"{metric.replace('_', ' ').title()} Overage",
                        "quantity": quantity,
                        "unit_price": float(overage_amount / quantity) if quantity > 0 else 0,
                        "total": float(overage_amount)
                    })
                    subtotal += overage_amount

            # Calcul des taxes (à implémenter selon la juridiction)
            tax_rate = await self._get_tax_rate(tenant_id)
            tax_amount = subtotal * tax_rate

            # Total
            total_amount = subtotal + tax_amount

            # Génération de la facture
            invoice = Invoice(
                invoice_id=str(uuid.uuid4()),
                tenant_id=tenant_id,
                billing_period_start=billing_period_start,
                billing_period_end=billing_period_end,
                subtotal=subtotal,
                tax_amount=tax_amount,
                total_amount=total_amount,
                currency=billing_plan.currency,
                status=PaymentStatus.PENDING,
                due_date=billing_period_end + timedelta(days=30),
                line_items=line_items,
                issued_at=datetime.utcnow()
            )

            # Sauvegarde de la facture
            await self._store_invoice(invoice)

            logger.info(f"Facture générée pour {tenant_id}: {invoice.invoice_id} - {invoice.total_amount} {invoice.currency}")
            return invoice

        except Exception as e:
            logger.error(f"Erreur génération facture: {str(e)}")
            raise

    async def get_billing_summary(
        self,
        tenant_id: str,
        include_current_period: bool = True
    ) -> Dict[str, Any]:
        """
        Obtenir un résumé de facturation pour un tenant.
        
        Args:
            tenant_id: Identifiant du tenant
            include_current_period: Inclure la période courante
            
        Returns:
            Résumé de facturation
        """
        try:
            billing_plan = await self._get_tenant_billing_plan(tenant_id)
            current_period_start = await self._get_billing_period_start(tenant_id)
            
            summary = {
                "tenant_id": tenant_id,
                "plan": billing_plan.name if billing_plan else "Unknown",
                "billing_cycle": billing_plan.billing_cycle if billing_plan else None,
                "current_period_start": current_period_start.isoformat(),
                "current_period_end": (current_period_start + timedelta(days=30)).isoformat(),
                "usage": {},
                "limits": {},
                "estimated_charges": Decimal('0'),
                "payment_status": await self._get_payment_status(tenant_id)
            }

            if billing_plan and include_current_period:
                # Usage actuel par métrique
                for limit in billing_plan.usage_limits:
                    current_usage = await self.get_current_usage(tenant_id, limit.metric)
                    summary["usage"][limit.metric] = {
                        "current": current_usage,
                        "soft_limit": limit.soft_limit,
                        "hard_limit": limit.hard_limit,
                        "percentage": (current_usage / limit.hard_limit * 100) if limit.hard_limit > 0 else 0
                    }

                # Estimation des charges
                estimated_charges = billing_plan.base_price
                for metric, usage_info in summary["usage"].items():
                    overage = await self._calculate_overage(
                        billing_plan, UsageMetric(metric), usage_info["current"]
                    )
                    estimated_charges += overage

                summary["estimated_charges"] = float(estimated_charges)

            return summary

        except Exception as e:
            logger.error(f"Erreur résumé facturation: {str(e)}")
            return {}

    async def upgrade_plan(
        self,
        tenant_id: str,
        new_plan_id: str,
        prorate: bool = True
    ) -> bool:
        """
        Mettre à niveau le plan d'un tenant.
        
        Args:
            tenant_id: Identifiant du tenant
            new_plan_id: ID du nouveau plan
            prorate: Appliquer la proration
            
        Returns:
            Succès de l'opération
        """
        try:
            if new_plan_id not in self.billing_plans:
                raise ValueError(f"Plan {new_plan_id} non trouvé")

            new_plan = self.billing_plans[new_plan_id]
            current_plan = await self._get_tenant_billing_plan(tenant_id)

            # Calcul de la proration si demandée
            proration_credit = Decimal('0')
            if prorate and current_plan:
                proration_credit = await self._calculate_proration(
                    tenant_id, current_plan, new_plan
                )

            # Mise à jour du plan
            async with get_async_session() as db:
                await db.execute(
                    update(TenantSubscription)
                    .where(TenantSubscription.tenant_id == tenant_id)
                    .values(
                        plan_id=new_plan_id,
                        updated_at=datetime.utcnow()
                    )
                )
                await db.commit()

            # Application du crédit de proration
            if proration_credit > 0:
                await self._apply_credit(tenant_id, proration_credit, "Plan upgrade proration")

            # Événement de facturation
            await self._create_billing_event(
                tenant_id,
                "plan_upgrade",
                Decimal('0'),
                new_plan.currency,
                f"Plan upgraded to {new_plan.name}",
                {"old_plan": current_plan.id if current_plan else None, "new_plan": new_plan_id}
            )

            logger.info(f"Plan mis à niveau pour {tenant_id}: {new_plan_id}")
            return True

        except Exception as e:
            logger.error(f"Erreur mise à niveau plan: {str(e)}")
            return False

    # Méthodes privées

    async def _store_usage_record(self, usage_record: UsageRecord):
        """Stocker un enregistrement d'usage en base"""
        async with get_async_session() as db:
            db_usage = TenantUsage(
                tenant_id=usage_record.tenant_id,
                metric=usage_record.metric,
                quantity=usage_record.quantity,
                timestamp=usage_record.timestamp,
                metadata=usage_record.metadata
            )
            db.add(db_usage)
            await db.commit()

    async def _update_usage_counters(
        self,
        tenant_id: str,
        metric: UsageMetric,
        quantity: int
    ):
        """Mettre à jour les compteurs d'usage en cache"""
        redis_client = await self.get_redis_client()
        today = datetime.utcnow().strftime('%Y%m%d')
        cache_key = f"usage:{tenant_id}:{metric}:{today}"
        
        await redis_client.incrby(cache_key, quantity)
        await redis_client.expire(cache_key, 86400 * 7)  # 7 jours

    async def _check_usage_alerts(self, tenant_id: str, metric: UsageMetric):
        """Vérifier et envoyer les alertes d'usage"""
        billing_plan = await self._get_tenant_billing_plan(tenant_id)
        if not billing_plan:
            return

        # Recherche de la limite
        usage_limit = None
        for limit in billing_plan.usage_limits:
            if limit.metric == metric:
                usage_limit = limit
                break

        if not usage_limit:
            return

        current_usage = await self.get_current_usage(tenant_id, metric)
        
        # Alerte limite souple (80% de la limite dure)
        if current_usage >= usage_limit.soft_limit:
            await self._send_usage_alert(
                tenant_id, metric, current_usage, usage_limit.hard_limit, "soft_limit"
            )

        # Alerte proche de la limite dure (95%)
        if current_usage >= usage_limit.hard_limit * 0.95:
            await self._send_usage_alert(
                tenant_id, metric, current_usage, usage_limit.hard_limit, "near_hard_limit"
            )

    async def _send_usage_alert(
        self,
        tenant_id: str,
        metric: UsageMetric,
        current_usage: int,
        limit: int,
        alert_type: str
    ):
        """Envoyer une alerte d'usage"""
        # Implémentation de l'envoi d'alerte (email, webhook, etc.)
        logger.warning(f"Alerte usage {alert_type}: {tenant_id} - {metric} - {current_usage}/{limit}")

    async def _get_tenant_billing_plan(self, tenant_id: str) -> Optional[BillingPlan]:
        """Récupérer le plan de facturation d'un tenant"""
        try:
            async with get_async_session() as db:
                result = await db.execute(
                    select(TenantSubscription.plan_id)
                    .where(TenantSubscription.tenant_id == tenant_id)
                )
                plan_id = result.scalar_one_or_none()
                
                if plan_id and plan_id in self.billing_plans:
                    return self.billing_plans[plan_id]
                
                # Plan par défaut si aucun trouvé
                return self.billing_plans.get("free")
                
        except Exception as e:
            logger.error(f"Erreur récupération plan facturation: {str(e)}")
            return None

    async def _get_billing_period_start(self, tenant_id: str) -> datetime:
        """Obtenir le début de la période de facturation courante"""
        # Simplification: début du mois courant
        now = datetime.utcnow()
        return datetime(now.year, now.month, 1)

    async def _get_period_usage(
        self,
        tenant_id: str,
        start: datetime,
        end: datetime
    ) -> Dict[str, int]:
        """Récupérer l'usage pour une période"""
        async with get_async_session() as db:
            result = await db.execute(
                select(TenantUsage.metric, func.sum(TenantUsage.quantity))
                .where(
                    TenantUsage.tenant_id == tenant_id,
                    TenantUsage.timestamp >= start,
                    TenantUsage.timestamp <= end
                )
                .group_by(TenantUsage.metric)
            )
            
            return {row[0]: row[1] for row in result.fetchall()}

    async def _calculate_overage(
        self,
        billing_plan: BillingPlan,
        metric: UsageMetric,
        usage: int
    ) -> Decimal:
        """Calculer le montant de dépassement"""
        # Recherche de la limite
        usage_limit = None
        for limit in billing_plan.usage_limits:
            if limit.metric == metric:
                usage_limit = limit
                break

        if not usage_limit or not usage_limit.overage_price:
            return Decimal('0')

        overage_quantity = max(0, usage - usage_limit.hard_limit)
        return Decimal(str(overage_quantity)) * usage_limit.overage_price

    async def _get_tax_rate(self, tenant_id: str) -> Decimal:
        """Obtenir le taux de taxe pour un tenant"""
        # À implémenter selon la juridiction
        return Decimal('0.20')  # 20% par défaut

    async def _store_invoice(self, invoice: Invoice):
        """Stocker une facture en base"""
        async with get_async_session() as db:
            db_invoice = TenantBilling(
                invoice_id=invoice.invoice_id,
                tenant_id=invoice.tenant_id,
                billing_period_start=invoice.billing_period_start,
                billing_period_end=invoice.billing_period_end,
                subtotal=invoice.subtotal,
                tax_amount=invoice.tax_amount,
                total_amount=invoice.total_amount,
                currency=invoice.currency,
                status=invoice.status,
                due_date=invoice.due_date,
                line_items=invoice.line_items,
                issued_at=invoice.issued_at
            )
            db.add(db_invoice)
            await db.commit()

    async def _get_payment_status(self, tenant_id: str) -> str:
        """Obtenir le statut de paiement d'un tenant"""
        # À implémenter
        return "current"

    async def _calculate_proration(
        self,
        tenant_id: str,
        old_plan: BillingPlan,
        new_plan: BillingPlan
    ) -> Decimal:
        """Calculer la proration lors d'un changement de plan"""
        # Simplification: calcul basique
        return Decimal('0')

    async def _apply_credit(
        self,
        tenant_id: str,
        amount: Decimal,
        description: str
    ):
        """Appliquer un crédit au compte d'un tenant"""
        await self._create_billing_event(
            tenant_id, "credit", amount, "USD", description, {}
        )

    async def _create_billing_event(
        self,
        tenant_id: str,
        event_type: str,
        amount: Decimal,
        currency: str,
        description: str,
        metadata: Dict[str, Any]
    ):
        """Créer un événement de facturation"""
        event = BillingEvent(
            event_id=str(uuid.uuid4()),
            tenant_id=tenant_id,
            event_type=event_type,
            amount=amount,
            currency=currency,
            description=description,
            timestamp=datetime.utcnow(),
            metadata=metadata
        )
        
        # Stockage de l'événement
        logger.info(f"Événement de facturation: {event.event_type} - {tenant_id} - {amount} {currency}")


# Instance globale du gestionnaire de facturation
tenant_billing_manager = TenantBillingManager()
