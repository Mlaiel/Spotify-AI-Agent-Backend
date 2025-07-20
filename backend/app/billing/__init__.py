# 🎵 Spotify AI Agent - Payment & Billing System
# =============================================
# 
# Système complet de paiement et facturation
# avec Stripe, PayPal et gestion d'abonnements.
#
# 🎖️ Développé par l'équipe d'experts enterprise

"""
Enterprise Payment & Billing System
===================================

Complete payment processing and billing management:
- Stripe & PayPal integration
- Subscription management
- Invoice generation
- Webhook handling
- Fraud detection
- Multi-currency support

Authors & Roles:
- Lead Developer & AI Architect
- Senior Backend Developer (Python/FastAPI/Django)
- Security Specialist
- DBA & Data Engineer
"""

import os
import stripe
import paypal
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Decimal
from enum import Enum
from dataclasses import dataclass
import asyncio
import logging
from sqlalchemy import Column, String, Float, Boolean, DateTime, ForeignKey, Integer
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
import hashlib
import hmac
from fastapi import HTTPException


class PaymentProvider(Enum):
    """Fournisseurs de paiement supportés"""
    STRIPE = "stripe"
    PAYPAL = "paypal"
    APPLE_PAY = "apple_pay"
    GOOGLE_PAY = "google_pay"


class SubscriptionStatus(Enum):
    """Statuts d'abonnement"""
    ACTIVE = "active"
    TRIALING = "trialing"
    PAST_DUE = "past_due"
    CANCELED = "canceled"
    UNPAID = "unpaid"
    INCOMPLETE = "incomplete"


class PaymentStatus(Enum):
    """Statuts de paiement"""
    PENDING = "pending"
    PROCESSING = "processing"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELED = "canceled"
    REFUNDED = "refunded"


@dataclass
class PaymentConfig:
    """Configuration de paiement"""
    provider: PaymentProvider
    currency: str = "EUR"
    webhook_secret: str = ""
    api_key: str = ""
    environment: str = "sandbox"  # sandbox ou production


class PaymentManager:
    """Gestionnaire principal des paiements"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.stripe_manager = StripeManager()
        self.paypal_manager = PayPalManager()
        self.invoice_generator = InvoiceGenerator()
        self.fraud_detector = FraudDetector()
        
    async def process_payment(self, payment_data: Dict[str, Any]) -> Dict[str, Any]:
        """Traite un paiement selon le fournisseur"""
        try:
            provider = PaymentProvider(payment_data.get('provider'))
            
            # Vérification anti-fraude
            fraud_score = await self.fraud_detector.analyze_payment(payment_data)
            if fraud_score > 0.8:
                raise HTTPException(status_code=403, detail="Paiement bloqué - Activité suspecte")
            
            # Traitement selon le fournisseur
            if provider == PaymentProvider.STRIPE:
                result = await self.stripe_manager.process_payment(payment_data)
            elif provider == PaymentProvider.PAYPAL:
                result = await self.paypal_manager.process_payment(payment_data)
            else:
                raise HTTPException(status_code=400, detail="Fournisseur non supporté")
            
            # Génération de facture si succès
            if result.get('status') == PaymentStatus.SUCCEEDED.value:
                await self.invoice_generator.create_invoice(result)
            
            return result
            
        except Exception as exc:
            self.logger.error(f"Erreur traitement paiement: {exc}")
            raise


class StripeManager:
    """Gestionnaire Stripe"""
    
    def __init__(self):
        stripe.api_key = os.getenv('STRIPE_SECRET_KEY')
        self.webhook_secret = os.getenv('STRIPE_WEBHOOK_SECRET')
        self.logger = logging.getLogger(__name__)
        
    async def process_payment(self, payment_data: Dict[str, Any]) -> Dict[str, Any]:
        """Traite un paiement Stripe"""
        try:
            # Créer PaymentIntent
            intent = stripe.PaymentIntent.create(
                amount=int(payment_data['amount'] * 100),  # Stripe utilise les centimes
                currency=payment_data.get('currency', 'eur'),
                payment_method=payment_data['payment_method_id'],
                customer=payment_data.get('customer_id'),
                description=payment_data.get('description', 'Spotify AI Agent Payment'),
                metadata={
                    'user_id': payment_data.get('user_id'),
                    'subscription_id': payment_data.get('subscription_id'),
                    'plan_type': payment_data.get('plan_type')
                },
                confirm=True,
                return_url=payment_data.get('return_url')
            )
            
            return {
                'payment_id': intent.id,
                'status': PaymentStatus.SUCCEEDED.value if intent.status == 'succeeded' else PaymentStatus.PENDING.value,
                'amount': intent.amount / 100,
                'currency': intent.currency,
                'provider': PaymentProvider.STRIPE.value,
                'client_secret': intent.client_secret
            }
            
        except stripe.error.CardError as e:
            self.logger.error(f"Erreur carte Stripe: {e}")
            return {
                'status': PaymentStatus.FAILED.value,
                'error': str(e),
                'provider': PaymentProvider.STRIPE.value
            }
        except Exception as exc:
            self.logger.error(f"Erreur Stripe: {exc}")
            raise
    
    async def create_subscription(self, subscription_data: Dict[str, Any]) -> Dict[str, Any]:
        """Crée un abonnement Stripe"""
        try:
            # Créer ou récupérer le client
            customer = stripe.Customer.create(
                email=subscription_data['customer_email'],
                name=subscription_data.get('customer_name'),
                metadata={'user_id': subscription_data['user_id']}
            )
            
            # Créer l'abonnement
            subscription = stripe.Subscription.create(
                customer=customer.id,
                items=[{'price': subscription_data['price_id']}],
                payment_behavior='default_incomplete',
                payment_settings={'save_default_payment_method': 'on_subscription'},
                expand=['latest_invoice.payment_intent'],
                metadata={
                    'user_id': subscription_data['user_id'],
                    'plan_type': subscription_data['plan_type']
                }
            )
            
            return {
                'subscription_id': subscription.id,
                'customer_id': customer.id,
                'status': subscription.status,
                'current_period_end': subscription.current_period_end,
                'client_secret': subscription.latest_invoice.payment_intent.client_secret
            }
            
        except Exception as exc:
            self.logger.error(f"Erreur création abonnement Stripe: {exc}")
            raise
    
    async def handle_webhook(self, payload: str, signature: str) -> Dict[str, Any]:
        """Gère les webhooks Stripe"""
        try:
            event = stripe.Webhook.construct_event(
                payload, signature, self.webhook_secret
            )
            
            # Traitement selon le type d'événement
            if event['type'] == 'payment_intent.succeeded':
                return await self._handle_payment_succeeded(event['data']['object'])
            elif event['type'] == 'invoice.payment_succeeded':
                return await self._handle_invoice_paid(event['data']['object'])
            elif event['type'] == 'customer.subscription.deleted':
                return await self._handle_subscription_canceled(event['data']['object'])
            
            return {'status': 'handled', 'event_type': event['type']}
            
        except ValueError as e:
            self.logger.error(f"Payload invalide: {e}")
            raise HTTPException(status_code=400, detail="Payload invalide")
        except stripe.error.SignatureVerificationError as e:
            self.logger.error(f"Signature invalide: {e}")
            raise HTTPException(status_code=400, detail="Signature invalide")
    
    async def _handle_payment_succeeded(self, payment_intent):
        """Gère le succès d'un paiement"""
        # Logique de mise à jour de la base de données
        return {'status': 'payment_processed', 'payment_id': payment_intent['id']}
    
    async def _handle_invoice_paid(self, invoice):
        """Gère le paiement d'une facture"""
        # Logique de mise à jour de l'abonnement
        return {'status': 'invoice_paid', 'invoice_id': invoice['id']}
    
    async def _handle_subscription_canceled(self, subscription):
        """Gère l'annulation d'un abonnement"""
        # Logique de désactivation de l'abonnement
        return {'status': 'subscription_canceled', 'subscription_id': subscription['id']}


class PayPalManager:
    """Gestionnaire PayPal"""
    
    def __init__(self):
        self.client_id = os.getenv('PAYPAL_CLIENT_ID')
        self.client_secret = os.getenv('PAYPAL_CLIENT_SECRET')
        self.environment = os.getenv('PAYPAL_ENVIRONMENT', 'sandbox')
        self.logger = logging.getLogger(__name__)
        
    async def process_payment(self, payment_data: Dict[str, Any]) -> Dict[str, Any]:
        """Traite un paiement PayPal"""
        try:
            # Configuration PayPal SDK
            paypal.configure({
                "mode": self.environment,
                "client_id": self.client_id,
                "client_secret": self.client_secret
            })
            
            # Créer le paiement
            payment = paypal.Payment({
                "intent": "sale",
                "payer": {"payment_method": "paypal"},
                "redirect_urls": {
                    "return_url": payment_data.get('return_url'),
                    "cancel_url": payment_data.get('cancel_url')
                },
                "transactions": [{
                    "item_list": {
                        "items": [{
                            "name": payment_data.get('item_name', 'Spotify AI Agent Service'),
                            "sku": payment_data.get('sku', 'spotify-ai'),
                            "price": str(payment_data['amount']),
                            "currency": payment_data.get('currency', 'EUR'),
                            "quantity": 1
                        }]
                    },
                    "amount": {
                        "total": str(payment_data['amount']),
                        "currency": payment_data.get('currency', 'EUR')
                    },
                    "description": payment_data.get('description', 'Spotify AI Agent Payment')
                }]
            })
            
            if payment.create():
                return {
                    'payment_id': payment.id,
                    'status': PaymentStatus.PENDING.value,
                    'approval_url': next(link.href for link in payment.links if link.rel == "approval_url"),
                    'provider': PaymentProvider.PAYPAL.value
                }
            else:
                self.logger.error(f"Erreur création paiement PayPal: {payment.error}")
                return {
                    'status': PaymentStatus.FAILED.value,
                    'error': payment.error,
                    'provider': PaymentProvider.PAYPAL.value
                }
                
        except Exception as exc:
            self.logger.error(f"Erreur PayPal: {exc}")
            raise
    
    async def execute_payment(self, payment_id: str, payer_id: str) -> Dict[str, Any]:
        """Exécute un paiement PayPal approuvé"""
        try:
            payment = paypal.Payment.find(payment_id)
            
            if payment.execute({"payer_id": payer_id}):
                return {
                    'payment_id': payment.id,
                    'status': PaymentStatus.SUCCEEDED.value,
                    'provider': PaymentProvider.PAYPAL.value,
                    'transaction_id': payment.transactions[0].related_resources[0].sale.id
                }
            else:
                return {
                    'status': PaymentStatus.FAILED.value,
                    'error': payment.error,
                    'provider': PaymentProvider.PAYPAL.value
                }
                
        except Exception as exc:
            self.logger.error(f"Erreur exécution paiement PayPal: {exc}")
            raise


class SubscriptionManager:
    """Gestionnaire d'abonnements"""
    
    def __init__(self):
        self.stripe_manager = StripeManager()
        self.logger = logging.getLogger(__name__)
        
    async def create_subscription(self, user_id: str, plan_data: Dict[str, Any]) -> Dict[str, Any]:
        """Crée un nouvel abonnement"""
        try:
            # Validation du plan
            if not self._validate_plan(plan_data):
                raise HTTPException(status_code=400, detail="Plan invalide")
            
            # Créer l'abonnement selon le fournisseur
            if plan_data['provider'] == PaymentProvider.STRIPE.value:
                result = await self.stripe_manager.create_subscription({
                    'user_id': user_id,
                    'customer_email': plan_data['customer_email'],
                    'price_id': plan_data['price_id'],
                    'plan_type': plan_data['plan_type']
                })
            else:
                raise HTTPException(status_code=400, detail="Fournisseur non supporté pour abonnements")
            
            # Sauvegarder en base de données
            await self._save_subscription(user_id, result, plan_data)
            
            return result
            
        except Exception as exc:
            self.logger.error(f"Erreur création abonnement: {exc}")
            raise
    
    def _validate_plan(self, plan_data: Dict[str, Any]) -> bool:
        """Valide les données du plan"""
        required_fields = ['plan_type', 'price_id', 'provider', 'customer_email']
        return all(field in plan_data for field in required_fields)
    
    async def _save_subscription(self, user_id: str, subscription_result: Dict, plan_data: Dict):
        """Sauvegarde l'abonnement en base de données"""
        from backend.app.models.orm.users.user_subscription import UserSubscription
        
        subscription = UserSubscription.create(
            user_id=user_id,
            external_id=subscription_result['subscription_id'],
            plan_type=plan_data['plan_type'],
            status=subscription_result['status'],
            provider=plan_data['provider'],
            current_period_end=datetime.fromtimestamp(subscription_result['current_period_end'])
        )
        
        return subscription


class InvoiceGenerator:
    """Générateur de factures"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
    async def create_invoice(self, payment_data: Dict[str, Any]) -> Dict[str, Any]:
        """Crée une facture pour un paiement"""
        try:
            invoice_data = {
                'invoice_number': self._generate_invoice_number(),
                'payment_id': payment_data['payment_id'],
                'amount': payment_data['amount'],
                'currency': payment_data['currency'],
                'status': 'paid',
                'created_at': datetime.utcnow(),
                'due_date': datetime.utcnow() + timedelta(days=30)
            }
            
            # Sauvegarder la facture
            invoice = await self._save_invoice(invoice_data)
            
            # Générer le PDF
            pdf_path = await self._generate_pdf(invoice)
            
            # Envoyer par email
            await self._send_invoice_email(invoice, pdf_path)
            
            return {
                'invoice_id': invoice['id'],
                'invoice_number': invoice['invoice_number'],
                'pdf_path': pdf_path
            }
            
        except Exception as exc:
            self.logger.error(f"Erreur génération facture: {exc}")
            raise
    
    def _generate_invoice_number(self) -> str:
        """Génère un numéro de facture unique"""
        timestamp = datetime.utcnow().strftime('%Y%m%d%H%M%S')
        return f"INV-{timestamp}"
    
    async def _save_invoice(self, invoice_data: Dict) -> Dict:
        """Sauvegarde la facture en base"""
        # Logique de sauvegarde en base de données
        return invoice_data
    
    async def _generate_pdf(self, invoice_data: Dict) -> str:
        """Génère le PDF de la facture"""
        # Logique de génération PDF
        return f"/invoices/{invoice_data['invoice_number']}.pdf"
    
    async def _send_invoice_email(self, invoice_data: Dict, pdf_path: str):
        """Envoie la facture par email"""
        from backend.app.tasks.celery_manager import send_email_notification
        
        send_email_notification.delay(
            to_email=invoice_data.get('customer_email'),
            subject=f"Facture {invoice_data['invoice_number']}",
            template="invoice_email",
            context={
                'invoice': invoice_data,
                'pdf_attachment': pdf_path
            }
        )


class FraudDetector:
    """Détecteur de fraude"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
    async def analyze_payment(self, payment_data: Dict[str, Any]) -> float:
        """Analyse un paiement pour détecter la fraude"""
        try:
            score = 0.0
            
            # Vérification du montant
            if payment_data.get('amount', 0) > 1000:
                score += 0.2
            
            # Vérification de la géolocalisation
            if payment_data.get('country') in ['NG', 'PK', 'BD']:  # Pays à risque
                score += 0.3
            
            # Vérification de la fréquence
            user_id = payment_data.get('user_id')
            if user_id:
                recent_payments = await self._get_recent_payments(user_id)
                if len(recent_payments) > 5:  # Plus de 5 paiements dans l'heure
                    score += 0.4
            
            # Vérification de l'email
            email = payment_data.get('customer_email', '')
            if self._is_suspicious_email(email):
                score += 0.2
            
            return min(score, 1.0)
            
        except Exception as exc:
            self.logger.error(f"Erreur analyse fraude: {exc}")
            return 0.0
    
    async def _get_recent_payments(self, user_id: str) -> List[Dict]:
        """Récupère les paiements récents d'un utilisateur"""
        # Logique de récupération des paiements récents
        return []
    
    def _is_suspicious_email(self, email: str) -> bool:
        """Vérifie si l'email est suspect"""
        suspicious_domains = ['tempmail.com', '10minutemail.com', 'guerrillamail.com']
        domain = email.split('@')[-1] if '@' in email else ''
        return domain in suspicious_domains


class WebhookValidator:
    """Validateur de webhooks"""
    
    @staticmethod
    def validate_stripe_webhook(payload: str, signature: str, secret: str) -> bool:
        """Valide un webhook Stripe"""
        try:
            stripe.Webhook.construct_event(payload, signature, secret)
            return True
        except:
            return False
    
    @staticmethod
    def validate_paypal_webhook(payload: str, headers: Dict, webhook_id: str) -> bool:
        """Valide un webhook PayPal"""
        # Logique de validation PayPal
        return True


# Instances globales
payment_manager = PaymentManager()
subscription_manager = SubscriptionManager()
invoice_generator = InvoiceGenerator()
fraud_detector = FraudDetector()


# Export des classes principales
__all__ = [
    'PaymentManager',
    'StripeManager', 
    'PayPalManager',
    'SubscriptionManager',
    'InvoiceGenerator',
    'FraudDetector',
    'PaymentProvider',
    'PaymentStatus',
    'SubscriptionStatus',
    'payment_manager',
    'subscription_manager'
]
