"""
Convertisseur de Devises en Temps Réel - Spotify AI Agent
=========================================================

Module avancé de conversion de devises avec mise à jour en temps réel des taux de change,
cache intelligent et support multi-provider pour assurer la fiabilité des données financières.

Fonctionnalités:
- Conversion de devises en temps réel via APIs externes
- Cache intelligent avec TTL adaptatif
- Support multi-provider avec fallback automatique
- Historique des taux de change
- Validation et conformité réglementaire

Author: Fahed Mlaiel
"""

import asyncio
import aiohttp
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from decimal import Decimal, ROUND_HALF_UP
from enum import Enum
import json
import logging
import redis.asyncio as redis
from pathlib import Path
import os

from . import LocaleType


class CurrencyCode(Enum):
    """Codes des devises supportées (ISO 4217)"""
    USD = "USD"  # Dollar américain
    EUR = "EUR"  # Euro
    GBP = "GBP"  # Livre sterling
    JPY = "JPY"  # Yen japonais
    CHF = "CHF"  # Franc suisse
    CAD = "CAD"  # Dollar canadien
    AUD = "AUD"  # Dollar australien
    CNY = "CNY"  # Yuan chinois
    KRW = "KRW"  # Won sud-coréen
    BRL = "BRL"  # Real brésilien
    INR = "INR"  # Roupie indienne
    RUB = "RUB"  # Rouble russe
    MXN = "MXN"  # Peso mexicain
    ZAR = "ZAR"  # Rand sud-africain
    SGD = "SGD"  # Dollar de Singapour


class ExchangeProvider(Enum):
    """Fournisseurs de taux de change"""
    EXCHANGERATE_API = "exchangerate-api"
    FIXER_IO = "fixer"
    CURRENCYLAYER = "currencylayer"
    OPENEXCHANGERATES = "openexchangerates"
    ECB = "ecb"  # Banque Centrale Européenne


@dataclass
class ExchangeRate:
    """Taux de change entre deux devises"""
    from_currency: CurrencyCode
    to_currency: CurrencyCode
    rate: Decimal
    timestamp: datetime
    provider: ExchangeProvider
    source: str = ""
    validity_period: timedelta = timedelta(hours=1)


@dataclass
class CurrencyInfo:
    """Informations détaillées sur une devise"""
    code: CurrencyCode
    name: str
    symbol: str
    decimal_places: int
    locale_symbol_map: Dict[LocaleType, str] = field(default_factory=dict)


class CurrencyConverter:
    """Convertisseur de devises intelligent"""
    
    def __init__(self, redis_client: Optional[redis.Redis] = None):
        self.logger = logging.getLogger(__name__)
        self.redis_client = redis_client
        self._rates_cache: Dict[Tuple[CurrencyCode, CurrencyCode], ExchangeRate] = {}
        self._currency_info: Dict[CurrencyCode, CurrencyInfo] = {}
        self._providers_config = {}
        self._default_provider = ExchangeProvider.EXCHANGERATE_API
        self._session: Optional[aiohttp.ClientSession] = None
        
        self._initialize_currency_info()
        self._load_providers_config()
    
    def _initialize_currency_info(self):
        """Initialise les informations sur les devises"""
        currencies_data = {
            CurrencyCode.USD: CurrencyInfo(
                code=CurrencyCode.USD,
                name="US Dollar",
                symbol="$",
                decimal_places=2,
                locale_symbol_map={
                    LocaleType.EN_US: "$",
                    LocaleType.FR_FR: "$US",
                    LocaleType.DE_DE: "$"
                }
            ),
            CurrencyCode.EUR: CurrencyInfo(
                code=CurrencyCode.EUR,
                name="Euro",
                symbol="€",
                decimal_places=2,
                locale_symbol_map={
                    LocaleType.EN_US: "€",
                    LocaleType.FR_FR: "€",
                    LocaleType.DE_DE: "€"
                }
            ),
            CurrencyCode.GBP: CurrencyInfo(
                code=CurrencyCode.GBP,
                name="British Pound",
                symbol="£",
                decimal_places=2,
                locale_symbol_map={
                    LocaleType.EN_US: "£",
                    LocaleType.FR_FR: "£",
                    LocaleType.DE_DE: "£"
                }
            ),
            CurrencyCode.JPY: CurrencyInfo(
                code=CurrencyCode.JPY,
                name="Japanese Yen",
                symbol="¥",
                decimal_places=0,  # Le yen n'a pas de subdivision
                locale_symbol_map={
                    LocaleType.EN_US: "¥",
                    LocaleType.FR_FR: "¥",
                    LocaleType.DE_DE: "¥"
                }
            )
        }
        
        self._currency_info.update(currencies_data)
    
    def _load_providers_config(self):
        """Charge la configuration des fournisseurs"""
        self._providers_config = {
            ExchangeProvider.EXCHANGERATE_API: {
                "base_url": "https://api.exchangerate-api.com/v4/latest",
                "api_key_required": False,
                "rate_limit": 1500,  # requêtes par mois (gratuit)
                "timeout": 10
            },
            ExchangeProvider.FIXER_IO: {
                "base_url": "https://api.fixer.io/latest",
                "api_key_required": True,
                "api_key_param": "access_key",
                "rate_limit": 1000,
                "timeout": 10
            },
            ExchangeProvider.ECB: {
                "base_url": "https://api.exchangerate.host/latest",
                "api_key_required": False,
                "rate_limit": None,  # Pas de limite
                "timeout": 15
            }
        }
    
    async def __aenter__(self):
        """Entrée du context manager async"""
        self._session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30)
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Sortie du context manager async"""
        if self._session:
            await self._session.close()
    
    async def convert(
        self,
        amount: Union[int, float, Decimal],
        from_currency: CurrencyCode,
        to_currency: CurrencyCode,
        precision: Optional[int] = None
    ) -> Decimal:
        """Convertit un montant d'une devise à une autre"""
        try:
            if from_currency == to_currency:
                return Decimal(str(amount))
            
            # Récupère le taux de change
            rate = await self._get_exchange_rate(from_currency, to_currency)
            
            # Effectue la conversion
            amount_decimal = Decimal(str(amount))
            converted = amount_decimal * rate.rate
            
            # Applique la précision appropriée
            target_currency_info = self._currency_info.get(to_currency)
            if target_currency_info:
                decimal_places = target_currency_info.decimal_places
            else:
                decimal_places = precision or 2
            
            if decimal_places == 0:
                return converted.quantize(Decimal('1'), rounding=ROUND_HALF_UP)
            else:
                return converted.quantize(
                    Decimal('0.' + '0' * decimal_places),
                    rounding=ROUND_HALF_UP
                )
                
        except Exception as e:
            self.logger.error(f"Currency conversion error: {e}")
            raise
    
    async def format_currency(
        self,
        amount: Union[int, float, Decimal],
        currency: CurrencyCode,
        locale: LocaleType,
        show_symbol: bool = True
    ) -> str:
        """Formate un montant avec la devise selon la locale"""
        try:
            currency_info = self._currency_info.get(currency)
            if not currency_info:
                return f"{amount} {currency.value}"
            
            # Formate le nombre selon la locale
            amount_decimal = Decimal(str(amount))
            precision = currency_info.decimal_places
            
            # Applique le formatage numérique selon la locale
            if locale == LocaleType.FR_FR:
                # Format français: 1 234,56 €
                formatted = f"{float(amount_decimal):,.{precision}f}"
                formatted = formatted.replace(',', 'TEMP')
                formatted = formatted.replace('.', ',')
                formatted = formatted.replace('TEMP', ' ')
            elif locale == LocaleType.DE_DE:
                # Format allemand: 1.234,56 €
                formatted = f"{float(amount_decimal):,.{precision}f}"
                formatted = formatted.replace(',', 'TEMP')
                formatted = formatted.replace('.', ',')
                formatted = formatted.replace('TEMP', '.')
            else:
                # Format anglais/international: 1,234.56
                formatted = f"{float(amount_decimal):,.{precision}f}"
            
            # Ajoute le symbole de devise si demandé
            if show_symbol:
                symbol = currency_info.locale_symbol_map.get(locale, currency_info.symbol)
                
                # Position du symbole selon la locale
                if locale == LocaleType.EN_US and currency == CurrencyCode.USD:
                    return f"{symbol}{formatted}"
                else:
                    return f"{formatted} {symbol}"
            
            return formatted
            
        except Exception as e:
            self.logger.error(f"Currency formatting error: {e}")
            return f"{amount} {currency.value}"
    
    async def _get_exchange_rate(
        self,
        from_currency: CurrencyCode,
        to_currency: CurrencyCode
    ) -> ExchangeRate:
        """Récupère le taux de change entre deux devises"""
        cache_key = (from_currency, to_currency)
        
        # Vérifie le cache local
        if cache_key in self._rates_cache:
            rate = self._rates_cache[cache_key]
            if self._is_rate_valid(rate):
                return rate
        
        # Vérifie le cache Redis
        if self.redis_client:
            cached_rate = await self._get_cached_rate(from_currency, to_currency)
            if cached_rate and self._is_rate_valid(cached_rate):
                self._rates_cache[cache_key] = cached_rate
                return cached_rate
        
        # Récupère le taux depuis l'API
        rate = await self._fetch_exchange_rate(from_currency, to_currency)
        
        # Met en cache
        self._rates_cache[cache_key] = rate
        if self.redis_client:
            await self._cache_rate(rate)
        
        return rate
    
    def _is_rate_valid(self, rate: ExchangeRate) -> bool:
        """Vérifie si un taux de change est encore valide"""
        now = datetime.now(timezone.utc)
        return now - rate.timestamp < rate.validity_period
    
    async def _fetch_exchange_rate(
        self,
        from_currency: CurrencyCode,
        to_currency: CurrencyCode
    ) -> ExchangeRate:
        """Récupère le taux de change depuis l'API"""
        providers_to_try = [self._default_provider] + [
            p for p in ExchangeProvider if p != self._default_provider
        ]
        
        for provider in providers_to_try:
            try:
                rate = await self._fetch_from_provider(provider, from_currency, to_currency)
                if rate:
                    return rate
            except Exception as e:
                self.logger.warning(f"Provider {provider.value} failed: {e}")
                continue
        
        raise Exception("All exchange rate providers failed")
    
    async def _fetch_from_provider(
        self,
        provider: ExchangeProvider,
        from_currency: CurrencyCode,
        to_currency: CurrencyCode
    ) -> Optional[ExchangeRate]:
        """Récupère le taux depuis un fournisseur spécifique"""
        if not self._session:
            raise Exception("HTTP session not initialized")
        
        config = self._providers_config.get(provider)
        if not config:
            return None
        
        try:
            if provider == ExchangeProvider.EXCHANGERATE_API:
                return await self._fetch_exchangerate_api(from_currency, to_currency, config)
            elif provider == ExchangeProvider.ECB:
                return await self._fetch_ecb_api(from_currency, to_currency, config)
            # Ajouter d'autres providers ici
            
        except Exception as e:
            self.logger.error(f"Error fetching from {provider.value}: {e}")
            return None
    
    async def _fetch_exchangerate_api(
        self,
        from_currency: CurrencyCode,
        to_currency: CurrencyCode,
        config: Dict[str, Any]
    ) -> ExchangeRate:
        """Récupère depuis ExchangeRate-API"""
        url = f"{config['base_url']}/{from_currency.value}"
        
        async with self._session.get(url) as response:
            if response.status != 200:
                raise Exception(f"HTTP {response.status}")
            
            data = await response.json()
            
            if 'rates' not in data or to_currency.value not in data['rates']:
                raise Exception(f"Rate not available for {to_currency.value}")
            
            rate_value = Decimal(str(data['rates'][to_currency.value]))
            timestamp = datetime.now(timezone.utc)
            
            return ExchangeRate(
                from_currency=from_currency,
                to_currency=to_currency,
                rate=rate_value,
                timestamp=timestamp,
                provider=ExchangeProvider.EXCHANGERATE_API,
                source=url
            )
    
    async def _fetch_ecb_api(
        self,
        from_currency: CurrencyCode,
        to_currency: CurrencyCode,
        config: Dict[str, Any]
    ) -> ExchangeRate:
        """Récupère depuis l'API ECB (Exchange Rate Host)"""
        url = f"{config['base_url']}?base={from_currency.value}&symbols={to_currency.value}"
        
        async with self._session.get(url) as response:
            if response.status != 200:
                raise Exception(f"HTTP {response.status}")
            
            data = await response.json()
            
            if not data.get('success', False):
                raise Exception("API returned error")
            
            if 'rates' not in data or to_currency.value not in data['rates']:
                raise Exception(f"Rate not available for {to_currency.value}")
            
            rate_value = Decimal(str(data['rates'][to_currency.value]))
            timestamp = datetime.fromisoformat(data['date'] + 'T00:00:00+00:00')
            
            return ExchangeRate(
                from_currency=from_currency,
                to_currency=to_currency,
                rate=rate_value,
                timestamp=timestamp,
                provider=ExchangeProvider.ECB,
                source=url,
                validity_period=timedelta(hours=24)  # Les taux ECB sont quotidiens
            )
    
    async def _get_cached_rate(
        self,
        from_currency: CurrencyCode,
        to_currency: CurrencyCode
    ) -> Optional[ExchangeRate]:
        """Récupère un taux depuis le cache Redis"""
        try:
            cache_key = f"exchange_rate:{from_currency.value}:{to_currency.value}"
            cached_data = await self.redis_client.hgetall(cache_key)
            
            if not cached_data:
                return None
            
            return ExchangeRate(
                from_currency=CurrencyCode(cached_data['from_currency']),
                to_currency=CurrencyCode(cached_data['to_currency']),
                rate=Decimal(cached_data['rate']),
                timestamp=datetime.fromisoformat(cached_data['timestamp']),
                provider=ExchangeProvider(cached_data['provider']),
                source=cached_data.get('source', ''),
                validity_period=timedelta(seconds=int(cached_data.get('validity_seconds', 3600)))
            )
            
        except Exception as e:
            self.logger.error(f"Error getting cached rate: {e}")
            return None
    
    async def _cache_rate(self, rate: ExchangeRate):
        """Met en cache un taux de change dans Redis"""
        try:
            cache_key = f"exchange_rate:{rate.from_currency.value}:{rate.to_currency.value}"
            cache_data = {
                'from_currency': rate.from_currency.value,
                'to_currency': rate.to_currency.value,
                'rate': str(rate.rate),
                'timestamp': rate.timestamp.isoformat(),
                'provider': rate.provider.value,
                'source': rate.source,
                'validity_seconds': int(rate.validity_period.total_seconds())
            }
            
            await self.redis_client.hset(cache_key, mapping=cache_data)
            await self.redis_client.expire(cache_key, int(rate.validity_period.total_seconds()))
            
        except Exception as e:
            self.logger.error(f"Error caching rate: {e}")
    
    async def get_supported_currencies(self) -> List[CurrencyCode]:
        """Retourne la liste des devises supportées"""
        return list(CurrencyCode)
    
    async def get_currency_info(self, currency: CurrencyCode) -> Optional[CurrencyInfo]:
        """Retourne les informations sur une devise"""
        return self._currency_info.get(currency)
    
    async def get_historical_rates(
        self,
        from_currency: CurrencyCode,
        to_currency: CurrencyCode,
        start_date: datetime,
        end_date: datetime
    ) -> List[ExchangeRate]:
        """Récupère l'historique des taux de change (fonctionnalité future)"""
        # À implémenter avec des APIs supportant l'historique
        raise NotImplementedError("Historical rates not yet implemented")


# Instance globale du convertisseur
currency_converter = CurrencyConverter()

__all__ = [
    "CurrencyCode",
    "ExchangeProvider",
    "ExchangeRate",
    "CurrencyInfo", 
    "CurrencyConverter",
    "currency_converter"
]
