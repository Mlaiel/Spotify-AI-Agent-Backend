"""
Module GraphQL industriel pour l’agent IA Spotify.
Expose : schéma, resolvers, mutations, subscriptions, scalaires custom.
"""

from .schema import schema
from .resolvers import query, mutation, subscription
from .mutations import mutation as advanced_mutation
from .subscriptions import subscription as advanced_subscription
from .scalars import datetime_scalar, json_scalar

__all__ = [
    "schema",
    "query",
    "mutation",
    "subscription",
    "advanced_mutation",
    "advanced_subscription",
    "datetime_scalar",
    "json_scalar"
]
