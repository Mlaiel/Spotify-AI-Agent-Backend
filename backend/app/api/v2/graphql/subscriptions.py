from ariadne import SubscriptionType
import asyncio
import logging

subscription = SubscriptionType()
logger = logging.getLogger("GraphQLSubscriptions")

@subscription.source("onAnalyticsUpdate")
async def on_analytics_update_generator(_, info, artistId):
    # Simuler notifications analytics temps réel
    for i in range(3):
        await asyncio.sleep(2)
        yield {"artistId": artistId, "listeners": 40000 + i*1000}

@subscription.field("onAnalyticsUpdate")
def on_analytics_update_resolver(event, info, artistId):
    logger.info(f"GraphQL: onAnalyticsUpdate pour {artistId}")
    return event
