"""
Exemples d'utilisation du gestionnaire de cache avancé
Auteur: Fahed Mlaiel - Lead Dev & Architecte IA
"""

import asyncio
from datetime import timedelta
from core.cache import (
    CacheManager, CacheKey, cache_manager, 
    cache_result, cache_key_for_tenant
)

async def example_basic_cache_usage():
    """Exemple d'utilisation basique du cache"""
    
    # Initialisation du gestionnaire de cache
    config = {
        "redis_url": "redis://localhost:6379",
        "redis_db": 0,
        "memory_max_size": 1000,
        "default_ttl": 3600,
        "redis_enabled": True
    }
    
    cache_mgr = CacheManager(config)
    await cache_mgr.initialize()
    
    # Stockage d'une configuration de tenant
    tenant_config = {
        "name": "ACME Corporation",
        "plan": "premium",
        "features": ["audio_processing", "analytics"],
        "quota": {"api_calls": 10000, "storage_gb": 100}
    }
    
    success = await cache_mgr.set_tenant_config("acme_corp", tenant_config, ttl=7200)
    print(f"Configuration stockée: {success}")
    
    # Récupération de la configuration
    retrieved_config = await cache_mgr.get_tenant_config("acme_corp")
    print(f"Configuration récupérée: {retrieved_config}")
    
    # Statistiques du cache
    stats = await cache_mgr.get_stats()
    print(f"Statistiques du cache: {stats}")
    
    # Nettoyage
    await cache_mgr.cleanup()

async def example_advanced_cache_usage():
    """Exemple d'utilisation avancée avec clés personnalisées"""
    
    await cache_manager.initialize()
    
    # Clé de cache personnalisée
    user_profile_key = CacheKey(
        namespace="users",
        tenant_id="acme_corp",
        entity_type="profile",
        entity_id="user_123",
        tags={"premium", "verified"}
    )
    
    user_profile = {
        "user_id": "user_123",
        "name": "John Doe",
        "email": "john@acme.com",
        "preferences": {
            "theme": "dark",
            "language": "fr",
            "notifications": True
        },
        "subscription": {
            "plan": "premium",
            "expires_at": "2025-12-31"
        }
    }
    
    # Stockage avec TTL personnalisé
    await cache_manager.cache.set(user_profile_key, user_profile, ttl=1800)
    
    # Récupération
    cached_profile = await cache_manager.cache.get(user_profile_key)
    print(f"Profil utilisateur: {cached_profile}")
    
    # Invalidation par pattern (tous les utilisateurs du tenant)
    pattern = CacheKey(
        namespace="users",
        tenant_id="acme_corp",
        entity_type="*",
        entity_id="*"
    ).to_pattern()
    
    invalidated_count = await cache_manager.cache.invalidate_pattern(pattern)
    print(f"Entrées invalidées: {invalidated_count}")

async def example_cache_decorator():
    """Exemple d'utilisation du décorateur de cache"""
    
    await cache_manager.initialize()
    
    async def expensive_computation(tenant_id: str, data_type: str) -> dict:
        """Fonction coûteuse à mettre en cache"""
        print(f"Calcul coûteux pour {tenant_id}...")
        await asyncio.sleep(2)  # Simulation de calcul long
        
        return {
            "tenant_id": tenant_id,
            "data_type": data_type,
            "result": f"Computed result for {tenant_id}",
            "computation_time": 2.0
        }
    
    # Utilisation avec le décorateur de cache
    cache_key = cache_key_for_tenant("acme_corp", "analytics", "monthly_report")
    
    # Premier appel - calcul effectué
    result1 = await cache_result(
        cache_key, 
        expensive_computation, 
        "acme_corp", 
        "monthly_report",
        ttl=3600
    )
    print(f"Premier appel: {result1}")
    
    # Deuxième appel - résultat depuis le cache
    result2 = await cache_result(
        cache_key, 
        expensive_computation, 
        "acme_corp", 
        "monthly_report",
        ttl=3600
    )
    print(f"Deuxième appel (cache): {result2}")

async def example_multi_level_cache():
    """Exemple de cache multi-niveaux"""
    
    from core.cache import MemoryCacheBackend, RedisCacheBackend, DistributedCache
    
    # Configuration de backends spécifiques
    l1_backend = MemoryCacheBackend(max_size=500)
    l2_backend = RedisCacheBackend("redis://localhost:6379", db=1)
    
    # Initialisation du backend Redis
    await l2_backend.initialize()
    
    # Cache distribué personnalisé
    distributed_cache = DistributedCache(
        l1_backend=l1_backend,
        l2_backend=l2_backend,
        default_ttl=1800,
        compression_threshold=512
    )
    
    # Test de stockage et récupération
    test_key = CacheKey(
        namespace="test",
        entity_type="performance",
        entity_id="multi_level"
    )
    
    large_data = {
        "data": "x" * 1000,  # Données importantes pour tester la compression
        "metadata": {"size": 1000, "type": "test"}
    }
    
    # Stockage
    await distributed_cache.set(test_key, large_data)
    
    # Récupération (devrait venir du L1)
    cached_data = await distributed_cache.get(test_key)
    print(f"Données récupérées: {len(str(cached_data))} caractères")
    
    # Statistiques détaillées
    stats = await distributed_cache.get_cache_stats()
    print(f"Statistiques multi-niveaux: {stats}")
    
    # Nettoyage
    await l2_backend.close()

async def example_cache_monitoring():
    """Exemple de monitoring du cache"""
    
    await cache_manager.initialize()
    
    # Simulation d'opérations multiples
    for i in range(100):
        key = cache_key_for_tenant(f"tenant_{i % 10}", "data", f"item_{i}")
        data = {"id": i, "value": f"data_{i}", "timestamp": "2025-01-19"}
        
        await cache_manager.cache.set(key, data, ttl=300)
        
        # Quelques récupérations
        if i % 3 == 0:
            await cache_manager.cache.get(key)
    
    # Statistiques finales
    final_stats = await cache_manager.get_stats()
    print(f"Statistiques finales: {final_stats}")
    
    # Calcul des métriques de performance
    distributed_stats = final_stats.get("distributed_cache", {})
    total_ops = distributed_stats.get("operations", 0)
    hit_rate = distributed_stats.get("total_hit_rate", 0)
    
    print(f"Opérations totales: {total_ops}")
    print(f"Taux de hit global: {hit_rate}%")

if __name__ == "__main__":
    async def run_examples():
        """Exécute tous les exemples"""
        print("=== Exemple d'utilisation basique ===")
        await example_basic_cache_usage()
        
        print("\n=== Exemple d'utilisation avancée ===")
        await example_advanced_cache_usage()
        
        print("\n=== Exemple de décorateur de cache ===")
        await example_cache_decorator()
        
        print("\n=== Exemple de cache multi-niveaux ===")
        await example_multi_level_cache()
        
        print("\n=== Exemple de monitoring ===")
        await example_cache_monitoring()
    
    # Exécution des exemples
    asyncio.run(run_examples())
