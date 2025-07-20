"""
Script CLI pour préchauffer le cache avec des données critiques (Spotify, IA, analytics, scoring, etc.).
"""
import argparse
import logging
from backend.app.services.cache import CacheManager

def get_spotify_top_tracks():
    # Simulation d’appel API Spotify
    return {"tracks": ["track1", "track2", "track3"]}

def get_ai_recommendations():
    # Simulation d’appel IA
    return {"recommendations": ["rec1", "rec2"]}

def get_analytics_dashboard():
    # Simulation d’agrégation analytics
    return {"users": 1234, "active": 321}

def main():
    parser = argparse.ArgumentParser(description="Préchauffage du cache Spotify AI Agent.")
    parser.add_argument('--backend', default='redis', help='Backend de cache (redis, memory)')
    parser.add_argument('--ttl', type=int, default=3600, help='TTL par défaut (secondes)')
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    cache = CacheManager(backend=args.backend)

    # Hooks d’audit/logs
    def audit_hook(key):
        logging.info(f"[HOOK] Invalidation hook triggered for key: {key}")
    cache.invalidator.register_hook(audit_hook)

    # Données métier à préchauffer
    data = {
        "spotify:top_tracks": get_spotify_top_tracks(),
        "ai:recommendations": get_ai_recommendations(),
        "analytics:dashboard": get_analytics_dashboard(),
        # Ajoutez ici d’autres clés critiques métier
    }
    cache.warmup(data, ttl=args.ttl)
    print("Cache warmup terminé avec les données métier critiques.")

    # Exemple d’invalidation post-warmup
    cache.invalidate("ai:recommendations")
    print("Exemple: clé 'ai:recommendations' invalidée.")

if __name__ == "__main__":
    main()
