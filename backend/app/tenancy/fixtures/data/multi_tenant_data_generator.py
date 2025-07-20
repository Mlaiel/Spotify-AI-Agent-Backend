"""
Multi-Tenant Test Data Generator for Spotify AI Agent
===================================================

This module generates realistic test data for multi-tenant scenarios,
providing comprehensive datasets for testing database configurations,
tenant isolation, and performance across different subscription tiers.

Features:
- Realistic user and music data generation
- Tenant-specific data isolation
- Performance testing datasets
- Compliance testing data
- Analytics and ML training data

Author: AI Assistant
Version: 1.0.0
"""

import json
import uuid
import random
import asyncio
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import csv
import yaml
from pathlib import Path
import faker
from faker.providers import internet, person, company
import asyncpg
import motor.motor_asyncio
import aioredis


class TenantTier(Enum):
    """Tenant subscription tiers."""
    FREE = "free"
    STANDARD = "standard"
    PREMIUM = "premium"
    ENTERPRISE = "enterprise"


class DataType(Enum):
    """Types of test data to generate."""
    USERS = "users"
    TRACKS = "tracks"
    ALBUMS = "albums"
    ARTISTS = "artists"
    PLAYLISTS = "playlists"
    LISTENING_HISTORY = "listening_history"
    RECOMMENDATIONS = "recommendations"
    ANALYTICS_EVENTS = "analytics_events"
    ML_FEATURES = "ml_features"


@dataclass
class TenantDataProfile:
    """Profile defining data generation parameters for a tenant."""
    tenant_id: str
    tenant_name: str
    tier: TenantTier
    
    # Data volume based on tier
    num_users: int
    num_tracks: int
    num_albums: int
    num_artists: int
    num_playlists: int
    num_listening_events: int
    
    # Data characteristics
    user_activity_level: float  # 0.0 to 1.0
    music_diversity: float  # 0.0 to 1.0
    geographic_spread: List[str]  # Country codes
    
    # Compliance requirements
    gdpr_compliant: bool = False
    pii_anonymization: bool = False
    
    # Performance testing
    generate_load_data: bool = False
    concurrent_users: int = 0


class MultiTenantDataGenerator:
    """
    Generates realistic test data for multi-tenant scenarios.
    
    Provides comprehensive data generation for testing:
    - Tenant isolation and security
    - Performance across different tiers
    - Compliance and data governance
    - Analytics and ML workflows
    """
    
    def __init__(self, output_path: str = "/app/tenancy/fixtures/data"):
        """
        Initialize the data generator.
        
        Args:
            output_path: Path where generated data files will be saved
        """
        self.output_path = Path(output_path)
        self.output_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize Faker for realistic data generation
        self.fake = faker.Faker(['en_US', 'en_GB', 'fr_FR', 'de_DE', 'es_ES'])
        self.fake.add_provider(internet)
        self.fake.add_provider(person)
        self.fake.add_provider(company)
        
        # Music genre and style data
        self.genres = [
            "Pop", "Rock", "Hip-Hop", "Electronic", "Classical", "Jazz", "Country",
            "R&B", "Reggae", "Blues", "Folk", "Punk", "Metal", "Indie", "Alternative",
            "World", "Latin", "Gospel", "Funk", "Disco", "House", "Techno", "Ambient"
        ]
        
        self.moods = [
            "Happy", "Sad", "Energetic", "Relaxed", "Romantic", "Aggressive",
            "Melancholic", "Uplifting", "Dark", "Dreamy", "Intense", "Peaceful"
        ]
        
        # Countries for geographic distribution
        self.countries = [
            "US", "GB", "FR", "DE", "ES", "IT", "NL", "SE", "NO", "DK",
            "CA", "AU", "JP", "KR", "BR", "MX", "IN", "CN", "RU", "PL"
        ]
        
        # Tier-specific configurations
        self.tier_configs = {
            TenantTier.FREE: {
                "max_users": 10,
                "max_tracks": 100,
                "max_albums": 20,
                "max_artists": 30,
                "max_playlists": 5,
                "max_listening_events": 1000,
                "activity_multiplier": 0.3,
                "diversity_factor": 0.5
            },
            TenantTier.STANDARD: {
                "max_users": 1000,
                "max_tracks": 10000,
                "max_albums": 2000,
                "max_artists": 3000,
                "max_playlists": 500,
                "max_listening_events": 100000,
                "activity_multiplier": 0.7,
                "diversity_factor": 0.7
            },
            TenantTier.PREMIUM: {
                "max_users": 10000,
                "max_tracks": 100000,
                "max_albums": 20000,
                "max_artists": 30000,
                "max_playlists": 5000,
                "max_listening_events": 1000000,
                "activity_multiplier": 0.9,
                "diversity_factor": 0.9
            },
            TenantTier.ENTERPRISE: {
                "max_users": 100000,
                "max_tracks": 1000000,
                "max_albums": 200000,
                "max_artists": 300000,
                "max_playlists": 50000,
                "max_listening_events": 10000000,
                "activity_multiplier": 1.0,
                "diversity_factor": 1.0
            }
        }
        
    def create_tenant_profile(
        self,
        tenant_id: str,
        tenant_name: str,
        tier: TenantTier,
        scale_factor: float = 1.0,
        **kwargs
    ) -> TenantDataProfile:
        """
        Create a data generation profile for a tenant.
        
        Args:
            tenant_id: Unique tenant identifier
            tenant_name: Human-readable tenant name
            tier: Subscription tier
            scale_factor: Scale data volumes (0.1 = 10% of tier max)
            **kwargs: Additional profile parameters
            
        Returns:
            Tenant data generation profile
        """
        config = self.tier_configs[tier]
        
        # Calculate data volumes based on tier and scale factor
        num_users = int(config["max_users"] * scale_factor)
        num_tracks = int(config["max_tracks"] * scale_factor)
        num_albums = int(config["max_albums"] * scale_factor)
        num_artists = int(config["max_artists"] * scale_factor)
        num_playlists = int(config["max_playlists"] * scale_factor)
        num_listening_events = int(config["max_listening_events"] * scale_factor)
        
        # Ensure minimum viable data
        num_users = max(1, num_users)
        num_tracks = max(10, num_tracks)
        num_albums = max(2, num_albums)
        num_artists = max(3, num_artists)
        num_playlists = max(1, num_playlists)
        num_listening_events = max(100, num_listening_events)
        
        return TenantDataProfile(
            tenant_id=tenant_id,
            tenant_name=tenant_name,
            tier=tier,
            num_users=num_users,
            num_tracks=num_tracks,
            num_albums=num_albums,
            num_artists=num_artists,
            num_playlists=num_playlists,
            num_listening_events=num_listening_events,
            user_activity_level=config["activity_multiplier"],
            music_diversity=config["diversity_factor"],
            geographic_spread=random.sample(self.countries, min(5, len(self.countries))),
            gdpr_compliant=tier in [TenantTier.PREMIUM, TenantTier.ENTERPRISE],
            pii_anonymization=tier == TenantTier.ENTERPRISE,
            generate_load_data=tier in [TenantTier.PREMIUM, TenantTier.ENTERPRISE],
            concurrent_users=num_users // 10,
            **kwargs
        )
        
    async def generate_tenant_data(
        self,
        profile: TenantDataProfile,
        data_types: Optional[List[DataType]] = None
    ) -> Dict[str, Any]:
        """
        Generate complete dataset for a tenant.
        
        Args:
            profile: Tenant data generation profile
            data_types: Specific data types to generate (default: all)
            
        Returns:
            Complete tenant dataset
        """
        if data_types is None:
            data_types = list(DataType)
            
        tenant_data = {
            "tenant_info": {
                "tenant_id": profile.tenant_id,
                "tenant_name": profile.tenant_name,
                "tier": profile.tier.value,
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "profile": profile.__dict__
            },
            "data": {}
        }
        
        # Generate data based on dependencies
        generated_data = {}
        
        # Step 1: Generate base entities
        if DataType.ARTISTS in data_types:
            generated_data["artists"] = await self._generate_artists(profile)
            
        if DataType.ALBUMS in data_types:
            artists = generated_data.get("artists", await self._generate_artists(profile))
            generated_data["albums"] = await self._generate_albums(profile, artists)
            
        if DataType.TRACKS in data_types:
            artists = generated_data.get("artists", await self._generate_artists(profile))
            albums = generated_data.get("albums", await self._generate_albums(profile, artists))
            generated_data["tracks"] = await self._generate_tracks(profile, artists, albums)
            
        if DataType.USERS in data_types:
            generated_data["users"] = await self._generate_users(profile)
            
        # Step 2: Generate relationship entities
        if DataType.PLAYLISTS in data_types:
            users = generated_data.get("users", await self._generate_users(profile))
            tracks = generated_data.get("tracks", [])
            if not tracks:
                artists = generated_data.get("artists", await self._generate_artists(profile))
                albums = generated_data.get("albums", await self._generate_albums(profile, artists))
                tracks = await self._generate_tracks(profile, artists, albums)
            generated_data["playlists"] = await self._generate_playlists(profile, users, tracks)
            
        # Step 3: Generate behavioral data
        if DataType.LISTENING_HISTORY in data_types:
            users = generated_data.get("users", await self._generate_users(profile))
            tracks = generated_data.get("tracks", [])
            if not tracks:
                artists = generated_data.get("artists", await self._generate_artists(profile))
                albums = generated_data.get("albums", await self._generate_albums(profile, artists))
                tracks = await self._generate_tracks(profile, artists, albums)
            generated_data["listening_history"] = await self._generate_listening_history(
                profile, users, tracks
            )
            
        # Step 4: Generate ML and analytics data
        if DataType.RECOMMENDATIONS in data_types:
            users = generated_data.get("users", await self._generate_users(profile))
            tracks = generated_data.get("tracks", [])
            if not tracks:
                artists = generated_data.get("artists", await self._generate_artists(profile))
                albums = generated_data.get("albums", await self._generate_albums(profile, artists))
                tracks = await self._generate_tracks(profile, artists, albums)
            generated_data["recommendations"] = await self._generate_recommendations(
                profile, users, tracks
            )
            
        if DataType.ANALYTICS_EVENTS in data_types:
            users = generated_data.get("users", await self._generate_users(profile))
            generated_data["analytics_events"] = await self._generate_analytics_events(
                profile, users
            )
            
        if DataType.ML_FEATURES in data_types:
            users = generated_data.get("users", await self._generate_users(profile))
            tracks = generated_data.get("tracks", [])
            if not tracks:
                artists = generated_data.get("artists", await self._generate_artists(profile))
                albums = generated_data.get("albums", await self._generate_albums(profile, artists))
                tracks = await self._generate_tracks(profile, artists, albums)
            generated_data["ml_features"] = await self._generate_ml_features(
                profile, users, tracks
            )
            
        tenant_data["data"] = generated_data
        return tenant_data
        
    async def _generate_artists(self, profile: TenantDataProfile) -> List[Dict[str, Any]]:
        """Generate artist data for tenant."""
        artists = []
        
        for i in range(profile.num_artists):
            artist_id = f"{profile.tenant_id}_artist_{i+1:06d}"
            
            # Generate realistic artist name
            if random.random() < 0.3:  # 30% chance of band name
                name = self.fake.company().replace("Inc.", "").replace("LLC", "").strip()
            else:  # 70% chance of person name
                name = self.fake.name()
                
            artist = {
                "artist_id": artist_id,
                "tenant_id": profile.tenant_id,
                "name": name,
                "biography": self.fake.text(max_nb_chars=500),
                "genres": random.sample(self.genres, random.randint(1, 3)),
                "country": random.choice(profile.geographic_spread),
                "formed_year": random.randint(1960, 2020),
                "is_verified": random.random() < 0.1,  # 10% verified
                "monthly_listeners": random.randint(1000, 10000000),
                "social_media": {
                    "spotify_url": f"https://open.spotify.com/artist/{artist_id}",
                    "website": self.fake.url() if random.random() < 0.6 else None,
                    "instagram": f"@{name.lower().replace(' ', '')}" if random.random() < 0.7 else None
                },
                "created_at": self.fake.date_time_between(start_date="-5y").isoformat(),
                "updated_at": datetime.now(timezone.utc).isoformat()
            }
            
            # GDPR compliance: anonymize if required
            if profile.pii_anonymization:
                artist["biography"] = "[ANONYMIZED]"
                if "website" in artist["social_media"]:
                    artist["social_media"]["website"] = "[ANONYMIZED]"
                    
            artists.append(artist)
            
        return artists
        
    async def _generate_albums(
        self,
        profile: TenantDataProfile,
        artists: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Generate album data for tenant."""
        albums = []
        
        for i in range(profile.num_albums):
            album_id = f"{profile.tenant_id}_album_{i+1:06d}"
            artist = random.choice(artists)
            
            album_types = ["album", "single", "compilation", "ep"]
            album_type = random.choice(album_types)
            
            # Track count based on album type
            track_counts = {
                "single": random.randint(1, 3),
                "ep": random.randint(4, 8),
                "album": random.randint(8, 20),
                "compilation": random.randint(15, 50)
            }
            
            release_date = self.fake.date_between(start_date="-10y", end_date="today")
            
            album = {
                "album_id": album_id,
                "tenant_id": profile.tenant_id,
                "artist_id": artist["artist_id"],
                "artist_name": artist["name"],
                "title": self.fake.catch_phrase(),
                "album_type": album_type,
                "release_date": release_date.isoformat(),
                "genres": artist["genres"][:random.randint(1, len(artist["genres"]))],
                "total_tracks": track_counts[album_type],
                "duration_ms": random.randint(120000, 4800000),  # 2min to 80min
                "label": self.fake.company(),
                "copyright": f"(C) {release_date.year} {self.fake.company()}",
                "is_explicit": random.random() < 0.1,
                "popularity": random.randint(0, 100),
                "markets": random.sample(self.countries, random.randint(5, 15)),
                "images": [
                    {
                        "url": f"https://i.scdn.co/image/{uuid.uuid4()}",
                        "height": 640,
                        "width": 640
                    },
                    {
                        "url": f"https://i.scdn.co/image/{uuid.uuid4()}",
                        "height": 300,
                        "width": 300
                    }
                ],
                "created_at": release_date.isoformat(),
                "updated_at": datetime.now(timezone.utc).isoformat()
            }
            
            albums.append(album)
            
        return albums
        
    async def _generate_tracks(
        self,
        profile: TenantDataProfile,
        artists: List[Dict[str, Any]],
        albums: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Generate track data for tenant."""
        tracks = []
        
        # First, generate tracks for existing albums
        album_tracks = {}
        for album in albums:
            album_tracks[album["album_id"]] = []
            
            for track_num in range(1, album["total_tracks"] + 1):
                track_id = f"{profile.tenant_id}_track_{len(tracks)+1:06d}"
                
                # Audio features for ML
                audio_features = {
                    "danceability": random.uniform(0.0, 1.0),
                    "energy": random.uniform(0.0, 1.0),
                    "key": random.randint(0, 11),
                    "loudness": random.uniform(-60.0, 0.0),
                    "mode": random.randint(0, 1),
                    "speechiness": random.uniform(0.0, 1.0),
                    "acousticness": random.uniform(0.0, 1.0),
                    "instrumentalness": random.uniform(0.0, 1.0),
                    "liveness": random.uniform(0.0, 1.0),
                    "valence": random.uniform(0.0, 1.0),
                    "tempo": random.uniform(60.0, 200.0),
                    "time_signature": random.choice([3, 4, 5])
                }
                
                track = {
                    "track_id": track_id,
                    "tenant_id": profile.tenant_id,
                    "name": self.fake.catch_phrase(),
                    "album_id": album["album_id"],
                    "album_name": album["title"],
                    "artist_id": album["artist_id"],
                    "artist_name": album["artist_name"],
                    "disc_number": 1,
                    "track_number": track_num,
                    "duration_ms": random.randint(30000, 600000),  # 30sec to 10min
                    "explicit": random.random() < 0.05,
                    "is_playable": True,
                    "popularity": random.randint(0, 100),
                    "preview_url": f"https://p.scdn.co/mp3-preview/{track_id}",
                    "external_urls": {
                        "spotify": f"https://open.spotify.com/track/{track_id}"
                    },
                    "genres": album["genres"],
                    "mood": random.choice(self.moods),
                    "audio_features": audio_features,
                    "markets": album["markets"],
                    "created_at": album["release_date"],
                    "updated_at": datetime.now(timezone.utc).isoformat()
                }
                
                tracks.append(track)
                album_tracks[album["album_id"]].append(track)
                
        # Generate additional standalone tracks if needed
        remaining_tracks = profile.num_tracks - len(tracks)
        if remaining_tracks > 0:
            for i in range(remaining_tracks):
                track_id = f"{profile.tenant_id}_track_{len(tracks)+1:06d}"
                artist = random.choice(artists)
                
                audio_features = {
                    "danceability": random.uniform(0.0, 1.0),
                    "energy": random.uniform(0.0, 1.0),
                    "key": random.randint(0, 11),
                    "loudness": random.uniform(-60.0, 0.0),
                    "mode": random.randint(0, 1),
                    "speechiness": random.uniform(0.0, 1.0),
                    "acousticness": random.uniform(0.0, 1.0),
                    "instrumentalness": random.uniform(0.0, 1.0),
                    "liveness": random.uniform(0.0, 1.0),
                    "valence": random.uniform(0.0, 1.0),
                    "tempo": random.uniform(60.0, 200.0),
                    "time_signature": random.choice([3, 4, 5])
                }
                
                track = {
                    "track_id": track_id,
                    "tenant_id": profile.tenant_id,
                    "name": self.fake.catch_phrase(),
                    "album_id": None,
                    "album_name": None,
                    "artist_id": artist["artist_id"],
                    "artist_name": artist["name"],
                    "disc_number": 1,
                    "track_number": 1,
                    "duration_ms": random.randint(30000, 600000),
                    "explicit": random.random() < 0.05,
                    "is_playable": True,
                    "popularity": random.randint(0, 100),
                    "preview_url": f"https://p.scdn.co/mp3-preview/{track_id}",
                    "external_urls": {
                        "spotify": f"https://open.spotify.com/track/{track_id}"
                    },
                    "genres": artist["genres"],
                    "mood": random.choice(self.moods),
                    "audio_features": audio_features,
                    "markets": random.sample(self.countries, random.randint(5, 15)),
                    "created_at": self.fake.date_time_between(start_date="-5y").isoformat(),
                    "updated_at": datetime.now(timezone.utc).isoformat()
                }
                
                tracks.append(track)
                
        return tracks
        
    async def _generate_users(self, profile: TenantDataProfile) -> List[Dict[str, Any]]:
        """Generate user data for tenant."""
        users = []
        
        for i in range(profile.num_users):
            user_id = f"{profile.tenant_id}_user_{i+1:06d}"
            
            # User demographics
            birth_date = self.fake.date_of_birth(minimum_age=13, maximum_age=80)
            country = random.choice(profile.geographic_spread)
            
            # Subscription status based on tenant tier
            subscription_map = {
                TenantTier.FREE: "free",
                TenantTier.STANDARD: "premium",
                TenantTier.PREMIUM: "premium",
                TenantTier.ENTERPRISE: "premium"
            }
            
            user = {
                "user_id": user_id,
                "tenant_id": profile.tenant_id,
                "display_name": self.fake.name(),
                "email": self.fake.email(),
                "birth_date": birth_date.isoformat(),
                "country": country,
                "subscription_type": subscription_map[profile.tier],
                "subscription_start_date": self.fake.date_between(start_date="-2y").isoformat(),
                "is_verified": random.random() < 0.3,
                "followers": random.randint(0, 10000),
                "following": random.randint(0, 1000),
                "preferences": {
                    "explicit_content": random.random() < 0.7,
                    "language": random.choice(["en", "fr", "de", "es", "it"]),
                    "preferred_genres": random.sample(self.genres, random.randint(3, 8)),
                    "discovery_mode": random.choice(["adventurous", "balanced", "familiar"])
                },
                "activity_metrics": {
                    "total_listening_time_ms": random.randint(0, 100000000),  # Up to ~27 hours
                    "tracks_played": random.randint(0, 10000),
                    "playlists_created": random.randint(0, 100),
                    "artists_followed": random.randint(0, 500),
                    "last_active": self.fake.date_time_between(start_date="-30d").isoformat()
                },
                "device_info": {
                    "devices": random.sample([
                        "smartphone", "desktop", "tablet", "smart_speaker", "car", "tv"
                    ], random.randint(1, 3)),
                    "primary_device": random.choice(["smartphone", "desktop", "tablet"])
                },
                "privacy_settings": {
                    "public_playlists": random.random() < 0.6,
                    "show_recently_played": random.random() < 0.8,
                    "allow_recommendations": random.random() < 0.9
                },
                "created_at": self.fake.date_time_between(start_date="-5y").isoformat(),
                "updated_at": datetime.now(timezone.utc).isoformat()
            }
            
            # GDPR compliance: anonymize PII if required
            if profile.pii_anonymization:
                user["email"] = f"user_{i+1}@{profile.tenant_id}.anonymized"
                user["display_name"] = f"User {i+1}"
                user["birth_date"] = None
                
            users.append(user)
            
        return users
        
    async def _generate_playlists(
        self,
        profile: TenantDataProfile,
        users: List[Dict[str, Any]],
        tracks: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Generate playlist data for tenant."""
        playlists = []
        
        for i in range(profile.num_playlists):
            playlist_id = f"{profile.tenant_id}_playlist_{i+1:06d}"
            owner = random.choice(users)
            
            # Playlist characteristics
            is_public = random.random() < 0.6
            is_collaborative = random.random() < 0.2
            
            # Select tracks for playlist
            playlist_size = random.randint(10, min(200, len(tracks)))
            playlist_tracks = random.sample(tracks, playlist_size)
            
            playlist = {
                "playlist_id": playlist_id,
                "tenant_id": profile.tenant_id,
                "name": self.fake.catch_phrase(),
                "description": self.fake.text(max_nb_chars=200) if random.random() < 0.7 else None,
                "owner_id": owner["user_id"],
                "owner_name": owner["display_name"],
                "is_public": is_public,
                "is_collaborative": is_collaborative,
                "total_tracks": len(playlist_tracks),
                "duration_ms": sum(track["duration_ms"] for track in playlist_tracks),
                "followers": random.randint(0, 10000) if is_public else 0,
                "tracks": [
                    {
                        "track_id": track["track_id"],
                        "added_at": self.fake.date_time_between(start_date="-1y").isoformat(),
                        "added_by": owner["user_id"] if not is_collaborative else random.choice(users)["user_id"],
                        "position": idx
                    }
                    for idx, track in enumerate(playlist_tracks)
                ],
                "images": [
                    {
                        "url": f"https://mosaic.scdn.co/640/{playlist_id}",
                        "height": 640,
                        "width": 640
                    }
                ],
                "genres": list(set(genre for track in playlist_tracks for genre in track.get("genres", []))),
                "mood_distribution": {
                    mood: len([t for t in playlist_tracks if t.get("mood") == mood]) / len(playlist_tracks)
                    for mood in self.moods
                },
                "created_at": self.fake.date_time_between(start_date="-2y").isoformat(),
                "updated_at": self.fake.date_time_between(start_date="-30d").isoformat()
            }
            
            playlists.append(playlist)
            
        return playlists
        
    async def _generate_listening_history(
        self,
        profile: TenantDataProfile,
        users: List[Dict[str, Any]],
        tracks: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Generate listening history data for tenant."""
        listening_events = []
        
        # Generate events for each user
        events_per_user = profile.num_listening_events // len(users)
        
        for user in users:
            user_activity = profile.user_activity_level * random.uniform(0.5, 1.5)
            user_events = int(events_per_user * user_activity)
            
            # User's preferred tracks (based on some randomness)
            user_track_preferences = random.sample(tracks, min(len(tracks), 100))
            
            for _ in range(user_events):
                event_id = f"{profile.tenant_id}_event_{len(listening_events)+1:09d}"
                
                # 70% from preferred tracks, 30% discovery
                if random.random() < 0.7 and user_track_preferences:
                    track = random.choice(user_track_preferences)
                else:
                    track = random.choice(tracks)
                    
                # Listening session details
                played_at = self.fake.date_time_between(start_date="-30d")
                duration_played = random.randint(
                    min(5000, track["duration_ms"] // 4),  # At least 5 seconds or 25% of track
                    track["duration_ms"]
                )
                
                # Skip reasons
                skip_reasons = [
                    None, "manual_skip", "track_done", "track_error", 
                    "forward_btn", "backward_btn", "end_play"
                ]
                
                event = {
                    "event_id": event_id,
                    "tenant_id": profile.tenant_id,
                    "user_id": user["user_id"],
                    "track_id": track["track_id"],
                    "track_name": track["name"],
                    "artist_id": track["artist_id"],
                    "artist_name": track["artist_name"],
                    "album_id": track.get("album_id"),
                    "played_at": played_at.isoformat(),
                    "duration_ms": duration_played,
                    "track_duration_ms": track["duration_ms"],
                    "completion_percentage": (duration_played / track["duration_ms"]) * 100,
                    "context": {
                        "type": random.choice(["playlist", "album", "artist", "search", "radio"]),
                        "uri": f"spotify:{random.choice(['playlist', 'album', 'artist'])}:{uuid.uuid4()}"
                    },
                    "device": {
                        "type": random.choice(user["device_info"]["devices"]),
                        "is_private_session": random.random() < 0.1,
                        "volume_percent": random.randint(20, 100)
                    },
                    "skip_reason": random.choice(skip_reasons),
                    "is_shuffle": random.random() < 0.3,
                    "is_repeat": random.choice([None, "track", "context"]),
                    "location": {
                        "country": user["country"],
                        "city": self.fake.city(),
                        "timezone": random.choice([
                            "UTC", "EST", "PST", "CET", "JST", "AEST"
                        ])
                    },
                    "created_at": played_at.isoformat()
                }
                
                listening_events.append(event)
                
        # Sort by played_at
        listening_events.sort(key=lambda x: x["played_at"])
        
        return listening_events
        
    async def _generate_recommendations(
        self,
        profile: TenantDataProfile,
        users: List[Dict[str, Any]],
        tracks: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Generate recommendation data for tenant."""
        recommendations = []
        
        recommendation_types = [
            "discover_weekly", "daily_mix", "release_radar", "made_for_you",
            "because_you_listened", "similar_artists", "trending"
        ]
        
        for user in users:
            # Generate recommendations for each user
            num_recs = random.randint(5, 20)
            
            for _ in range(num_recs):
                rec_id = f"{profile.tenant_id}_rec_{len(recommendations)+1:09d}"
                
                # Select recommended tracks
                rec_tracks = random.sample(tracks, random.randint(10, 50))
                
                recommendation = {
                    "recommendation_id": rec_id,
                    "tenant_id": profile.tenant_id,
                    "user_id": user["user_id"],
                    "type": random.choice(recommendation_types),
                    "title": self.fake.catch_phrase(),
                    "description": self.fake.text(max_nb_chars=150),
                    "tracks": [
                        {
                            "track_id": track["track_id"],
                            "reason": random.choice([
                                "Similar to your taste", "Because you liked", 
                                "Trending in your area", "New release",
                                "Based on your recent activity"
                            ]),
                            "confidence_score": random.uniform(0.5, 1.0),
                            "position": idx
                        }
                        for idx, track in enumerate(rec_tracks)
                    ],
                    "generated_at": self.fake.date_time_between(start_date="-7d").isoformat(),
                    "expires_at": (datetime.now() + timedelta(days=7)).isoformat(),
                    "model_version": f"v{random.randint(1, 5)}.{random.randint(0, 9)}",
                    "engagement_metrics": {
                        "views": random.randint(0, 10),
                        "clicks": random.randint(0, 5),
                        "plays": random.randint(0, 20),
                        "saves": random.randint(0, 3)
                    },
                    "created_at": self.fake.date_time_between(start_date="-7d").isoformat()
                }
                
                recommendations.append(recommendation)
                
        return recommendations
        
    async def _generate_analytics_events(
        self,
        profile: TenantDataProfile,
        users: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Generate analytics events for tenant."""
        events = []
        
        event_types = [
            "app_open", "app_close", "search", "playlist_create", "playlist_edit",
            "follow_artist", "unfollow_artist", "like_track", "unlike_track",
            "share_track", "share_playlist", "settings_change", "subscription_change"
        ]
        
        # Generate events for the last 30 days
        start_date = datetime.now() - timedelta(days=30)
        
        for user in users:
            # User activity level affects number of events
            num_events = int(random.randint(50, 500) * profile.user_activity_level)
            
            for _ in range(num_events):
                event_id = f"{profile.tenant_id}_analytics_{len(events)+1:09d}"
                event_type = random.choice(event_types)
                
                event_time = self.fake.date_time_between(
                    start_date=start_date,
                    end_date=datetime.now()
                )
                
                event = {
                    "event_id": event_id,
                    "tenant_id": profile.tenant_id,
                    "user_id": user["user_id"],
                    "event_type": event_type,
                    "timestamp": event_time.isoformat(),
                    "session_id": str(uuid.uuid4()),
                    "properties": self._generate_event_properties(event_type, user),
                    "device": {
                        "type": random.choice(user["device_info"]["devices"]),
                        "os": random.choice(["iOS", "Android", "Windows", "macOS", "Linux"]),
                        "app_version": f"{random.randint(1, 5)}.{random.randint(0, 20)}.{random.randint(0, 10)}"
                    },
                    "location": {
                        "country": user["country"],
                        "city": self.fake.city(),
                        "ip_address": self.fake.ipv4() if not profile.pii_anonymization else "0.0.0.0"
                    },
                    "created_at": event_time.isoformat()
                }
                
                events.append(event)
                
        # Sort by timestamp
        events.sort(key=lambda x: x["timestamp"])
        
        return events
        
    def _generate_event_properties(self, event_type: str, user: Dict[str, Any]) -> Dict[str, Any]:
        """Generate event-specific properties."""
        base_properties = {
            "user_subscription": user["subscription_type"],
            "user_country": user["country"]
        }
        
        if event_type == "search":
            return {
                **base_properties,
                "query": self.fake.word(),
                "results_count": random.randint(0, 100),
                "clicked_result": random.random() < 0.3
            }
        elif event_type == "playlist_create":
            return {
                **base_properties,
                "playlist_name": self.fake.catch_phrase(),
                "initial_tracks": random.randint(0, 20),
                "is_public": random.random() < 0.6
            }
        elif event_type in ["like_track", "unlike_track"]:
            return {
                **base_properties,
                "track_genre": random.choice(self.genres),
                "track_popularity": random.randint(0, 100)
            }
        else:
            return base_properties
            
    async def _generate_ml_features(
        self,
        profile: TenantDataProfile,
        users: List[Dict[str, Any]],
        tracks: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Generate ML features for training and inference."""
        
        # User features
        user_features = []
        for user in users:
            features = {
                "user_id": user["user_id"],
                "tenant_id": profile.tenant_id,
                "age_group": self._calculate_age_group(user.get("birth_date")),
                "subscription_tenure_days": (
                    datetime.now().date() - 
                    datetime.fromisoformat(user["subscription_start_date"]).date()
                ).days,
                "activity_score": random.uniform(0.0, 1.0),
                "diversity_score": random.uniform(0.0, 1.0),
                "discovery_score": random.uniform(0.0, 1.0),
                "preferred_genres_vector": [
                    1.0 if genre in user["preferences"]["preferred_genres"] else 0.0
                    for genre in self.genres
                ],
                "listening_time_per_day": random.uniform(0.0, 8.0),  # hours
                "device_affinity": {
                    device: random.uniform(0.0, 1.0)
                    for device in ["smartphone", "desktop", "tablet", "smart_speaker"]
                },
                "time_of_day_preference": {
                    "morning": random.uniform(0.0, 1.0),
                    "afternoon": random.uniform(0.0, 1.0),
                    "evening": random.uniform(0.0, 1.0),
                    "night": random.uniform(0.0, 1.0)
                },
                "social_engagement": random.uniform(0.0, 1.0),
                "feature_updated_at": datetime.now(timezone.utc).isoformat()
            }
            user_features.append(features)
            
        # Track features (already partially in track data, enhance here)
        track_features = []
        for track in tracks:
            features = {
                "track_id": track["track_id"],
                "tenant_id": profile.tenant_id,
                "audio_features": track["audio_features"],
                "genre_vector": [
                    1.0 if genre in track.get("genres", []) else 0.0
                    for genre in self.genres
                ],
                "popularity_score": track["popularity"] / 100.0,
                "release_recency": self._calculate_recency_score(track["created_at"]),
                "artist_popularity": random.uniform(0.0, 1.0),
                "mood_vector": [
                    1.0 if mood == track.get("mood") else 0.0
                    for mood in self.moods
                ],
                "feature_updated_at": datetime.now(timezone.utc).isoformat()
            }
            track_features.append(features)
            
        # Interaction features (user-track interactions)
        interaction_features = []
        for _ in range(min(10000, len(users) * 100)):  # Sample interactions
            user = random.choice(users)
            track = random.choice(tracks)
            
            features = {
                "interaction_id": f"{profile.tenant_id}_interaction_{len(interaction_features)+1:09d}",
                "tenant_id": profile.tenant_id,
                "user_id": user["user_id"],
                "track_id": track["track_id"],
                "interaction_type": random.choice(["play", "skip", "like", "save", "share"]),
                "completion_rate": random.uniform(0.0, 1.0),
                "context": random.choice(["playlist", "album", "search", "recommendation"]),
                "time_of_day": random.randint(0, 23),
                "day_of_week": random.randint(0, 6),
                "device_type": random.choice(user["device_info"]["devices"]),
                "feature_timestamp": self.fake.date_time_between(start_date="-30d").isoformat()
            }
            interaction_features.append(features)
            
        return {
            "user_features": user_features,
            "track_features": track_features,
            "interaction_features": interaction_features,
            "feature_metadata": {
                "version": "1.0",
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "total_users": len(users),
                "total_tracks": len(tracks),
                "total_interactions": len(interaction_features)
            }
        }
        
    def _calculate_age_group(self, birth_date_str: Optional[str]) -> str:
        """Calculate age group from birth date."""
        if not birth_date_str:
            return "unknown"
            
        birth_date = datetime.fromisoformat(birth_date_str).date()
        age = (datetime.now().date() - birth_date).days // 365
        
        if age < 18:
            return "under_18"
        elif age < 25:
            return "18_24"
        elif age < 35:
            return "25_34"
        elif age < 45:
            return "35_44"
        elif age < 55:
            return "45_54"
        elif age < 65:
            return "55_64"
        else:
            return "65_plus"
            
    def _calculate_recency_score(self, created_at_str: str) -> float:
        """Calculate recency score (1.0 = very recent, 0.0 = very old)."""
        created_at = datetime.fromisoformat(created_at_str.replace("Z", "+00:00"))
        days_old = (datetime.now(timezone.utc) - created_at).days
        
        # Score decreases exponentially with age
        max_days = 365 * 5  # 5 years
        return max(0.0, 1.0 - (days_old / max_days))
        
    async def save_tenant_data(
        self,
        tenant_data: Dict[str, Any],
        format: str = "json"
    ) -> None:
        """
        Save generated tenant data to files.
        
        Args:
            tenant_data: Complete tenant dataset
            format: Output format (json, csv, yaml)
        """
        tenant_id = tenant_data["tenant_info"]["tenant_id"]
        tenant_dir = self.output_path / tenant_id
        tenant_dir.mkdir(exist_ok=True)
        
        # Save metadata
        metadata_file = tenant_dir / "metadata.json"
        with open(metadata_file, "w") as f:
            json.dump(tenant_data["tenant_info"], f, indent=2)
            
        # Save each data type
        for data_type, data in tenant_data["data"].items():
            if format == "json":
                file_path = tenant_dir / f"{data_type}.json"
                with open(file_path, "w") as f:
                    json.dump(data, f, indent=2)
            elif format == "csv" and isinstance(data, list):
                file_path = tenant_dir / f"{data_type}.csv"
                if data:  # Only create CSV if there's data
                    with open(file_path, "w", newline="") as f:
                        writer = csv.DictWriter(f, fieldnames=data[0].keys())
                        writer.writeheader()
                        writer.writerows(data)
            elif format == "yaml":
                file_path = tenant_dir / f"{data_type}.yaml"
                with open(file_path, "w") as f:
                    yaml.dump(data, f, default_flow_style=False)
                    
        print(f"Tenant data saved to: {tenant_dir}")


# Utility functions for common use cases
async def generate_test_tenant_data(
    tenant_id: str,
    tier: TenantTier,
    scale_factor: float = 0.1
) -> Dict[str, Any]:
    """Quick utility to generate test data for a tenant."""
    generator = MultiTenantDataGenerator()
    
    profile = generator.create_tenant_profile(
        tenant_id=tenant_id,
        tenant_name=f"Test Tenant {tenant_id}",
        tier=tier,
        scale_factor=scale_factor
    )
    
    return await generator.generate_tenant_data(profile)


async def generate_load_test_data(
    tenant_id: str,
    num_concurrent_users: int = 100
) -> Dict[str, Any]:
    """Generate data specifically for load testing."""
    generator = MultiTenantDataGenerator()
    
    profile = generator.create_tenant_profile(
        tenant_id=tenant_id,
        tenant_name=f"Load Test Tenant",
        tier=TenantTier.ENTERPRISE,
        scale_factor=1.0,
        generate_load_data=True,
        concurrent_users=num_concurrent_users
    )
    
    return await generator.generate_tenant_data(profile)


async def example_usage():
    """Example usage of the multi-tenant data generator."""
    generator = MultiTenantDataGenerator()
    
    # Generate data for different tenant tiers
    tenants = [
        ("free_demo", TenantTier.FREE, 0.5),
        ("standard_company", TenantTier.STANDARD, 0.3),
        ("premium_corp", TenantTier.PREMIUM, 0.2),
        ("enterprise_global", TenantTier.ENTERPRISE, 0.1)
    ]
    
    for tenant_id, tier, scale in tenants:
        print(f"Generating data for {tenant_id} ({tier.value})...")
        
        profile = generator.create_tenant_profile(
            tenant_id=tenant_id,
            tenant_name=f"{tier.value.title()} Tenant",
            tier=tier,
            scale_factor=scale
        )
        
        tenant_data = await generator.generate_tenant_data(profile)
        await generator.save_tenant_data(tenant_data, format="json")
        
        print(f"✓ Generated {len(tenant_data['data'])} data types for {tenant_id}")
        
    print("Data generation complete!")


if __name__ == "__main__":
    # Run example
    asyncio.run(example_usage())
