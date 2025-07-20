"""
Template System Integration Test Suite

Umfassende Integrationstests für das Enterprise Template System,
die alle Komponenten zusammen testen und realistische Szenarien abdecken.
"""

import asyncio
import pytest
import json
import tempfile
import shutil
from pathlib import Path
from typing import Dict, Any, List
from unittest.mock import Mock, AsyncMock, patch

from backend.app.tenancy.fixtures.templates import (
    TemplateEngine,
    TemplateManager,
    TemplateValidationEngine,
    TemplateProcessingPipeline,
    TenantTemplateGenerator,
    UserTemplateGenerator,
    ContentTemplateGenerator,
    MigrationManager
)


class TestTemplateSystemIntegration:
    """Integrationstests für das Template System."""

    @pytest.fixture
    async def setup_system(self):
        """Template System für Tests einrichten."""
        # Temporäres Verzeichnis für Tests
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)
        
        # Template System Komponenten initialisieren
        self.engine = TemplateEngine(enable_monitoring=True)
        self.manager = TemplateManager(self.engine)
        self.validator = TemplateValidationEngine()
        self.processor = TemplateProcessingPipeline()
        self.migration_manager = MigrationManager()
        
        # Generatoren initialisieren
        self.tenant_generator = TenantTemplateGenerator()
        self.user_generator = UserTemplateGenerator()
        self.content_generator = ContentTemplateGenerator()
        
        # Test-Templates erstellen
        await self._create_test_templates()
        
        yield
        
        # Cleanup
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    async def _create_test_templates(self):
        """Test-Templates für Integration erstellen."""
        # Basis Tenant Template
        self.tenant_template = {
            "_metadata": {
                "template_type": "tenant_init",
                "template_version": "1.0.0",
                "schema_version": "2024.1"
            },
            "tenant_id": "{{ tenant_id }}",
            "tenant_name": "{{ tenant_name }}",
            "tier": "{{ tier | default('professional') }}",
            "configuration": {
                "limits": {
                    "max_users": "{{ max_users | default(100) }}",
                    "storage_gb": "{{ storage_gb | default(50) }}"
                },
                "features": {
                    "enabled": ["{{ feature_1 }}", "{{ feature_2 }}"]
                }
            }
        }
        
        # Basis User Template
        self.user_template = {
            "_metadata": {
                "template_type": "user_profile",
                "template_version": "2.0.0",
                "schema_version": "2024.1"
            },
            "user_id": "{{ user_id }}",
            "tenant_id": "{{ tenant_id }}",
            "profile": {
                "username": "{{ username }}",
                "email": "{{ email | validate_email }}",
                "preferences": {
                    "theme": "{{ theme | default('dark') }}"
                }
            },
            "music_profile": {
                "spotify_connected": "{{ spotify_connected | default(false) }}",
                "favorite_genres": ["{{ genre_1 }}", "{{ genre_2 }}"]
            }
        }
        
        # Template-Dateien speichern
        tenant_file = self.temp_path / "tenant_template.json"
        user_file = self.temp_path / "user_template.json"
        
        with open(tenant_file, "w") as f:
            json.dump(self.tenant_template, f, indent=2)
        
        with open(user_file, "w") as f:
            json.dump(self.user_template, f, indent=2)
        
        self.tenant_template_path = str(tenant_file)
        self.user_template_path = str(user_file)

    @pytest.mark.asyncio
    async def test_complete_tenant_workflow(self, setup_system):
        """Testet kompletten Tenant-Erstellungsworkflow."""
        print("🏢 Testing Complete Tenant Workflow...")
        
        # 1. Template generieren
        generated_template = await self.tenant_generator.generate_tenant_template(
            tenant_id="integration_test_tenant",
            tier="enterprise",
            features=["advanced_ai", "full_collaboration", "analytics"],
            custom_config={
                "max_users": 500,
                "storage_gb": 100,
                "ai_sessions_per_month": 5000
            }
        )
        
        assert generated_template is not None
        assert generated_template["tenant_id"] == "integration_test_tenant"
        assert generated_template["tier"] == "enterprise"
        assert generated_template["configuration"]["limits"]["max_users"] == 500
        
        # 2. Template validieren
        validation_result = await self.validator.validate_template(
            template_data=generated_template,
            template_type="tenant_init",
            schema_version="2024.1"
        )
        
        assert validation_result.is_valid
        assert len(validation_result.errors) == 0
        
        # 3. Template verarbeiten
        self.processor.add_stage("compression")
        self.processor.add_stage("security_scan")
        self.processor.add_stage("performance_optimization")
        
        processed_template = await self.processor.process_template(
            template_data=generated_template,
            template_type="tenant_init"
        )
        
        assert processed_template is not None
        assert "_processed_metadata" in processed_template
        
        # 4. Template speichern und laden
        template_path = self.temp_path / "processed_tenant.json"
        await self.manager.save_template(str(template_path), processed_template)
        
        loaded_template = await self.manager.load_template(str(template_path))
        assert loaded_template["tenant_id"] == "integration_test_tenant"
        
        print("✅ Tenant Workflow erfolgreich abgeschlossen")

    @pytest.mark.asyncio
    async def test_complete_user_workflow(self, setup_system):
        """Testet kompletten User-Erstellungsworkflow."""
        print("👤 Testing Complete User Workflow...")
        
        # 1. User Template generieren
        user_template = await self.user_generator.generate_user_template(
            user_type="music_enthusiast",
            subscription_tier="premium",
            music_preferences={
                "genres": ["jazz", "electronic", "classical"],
                "discovery_mode": "adventurous",
                "ai_recommendations": True
            },
            ai_settings={
                "personality": "enthusiastic",
                "response_length": "detailed",
                "include_examples": True
            }
        )
        
        assert user_template is not None
        assert user_template["music_profile"]["preferences"]["favorite_genres"] == ["jazz", "electronic", "classical"]
        assert user_template["ai_settings"]["conversation_preferences"]["personality_type"] == "enthusiastic"
        
        # 2. Template mit Benutzerdaten rendern
        context = {
            "user_id": "test_user_001",
            "tenant_id": "integration_test_tenant",
            "username": "jazz_lover_42",
            "email": "test@example.com",
            "spotify_connected": True,
            "subscription_plan": "premium"
        }
        
        rendered_user = await self.engine.render_template(user_template, context)
        
        assert rendered_user["user_id"] == "test_user_001"
        assert rendered_user["profile"]["username"] == "jazz_lover_42"
        assert rendered_user["profile"]["email"] == "test@example.com"
        
        # 3. Validierung
        validation_result = await self.validator.validate_template(
            template_data=rendered_user,
            template_type="user_profile",
            schema_version="2024.1"
        )
        
        assert validation_result.is_valid
        
        # 4. Caching testen
        cache_key = f"user_profile_{context['user_id']}"
        await self.engine.cache.set(cache_key, rendered_user)
        
        cached_user = await self.engine.cache.get(cache_key)
        assert cached_user["user_id"] == "test_user_001"
        
        print("✅ User Workflow erfolgreich abgeschlossen")

    @pytest.mark.asyncio
    async def test_playlist_generation_and_processing(self, setup_system):
        """Testet AI-Playlist-Generierung und -Verarbeitung."""
        print("🎵 Testing Playlist Generation and Processing...")
        
        # 1. Playlist Template generieren
        playlist_template = await self.content_generator.generate_playlist_template(
            playlist_type="ai_generated",
            mood="energetic",
            activity="workout",
            genre_preferences=["electronic", "rock", "pop"],
            duration_minutes=45,
            energy_level=0.9
        )
        
        assert playlist_template is not None
        assert playlist_template["basic_info"]["mood"] == "energetic"
        assert playlist_template["basic_info"]["activity"] == "workout"
        
        # 2. Mit spezifischen Daten rendern
        context = {
            "playlist_id": "pl_workout_001",
            "creator_user_id": "test_user_001",
            "tenant_id": "integration_test_tenant",
            "playlist_name": "Ultimate Workout Mix",
            "target_mood": "energetic",
            "energy_level": 0.9,
            "danceability": 0.8,
            "tempo": 140,
            "seed_track_1_id": "4iV5W9uYEdYUVa79Axb7Rh",
            "seed_track_1_name": "Stronger",
            "seed_track_1_artist": "Kanye West"
        }
        
        rendered_playlist = await self.engine.render_template(playlist_template, context)
        
        assert rendered_playlist["playlist_id"] == "pl_workout_001"
        assert rendered_playlist["basic_info"]["name"] == "Ultimate Workout Mix"
        assert rendered_playlist["ai_generation"]["generation_parameters"]["energy"] == 0.9
        
        # 3. AI Enhancement Processing
        self.processor.add_stage("ai_enhancement")
        
        with patch('backend.app.tenancy.fixtures.templates.processors.ai_client') as mock_ai:
            mock_ai.generate_content = AsyncMock(return_value={
                "enhanced_description": "AI-enhanced energetic workout playlist",
                "suggested_tags": ["high-energy", "workout", "motivation"]
            })
            
            enhanced_playlist = await self.processor.process_template(
                template_data=rendered_playlist,
                template_type="content_playlist"
            )
            
            assert enhanced_playlist is not None
            # Verify AI enhancement was applied
            assert "_processed_metadata" in enhanced_playlist
        
        print("✅ Playlist Generation erfolgreich abgeschlossen")

    @pytest.mark.asyncio
    async def test_template_migration_workflow(self, setup_system):
        """Testet Template-Migration zwischen Versionen."""
        print("🔄 Testing Template Migration Workflow...")
        
        # 1. Altes Template (v1.0.0) erstellen
        old_template = {
            "_metadata": {
                "template_type": "user_profile",
                "template_version": "1.0.0",
                "schema_version": "2023.1"
            },
            "user_id": "migration_test_user",
            "basic_info": {
                "username": "old_user",
                "email": "old@example.com"
            },
            "preferences": {
                "theme": "light"
            }
        }
        
        # 2. Migration auf neue Version (v2.0.0)
        migrated_template = await self.migration_manager.migrate_template(
            template_data=old_template,
            from_version="1.0.0",
            to_version="2.0.0",
            template_type="user_profile"
        )
        
        assert migrated_template is not None
        assert migrated_template["_metadata"]["template_version"] == "2.0.0"
        assert migrated_template["_metadata"]["schema_version"] == "2024.1"
        
        # Verify migration transformed structure correctly
        assert "profile" in migrated_template  # New structure
        assert migrated_template["profile"]["username"] == "old_user"
        assert migrated_template["profile"]["email"] == "old@example.com"
        
        # 3. Validierung der migrierten Daten
        validation_result = await self.validator.validate_template(
            template_data=migrated_template,
            template_type="user_profile",
            schema_version="2024.1"
        )
        
        assert validation_result.is_valid
        
        print("✅ Template Migration erfolgreich abgeschlossen")

    @pytest.mark.asyncio
    async def test_multi_tenant_isolation(self, setup_system):
        """Testet Multi-Tenant-Isolation im Template System."""
        print("🏢 Testing Multi-Tenant Isolation...")
        
        # 1. Templates für verschiedene Tenants erstellen
        tenant_a_template = await self.tenant_generator.generate_tenant_template(
            tenant_id="tenant_a",
            tier="professional",
            features=["basic_ai", "collaboration"]
        )
        
        tenant_b_template = await self.tenant_generator.generate_tenant_template(
            tenant_id="tenant_b",
            tier="enterprise",
            features=["advanced_ai", "full_collaboration", "analytics"]
        )
        
        # 2. Tenant-spezifische User Templates
        user_a_template = await self.user_generator.generate_user_template(
            user_type="casual_listener",
            subscription_tier="basic"
        )
        
        user_b_template = await self.user_generator.generate_user_template(
            user_type="music_professional",
            subscription_tier="enterprise"
        )
        
        # 3. Rendering mit Tenant-spezifischen Kontexten
        context_a = {
            "user_id": "user_a_001",
            "tenant_id": "tenant_a",
            "username": "casual_user",
            "subscription_plan": "basic"
        }
        
        context_b = {
            "user_id": "user_b_001",
            "tenant_id": "tenant_b",
            "username": "pro_user",
            "subscription_plan": "enterprise"
        }
        
        rendered_user_a = await self.engine.render_template(user_a_template, context_a)
        rendered_user_b = await self.engine.render_template(user_b_template, context_b)
        
        # 4. Isolation verifizieren
        assert rendered_user_a["tenant_id"] == "tenant_a"
        assert rendered_user_b["tenant_id"] == "tenant_b"
        assert rendered_user_a["subscription"]["plan"] == "basic"
        assert rendered_user_b["subscription"]["plan"] == "enterprise"
        
        # 5. Cache-Isolation testen
        await self.engine.cache.set(f"tenant_a:user_{context_a['user_id']}", rendered_user_a)
        await self.engine.cache.set(f"tenant_b:user_{context_b['user_id']}", rendered_user_b)
        
        cached_a = await self.engine.cache.get(f"tenant_a:user_{context_a['user_id']}")
        cached_b = await self.engine.cache.get(f"tenant_b:user_{context_b['user_id']}")
        
        assert cached_a["tenant_id"] != cached_b["tenant_id"]
        
        print("✅ Multi-Tenant Isolation erfolgreich verifiziert")

    @pytest.mark.asyncio
    async def test_performance_under_load(self, setup_system):
        """Testet Performance unter Last."""
        print("⚡ Testing Performance Under Load...")
        
        # 1. Concurrent Template Rendering
        async def render_task(task_id: int):
            context = {
                "user_id": f"load_test_user_{task_id}",
                "tenant_id": "load_test_tenant",
                "username": f"user_{task_id}",
                "email": f"user_{task_id}@example.com"
            }
            
            template = await self.manager.load_template(self.user_template_path)
            return await self.engine.render_template(template, context)
        
        # 50 gleichzeitige Rendering-Operationen
        tasks = [render_task(i) for i in range(50)]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Erfolgsrate prüfen
        successful_results = [r for r in results if not isinstance(r, Exception)]
        success_rate = len(successful_results) / len(results)
        
        assert success_rate >= 0.95  # Mind. 95% Erfolgsrate
        
        # 2. Cache Performance unter Last
        cache_tasks = []
        for i in range(100):
            cache_tasks.append(
                self.engine.cache.set(f"load_test_key_{i}", {"data": f"test_data_{i}"})
            )
        
        await asyncio.gather(*cache_tasks)
        
        # Cache Hit Rate testen
        hit_count = 0
        for i in range(100):
            cached_data = await self.engine.cache.get(f"load_test_key_{i}")
            if cached_data is not None:
                hit_count += 1
        
        cache_hit_rate = hit_count / 100
        assert cache_hit_rate >= 0.9  # Mind. 90% Cache Hit Rate
        
        print(f"✅ Performance Test: {success_rate*100:.1f}% Success Rate, {cache_hit_rate*100:.1f}% Cache Hit Rate")

    @pytest.mark.asyncio
    async def test_error_handling_and_recovery(self, setup_system):
        """Testet Fehlerbehandlung und Recovery."""
        print("🛡️ Testing Error Handling and Recovery...")
        
        # 1. Ungültiges Template testen
        invalid_template = {
            "_metadata": {
                "template_type": "invalid_type",
                "template_version": "999.0.0"
            },
            "invalid_field": "{{ invalid_function() }}"
        }
        
        # Fehler beim Rendering erwarten
        with pytest.raises(Exception):
            await self.engine.render_template(invalid_template, {})
        
        # 2. Validierungsfehler testen
        invalid_user_template = {
            "_metadata": {
                "template_type": "user_profile",
                "template_version": "2.0.0"
            },
            "user_id": "",  # Ungültig - leer
            "profile": {
                "email": "invalid-email"  # Ungültiges Format
            }
        }
        
        validation_result = await self.validator.validate_template(
            template_data=invalid_user_template,
            template_type="user_profile",
            schema_version="2024.1"
        )
        
        assert not validation_result.is_valid
        assert len(validation_result.errors) > 0
        
        # 3. Cache Fehler und Recovery
        # Simuliere Cache-Fehler
        with patch.object(self.engine.cache, 'get', side_effect=Exception("Cache error")):
            # System sollte gracefully fallback
            template = await self.manager.load_template(self.user_template_path)
            assert template is not None  # Sollte trotz Cache-Fehler funktionieren
        
        # 4. Recovery nach Fehlern testen
        # Nach Fehlern sollte System normal weiterarbeiten
        valid_context = {
            "user_id": "recovery_test_user",
            "tenant_id": "recovery_tenant",
            "username": "recovery_user",
            "email": "recovery@example.com"
        }
        
        template = await self.manager.load_template(self.user_template_path)
        result = await self.engine.render_template(template, valid_context)
        
        assert result["user_id"] == "recovery_test_user"
        
        print("✅ Error Handling and Recovery erfolgreich getestet")

    @pytest.mark.asyncio
    async def test_backup_and_restore_workflow(self, setup_system):
        """Testet Backup und Restore Workflow."""
        print("💾 Testing Backup and Restore Workflow...")
        
        # 1. Template-Daten für Backup vorbereiten
        test_template = {
            "_metadata": {
                "template_type": "user_profile",
                "template_version": "2.0.0",
                "created_at": "2024-01-01T00:00:00Z"
            },
            "user_id": "backup_test_user",
            "profile": {
                "username": "backup_user",
                "email": "backup@example.com"
            }
        }
        
        # 2. Template speichern
        template_path = self.temp_path / "backup_test_template.json"
        await self.manager.save_template(str(template_path), test_template)
        
        # 3. Backup erstellen
        backup_data = await self.manager.create_backup(
            template_path=str(template_path),
            include_metadata=True
        )
        
        assert backup_data is not None
        assert "template_data" in backup_data
        assert "metadata" in backup_data
        assert backup_data["template_data"]["user_id"] == "backup_test_user"
        
        # 4. Template löschen (simuliert Datenverlust)
        template_path.unlink()
        assert not template_path.exists()
        
        # 5. Aus Backup wiederherstellen
        restore_path = self.temp_path / "restored_template.json"
        restored_template = await self.manager.restore_from_backup(
            backup_data=backup_data,
            target_path=str(restore_path)
        )
        
        assert restored_template is not None
        assert restored_template["user_id"] == "backup_test_user"
        assert restore_path.exists()
        
        # 6. Wiederhergestelltes Template validieren
        loaded_restored = await self.manager.load_template(str(restore_path))
        assert loaded_restored["profile"]["username"] == "backup_user"
        
        print("✅ Backup and Restore Workflow erfolgreich abgeschlossen")


# Test Runner
async def run_integration_tests():
    """Führt alle Integrationstests aus."""
    print("=" * 80)
    print("🧪 TEMPLATE SYSTEM INTEGRATION TESTS")
    print("=" * 80)
    
    test_suite = TestTemplateSystemIntegration()
    
    # Setup
    print("🔧 Setting up test environment...")
    await test_suite.setup_system()
    
    try:
        # Alle Tests ausführen
        test_methods = [
            test_suite.test_complete_tenant_workflow,
            test_suite.test_complete_user_workflow,
            test_suite.test_playlist_generation_and_processing,
            test_suite.test_template_migration_workflow,
            test_suite.test_multi_tenant_isolation,
            test_suite.test_performance_under_load,
            test_suite.test_error_handling_and_recovery,
            test_suite.test_backup_and_restore_workflow
        ]
        
        passed_tests = 0
        total_tests = len(test_methods)
        
        for test_method in test_methods:
            try:
                await test_method()
                passed_tests += 1
            except Exception as e:
                print(f"❌ Test failed: {test_method.__name__} - {e}")
        
        print("\n" + "=" * 80)
        print("📊 INTEGRATION TEST RESULTS")
        print("=" * 80)
        print(f"Tests passed: {passed_tests}/{total_tests}")
        print(f"Success rate: {(passed_tests/total_tests)*100:.1f}%")
        
        if passed_tests == total_tests:
            print("🎉 All integration tests passed!")
        else:
            print(f"⚠️  {total_tests - passed_tests} tests failed")
            
    finally:
        # Cleanup wird automatisch durch fixture gemacht
        print("🧹 Cleaning up test environment...")


if __name__ == "__main__":
    asyncio.run(run_integration_tests())
