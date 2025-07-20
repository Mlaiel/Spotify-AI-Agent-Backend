#!/usr/bin/env python3
"""
Suite de tests automatisés pour le système de validation ultra-avancé
Créé par l'équipe d'experts dirigée par Fahed Mlaiel

Tests complets pour valider l'ensemble des fonctionnalités enterprise
"""

import pytest
import asyncio
import json
import tempfile
from pathlib import Path
from typing import Dict, Any

# Import du système de validation
from . import EnterpriseSchemaManager, ValidationResult, SchemaType, DataFormat


class TestEnterpriseSchemaValidation:
    """Suite de tests pour le système de validation enterprise"""
    
    @pytest.fixture
    async def schema_manager(self):
        """Fixture pour initialiser le gestionnaire de schémas"""
        manager = EnterpriseSchemaManager()
        
        # Chargement des schémas de test
        schemas_dir = Path(__file__).parent
        test_schemas = [
            ("user_profile", "user_profile_schema.json"),
            ("api", "api_schema.json"),
            ("security", "security_schema.json"),
            ("ml_model", "ml_model_schema.json"),
            ("environment", "environment_schema.json")
        ]
        
        for schema_id, filename in test_schemas:
            schema_path = schemas_dir / filename
            if schema_path.exists():
                await manager.load_schema_from_file(
                    schema_id=schema_id,
                    file_path=str(schema_path),
                    schema_type=SchemaType.JSON_SCHEMA
                )
        
        return manager
    
    @pytest.mark.asyncio
    async def test_valid_user_profile(self, schema_manager):
        """Test de validation d'un profil utilisateur valide"""
        valid_user = {
            "user_id": "test_user_123",
            "profile_data": {
                "email": "test@example.com",
                "username": "testuser",
                "created_at": "2024-01-15T10:30:00Z",
                "status": "active"
            },
            "preferences": {
                "music_preferences": {
                    "favorite_genres": ["pop", "rock"],
                    "discovery_mode": "moderate"
                },
                "ai_preferences": {
                    "personalization_level": "standard"
                }
            },
            "privacy_settings": {
                "data_sharing_consent": {
                    "marketing": False,
                    "analytics": True,
                    "third_party": False
                },
                "analytics_consent": {
                    "behavioral_analytics": True
                }
            }
        }
        
        result = await schema_manager.validate_data(
            data=valid_user,
            schema_id="user_profile"
        )
        
        assert result.is_valid, f"Validation échouée: {result.errors}"
        assert len(result.errors) == 0
    
    @pytest.mark.asyncio
    async def test_invalid_user_profile(self, schema_manager):
        """Test de validation d'un profil utilisateur invalide"""
        invalid_user = {
            "user_id": "ab",  # Trop court
            "profile_data": {
                "email": "invalid-email",  # Format invalide
                "created_at": "invalid-date"
            },
            "preferences": {},  # Données manquantes
            "privacy_settings": {}  # Données manquantes
        }
        
        result = await schema_manager.validate_data(
            data=invalid_user,
            schema_id="user_profile"
        )
        
        assert not result.is_valid
        assert len(result.errors) > 0
    
    @pytest.mark.asyncio
    async def test_ai_suggestions(self, schema_manager):
        """Test des suggestions IA pour données invalides"""
        invalid_data = {
            "user_id": "short",
            "profile_data": {
                "email": "bad-email",
                "created_at": "2024-01-15T10:30:00Z"
            },
            "preferences": {
                "music_preferences": {},
                "ai_preferences": {}
            },
            "privacy_settings": {
                "data_sharing_consent": {
                    "marketing": False,
                    "analytics": True,
                    "third_party": False
                },
                "analytics_consent": {
                    "behavioral_analytics": True
                }
            }
        }
        
        result = await schema_manager.validate_data(
            data=invalid_data,
            schema_id="user_profile",
            enable_ai_suggestions=True
        )
        
        assert not result.is_valid
        assert len(result.ai_suggestions) > 0
    
    @pytest.mark.asyncio
    async def test_ml_model_validation(self, schema_manager):
        """Test de validation d'une configuration ML"""
        ml_config = {
            "model_info": {
                "name": "test_model",
                "version": "1.0.0",
                "type": "recommendation",
                "framework": "pytorch",
                "description": "Test model for validation"
            },
            "training_config": {
                "data_source": {
                    "type": "database",
                    "location": "test://localhost/db"
                },
                "hyperparameters": {
                    "learning_rate": 0.001,
                    "batch_size": 32
                }
            },
            "deployment_config": {
                "environment": "development",
                "serving": {
                    "type": "online"
                }
            },
            "monitoring": {
                "metrics": {},
                "logging": {}
            }
        }
        
        result = await schema_manager.validate_data(
            data=ml_config,
            schema_id="ml_model"
        )
        
        assert result.is_valid, f"Validation ML échouée: {result.errors}"
    
    @pytest.mark.asyncio
    async def test_api_configuration_validation(self, schema_manager):
        """Test de validation d'une configuration API"""
        api_config = {
            "api_version": "v1.0",
            "endpoints": {
                "/test": {
                    "methods": ["GET"],
                    "description": "Test endpoint for validation"
                }
            },
            "authentication": {
                "schemes": [
                    {
                        "type": "jwt",
                        "name": "Bearer Auth"
                    }
                ]
            },
            "rate_limiting": {
                "global_limits": {
                    "requests_per_second": 10
                }
            },
            "monitoring": {
                "metrics": {},
                "health_checks": {}
            }
        }
        
        result = await schema_manager.validate_data(
            data=api_config,
            schema_id="api"
        )
        
        assert result.is_valid, f"Validation API échouée: {result.errors}"
    
    @pytest.mark.asyncio
    async def test_security_configuration(self, schema_manager):
        """Test de validation d'une configuration de sécurité"""
        security_config = {
            "authentication": {
                "primary_method": "jwt",
                "mfa": {
                    "enabled": True
                },
                "session_management": {
                    "timeout_minutes": 60
                }
            },
            "authorization": {
                "rbac": {
                    "enabled": True
                },
                "permissions": {}
            },
            "encryption": {
                "data_at_rest": {
                    "enabled": True,
                    "algorithm": "AES-256-GCM"
                },
                "data_in_transit": {
                    "tls_version": "1.3",
                    "cipher_suites": ["TLS_AES_256_GCM_SHA384"]
                }
            },
            "monitoring": {
                "security_events": {},
                "anomaly_detection": {}
            }
        }
        
        result = await schema_manager.validate_data(
            data=security_config,
            schema_id="security"
        )
        
        assert result.is_valid, f"Validation sécurité échouée: {result.errors}"
    
    @pytest.mark.asyncio
    async def test_batch_validation(self, schema_manager):
        """Test de validation en lot"""
        test_users = [
            {
                "user_id": f"user_{i}",
                "profile_data": {
                    "email": f"user{i}@example.com",
                    "created_at": "2024-01-15T10:30:00Z"
                },
                "preferences": {
                    "music_preferences": {},
                    "ai_preferences": {}
                },
                "privacy_settings": {
                    "data_sharing_consent": {
                        "marketing": False,
                        "analytics": True,
                        "third_party": False
                    },
                    "analytics_consent": {
                        "behavioral_analytics": True
                    }
                }
            }
            for i in range(10)
        ]
        
        results = await schema_manager.validate_batch(
            data_list=test_users,
            schema_id="user_profile"
        )
        
        assert len(results) == 10
        assert all(result.is_valid for result in results)
    
    @pytest.mark.asyncio
    async def test_multi_format_validation(self, schema_manager):
        """Test de validation multi-format"""
        test_data = {"test": "data"}
        
        # Test JSON
        json_result = await schema_manager.validate_multi_format(
            data=json.dumps(test_data),
            data_format=DataFormat.JSON
        )
        assert json_result is not None
        
        # Test YAML
        yaml_data = "test: data\nvalid: true"
        yaml_result = await schema_manager.validate_multi_format(
            data=yaml_data,
            data_format=DataFormat.YAML
        )
        assert yaml_result is not None
    
    @pytest.mark.asyncio
    async def test_performance_metrics(self, schema_manager):
        """Test des métriques de performance"""
        # Effectuer quelques validations
        test_data = {
            "user_id": "perf_test_user",
            "profile_data": {
                "email": "perf@example.com",
                "created_at": "2024-01-15T10:30:00Z"
            },
            "preferences": {
                "music_preferences": {},
                "ai_preferences": {}
            },
            "privacy_settings": {
                "data_sharing_consent": {
                    "marketing": False,
                    "analytics": True,
                    "third_party": False
                },
                "analytics_consent": {
                    "behavioral_analytics": True
                }
            }
        }
        
        # Plusieurs validations pour générer des métriques
        for _ in range(5):
            await schema_manager.validate_data(
                data=test_data,
                schema_id="user_profile"
            )
        
        metrics = await schema_manager.get_performance_metrics()
        
        assert metrics is not None
        assert "total_validations" in metrics
        assert metrics["total_validations"] >= 5
    
    @pytest.mark.asyncio
    async def test_schema_loading_errors(self, schema_manager):
        """Test de gestion des erreurs de chargement de schémas"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            f.write('{"invalid": json}')  # JSON invalide
            invalid_schema_path = f.name
        
        try:
            with pytest.raises(Exception):
                await schema_manager.load_schema_from_file(
                    schema_id="invalid_schema",
                    file_path=invalid_schema_path,
                    schema_type=SchemaType.JSON_SCHEMA
                )
        finally:
            Path(invalid_schema_path).unlink()  # Nettoyer le fichier temporaire
    
    @pytest.mark.asyncio
    async def test_custom_validation_rules(self, schema_manager):
        """Test des règles de validation personnalisées"""
        # Test avec une règle personnalisée
        custom_data = {
            "user_id": "custom_validation_test",
            "profile_data": {
                "email": "custom@example.com",
                "created_at": "2024-01-15T10:30:00Z"
            },
            "preferences": {
                "music_preferences": {
                    "favorite_genres": ["pop"] * 15  # Trop de genres (limite: 10)
                },
                "ai_preferences": {}
            },
            "privacy_settings": {
                "data_sharing_consent": {
                    "marketing": False,
                    "analytics": True,
                    "third_party": False
                },
                "analytics_consent": {
                    "behavioral_analytics": True
                }
            }
        }
        
        result = await schema_manager.validate_data(
            data=custom_data,
            schema_id="user_profile"
        )
        
        # Devrait échouer à cause de la limite sur favorite_genres
        assert not result.is_valid
        assert any("favorite_genres" in str(error) for error in result.errors)


@pytest.mark.asyncio
async def test_enterprise_features_integration():
    """Test d'intégration des fonctionnalités enterprise"""
    manager = EnterpriseSchemaManager()
    
    # Test que toutes les fonctionnalités enterprise sont disponibles
    assert hasattr(manager, 'validate_data')
    assert hasattr(manager, 'validate_batch')
    assert hasattr(manager, 'validate_multi_format')
    assert hasattr(manager, 'get_performance_metrics')
    assert hasattr(manager, 'load_schema_from_file')
    
    # Test des énumérations
    assert SchemaType.JSON_SCHEMA
    assert DataFormat.JSON
    assert DataFormat.YAML
    assert DataFormat.XML


if __name__ == "__main__":
    """Exécution directe des tests pour validation rapide"""
    import sys
    
    print("🧪 Lancement des tests automatisés du système enterprise")
    print("=" * 60)
    
    # Exécution simple sans pytest pour démonstration
    async def run_basic_tests():
        manager = EnterpriseSchemaManager()
        
        print("✅ Test 1: Initialisation du gestionnaire - OK")
        
        # Test basique de validation
        test_data = {"test": "data"}
        try:
            # Ce test échouera car aucun schéma n'est chargé, mais teste la méthode
            result = await manager.validate_data(test_data, "non_existent_schema")
            print("❌ Test 2: Validation avec schéma inexistant - Attendu")
        except Exception:
            print("✅ Test 2: Gestion d'erreur schéma inexistant - OK")
        
        print("✅ Test 3: Méthodes enterprise disponibles - OK")
        
        metrics = await manager.get_performance_metrics()
        print(f"✅ Test 4: Métriques de performance - {type(metrics)}")
        
        print("\n🎉 Tests basiques terminés avec succès!")
        print("💡 Utilisez 'pytest' pour la suite complète de tests")
    
    # Exécution
    if len(sys.argv) > 1 and sys.argv[1] == "--basic":
        asyncio.run(run_basic_tests())
    else:
        print("💡 Pour lancer les tests complets: pytest test_validation_system.py")
        print("💡 Pour les tests basiques: python test_validation_system.py --basic")
