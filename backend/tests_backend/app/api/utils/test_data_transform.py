"""
🎵 Spotify AI Agent - Tests Data Transform Module
================================================

Tests enterprise complets pour le module data_transform
avec validation de transformation, sécurité et performance.

🎖️ Développé par l'équipe d'experts enterprise
"""

import pytest
import json
from datetime import datetime, date
from decimal import Decimal
from typing import Dict, Any, List
from unittest.mock import patch, Mock

# Import du module à tester
from backend.app.api.utils.data_transform import (
    transform_data,
    validate_data_structure,
    normalize_data,
    sanitize_input,
    deep_merge,
    flatten_dict,
    unflatten_dict,
    filter_dict,
    safe_cast,
    serialize_for_json
)

from . import TestUtils, security_test, performance_test, integration_test


class TestDataTransform:
    """Tests pour le module data_transform"""
    
    def test_transform_data_basic(self, sample_data):
        """Test transformation basique de données"""
        schema = {
            'user_id': int,
            'username': str,
            'email': str
        }
        
        result = transform_data(sample_data, schema)
        
        assert isinstance(result['user_id'], int)
        assert isinstance(result['username'], str)
        assert isinstance(result['email'], str)
        assert result['user_id'] == 12345
        assert result['username'] == 'test_user'
        assert result['email'] == 'test@example.com'
    
    def test_transform_data_with_defaults(self):
        """Test transformation avec valeurs par défaut"""
        data = {'name': 'John'}
        schema = {
            'name': str,
            'age': int,
            'active': bool
        }
        defaults = {
            'age': 25,
            'active': True
        }
        
        result = transform_data(data, schema, defaults)
        
        assert result['name'] == 'John'
        assert result['age'] == 25
        assert result['active'] is True
    
    def test_transform_data_type_conversion(self):
        """Test conversion de types"""
        data = {
            'id': '123',
            'price': '19.99',
            'active': 'true',
            'count': '0'
        }
        schema = {
            'id': int,
            'price': float,
            'active': bool,
            'count': int
        }
        
        result = transform_data(data, schema)
        
        assert result['id'] == 123
        assert result['price'] == 19.99
        assert result['active'] is True
        assert result['count'] == 0
    
    @performance_test
    def test_transform_data_performance(self):
        """Test performance de transformation"""
        large_data = TestUtils.generate_large_dataset(1000)
        schema = {
            'id': int,
            'name': str,
            'value': float,
            'active': bool
        }
        
        def transform_large():
            return [transform_data(item, schema) for item in large_data]
        
        TestUtils.assert_performance(transform_large, max_time_ms=200)
    
    def test_validate_data_structure_valid(self, sample_data):
        """Test validation structure valide"""
        schema = {
            'user_id': {'type': int, 'required': True},
            'username': {'type': str, 'required': True, 'min_length': 3},
            'email': {'type': str, 'required': True, 'pattern': r'^[^@]+@[^@]+\.[^@]+$'},
            'metadata': {'type': dict, 'required': False}
        }
        
        result = validate_data_structure(sample_data, schema)
        
        assert result['valid'] is True
        assert len(result['errors']) == 0
    
    def test_validate_data_structure_invalid(self):
        """Test validation structure invalide"""
        data = {
            'user_id': 'not_a_number',
            'username': 'ab',  # Trop court
            'email': 'invalid_email'
        }
        schema = {
            'user_id': {'type': int, 'required': True},
            'username': {'type': str, 'required': True, 'min_length': 3},
            'email': {'type': str, 'required': True, 'pattern': r'^[^@]+@[^@]+\.[^@]+$'}
        }
        
        result = validate_data_structure(data, schema)
        
        assert result['valid'] is False
        assert len(result['errors']) >= 3
    
    def test_validate_data_structure_nested(self):
        """Test validation structure imbriquée"""
        data = {
            'user': {
                'name': 'John',
                'profile': {
                    'age': 30,
                    'city': 'Paris'
                }
            }
        }
        schema = {
            'user': {
                'type': dict,
                'schema': {
                    'name': {'type': str, 'required': True},
                    'profile': {
                        'type': dict,
                        'schema': {
                            'age': {'type': int, 'min': 0, 'max': 120},
                            'city': {'type': str, 'required': True}
                        }
                    }
                }
            }
        }
        
        result = validate_data_structure(data, schema)
        
        assert result['valid'] is True
    
    def test_normalize_data_basic(self):
        """Test normalisation basique"""
        data = {
            'NAME': 'John Doe',
            'email': '  JOHN@EXAMPLE.COM  ',
            'Age': '30'
        }
        
        result = normalize_data(data)
        
        assert result['name'] == 'John Doe'
        assert result['email'] == 'john@example.com'
        assert result['age'] == '30'
    
    def test_normalize_data_custom_rules(self):
        """Test normalisation avec règles personnalisées"""
        data = {
            'phone': '+33 1 23 45 67 89',
            'postal_code': '75001'
        }
        rules = {
            'phone': lambda x: x.replace(' ', '').replace('+', '00'),
            'postal_code': lambda x: x.zfill(5)
        }
        
        result = normalize_data(data, rules)
        
        assert result['phone'] == '0033123456789'
        assert result['postal_code'] == '75001'
    
    @security_test
    def test_sanitize_input_xss_protection(self):
        """Test protection XSS"""
        malicious_data = {
            'comment': '<script>alert("XSS")</script>Hello',
            'name': '<img src="x" onerror="alert(1)">John',
            'description': 'Normal text'
        }
        
        result = sanitize_input(malicious_data)
        
        assert '<script>' not in result['comment']
        assert 'alert' not in result['comment']
        assert result['comment'] == 'Hello'
        assert '<img' not in result['name']
        assert result['name'] == 'John'
        assert result['description'] == 'Normal text'
    
    @security_test
    def test_sanitize_input_sql_injection(self):
        """Test protection injection SQL"""
        data = {
            'username': "admin'; DROP TABLE users; --",
            'search': "'; SELECT * FROM passwords WHERE '1'='1"
        }
        
        result = sanitize_input(data)
        
        assert 'DROP TABLE' not in result['username']
        assert 'SELECT' not in result['search']
        assert '--' not in result['username']
    
    def test_deep_merge_basic(self):
        """Test fusion profonde basique"""
        dict1 = {
            'a': 1,
            'b': {
                'c': 2,
                'd': 3
            }
        }
        dict2 = {
            'b': {
                'c': 4,
                'e': 5
            },
            'f': 6
        }
        
        result = deep_merge(dict1, dict2)
        
        assert result['a'] == 1
        assert result['b']['c'] == 4  # Écrasé
        assert result['b']['d'] == 3  # Conservé
        assert result['b']['e'] == 5  # Ajouté
        assert result['f'] == 6
    
    def test_deep_merge_with_lists(self):
        """Test fusion avec listes"""
        dict1 = {
            'tags': ['python', 'django'],
            'config': {
                'features': ['auth', 'api']
            }
        }
        dict2 = {
            'tags': ['fastapi'],
            'config': {
                'features': ['cache']
            }
        }
        
        # Mode extend pour les listes
        result = deep_merge(dict1, dict2, list_strategy='extend')
        
        assert 'python' in result['tags']
        assert 'fastapi' in result['tags']
        assert 'auth' in result['config']['features']
        assert 'cache' in result['config']['features']
    
    def test_deep_merge_conflict_resolution(self):
        """Test résolution de conflits"""
        dict1 = {'value': 10}
        dict2 = {'value': 20}
        
        # Stratégie override (défaut)
        result1 = deep_merge(dict1, dict2)
        assert result1['value'] == 20
        
        # Stratégie keep_first
        result2 = deep_merge(dict1, dict2, conflict_strategy='keep_first')
        assert result2['value'] == 10
    
    def test_flatten_dict_basic(self):
        """Test aplatissement basique"""
        nested = {
            'user': {
                'profile': {
                    'name': 'John',
                    'age': 30
                },
                'settings': {
                    'theme': 'dark'
                }
            },
            'active': True
        }
        
        result = flatten_dict(nested)
        
        assert result['user.profile.name'] == 'John'
        assert result['user.profile.age'] == 30
        assert result['user.settings.theme'] == 'dark'
        assert result['active'] is True
    
    def test_flatten_dict_custom_separator(self):
        """Test aplatissement avec séparateur personnalisé"""
        nested = {
            'a': {
                'b': {
                    'c': 'value'
                }
            }
        }
        
        result = flatten_dict(nested, separator='/')
        
        assert result['a/b/c'] == 'value'
    
    def test_unflatten_dict_basic(self):
        """Test reconstruction dictionnaire"""
        flat = {
            'user.name': 'John',
            'user.age': 30,
            'user.profile.city': 'Paris',
            'active': True
        }
        
        result = unflatten_dict(flat)
        
        assert result['user']['name'] == 'John'
        assert result['user']['age'] == 30
        assert result['user']['profile']['city'] == 'Paris'
        assert result['active'] is True
    
    def test_flatten_unflatten_roundtrip(self):
        """Test aller-retour aplatissement/reconstruction"""
        original = {
            'level1': {
                'level2': {
                    'level3': {
                        'value': 'test'
                    },
                    'array': [1, 2, 3]
                }
            }
        }
        
        flattened = flatten_dict(original)
        reconstructed = unflatten_dict(flattened)
        
        assert reconstructed == original
    
    def test_filter_dict_include(self):
        """Test filtrage par inclusion"""
        data = {
            'id': 1,
            'name': 'John',
            'email': 'john@example.com',
            'password': 'secret',
            'internal_id': 'xyz'
        }
        
        result = filter_dict(data, include=['id', 'name', 'email'])
        
        assert 'id' in result
        assert 'name' in result
        assert 'email' in result
        assert 'password' not in result
        assert 'internal_id' not in result
    
    def test_filter_dict_exclude(self):
        """Test filtrage par exclusion"""
        data = {
            'id': 1,
            'name': 'John',
            'email': 'john@example.com',
            'password': 'secret',
            'internal_id': 'xyz'
        }
        
        result = filter_dict(data, exclude=['password', 'internal_id'])
        
        assert 'id' in result
        assert 'name' in result
        assert 'email' in result
        assert 'password' not in result
        assert 'internal_id' not in result
    
    def test_filter_dict_predicate(self):
        """Test filtrage avec prédicat"""
        data = {
            'public_name': 'John',
            'private_key': 'secret',
            'public_email': 'john@example.com',
            'private_data': 'confidential'
        }
        
        # Garder seulement les clés publiques
        result = filter_dict(data, predicate=lambda k, v: k.startswith('public_'))
        
        assert 'public_name' in result
        assert 'public_email' in result
        assert 'private_key' not in result
        assert 'private_data' not in result
    
    def test_safe_cast_valid_conversions(self):
        """Test conversions sécurisées valides"""
        assert safe_cast('123', int) == 123
        assert safe_cast('19.99', float) == 19.99
        assert safe_cast('true', bool) is True
        assert safe_cast('false', bool) is False
        assert safe_cast('hello', str) == 'hello'
    
    def test_safe_cast_invalid_conversions(self):
        """Test conversions sécurisées invalides"""
        assert safe_cast('not_a_number', int) is None
        assert safe_cast('not_a_float', float) is None
        assert safe_cast('not_a_bool', bool) is None
    
    def test_safe_cast_with_default(self):
        """Test conversions avec valeur par défaut"""
        assert safe_cast('invalid', int, default=0) == 0
        assert safe_cast('invalid', float, default=0.0) == 0.0
        assert safe_cast('invalid', bool, default=False) is False
    
    def test_serialize_for_json_basic_types(self):
        """Test sérialisation types basiques"""
        data = {
            'string': 'hello',
            'integer': 123,
            'float': 19.99,
            'boolean': True,
            'null': None,
            'list': [1, 2, 3],
            'dict': {'key': 'value'}
        }
        
        result = serialize_for_json(data)
        json_str = json.dumps(result)  # Doit fonctionner sans erreur
        
        assert isinstance(json_str, str)
    
    def test_serialize_for_json_complex_types(self):
        """Test sérialisation types complexes"""
        data = {
            'datetime': datetime(2025, 7, 14, 10, 30, 0),
            'date': date(2025, 7, 14),
            'decimal': Decimal('19.99'),
            'set': {1, 2, 3},
            'custom_object': Mock()
        }
        
        result = serialize_for_json(data)
        
        assert isinstance(result['datetime'], str)
        assert result['datetime'] == '2025-07-14T10:30:00'
        assert isinstance(result['date'], str)
        assert result['date'] == '2025-07-14'
        assert isinstance(result['decimal'], float)
        assert result['decimal'] == 19.99
        assert isinstance(result['set'], list)
        assert set(result['set']) == {1, 2, 3}
    
    @performance_test
    def test_serialize_for_json_performance(self):
        """Test performance sérialisation"""
        large_data = {
            'items': TestUtils.generate_large_dataset(1000),
            'timestamp': datetime.now(),
            'metadata': {
                'complex': {
                    'nested': {
                        'data': list(range(100))
                    }
                }
            }
        }
        
        def serialize_large():
            return serialize_for_json(large_data)
        
        TestUtils.assert_performance(serialize_large, max_time_ms=500)
    
    @integration_test
    def test_data_pipeline_integration(self, sample_data):
        """Test intégration pipeline de données"""
        # Pipeline complet: validation -> normalisation -> assainissement -> transformation
        
        # 1. Validation
        schema = {
            'user_id': {'type': int, 'required': True},
            'username': {'type': str, 'required': True},
            'email': {'type': str, 'required': True}
        }
        validation_result = validate_data_structure(sample_data, schema)
        assert validation_result['valid'] is True
        
        # 2. Normalisation
        normalized = normalize_data(sample_data)
        
        # 3. Assainissement
        sanitized = sanitize_input(normalized)
        
        # 4. Transformation
        transform_schema = {
            'user_id': int,
            'username': str,
            'email': str
        }
        transformed = transform_data(sanitized, transform_schema)
        
        # 5. Sérialisation
        serialized = serialize_for_json(transformed)
        
        # Vérifications finales
        assert isinstance(serialized['user_id'], int)
        assert isinstance(serialized['username'], str)
        assert isinstance(serialized['email'], str)
    
    @performance_test
    def test_memory_efficiency(self):
        """Test efficacité mémoire"""
        def process_large_dataset():
            # Générer et traiter un grand dataset
            data = TestUtils.generate_large_dataset(5000)
            
            # Pipeline de traitement
            for item in data:
                normalized = normalize_data(item)
                sanitized = sanitize_input(normalized)
                serialized = serialize_for_json(sanitized)
            
            return len(data)
        
        TestUtils.assert_memory_usage(process_large_dataset, max_memory_mb=100)


# Tests d'erreurs et cas limites
class TestDataTransformEdgeCases:
    """Tests pour les cas limites et gestion d'erreurs"""
    
    def test_transform_data_empty_dict(self):
        """Test transformation dictionnaire vide"""
        result = transform_data({}, {})
        assert result == {}
    
    def test_transform_data_none_values(self):
        """Test transformation avec valeurs None"""
        data = {'key': None}
        schema = {'key': str}
        
        result = transform_data(data, schema)
        assert result['key'] is None
    
    def test_deep_merge_empty_dicts(self):
        """Test fusion dictionnaires vides"""
        result = deep_merge({}, {})
        assert result == {}
    
    def test_flatten_dict_empty(self):
        """Test aplatissement dictionnaire vide"""
        result = flatten_dict({})
        assert result == {}
    
    def test_sanitize_input_none(self):
        """Test assainissement avec None"""
        result = sanitize_input(None)
        assert result is None
    
    def test_serialize_for_json_recursive_reference(self):
        """Test sérialisation avec référence circulaire"""
        data = {'key': 'value'}
        data['self'] = data  # Référence circulaire
        
        # Ne doit pas planter
        result = serialize_for_json(data)
        assert 'key' in result


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
