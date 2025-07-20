from unittest.mock import Mock

import pytest
from unittest.mock import MagicMock

@pytest.fixture(scope="session")
def es_mock():
    """Mock Elasticsearch pour tests mappings."""
    es = MagicMock()
    es.indices = MagicMock()
    es.search = MagicMock()
    es.index = MagicMock()
    return es

def validate_mapping(es_mock, index_name, expected_mapping):
    """Valide la structure d’un mapping Elasticsearch simulé."""
    actual_mapping = es_mock.indices.get_mapping(index=index_name)
    assert actual_mapping == expected_mapping, f"Le mapping de {index_name} ne correspond pas au mapping attendu."
