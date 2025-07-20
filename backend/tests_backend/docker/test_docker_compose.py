from unittest.mock import Mock
def test_compose_exists():
    assert os.path.isfile(COMPOSE), f"Fichier introuvable: {COMPOSE}"

def test_compose_yaml_valid():
    with open(COMPOSE, 'r') as f:
        try:
            data = yaml.safe_load(f)
        except yaml.YAMLError as e:
            pytest.fail(f"YAML invalide: {e}")
    assert isinstance(data, dict), "Le fichier doit être un dictionnaire YAML."

def test_compose_services_defined():
    with open(COMPOSE, 'r') as f:
        data = yaml.safe_load(f)
    assert 'services' in data, "Section 'services' manquante."
    assert len(data['services']) >= 1, "Au moins un service attendu."

def test_compose_no_secrets():
    with open(COMPOSE, 'r') as f:
        content = f.read()
    forbidden = ['password', 'secret', 'AWS_ACCESS_KEY', 'AWS_SECRET', 'token']
    for word in forbidden:
        assert word not in content, f"Secret détecté dans le fichier: {word}"
