from unittest.mock import Mock
def test_cache_warmup_script_exists():
    assert os.path.isfile(SCRIPT), f"Fichier introuvable: {SCRIPT}"

def test_cache_warmup_script_importable():
    spec = importlib.util.spec_from_file_location("cache_warmup", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as e:
        pytest.fail(f"Erreur d'import/cache_warmup.py: {e}")

def test_cache_warmup_no_secrets_in_code():
    with open(SCRIPT, 'r') as f:
        content = f.read()
    forbidden = ['password', 'secret', 'AWS_ACCESS_KEY', 'AWS_SECRET', 'token']
    for word in forbidden:
        assert word not in content, f"Secret détecté dans le code: {word}"
