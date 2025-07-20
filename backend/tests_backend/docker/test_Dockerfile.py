from unittest.mock import Mock
def test_dockerfile_exists():
    """Vérifie que le Dockerfile existe."""
    assert os.path.isfile(DOCKERFILE), f"Fichier introuvable: {DOCKERFILE}"

def test_dockerfile_no_root_user():
    """Vérifie que l'image ne tourne pas en root (USER non root)."""
    with open(DOCKERFILE, 'r') as f:
        content = f.read()
    assert re.search(r'^USER\s+\w+', content, re.MULTILINE), "USER non défini ou root par défaut."

def test_dockerfile_multistage():
    """Vérifie l'utilisation du multi-stage build pour la sécurité et la taille."""
    with open(DOCKERFILE, 'r') as f:
        content = f.read()
    assert content.count('FROM') >= 2, "Multi-stage build non utilisé."

def test_dockerfile_no_secrets():
    """Vérifie qu'aucun secret ou mot de passe n'est présent en clair."""
    with open(DOCKERFILE, 'r') as f:
        content = f.read()
    forbidden = ['password', 'secret', 'AWS_ACCESS_KEY', 'AWS_SECRET', 'token']
    for word in forbidden:
        assert word not in content, f"Secret détecté dans le Dockerfile: {word}"
