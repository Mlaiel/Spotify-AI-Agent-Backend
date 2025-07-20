from unittest.mock import Mock
def test_nginx_conf_exists():
    """Vérifie que le fichier nginx.conf existe."""
    assert os.path.isfile(NGINX_CONF), f"Fichier introuvable: {NGINX_CONF}"

def test_nginx_conf_syntax():
    """Vérifie que le fichier nginx.conf ne contient pas d'erreur de syntaxe basique."""
    with open(NGINX_CONF, 'r') as f:
        content = f.read()
    assert 'server' in content, "Bloc 'server' manquant."
    assert 'listen' in content, "Directive 'listen' manquante."

def test_nginx_conf_security_headers():
    """Vérifie la présence des headers de sécurité essentiels."""
    with open(NGINX_CONF, 'r') as f:
        content = f.read()
    for header in ['X-Frame-Options', 'X-Content-Type-Options', 'Strict-Transport-Security']:
        assert header in content, f"Header de sécurité manquant: {header}"

def test_nginx_conf_no_default_server():
    """Vérifie qu'aucun serveur par défaut n'est exposé publiquement."""
    with open(NGINX_CONF, 'r') as f:
        content = f.read()
    assert 'default_server' not in content, "Le serveur par défaut ne doit pas être exposé."
