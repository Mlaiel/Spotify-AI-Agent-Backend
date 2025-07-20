from unittest.mock import Mock
def test_supervisor_conf_exists():
    """Vérifie que le fichier supervisor.conf existe."""
    assert os.path.isfile(SUPERVISOR_CONF), f"Fichier introuvable: {SUPERVISOR_CONF}"

def test_supervisor_conf_syntax():
    """Vérifie que le fichier supervisor.conf contient les sections essentielles."""
    with open(SUPERVISOR_CONF, 'r') as f:
        content = f.read()
    assert '[supervisord]' in content, "Section [supervisord] manquante."
    assert '[program:' in content, "Section [program] manquante."

def test_supervisor_conf_no_insecure_commands():
    """Vérifie qu'aucune commande dangereuse n'est présente dans la conf."""
    with open(SUPERVISOR_CONF, 'r') as f:
        content = f.read()
    forbidden = ['rm -rf', 'chmod 777', 'curl ', 'wget ']
    for cmd in forbidden:
        assert cmd not in content, f"Commande dangereuse détectée: {cmd}"
