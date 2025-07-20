from unittest.mock import Mock
def test_clean_script_exists():
    assert os.path.isfile(CLEAN_SH), f"Fichier introuvable: {CLEAN_SH}"

def test_clean_script_permissions():
    st = os.stat(CLEAN_SH)
    assert stat.S_IMODE(st.st_mode) & 0o111, "Le script doit être exécutable."

def test_clean_script_runs():
    result = subprocess.run([CLEAN_SH, '--dry-run'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=10)
    assert result.returncode == 0, f"Le script clean.sh a échoué: {result.stderr.decode()}"

def test_clean_script_no_unprotected_rm_rf():
    with open(CLEAN_SH, 'r') as f:
        content = f.read()
    assert 'rm -rf /' not in content, "Commande dangereuse détectée: rm -rf /"
