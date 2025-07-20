from unittest.mock import Mock
def test_audit_script_exists():
    assert os.path.isfile(AUDIT_SH), f"Fichier introuvable: {AUDIT_SH}"

def test_audit_script_permissions():
    st = os.stat(AUDIT_SH)
    assert stat.S_IMODE(st.st_mode) & 0o111, "Le script doit être exécutable."

def test_audit_script_runs():
    result = subprocess.run([AUDIT_SH], stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=10)
    assert result.returncode == 0, f"Le script audit.sh a échoué: {result.stderr.decode()}"

def test_audit_script_no_secrets_in_output():
    result = subprocess.run([AUDIT_SH], stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=10)
    output = result.stdout.decode() + result.stderr.decode()
    forbidden = ['password', 'secret', 'AWS_ACCESS_KEY', 'AWS_SECRET', 'token']
    for word in forbidden:
        assert word not in output, f"Secret détecté dans la sortie du script: {word}"
