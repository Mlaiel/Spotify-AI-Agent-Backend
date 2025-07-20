from unittest.mock import Mock
def test_backup_script_exists():
    assert os.path.isfile(SCRIPT), f"Fichier introuvable: {SCRIPT}"

def test_backup_script_permissions():
    st = os.stat(SCRIPT)
    assert stat.S_IMODE(st.st_mode) & 0o111, "Le script doit être exécutable."

def test_backup_script_runs():
    result = subprocess.run([SCRIPT, '--dry-run'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=20)
    assert result.returncode == 0, f"Le script backup.sh a échoué: {result.stderr.decode()}"

def test_backup_script_no_secrets_in_output():
    result = subprocess.run([SCRIPT, '--dry-run'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=20)
    output = result.stdout.decode() + result.stderr.decode()
    forbidden = ['password', 'secret', 'AWS_ACCESS_KEY', 'AWS_SECRET', 'token']
    for word in forbidden:
        assert word not in output, f"Secret détecté dans la sortie du script: {word}"
