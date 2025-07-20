from unittest.mock import Mock
def test_migrate_script_exists():
    assert os.path.isfile(MIGRATE_SH), f"Fichier introuvable: {MIGRATE_SH}"

def test_migrate_script_permissions():
    st = os.stat(MIGRATE_SH)
    assert stat.S_IMODE(st.st_mode) & 0o111, "Le script doit être exécutable."

def test_migrate_script_runs():
    result = subprocess.run([MIGRATE_SH, '--dry-run'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=15)
    assert result.returncode == 0, f"Le script migrate.sh a échoué: {result.stderr.decode()}"

def test_migrate_script_no_secrets_in_output():
    result = subprocess.run([MIGRATE_SH, '--dry-run'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=15)
    output = result.stdout.decode() + result.stderr.decode()
    forbidden = ['password', 'secret', 'AWS_ACCESS_KEY', 'AWS_SECRET', 'token']
    for word in forbidden:
        assert word not in output, f"Secret détecté dans la sortie du script: {word}"
