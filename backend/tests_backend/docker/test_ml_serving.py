from unittest.mock import Mock
def test_ml_serving_script_exists():
    assert os.path.isfile(ML_SERVING_SH), f"Fichier introuvable: {ML_SERVING_SH}"

def test_ml_serving_script_permissions():
    st = os.stat(ML_SERVING_SH)
    assert stat.S_IMODE(st.st_mode) & 0o111, "Le script doit être exécutable."

def test_ml_serving_script_runs():
    result = subprocess.run([ML_SERVING_SH, '--dry-run'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=20)
    assert result.returncode == 0, f"Le script ml_serving.sh a échoué: {result.stderr.decode()}"

def test_ml_serving_script_no_secrets_in_output():
    result = subprocess.run([ML_SERVING_SH, '--dry-run'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=20)
    output = result.stdout.decode() + result.stderr.decode()
    forbidden = ['password', 'secret', 'AWS_ACCESS_KEY', 'AWS_SECRET', 'token']
    for word in forbidden:
        assert word not in output, f"Secret détecté dans la sortie du script: {word}"
