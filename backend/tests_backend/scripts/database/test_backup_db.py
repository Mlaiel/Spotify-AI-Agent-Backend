from unittest.mock import Mock
import os
import stat
import subprocess
import pytest

BACKUP_DB_SH = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../scripts/database/backup_db.sh'))

def test_backup_db_script_exists():
    assert os.path.isfile(BACKUP_DB_SH), f"Fichier introuvable: {BACKUP_DB_SH}"

def test_backup_db_script_permissions():
    st = os.stat(BACKUP_DB_SH)
    assert stat.S_IMODE(st.st_mode) & 0o111, "Le script doit être exécutable."

def test_backup_db_script_runs():
    result = subprocess.run([BACKUP_DB_SH, '--dry-run'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=20)
    assert result.returncode == 0, f"Le script backup_db.sh a échoué: {result.stderr.decode()}"

def test_backup_db_script_no_secrets_in_output():
    result = subprocess.run([BACKUP_DB_SH, '--dry-run'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=20)
    output = result.stdout.decode() + result.stderr.decode()
    forbidden = ['password', 'secret', 'AWS_ACCESS_KEY', 'AWS_SECRET', 'token']
    for word in forbidden:
        assert word not in output, f"Secret détecté dans la sortie du script: {word}"
