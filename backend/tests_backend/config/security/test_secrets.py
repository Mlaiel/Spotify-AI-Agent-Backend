from unittest.mock import Mock
def test_secrets_file_exists():
    """Testet, ob die secrets.encrypted Datei existiert."""
    assert os.path.isfile(SECRETS_PATH), f"Datei nicht gefunden: {SECRETS_PATH}"

def test_secrets_file_permissions():
    """Stellt sicher, dass die Datei nur vom Besitzer lesbar/schreibbar ist (600)."""
    st = os.stat(SECRETS_PATH)
    assert stat.S_IMODE(st.st_mode) == 0o600, f"Unsichere Dateirechte: {oct(stat.S_IMODE(st.st_mode))}"

def test_secrets_file_not_empty():
    """Stellt sicher, dass die Datei nicht leer ist."""
    assert os.path.getsize(SECRETS_PATH) > 0, "Die secrets.encrypted Datei ist leer."

def test_secrets_file_no_plaintext_leaks():
    """Überprüft, dass keine Klartext-Geheimnisse in der Datei stehen (z.B. 'password', 'secret', 'key')."""
    with open(SECRETS_PATH, 'rb') as f:
        content = f.read()
    # Suche nach typischen Klartext-Mustern (nur als Beispiel, kann erweitert werden)
    forbidden = [b'password', b'secret', b'key', b'api', b'token']
    for word in forbidden:
        assert word not in content, f"Klartext-Geheimnis gefunden: {word.decode()}"

# Optional: Test auf Verschlüsselungsheader oder Format
def test_secrets_file_encryption_header():
    """Prüft, ob die Datei mit einem erwarteten Verschlüsselungsheader beginnt (z.B. 'Salted__')."""
    with open(SECRETS_PATH, 'rb') as f:
        header = f.read(8)
    assert header == b'Salted__', "Kein gültiger Verschlüsselungsheader gefunden."
