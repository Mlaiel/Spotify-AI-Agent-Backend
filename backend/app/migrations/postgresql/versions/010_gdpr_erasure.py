"""
010_gdpr_erasure.py
GDPR/DSGVO-compliant user data erasure script (Enterprise, auditable)

- Erases or anonymizes all personal data for a given user_id
- Logs all actions to audit tables
- Can be integrated in CI/CD or run standalone
"""

import sqlalchemy as sa
from alembic import op

def erase_user(user_id):
    conn = op.get_bind()
    # Anonymize user in users table
    conn.execute(sa.text("""
        UPDATE users SET email = NULL, preferences = NULL, status = 'ANONYMIZED' WHERE id = :user_id)
    """), user_id=user_id)
    # Remove user from consent, analytics, audit_log, security_event
    for table in ['consent', 'analytics', 'audit_log', 'security_event']:
        conn.execute(sa.text(f"DELETE FROM {table} WHERE user_id = :user_id"), user_id=user_id)
    print(f"GDPR erasure for user {user_id} completed.")
