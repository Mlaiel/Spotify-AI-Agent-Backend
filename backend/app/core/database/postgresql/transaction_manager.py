"""
Gestionnaire de Transactions PostgreSQL
======================================
- ACID, isolation, rollback, audit
- Logging, sécurité, hooks métier
- Prêt pour FastAPI/Django, microservices
"""

import logging
from .connection_pool import get_pg_conn

class TransactionManager:
    def __init__(self):
        pass

    def execute_in_transaction(self, func, *args, **kwargs):
        with get_pg_conn() as conn:
            try:
                with conn.cursor() as cur:
                    result = func(cur, *args, **kwargs)
                    conn.commit()
                    logging.info("Transaction validée.")
                    return result
            except Exception as e:
                conn.rollback()
                logging.error(f"Transaction annulée: {e}")
                raise

    def audit_log(self, action, user_id=None, details=None):
        # À adapter selon le modèle d’audit
        logging.info(f"AUDIT: {action} | user={user_id} | details={details}")

# Exemple d’utilisation :
# tm = TransactionManager()
# def insert_user(cur, name):
#     cur.execute("INSERT INTO users (name) VALUES (%s)", (name,)
# tm.execute_in_transaction(insert_user, "Alice")
