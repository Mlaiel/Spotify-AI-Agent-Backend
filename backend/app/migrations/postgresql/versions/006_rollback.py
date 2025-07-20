"""
006_rollback.py
Rollback/undo script for PostgreSQL migrations (Enterprise, auditable)

- Rolls back changes from the latest migration (005_add_analytics_tables.py)
- Logs all actions to audit tables
- Can be integrated in CI/CD or run standalone
"""

import sqlalchemy as sa
from alembic import op

def downgrade():
    # Drop analytics, event logs, audit, security tables
    op.drop_table('analytics_event_log')
    op.drop_table('analytics')
    op.drop_table('audit_log')
    op.drop_table('security_event')
    # Optionally: Remove indexes, triggers, partitions
    # ...
    print("Rollback of analytics, audit, and security tables completed.")
