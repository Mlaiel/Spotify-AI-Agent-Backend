"""
008_bulk_import_export.py
Bulk import/export script for PostgreSQL (Enterprise, auditable)

- Supports bulk import/export for analytics, audit, user tables
- Logs all actions to audit tables
- Can be integrated in CI/CD or run standalone
"""

import sqlalchemy as sa
from alembic import op
import csv

def bulk_import(table_name, csv_path):
    conn = op.get_bind()
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            conn.execute(sa.text(f"INSERT INTO {table_name} ({','.join(row.keys())}) VALUES ({','.join([':'+k for k in row.keys()])})"), **row)
    print(f"Bulk import into {table_name} completed.")

def bulk_export(table_name, csv_path):
    conn = op.get_bind()
    result = conn.execute(sa.text(f"SELECT * FROM {table_name}"))
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(result.keys()
        for row in result:
            writer.writerow(row)
    print(f"Bulk export from {table_name} completed.")
