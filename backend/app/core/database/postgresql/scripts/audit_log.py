"""
Script d’audit PostgreSQL
- Analyse les logs d’audit, détecte les accès suspects, génère un rapport
"""

import os
import re
from datetime import datetime

LOG_PATH = os.getenv("PG_AUDIT_LOG", "/var/log/postgresql/postgresql.log")
REPORT_PATH = f"./audit_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"

with open(LOG_PATH, "r") as f:
    lines = f.readlines()

suspicious = [l for l in lines if re.search(r"(failed|denied|unauthorized|error)", l, re.I)]

with open(REPORT_PATH, "w") as f:
    f.write("Rapport d’audit PostgreSQL\n===========================\n")
    f.write(f"Total lignes analysées: {len(lines)}\n")
    f.write(f"Accès suspects: {len(suspicious)}\n\n")
    for l in suspicious:
        f.write(l)

print(f"Rapport d’audit généré: {REPORT_PATH}")
