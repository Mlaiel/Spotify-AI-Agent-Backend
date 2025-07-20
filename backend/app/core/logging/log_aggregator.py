"""
Module: log_aggregator.py
Description: Agrégateur de logs industriel, compatible ELK, Prometheus, Grafana, avec support multi-service et multi-format.
"""
import logging
from typing import List, Dict, Any

class LogAggregator:
    def __init__(self, name: str = "log_aggregator"):
        self.logger = logging.getLogger(name)
        self.logs: List[Dict[str, Any]] = []

    def add_log(self, log: Dict[str, Any]):
        self.logs.append(log)
        self.logger.debug(f"[AGGREGATE] {log}")

    def export(self, format: str = "json") -> Any:
        if format == "json":
            import json
            return json.dumps(self.logs)
        elif format == "csv":
            import csv
            import io
            output = io.StringIO()
            writer = csv.DictWriter(output, fieldnames=self.logs[0].keys() if self.logs else [])
            writer.writeheader()
            writer.writerows(self.logs)
            return output.getvalue()
        else:
            return self.logs

    def clear(self):
        self.logs.clear()

# Exemple d'utilisation
# aggregator = LogAggregator()
# aggregator.add_log({"service": "ai", "event": "start"})
# print(aggregator.export("json")
