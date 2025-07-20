import logging
import requests

class SentryAlerter:
    """
    Envoi d'alertes critiques à Sentry via HTTP API.
    """
    def __init__(self, sentry_dsn: str):
        self.sentry_dsn = sentry_dsn

    def alert(self, message: str, level: str = "error"):
        # Pour la prod, utiliser le SDK Sentry officiel (sentry_sdk)
        logging.error(f"SENTRY ALERT [{level}]: {message}")
        # Exemple d'appel API (à adapter selon Sentry)
        # requests.post(self.sentry_dsn, json={"message": message, "level": level})

class PrometheusAlerter:
    """
    Envoi d'alertes à Prometheus Alertmanager (webhook).
    """
    def __init__(self, alertmanager_url: str):
        self.alertmanager_url = alertmanager_url

    def alert(self, message: str, severity: str = "critical"):
        payload = {
            "alerts": [{
                "labels": {"alertname": "WebSocketCritical", "severity": severity},
                "annotations": {"summary": message}
            }]
        }
        logging.error(f"PROM ALERT [{severity}]: {message}")
        # requests.post(self.alertmanager_url, json=payload)

# Exemple d'utilisation :
# sentry = SentryAlerter(sentry_dsn="...")
# sentry.alert("Erreur critique WebSocket")
# prom = PrometheusAlerter(alertmanager_url="...")
# prom.alert("Trop d'erreurs WebSocket")
