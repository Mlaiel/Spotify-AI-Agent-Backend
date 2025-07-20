from unittest.mock import Mock

import pytest
from . import load_env_file, validate_env_vars, check_no_weak_secrets, check_format
from pathlib import Path

ENV_FILE = Path(__file__).parent / ".env.production"
REQUIRED_VARS = [
    "SECRET_KEY", "ALLOWED_HOSTS", "DEBUG", "LOG_LEVEL",
    "POSTGRES_HOST", "POSTGRES_PORT", "POSTGRES_DB", "POSTGRES_USER", "POSTGRES_PASSWORD",
    "REDIS_URL", "CACHE_TTL",
    "CELERY_BROKER_URL", "CELERY_RESULT_BACKEND", "CELERY_TASK_ALWAYS_EAGER",
    "SENTRY_DSN", "PROMETHEUS_METRICS_PORT", "OTEL_EXPORTER_OTLP_ENDPOINT", "OTEL_SERVICE_NAME",
    "SPOTIFY_CLIENT_ID", "SPOTIFY_CLIENT_SECRET", "SPOTIFY_REDIRECT_URI", "EMAIL_HOST"
]

def test_env_production_exists():
    assert ENV_FILE.exists(), f"{ENV_FILE} does not exist"

def test_env_production_required_vars():
    env = load_env_file(ENV_FILE)
    missing = validate_env_vars(env, REQUIRED_VARS)
    assert not missing, f"Missing variables: {missing}"

def test_env_production_debug_and_log_level():
    env = load_env_file(ENV_FILE)
    assert env["DEBUG"].lower() == "false"
    assert env["LOG_LEVEL"] == "INFO"

def test_env_production_postgres_config():
    env = load_env_file(ENV_FILE)
    assert env["POSTGRES_HOST"] == "prod-db-host"
    assert env["POSTGRES_PORT"] == "5432"
    assert env["POSTGRES_DB"] == "spotify_ai_agent"
    assert env["POSTGRES_USER"] == "prod_user"
    assert env["POSTGRES_PASSWORD"] == "prod-db-password"

def test_env_production_spotify_config():
    env = load_env_file(ENV_FILE)
    assert env["SPOTIFY_CLIENT_ID"].startswith("prod-")
    assert env["SPOTIFY_CLIENT_SECRET"].startswith("prod-")
    assert env["SPOTIFY_REDIRECT_URI"].startswith("https://")

def test_env_production_email():
    env = load_env_file(ENV_FILE)
    assert env["EMAIL_HOST"] == "smtp.sendgrid.net"

def test_env_production_no_plain_secrets():
    env = load_env_file(ENV_FILE)
    check_no_weak_secrets(env)

def test_env_production_format():
    env = load_env_file(ENV_FILE)
    check_format(env, ["POSTGRES_USER", "POSTGRES_DB"])
