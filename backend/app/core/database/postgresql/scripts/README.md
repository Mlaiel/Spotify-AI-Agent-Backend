# PostgreSQL Scripts – Spotify AI Agent (EN)

This folder contains industrial scripts for automation, audit, monitoring, backup, and business data seeding.

## Included Scripts

- **seed_users.py**: Add test users (AI profiles, roles, emails)
- **seed_analytics.py**: Generate simulated analytics data for AI
- **seed_music.py**: Add music tracks, artists, genres, durations, popularity
- **audit_log.py**: Analyze PostgreSQL logs, detect suspicious access, generate audit report
- **monitoring.py**: Check health, size, connections, locks, slow queries
- **auto_backup.py**: Secure automatic backup with rotation, logs, ready for cron/CI/CD
- **alert_slow_queries.py**: Detect and alert on slow queries (>5min)

## Usage

```bash
python3 seed_users.py
python3 seed_analytics.py
python3 seed_music.py
python3 audit_log.py
python3 monitoring.py
python3 auto_backup.py
python3 alert_slow_queries.py
```

> All scripts are ready for automation (CI/CD, cron, etc.) and follow security best practices (no hardcoded credentials, logs, audit).
