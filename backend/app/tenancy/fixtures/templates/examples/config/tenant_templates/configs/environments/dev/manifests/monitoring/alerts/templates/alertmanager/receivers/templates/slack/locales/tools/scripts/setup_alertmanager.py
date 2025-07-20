#!/bin/bash
"""
Spotify AI Agent - Advanced Alertmanager Setup Script
=====================================================

Enterprise-grade Alertmanager configuration and deployment script.
Automatically configures multi-tenant alerting with:
- Advanced routing rules
- Slack integration
- High availability setup
- Security best practices
- Performance optimization

Author: Fahed Mlaiel (Lead Developer + AI Architect)
Team: Expert Development Team
"""

set -euo pipefail

# Configuration
ALERTMANAGER_VERSION="${ALERTMANAGER_VERSION:-0.25.0}"
ALERTMANAGER_PORT="${ALERTMANAGER_PORT:-9093}"
ALERTMANAGER_CONFIG_DIR="${ALERTMANAGER_CONFIG_DIR:-/etc/alertmanager}"
ALERTMANAGER_DATA_DIR="${ALERTMANAGER_DATA_DIR:-/var/lib/alertmanager}"
ALERTMANAGER_USER="${ALERTMANAGER_USER:-alertmanager}"
SLACK_WEBHOOK_URL="${SLACK_WEBHOOK_URL:-}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if running as root
check_root() {
    if [[ $EUID -ne 0 ]]; then
        log_error "This script must be run as root for system installation"
        exit 1
    fi
}

# Create alertmanager user and directories
create_user_and_dirs() {
    log_info "Creating alertmanager user and directories..."
    
    # Create user
    if ! id "$ALERTMANAGER_USER" &>/dev/null; then
        useradd --no-create-home --shell /bin/false "$ALERTMANAGER_USER"
        log_success "Created user: $ALERTMANAGER_USER"
    else
        log_info "User $ALERTMANAGER_USER already exists"
    fi
    
    # Create directories
    mkdir -p "$ALERTMANAGER_CONFIG_DIR"
    mkdir -p "$ALERTMANAGER_DATA_DIR"
    mkdir -p /var/log/alertmanager
    
    # Set ownership
    chown -R "$ALERTMANAGER_USER:$ALERTMANAGER_USER" "$ALERTMANAGER_CONFIG_DIR"
    chown -R "$ALERTMANAGER_USER:$ALERTMANAGER_USER" "$ALERTMANAGER_DATA_DIR"
    chown -R "$ALERTMANAGER_USER:$ALERTMANAGER_USER" /var/log/alertmanager
    
    log_success "Directories created and configured"
}

# Download and install Alertmanager
install_alertmanager() {
    log_info "Installing Alertmanager v$ALERTMANAGER_VERSION..."
    
    # Download
    cd /tmp
    wget "https://github.com/prometheus/alertmanager/releases/download/v$ALERTMANAGER_VERSION/alertmanager-$ALERTMANAGER_VERSION.linux-amd64.tar.gz"
    
    # Extract
    tar xzf "alertmanager-$ALERTMANAGER_VERSION.linux-amd64.tar.gz"
    
    # Install binaries
    cp "alertmanager-$ALERTMANAGER_VERSION.linux-amd64/alertmanager" /usr/local/bin/
    cp "alertmanager-$ALERTMANAGER_VERSION.linux-amd64/amtool" /usr/local/bin/
    
    # Set permissions
    chmod +x /usr/local/bin/alertmanager
    chmod +x /usr/local/bin/amtool
    
    # Cleanup
    rm -rf "alertmanager-$ALERTMANAGER_VERSION.linux-amd64"*
    
    log_success "Alertmanager installed successfully"
}

# Create main configuration file
create_main_config() {
    log_info "Creating main Alertmanager configuration..."
    
    cat > "$ALERTMANAGER_CONFIG_DIR/alertmanager.yml" << 'EOF'
global:
  # Global SMTP configuration
  smtp_smarthost: 'localhost:587'
  smtp_from: 'alertmanager@spotify-ai-agent.com'
  smtp_auth_username: 'alertmanager@spotify-ai-agent.com'
  smtp_auth_password: '${SMTP_PASSWORD}'
  
  # Global Slack configuration  
  slack_api_url: '${SLACK_WEBHOOK_URL}'
  
  # Resolve timeout
  resolve_timeout: 5m
  
  # HTTP configuration
  http_config:
    tls_config:
      insecure_skip_verify: false

# Templates for notifications
templates:
  - '/etc/alertmanager/templates/*.tmpl'

# Routing configuration
route:
  group_by: ['alertname', 'cluster', 'service']
  group_wait: 30s
  group_interval: 5m
  repeat_interval: 12h
  receiver: 'default-receiver'
  
  # Tenant-specific routing
  routes:
    # Critical alerts - immediate notification
    - match:
        severity: critical
      receiver: 'critical-alerts'
      group_wait: 0s
      repeat_interval: 5m
      
    # High severity alerts
    - match:
        severity: high
      receiver: 'high-severity-alerts'
      group_wait: 30s
      repeat_interval: 1h
      
    # Warning alerts
    - match:
        severity: warning
      receiver: 'warning-alerts'
      group_wait: 2m
      repeat_interval: 4h
      
    # Info alerts
    - match:
        severity: info
      receiver: 'info-alerts'
      group_wait: 5m
      repeat_interval: 24h
      
    # Tenant-specific routes
    - match:
        tenant_id: spotify-ai-tenant-001
      receiver: 'tenant-001-slack'
      routes:
        - match:
            severity: critical
          receiver: 'tenant-001-critical'
    
    # Infrastructure alerts
    - match:
        team: infrastructure
      receiver: 'infrastructure-team'
      
    # SRE team alerts
    - match:
        team: sre
      receiver: 'sre-team'

# Inhibition rules
inhibit_rules:
  # Inhibit warning alerts if critical alert is firing
  - source_match:
      severity: 'critical'
    target_match:
      severity: 'warning'
    equal: ['alertname', 'instance']
    
  # Inhibit info alerts if warning alert is firing  
  - source_match:
      severity: 'warning'
    target_match:
      severity: 'info'
    equal: ['alertname', 'instance']

# Receivers configuration
receivers:
  # Default receiver
  - name: 'default-receiver'
    slack_configs:
      - api_url: '${SLACK_WEBHOOK_URL}'
        channel: '#alerts-default'
        username: 'AlertManager'
        icon_emoji: ':warning:'
        title: 'Alert: {{ range .Alerts }}{{ .Annotations.summary }}{{ end }}'
        text: '{{ range .Alerts }}{{ .Annotations.description }}{{ end }}'
        
  # Critical alerts receiver
  - name: 'critical-alerts'
    slack_configs:
      - api_url: '${SLACK_WEBHOOK_URL}'
        channel: '#alerts-critical'
        username: 'AlertManager'
        icon_emoji: ':rotating_light:'
        title: ':rotating_light: CRITICAL ALERT'
        color: 'danger'
        text: |
          *Alert:* {{ range .Alerts }}{{ .Annotations.summary }}{{ end }}
          *Severity:* {{ range .Alerts }}{{ .Labels.severity }}{{ end }}
          *Tenant:* {{ range .Alerts }}{{ .Labels.tenant_id }}{{ end }}
          *Details:* {{ range .Alerts }}{{ .Annotations.description }}{{ end }}
          *Runbook:* {{ range .Alerts }}{{ .Annotations.runbook_url }}{{ end }}
        fields:
          - title: "Instance"
            value: "{{ range .Alerts }}{{ .Labels.instance }}{{ end }}"
            short: true
          - title: "Job"
            value: "{{ range .Alerts }}{{ .Labels.job }}{{ end }}"
            short: true
        actions:
          - type: button
            text: 'Acknowledge'
            url: '{{ template "slack.default.actionURL" . }}'
    
    # Also send email for critical alerts
    email_configs:
      - to: 'sre-team@spotify-ai-agent.com'
        subject: 'CRITICAL ALERT: {{ range .Alerts }}{{ .Annotations.summary }}{{ end }}'
        body: |
          CRITICAL ALERT DETECTED
          
          Alert: {{ range .Alerts }}{{ .Annotations.summary }}{{ end }}
          Severity: {{ range .Alerts }}{{ .Labels.severity }}{{ end }}
          Tenant: {{ range .Alerts }}{{ .Labels.tenant_id }}{{ end }}
          Instance: {{ range .Alerts }}{{ .Labels.instance }}{{ end }}
          
          Description:
          {{ range .Alerts }}{{ .Annotations.description }}{{ end }}
          
          Runbook: {{ range .Alerts }}{{ .Annotations.runbook_url }}{{ end }}
          
          Time: {{ range .Alerts }}{{ .StartsAt }}{{ end }}
          
  # High severity alerts
  - name: 'high-severity-alerts'
    slack_configs:
      - api_url: '${SLACK_WEBHOOK_URL}'
        channel: '#alerts-high'
        username: 'AlertManager'
        icon_emoji: ':exclamation:'
        title: ':exclamation: High Severity Alert'
        color: 'warning'
        text: |
          *Alert:* {{ range .Alerts }}{{ .Annotations.summary }}{{ end }}
          *Tenant:* {{ range .Alerts }}{{ .Labels.tenant_id }}{{ end }}
          *Details:* {{ range .Alerts }}{{ .Annotations.description }}{{ end }}
          
  # Warning alerts
  - name: 'warning-alerts'
    slack_configs:
      - api_url: '${SLACK_WEBHOOK_URL}'
        channel: '#alerts-warning'
        username: 'AlertManager'
        icon_emoji: ':warning:'
        title: ':warning: Warning Alert'
        color: 'warning'
        text: |
          *Alert:* {{ range .Alerts }}{{ .Annotations.summary }}{{ end }}
          *Tenant:* {{ range .Alerts }}{{ .Labels.tenant_id }}{{ end }}
          
  # Info alerts
  - name: 'info-alerts'
    slack_configs:
      - api_url: '${SLACK_WEBHOOK_URL}'
        channel: '#alerts-info'
        username: 'AlertManager'
        icon_emoji: ':information_source:'
        title: ':information_source: Info Alert'
        color: 'good'
        text: |
          *Alert:* {{ range .Alerts }}{{ .Annotations.summary }}{{ end }}
          
  # Tenant-specific receivers
  - name: 'tenant-001-slack'
    slack_configs:
      - api_url: '${SLACK_WEBHOOK_URL}'
        channel: '#tenant-001-alerts'
        username: 'Tenant001-AlertManager'
        icon_emoji: ':bell:'
        
  - name: 'tenant-001-critical'
    slack_configs:
      - api_url: '${SLACK_WEBHOOK_URL}'
        channel: '#tenant-001-critical'
        username: 'Tenant001-Critical'
        icon_emoji: ':rotating_light:'
        
  # Team-specific receivers
  - name: 'infrastructure-team'
    slack_configs:
      - api_url: '${SLACK_WEBHOOK_URL}'
        channel: '#team-infrastructure'
        username: 'InfraAlerts'
        icon_emoji: ':gear:'
        
  - name: 'sre-team'
    slack_configs:
      - api_url: '${SLACK_WEBHOOK_URL}'
        channel: '#team-sre'
        username: 'SREAlerts'
        icon_emoji: ':fire_engine:'

# Silence configuration - for maintenance windows
EOF

    log_success "Main configuration created"
}

# Create notification templates
create_templates() {
    log_info "Creating notification templates..."
    
    mkdir -p "$ALERTMANAGER_CONFIG_DIR/templates"
    
    # Slack template
    cat > "$ALERTMANAGER_CONFIG_DIR/templates/slack.tmpl" << 'EOF'
{{ define "__alert_severity_prefix" }}{{ if ne .Status "firing" }}:white_check_mark:{{ else }}{{ if eq .Labels.severity "critical" }}:rotating_light:{{ else if eq .Labels.severity "warning" }}:warning:{{ else if eq .Labels.severity "info" }}:information_source:{{ else }}:question:{{ end }}{{ end }}{{ end }}

{{ define "__alert_severity_prefix_title" }}{{ if ne .Status "firing" }}:white_check_mark:{{ else }}{{ if eq .Labels.severity "critical" }}:rotating_light:{{ else if eq .Labels.severity "warning" }}:warning:{{ else if eq .Labels.severity "info" }}:information_source:{{ else }}:question:{{ end }}{{ end }}{{ end }}

{{ define "slack.default.title" }}{{ template "__alert_severity_prefix_title" . }} {{ .GroupLabels.alertname }}{{ if .CommonLabels.cluster }} on {{ .CommonLabels.cluster }}{{ end }}{{ end }}

{{ define "slack.default.text" }}
{{ if ne .CommonAnnotations.summary "" }}*Summary:* {{ .CommonAnnotations.summary }}{{ end }}
{{ if ne .CommonAnnotations.description "" }}*Description:* {{ .CommonAnnotations.description }}{{ end }}
*Severity:* `{{ .CommonLabels.severity | toUpper }}`
{{ if ne .CommonLabels.tenant_id "" }}*Tenant:* `{{ .CommonLabels.tenant_id }}`{{ end }}
*Environment:* `{{ .CommonLabels.environment | default "unknown" }}`
{{ if .CommonLabels.instance }}*Instance:* `{{ .CommonLabels.instance }}`{{ end }}
{{ if .CommonLabels.job }}*Job:* `{{ .CommonLabels.job }}`{{ end }}

*Alerts Firing:* {{ .Alerts.Firing | len }}
{{ range .Alerts.Firing }}• {{ .Annotations.summary }}{{ if .Labels.instance }} on {{ .Labels.instance }}{{ end }}
{{ end }}
{{ if .Alerts.Resolved }}
*Alerts Resolved:* {{ .Alerts.Resolved | len }}
{{ range .Alerts.Resolved }}• {{ .Annotations.summary }}{{ if .Labels.instance }} on {{ .Labels.instance }}{{ end }}
{{ end }}{{ end }}
{{ if ne .CommonAnnotations.runbook_url "" }}
*Runbook:* {{ .CommonAnnotations.runbook_url }}{{ end }}
{{ end }}

{{ define "slack.default.actionURL" }}{{ if .CommonAnnotations.dashboard_url }}{{ .CommonAnnotations.dashboard_url }}{{ else }}{{ .ExternalURL }}{{ end }}{{ end }}

{{ define "slack.default.color" }}{{ if eq .Status "firing" }}{{ if eq .CommonLabels.severity "critical" }}danger{{ else if eq .CommonLabels.severity "warning" }}warning{{ else }}good{{ end }}{{ else }}good{{ end }}{{ end }}
EOF

    # Email template
    cat > "$ALERTMANAGER_CONFIG_DIR/templates/email.tmpl" << 'EOF'
{{ define "email.default.subject" }}[{{ .Status | toUpper }}] {{ .GroupLabels.alertname }}{{ if .CommonLabels.cluster }} - {{ .CommonLabels.cluster }}{{ end }}{{ end }}

{{ define "email.default.html" }}
<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<style>
body { font-family: Arial, sans-serif; margin: 0; padding: 20px; background-color: #f5f5f5; }
.container { max-width: 600px; margin: 0 auto; background-color: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
.header { padding: 10px 0; border-bottom: 2px solid #e0e0e0; margin-bottom: 20px; }
.alert-critical { border-left: 4px solid #d32f2f; }
.alert-warning { border-left: 4px solid #f57c00; }
.alert-info { border-left: 4px solid #1976d2; }
.alert-resolved { border-left: 4px solid #388e3c; }
.alert-item { margin: 10px 0; padding: 10px; background-color: #f9f9f9; border-radius: 4px; }
.meta { color: #666; font-size: 12px; margin-top: 20px; padding-top: 10px; border-top: 1px solid #e0e0e0; }
.button { display: inline-block; padding: 8px 16px; background-color: #1976d2; color: white; text-decoration: none; border-radius: 4px; margin: 5px 0; }
</style>
</head>
<body>
<div class="container">
  <div class="header">
    <h2>Alert Notification</h2>
  </div>
  
  {{ if eq .Status "firing" }}
  <div class="alert-{{ .CommonLabels.severity }}">
    <h3>🚨 Alert Firing</h3>
  {{ else }}
  <div class="alert-resolved">
    <h3>✅ Alert Resolved</h3>
  {{ end }}
    
    <p><strong>Alert Name:</strong> {{ .GroupLabels.alertname }}</p>
    <p><strong>Severity:</strong> {{ .CommonLabels.severity | toUpper }}</p>
    {{ if .CommonLabels.tenant_id }}<p><strong>Tenant:</strong> {{ .CommonLabels.tenant_id }}</p>{{ end }}
    {{ if .CommonLabels.cluster }}<p><strong>Cluster:</strong> {{ .CommonLabels.cluster }}</p>{{ end }}
    {{ if .CommonLabels.environment }}<p><strong>Environment:</strong> {{ .CommonLabels.environment }}</p>{{ end }}
    
    {{ if .CommonAnnotations.summary }}
    <p><strong>Summary:</strong> {{ .CommonAnnotations.summary }}</p>
    {{ end }}
    
    {{ if .CommonAnnotations.description }}
    <p><strong>Description:</strong> {{ .CommonAnnotations.description }}</p>
    {{ end }}
  </div>
  
  {{ if .Alerts.Firing }}
  <h4>Firing Alerts ({{ .Alerts.Firing | len }})</h4>
  {{ range .Alerts.Firing }}
  <div class="alert-item alert-{{ .Labels.severity }}">
    <p><strong>{{ .Annotations.summary }}</strong></p>
    {{ if .Labels.instance }}<p>Instance: {{ .Labels.instance }}</p>{{ end }}
    {{ if .Labels.job }}<p>Job: {{ .Labels.job }}</p>{{ end }}
    <p>Started: {{ .StartsAt.Format "2006-01-02 15:04:05 UTC" }}</p>
  </div>
  {{ end }}
  {{ end }}
  
  {{ if .Alerts.Resolved }}
  <h4>Resolved Alerts ({{ .Alerts.Resolved | len }})</h4>
  {{ range .Alerts.Resolved }}
  <div class="alert-item alert-resolved">
    <p><strong>{{ .Annotations.summary }}</strong></p>
    {{ if .Labels.instance }}<p>Instance: {{ .Labels.instance }}</p>{{ end }}
    <p>Resolved: {{ .EndsAt.Format "2006-01-02 15:04:05 UTC" }}</p>
  </div>
  {{ end }}
  {{ end }}
  
  {{ if .CommonAnnotations.runbook_url }}
  <p><a href="{{ .CommonAnnotations.runbook_url }}" class="button">View Runbook</a></p>
  {{ end }}
  
  {{ if .CommonAnnotations.dashboard_url }}
  <p><a href="{{ .CommonAnnotations.dashboard_url }}" class="button">View Dashboard</a></p>
  {{ end }}
  
  <div class="meta">
    <p>This alert was generated by Spotify AI Agent Monitoring System</p>
    <p>Alert Manager: <a href="{{ .ExternalURL }}">{{ .ExternalURL }}</a></p>
  </div>
</div>
</body>
</html>
{{ end }}
EOF

    log_success "Notification templates created"
}

# Create systemd service
create_systemd_service() {
    log_info "Creating systemd service..."
    
    cat > /etc/systemd/system/alertmanager.service << EOF
[Unit]
Description=Alertmanager
Wants=network-online.target
After=network-online.target

[Service]
User=$ALERTMANAGER_USER
Group=$ALERTMANAGER_USER
Type=simple
Restart=always
RestartSec=5
ExecStart=/usr/local/bin/alertmanager \\
    --config.file=$ALERTMANAGER_CONFIG_DIR/alertmanager.yml \\
    --storage.path=$ALERTMANAGER_DATA_DIR \\
    --web.listen-address=0.0.0.0:$ALERTMANAGER_PORT \\
    --web.external-url=http://localhost:$ALERTMANAGER_PORT \\
    --cluster.listen-address=0.0.0.0:9094 \\
    --log.level=info \\
    --log.format=json

StandardOutput=journal
StandardError=journal
SyslogIdentifier=alertmanager

# Security settings
NoNewPrivileges=yes
PrivateTmp=yes
ProtectHome=yes
ProtectSystem=strict
ReadWritePaths=$ALERTMANAGER_DATA_DIR
ProtectKernelTunables=yes
ProtectKernelModules=yes
ProtectControlGroups=yes

[Install]
WantedBy=multi-user.target
EOF

    # Reload systemd and enable service
    systemctl daemon-reload
    systemctl enable alertmanager
    
    log_success "Systemd service created and enabled"
}

# Create configuration validation script
create_validation_script() {
    log_info "Creating configuration validation script..."
    
    cat > "$ALERTMANAGER_CONFIG_DIR/validate-config.sh" << 'EOF'
#!/bin/bash
# Alertmanager configuration validation script

CONFIG_FILE="${1:-/etc/alertmanager/alertmanager.yml}"

echo "Validating Alertmanager configuration: $CONFIG_FILE"

# Check if config file exists
if [[ ! -f "$CONFIG_FILE" ]]; then
    echo "ERROR: Configuration file not found: $CONFIG_FILE"
    exit 1
fi

# Validate using amtool
if command -v amtool &> /dev/null; then
    echo "Running amtool validation..."
    if amtool config check "$CONFIG_FILE"; then
        echo "✅ Configuration is valid"
        exit 0
    else
        echo "❌ Configuration validation failed"
        exit 1
    fi
else
    echo "WARNING: amtool not found, skipping validation"
    echo "✅ Basic file check passed"
    exit 0
fi
EOF

    chmod +x "$ALERTMANAGER_CONFIG_DIR/validate-config.sh"
    
    log_success "Validation script created"
}

# Create backup and maintenance scripts
create_maintenance_scripts() {
    log_info "Creating maintenance scripts..."
    
    # Backup script
    cat > "$ALERTMANAGER_CONFIG_DIR/backup-config.sh" << 'EOF'
#!/bin/bash
# Alertmanager configuration backup script

BACKUP_DIR="/var/backups/alertmanager"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
CONFIG_DIR="/etc/alertmanager"

mkdir -p "$BACKUP_DIR"

echo "Creating backup: alertmanager_config_$TIMESTAMP.tar.gz"
tar -czf "$BACKUP_DIR/alertmanager_config_$TIMESTAMP.tar.gz" -C "$CONFIG_DIR" .

echo "Backup created successfully"

# Keep only last 10 backups
ls -t "$BACKUP_DIR"/alertmanager_config_*.tar.gz | tail -n +11 | xargs -r rm

echo "Old backups cleaned up"
EOF

    # Reload script
    cat > "$ALERTMANAGER_CONFIG_DIR/reload-config.sh" << 'EOF'
#!/bin/bash
# Alertmanager configuration reload script

CONFIG_FILE="/etc/alertmanager/alertmanager.yml"

echo "Validating configuration..."
if ! /etc/alertmanager/validate-config.sh "$CONFIG_FILE"; then
    echo "ERROR: Configuration validation failed, not reloading"
    exit 1
fi

echo "Reloading Alertmanager configuration..."
if curl -X POST http://localhost:9093/-/reload; then
    echo "✅ Configuration reloaded successfully"
else
    echo "❌ Failed to reload configuration"
    exit 1
fi
EOF

    # Health check script
    cat > "$ALERTMANAGER_CONFIG_DIR/health-check.sh" << 'EOF'
#!/bin/bash
# Alertmanager health check script

ALERTMANAGER_URL="http://localhost:9093"

echo "Checking Alertmanager health..."

# Check if service is running
if ! systemctl is-active --quiet alertmanager; then
    echo "❌ Alertmanager service is not running"
    exit 1
fi

# Check HTTP endpoint
if curl -f -s "$ALERTMANAGER_URL/-/healthy" > /dev/null; then
    echo "✅ Alertmanager is healthy"
else
    echo "❌ Alertmanager health check failed"
    exit 1
fi

# Check configuration
if curl -f -s "$ALERTMANAGER_URL/api/v1/status" > /dev/null; then
    echo "✅ Alertmanager API is responding"
else
    echo "❌ Alertmanager API is not responding"
    exit 1
fi

echo "✅ All health checks passed"
EOF

    # Make scripts executable
    chmod +x "$ALERTMANAGER_CONFIG_DIR/backup-config.sh"
    chmod +x "$ALERTMANAGER_CONFIG_DIR/reload-config.sh"
    chmod +x "$ALERTMANAGER_CONFIG_DIR/health-check.sh"
    
    log_success "Maintenance scripts created"
}

# Configure environment variables
configure_environment() {
    log_info "Configuring environment variables..."
    
    cat > "$ALERTMANAGER_CONFIG_DIR/environment" << EOF
# Alertmanager environment variables
ALERTMANAGER_CONFIG_FILE=$ALERTMANAGER_CONFIG_DIR/alertmanager.yml
ALERTMANAGER_DATA_DIR=$ALERTMANAGER_DATA_DIR
ALERTMANAGER_LOG_LEVEL=info
SMTP_PASSWORD=${SMTP_PASSWORD:-}
SLACK_WEBHOOK_URL=${SLACK_WEBHOOK_URL:-}
EOF

    # Update systemd service to use environment file
    cat >> /etc/systemd/system/alertmanager.service << EOF

# Environment
EnvironmentFile=-$ALERTMANAGER_CONFIG_DIR/environment
EOF

    systemctl daemon-reload
    
    log_success "Environment configuration completed"
}

# Validate configuration
validate_configuration() {
    log_info "Validating Alertmanager configuration..."
    
    if "$ALERTMANAGER_CONFIG_DIR/validate-config.sh"; then
        log_success "Configuration validation passed"
    else
        log_error "Configuration validation failed"
        exit 1
    fi
}

# Start and test service
start_and_test() {
    log_info "Starting and testing Alertmanager service..."
    
    # Start service
    systemctl start alertmanager
    
    # Wait a moment for startup
    sleep 5
    
    # Test health
    if "$ALERTMANAGER_CONFIG_DIR/health-check.sh"; then
        log_success "Alertmanager is running and healthy"
    else
        log_error "Alertmanager health check failed"
        systemctl status alertmanager
        exit 1
    fi
    
    # Display status
    systemctl status alertmanager --no-pager -l
}

# Create firewall rules (if UFW is available)
configure_firewall() {
    if command -v ufw &> /dev/null; then
        log_info "Configuring firewall rules..."
        ufw allow "$ALERTMANAGER_PORT/tcp" comment "Alertmanager"
        log_success "Firewall rules configured"
    else
        log_warning "UFW not found, skipping firewall configuration"
    fi
}

# Print setup summary
print_summary() {
    log_success "Alertmanager setup completed successfully!"
    echo
    echo "📊 Setup Summary:"
    echo "=================="
    echo "• Version: $ALERTMANAGER_VERSION"
    echo "• Port: $ALERTMANAGER_PORT"
    echo "• Config: $ALERTMANAGER_CONFIG_DIR"
    echo "• Data: $ALERTMANAGER_DATA_DIR"
    echo "• User: $ALERTMANAGER_USER"
    echo
    echo "🔗 Useful URLs:"
    echo "• Web UI: http://localhost:$ALERTMANAGER_PORT"
    echo "• Health: http://localhost:$ALERTMANAGER_PORT/-/healthy"
    echo "• API: http://localhost:$ALERTMANAGER_PORT/api/v1"
    echo
    echo "🛠️ Useful Commands:"
    echo "• Status: systemctl status alertmanager"
    echo "• Logs: journalctl -u alertmanager -f"
    echo "• Reload: $ALERTMANAGER_CONFIG_DIR/reload-config.sh"
    echo "• Health: $ALERTMANAGER_CONFIG_DIR/health-check.sh"
    echo "• Backup: $ALERTMANAGER_CONFIG_DIR/backup-config.sh"
    echo
    if [[ -z "$SLACK_WEBHOOK_URL" ]]; then
        log_warning "Remember to set SLACK_WEBHOOK_URL environment variable for Slack notifications"
    fi
}

# Main execution
main() {
    log_info "Starting Alertmanager setup..."
    
    check_root
    create_user_and_dirs
    install_alertmanager
    create_main_config
    create_templates
    create_systemd_service
    create_validation_script
    create_maintenance_scripts
    configure_environment
    validate_configuration
    configure_firewall
    start_and_test
    print_summary
}

# Run main function
main "$@"
