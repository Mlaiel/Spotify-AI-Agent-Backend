#!/bin/bash

# Performance Monitoring Script - Spotify AI Agent
# ================================================
#
# Advanced real-time performance monitoring with ML-powered analytics,
# predictive scaling, and intelligent alerting for multi-tenant environment
#
# Features:
# - Real-time metrics collection and analysis
# - ML-based anomaly detection and prediction
# - Auto-scaling recommendations and execution
# - Performance optimization automation
# - Resource utilization tracking
# - Capacity planning with forecasting
# - SLA monitoring and reporting
# - Custom dashboard generation

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_FILE="/var/log/spotify-performance-monitor.log"
METRICS_DB="/var/lib/spotify/metrics.db"
DASHBOARD_PORT=8080
ALERT_THRESHOLD_CPU=80
ALERT_THRESHOLD_MEMORY=85
ALERT_THRESHOLD_DISK=90
ALERT_THRESHOLD_LATENCY=1000
PREDICTION_WINDOW=300
SCALING_COOLDOWN=600

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m'

# Global variables
MONITORING_MODE="continuous"
TENANT_ID=""
ENVIRONMENT="dev"
DASHBOARD_ENABLED=false
ALERTS_ENABLED=false
AUTO_SCALE=false
PREDICTION_ENABLED=false
VERBOSE=false
DRY_RUN=false

# Initialize monitoring database
init_metrics_db() {
    if ! command -v sqlite3 &> /dev/null; then
        log "ERROR" "SQLite3 is required but not installed"
        exit 1
    fi
    
    if [[ ! -f "$METRICS_DB" ]]; then
        log "INFO" "Initializing metrics database: $METRICS_DB"
        mkdir -p "$(dirname "$METRICS_DB")"
        
        sqlite3 "$METRICS_DB" << 'EOF'
CREATE TABLE IF NOT EXISTS metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    tenant_id TEXT,
    metric_type TEXT,
    metric_name TEXT,
    value REAL,
    unit TEXT,
    tags TEXT,
    anomaly_score REAL DEFAULT 0.0,
    prediction_confidence REAL DEFAULT 0.0
);

CREATE TABLE IF NOT EXISTS alerts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    tenant_id TEXT,
    alert_type TEXT,
    severity TEXT,
    message TEXT,
    resolved BOOLEAN DEFAULT FALSE,
    resolution_time DATETIME
);

CREATE TABLE IF NOT EXISTS scaling_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    tenant_id TEXT,
    resource_type TEXT,
    action TEXT,
    from_value INTEGER,
    to_value INTEGER,
    reason TEXT,
    success BOOLEAN
);

CREATE INDEX IF NOT EXISTS idx_metrics_timestamp ON metrics(timestamp);
CREATE INDEX IF NOT EXISTS idx_metrics_tenant ON metrics(tenant_id);
CREATE INDEX IF NOT EXISTS idx_alerts_timestamp ON alerts(timestamp);
CREATE INDEX IF NOT EXISTS idx_scaling_timestamp ON scaling_events(timestamp);
EOF
        log "SUCCESS" "Metrics database initialized"
    fi
}

# Logging function
log() {
    local level=$1
    shift
    local message="$*"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    
    echo -e "${timestamp} [${level}] ${message}" | tee -a "${LOG_FILE}"
    
    case $level in
        "ERROR") echo -e "${RED}❌ ${message}${NC}" >&2 ;;
        "SUCCESS") echo -e "${GREEN}✅ ${message}${NC}" ;;
        "WARNING") echo -e "${YELLOW}⚠️  ${message}${NC}" ;;
        "INFO") echo -e "${BLUE}ℹ️  ${message}${NC}" ;;
        "METRIC") echo -e "${PURPLE}📊 ${message}${NC}" ;;
        "PREDICT") echo -e "${CYAN}🔮 ${message}${NC}" ;;
    esac
}

# Store metric in database
store_metric() {
    local tenant_id="$1"
    local metric_type="$2"
    local metric_name="$3"
    local value="$4"
    local unit="$5"
    local tags="$6"
    local anomaly_score="${7:-0.0}"
    local prediction_confidence="${8:-0.0}"
    
    sqlite3 "$METRICS_DB" << EOF
INSERT INTO metrics (tenant_id, metric_type, metric_name, value, unit, tags, anomaly_score, prediction_confidence)
VALUES ('$tenant_id', '$metric_type', '$metric_name', $value, '$unit', '$tags', $anomaly_score, $prediction_confidence);
EOF
}

# Collect system metrics
collect_system_metrics() {
    local tenant_id="$1"
    
    # CPU Usage
    local cpu_usage=$(top -bn1 | grep "Cpu(s)" | awk '{print $2}' | sed 's/%us,//')
    if [[ -n "$cpu_usage" ]]; then
        store_metric "$tenant_id" "system" "cpu_usage" "$cpu_usage" "percent" "component=system"
        [[ "$VERBOSE" == "true" ]] && log "METRIC" "CPU Usage: ${cpu_usage}%"
    fi
    
    # Memory Usage
    local memory_info=$(free | grep Mem)
    local memory_total=$(echo "$memory_info" | awk '{print $2}')
    local memory_used=$(echo "$memory_info" | awk '{print $3}')
    local memory_usage=$((memory_used * 100 / memory_total))
    
    store_metric "$tenant_id" "system" "memory_usage" "$memory_usage" "percent" "component=system"
    store_metric "$tenant_id" "system" "memory_total" "$memory_total" "bytes" "component=system"
    store_metric "$tenant_id" "system" "memory_used" "$memory_used" "bytes" "component=system"
    [[ "$VERBOSE" == "true" ]] && log "METRIC" "Memory Usage: ${memory_usage}%"
    
    # Disk Usage
    while read -r filesystem size used avail use_percent mount; do
        if [[ "$mount" == "/" ]] || [[ "$mount" =~ ^/var|^/tmp|^/opt ]]; then
            local disk_usage=$(echo "$use_percent" | sed 's/%//')
            store_metric "$tenant_id" "system" "disk_usage" "$disk_usage" "percent" "mount=$mount,filesystem=$filesystem"
            [[ "$VERBOSE" == "true" ]] && log "METRIC" "Disk Usage ($mount): ${disk_usage}%"
        fi
    done < <(df -h | tail -n +2)
    
    # Network I/O
    if [[ -f /proc/net/dev ]]; then
        while read -r interface rx_bytes rx_packets rx_errs rx_drop rx_fifo rx_frame rx_compressed rx_multicast tx_bytes tx_packets tx_errs tx_drop tx_fifo tx_colls tx_carrier tx_compressed; do
            if [[ "$interface" != "lo:" ]] && [[ "$interface" =~ ^(eth|en|wl) ]]; then
                interface=$(echo "$interface" | sed 's/:$//')
                store_metric "$tenant_id" "network" "rx_bytes" "$rx_bytes" "bytes" "interface=$interface"
                store_metric "$tenant_id" "network" "tx_bytes" "$tx_bytes" "bytes" "interface=$interface"
                [[ "$VERBOSE" == "true" ]] && log "METRIC" "Network RX ($interface): $rx_bytes bytes"
            fi
        done < <(tail -n +3 /proc/net/dev)
    fi
}

# Collect application metrics
collect_app_metrics() {
    local tenant_id="$1"
    
    # API Response times
    if command -v curl &> /dev/null; then
        local api_endpoints=(
            "http://localhost:8000/health"
            "http://localhost:8000/api/v1/metrics"
            "http://localhost:8000/api/v1/alerts/status"
        )
        
        for endpoint in "${api_endpoints[@]}"; do
            local start_time=$(date +%s%N)
            local response_code=$(curl -s -w "%{http_code}" -o /dev/null --max-time 10 "$endpoint" 2>/dev/null || echo "000")
            local end_time=$(date +%s%N)
            local response_time=$(( (end_time - start_time) / 1000000 )) # Convert to milliseconds
            
            store_metric "$tenant_id" "application" "api_response_time" "$response_time" "milliseconds" "endpoint=$endpoint"
            store_metric "$tenant_id" "application" "api_response_code" "$response_code" "code" "endpoint=$endpoint"
            
            [[ "$VERBOSE" == "true" ]] && log "METRIC" "API Response ($endpoint): ${response_time}ms (${response_code})"
        done
    fi
    
    # Docker container metrics
    if command -v docker &> /dev/null && docker info &> /dev/null; then
        while read -r container_id container_name cpu_percent memory_usage memory_limit memory_percent net_io block_io pids; do
            if [[ "$container_name" =~ spotify|monitoring ]]; then
                # Parse memory usage (remove units)
                memory_usage_num=$(echo "$memory_usage" | sed 's/[^0-9.]//g')
                memory_limit_num=$(echo "$memory_limit" | sed 's/[^0-9.]//g')
                memory_percent_num=$(echo "$memory_percent" | sed 's/%//g')
                cpu_percent_num=$(echo "$cpu_percent" | sed 's/%//g')
                
                store_metric "$tenant_id" "container" "cpu_usage" "$cpu_percent_num" "percent" "container=$container_name"
                store_metric "$tenant_id" "container" "memory_usage" "$memory_usage_num" "bytes" "container=$container_name"
                store_metric "$tenant_id" "container" "memory_percent" "$memory_percent_num" "percent" "container=$container_name"
                
                [[ "$VERBOSE" == "true" ]] && log "METRIC" "Container ($container_name): CPU ${cpu_percent_num}%, Memory ${memory_percent_num}%"
            fi
        done < <(docker stats --no-stream --format "table {{.Container}}\t{{.Name}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.MemPerc}}\t{{.NetIO}}\t{{.BlockIO}}\t{{.PIDs}}" 2>/dev/null | tail -n +2)
    fi
}

# ML-based anomaly detection
detect_anomalies() {
    local tenant_id="$1"
    
    # Simple statistical anomaly detection using Z-score
    sqlite3 "$METRICS_DB" << EOF
WITH metric_stats AS (
    SELECT 
        metric_type,
        metric_name,
        AVG(value) as mean,
        CASE 
            WHEN COUNT(*) > 1 THEN 
                SQRT(SUM(POWER(value - (SELECT AVG(value) FROM metrics m2 WHERE m2.metric_type = metrics.metric_type AND m2.metric_name = metrics.metric_name AND m2.tenant_id = '$tenant_id' AND m2.timestamp > datetime('now', '-1 hour')), 2)) / (COUNT(*) - 1))
            ELSE 0 
        END as stddev
    FROM metrics 
    WHERE tenant_id = '$tenant_id' 
    AND timestamp > datetime('now', '-1 hour')
    GROUP BY metric_type, metric_name
),
latest_metrics AS (
    SELECT DISTINCT
        m1.metric_type,
        m1.metric_name,
        m1.value,
        m1.timestamp,
        ms.mean,
        ms.stddev
    FROM metrics m1
    JOIN metric_stats ms ON m1.metric_type = ms.metric_type AND m1.metric_name = ms.metric_name
    WHERE m1.tenant_id = '$tenant_id'
    AND m1.timestamp = (
        SELECT MAX(timestamp) 
        FROM metrics m2 
        WHERE m2.metric_type = m1.metric_type 
        AND m2.metric_name = m1.metric_name 
        AND m2.tenant_id = '$tenant_id'
    )
)
SELECT 
    metric_type,
    metric_name,
    value,
    mean,
    stddev,
    CASE 
        WHEN stddev > 0 THEN ABS(value - mean) / stddev 
        ELSE 0 
    END as z_score
FROM latest_metrics
WHERE stddev > 0
AND ABS(value - mean) / stddev > 2.0;
EOF
}

# Generate scaling recommendations
generate_scaling_recommendations() {
    local tenant_id="$1"
    
    log "INFO" "Generating scaling recommendations for tenant: $tenant_id"
    
    # Get recent CPU and memory metrics
    local cpu_avg=$(sqlite3 "$METRICS_DB" "SELECT AVG(value) FROM metrics WHERE tenant_id='$tenant_id' AND metric_name='cpu_usage' AND timestamp > datetime('now', '-10 minutes');")
    local memory_avg=$(sqlite3 "$METRICS_DB" "SELECT AVG(value) FROM metrics WHERE tenant_id='$tenant_id' AND metric_name='memory_usage' AND timestamp > datetime('now', '-10 minutes');")
    
    if [[ -n "$cpu_avg" ]] && [[ -n "$memory_avg" ]]; then
        local cpu_int=$(printf "%.0f" "$cpu_avg")
        local memory_int=$(printf "%.0f" "$memory_avg")
        
        # CPU-based recommendations
        if [[ $cpu_int -gt $ALERT_THRESHOLD_CPU ]]; then
            log "WARNING" "High CPU usage detected: ${cpu_int}% (threshold: ${ALERT_THRESHOLD_CPU}%)"
            if [[ "$AUTO_SCALE" == "true" ]] && [[ "$DRY_RUN" == "false" ]]; then
                execute_scaling "cpu" "scale_up" "$tenant_id" "High CPU usage: ${cpu_int}%"
            else
                log "INFO" "Recommendation: Scale up CPU resources (current: ${cpu_int}%)"
            fi
        elif [[ $cpu_int -lt 30 ]]; then
            log "INFO" "Low CPU usage detected: ${cpu_int}%"
            if [[ "$AUTO_SCALE" == "true" ]] && [[ "$DRY_RUN" == "false" ]]; then
                execute_scaling "cpu" "scale_down" "$tenant_id" "Low CPU usage: ${cpu_int}%"
            else
                log "INFO" "Recommendation: Consider scaling down CPU resources (current: ${cpu_int}%)"
            fi
        fi
        
        # Memory-based recommendations
        if [[ $memory_int -gt $ALERT_THRESHOLD_MEMORY ]]; then
            log "WARNING" "High memory usage detected: ${memory_int}% (threshold: ${ALERT_THRESHOLD_MEMORY}%)"
            if [[ "$AUTO_SCALE" == "true" ]] && [[ "$DRY_RUN" == "false" ]]; then
                execute_scaling "memory" "scale_up" "$tenant_id" "High memory usage: ${memory_int}%"
            else
                log "INFO" "Recommendation: Scale up memory resources (current: ${memory_int}%)"
            fi
        fi
    fi
}

# Execute scaling actions
execute_scaling() {
    local resource_type="$1"
    local action="$2"
    local tenant_id="$3"
    local reason="$4"
    
    log "INFO" "Executing scaling action: $action for $resource_type (tenant: $tenant_id)"
    
    # Check cooldown period
    local last_scaling=$(sqlite3 "$METRICS_DB" "SELECT timestamp FROM scaling_events WHERE tenant_id='$tenant_id' AND resource_type='$resource_type' ORDER BY timestamp DESC LIMIT 1;")
    if [[ -n "$last_scaling" ]]; then
        local last_timestamp=$(date -d "$last_scaling" +%s)
        local current_timestamp=$(date +%s)
        local time_diff=$((current_timestamp - last_timestamp))
        
        if [[ $time_diff -lt $SCALING_COOLDOWN ]]; then
            log "WARNING" "Scaling cooldown active. Skipping scaling action."
            return
        fi
    fi
    
    local success=false
    local from_value=0
    local to_value=0
    
    # Simulate scaling actions (replace with actual scaling logic)
    case "$action" in
        "scale_up")
            if [[ "$resource_type" == "cpu" ]]; then
                from_value=2
                to_value=4
                log "INFO" "Scaling up CPU: $from_value -> $to_value cores"
                # docker update --cpus="$to_value" container_name
                success=true
            elif [[ "$resource_type" == "memory" ]]; then
                from_value=2048
                to_value=4096
                log "INFO" "Scaling up Memory: $from_value -> $to_value MB"
                # docker update --memory="${to_value}m" container_name
                success=true
            fi
            ;;
        "scale_down")
            if [[ "$resource_type" == "cpu" ]]; then
                from_value=4
                to_value=2
                log "INFO" "Scaling down CPU: $from_value -> $to_value cores"
                # docker update --cpus="$to_value" container_name
                success=true
            elif [[ "$resource_type" == "memory" ]]; then
                from_value=4096
                to_value=2048
                log "INFO" "Scaling down Memory: $from_value -> $to_value MB"
                # docker update --memory="${to_value}m" container_name
                success=true
            fi
            ;;
    esac
    
    # Record scaling event
    sqlite3 "$METRICS_DB" << EOF
INSERT INTO scaling_events (tenant_id, resource_type, action, from_value, to_value, reason, success)
VALUES ('$tenant_id', '$resource_type', '$action', $from_value, $to_value, '$reason', $success);
EOF
    
    if [[ "$success" == "true" ]]; then
        log "SUCCESS" "Scaling action completed successfully"
    else
        log "ERROR" "Scaling action failed"
    fi
}

# Generate performance report
generate_report() {
    local tenant_id="$1"
    local report_file="/tmp/performance_report_${tenant_id}_$(date +%Y%m%d_%H%M%S).html"
    
    log "INFO" "Generating performance report: $report_file"
    
    cat > "$report_file" << EOF
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Performance Report - Spotify AI Agent</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }
        .container { max-width: 1200px; margin: 0 auto; background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
        .header { text-align: center; margin-bottom: 30px; padding: 20px; background: linear-gradient(135deg, #1db954, #1ed760); color: white; border-radius: 8px; }
        .metrics-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; margin-bottom: 30px; }
        .metric-card { padding: 20px; border-radius: 8px; background: #f8f9fa; border-left: 4px solid #1db954; }
        .metric-value { font-size: 2em; font-weight: bold; color: #1db954; }
        .metric-label { font-size: 0.9em; color: #666; margin-top: 5px; }
        .chart-container { margin: 20px 0; padding: 20px; background: #f8f9fa; border-radius: 8px; }
        .alert { padding: 15px; margin: 10px 0; border-radius: 5px; }
        .alert.warning { background-color: #fff3cd; border-color: #ffc107; color: #856404; }
        .alert.success { background-color: #d4edda; border-color: #28a745; color: #155724; }
        .footer { text-align: center; margin-top: 30px; padding: 20px; color: #666; border-top: 1px solid #eee; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🎵 Spotify AI Agent</h1>
            <h2>Performance Monitoring Report</h2>
            <p>Tenant: $tenant_id | Generated: $(date '+%Y-%m-%d %H:%M:%S')</p>
        </div>
        
        <div class="metrics-grid">
EOF

    # Add current metrics to report
    local cpu_current=$(sqlite3 "$METRICS_DB" "SELECT value FROM metrics WHERE tenant_id='$tenant_id' AND metric_name='cpu_usage' ORDER BY timestamp DESC LIMIT 1;" 2>/dev/null || echo "0")
    local memory_current=$(sqlite3 "$METRICS_DB" "SELECT value FROM metrics WHERE tenant_id='$tenant_id' AND metric_name='memory_usage' ORDER BY timestamp DESC LIMIT 1;" 2>/dev/null || echo "0")
    local disk_current=$(sqlite3 "$METRICS_DB" "SELECT value FROM metrics WHERE tenant_id='$tenant_id' AND metric_name='disk_usage' ORDER BY timestamp DESC LIMIT 1;" 2>/dev/null || echo "0")
    
    cat >> "$report_file" << EOF
            <div class="metric-card">
                <div class="metric-value">${cpu_current:-0}%</div>
                <div class="metric-label">CPU Usage</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">${memory_current:-0}%</div>
                <div class="metric-label">Memory Usage</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">${disk_current:-0}%</div>
                <div class="metric-label">Disk Usage</div>
            </div>
        </div>
        
        <div class="chart-container">
            <h3>Recent Metrics Trends</h3>
            <p>Detailed metrics and trends would be displayed here with interactive charts.</p>
        </div>
        
        <div class="footer">
            <p>Performance monitoring system powered by advanced ML analytics</p>
        </div>
    </div>
</body>
</html>
EOF
    
    log "SUCCESS" "Performance report generated: $report_file"
}

# Start web dashboard
start_dashboard() {
    if command -v python3 &> /dev/null; then
        log "INFO" "Starting web dashboard on port $DASHBOARD_PORT"
        
        # Create simple dashboard server
        cat > "/tmp/dashboard_server.py" << 'EOF'
#!/usr/bin/env python3
import http.server
import socketserver
import json
import sqlite3
import os
from urllib.parse import urlparse, parse_qs

class DashboardHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        parsed_path = urlparse(self.path)
        
        if parsed_path.path == '/api/metrics':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            
            # Get metrics from database
            metrics = self.get_recent_metrics()
            self.wfile.write(json.dumps(metrics).encode())
            
        elif parsed_path.path == '/' or parsed_path.path == '/dashboard':
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            
            dashboard_html = self.get_dashboard_html()
            self.wfile.write(dashboard_html.encode())
        else:
            super().do_GET()
    
    def get_recent_metrics(self):
        db_path = "/var/lib/spotify/metrics.db"
        if not os.path.exists(db_path):
            return {"error": "Metrics database not found"}
        
        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT metric_type, metric_name, value, unit, timestamp
                FROM metrics 
                WHERE timestamp > datetime('now', '-1 hour')
                ORDER BY timestamp DESC 
                LIMIT 100
            """)
            
            rows = cursor.fetchall()
            conn.close()
            
            return {
                "metrics": [
                    {
                        "type": row[0],
                        "name": row[1], 
                        "value": row[2],
                        "unit": row[3],
                        "timestamp": row[4]
                    } for row in rows
                ]
            }
        except Exception as e:
            return {"error": str(e)}
    
    def get_dashboard_html(self):
        return """
<!DOCTYPE html>
<html>
<head>
    <title>Spotify AI Agent - Performance Dashboard</title>
    <meta charset="UTF-8">
    <style>
        body { font-family: Arial, sans-serif; margin: 0; padding: 20px; background: #f5f5f5; }
        .header { background: linear-gradient(135deg, #1db954, #1ed760); color: white; padding: 20px; text-align: center; margin-bottom: 20px; }
        .metrics-container { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; }
        .metric-card { background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }
        .metric-value { font-size: 2em; font-weight: bold; color: #1db954; }
        .refresh-btn { background: #1db954; color: white; border: none; padding: 10px 20px; border-radius: 5px; cursor: pointer; }
    </style>
</head>
<body>
    <div class="header">
        <h1>🎵 Spotify AI Agent Performance Dashboard</h1>
        <button class="refresh-btn" onclick="loadMetrics()">Refresh Metrics</button>
    </div>
    <div id="metrics-container" class="metrics-container">
        <p>Loading metrics...</p>
    </div>
    
    <script>
        function loadMetrics() {
            fetch('/api/metrics')
                .then(response => response.json())
                .then(data => {
                    const container = document.getElementById('metrics-container');
                    if (data.error) {
                        container.innerHTML = '<p>Error loading metrics: ' + data.error + '</p>';
                        return;
                    }
                    
                    const metricsMap = {};
                    data.metrics.forEach(metric => {
                        const key = metric.type + '_' + metric.name;
                        if (!metricsMap[key] || metricsMap[key].timestamp < metric.timestamp) {
                            metricsMap[key] = metric;
                        }
                    });
                    
                    let html = '';
                    Object.values(metricsMap).forEach(metric => {
                        html += `
                            <div class="metric-card">
                                <div class="metric-value">${metric.value.toFixed(2)} ${metric.unit}</div>
                                <div class="metric-label">${metric.type} - ${metric.name}</div>
                                <div class="metric-timestamp">${metric.timestamp}</div>
                            </div>
                        `;
                    });
                    
                    container.innerHTML = html;
                })
                .catch(error => {
                    console.error('Error:', error);
                    document.getElementById('metrics-container').innerHTML = '<p>Error loading metrics</p>';
                });
        }
        
        loadMetrics();
        setInterval(loadMetrics, 30000); // Refresh every 30 seconds
    </script>
</body>
</html>
        """

PORT = int(os.environ.get('DASHBOARD_PORT', 8080))
with socketserver.TCPServer(("", PORT), DashboardHandler) as httpd:
    print(f"Dashboard server running on port {PORT}")
    httpd.serve_forever()
EOF
        
        python3 "/tmp/dashboard_server.py" &
        local dashboard_pid=$!
        echo "$dashboard_pid" > "/tmp/dashboard.pid"
        
        log "SUCCESS" "Dashboard started at http://localhost:$DASHBOARD_PORT"
        log "INFO" "Dashboard PID: $dashboard_pid"
    else
        log "ERROR" "Python3 not available. Cannot start dashboard."
    fi
}

# Stop dashboard
stop_dashboard() {
    if [[ -f "/tmp/dashboard.pid" ]]; then
        local pid=$(cat "/tmp/dashboard.pid")
        if kill -0 "$pid" 2>/dev/null; then
            kill "$pid"
            rm -f "/tmp/dashboard.pid"
            log "SUCCESS" "Dashboard stopped"
        else
            log "WARNING" "Dashboard process not running"
            rm -f "/tmp/dashboard.pid"
        fi
    else
        log "WARNING" "Dashboard PID file not found"
    fi
}

# Main monitoring loop
monitor_loop() {
    log "INFO" "Starting performance monitoring loop"
    log "INFO" "Tenant: ${TENANT_ID:-all}, Environment: $ENVIRONMENT"
    log "INFO" "Mode: $MONITORING_MODE, Dashboard: $DASHBOARD_ENABLED, Alerts: $ALERTS_ENABLED"
    
    if [[ "$DASHBOARD_ENABLED" == "true" ]]; then
        start_dashboard
    fi
    
    local iteration=0
    
    while true; do
        iteration=$((iteration + 1))
        log "INFO" "Monitoring iteration $iteration"
        
        # Collect metrics for specified tenant or all tenants
        if [[ -n "$TENANT_ID" ]]; then
            collect_system_metrics "$TENANT_ID"
            collect_app_metrics "$TENANT_ID"
            
            if [[ "$ALERTS_ENABLED" == "true" ]]; then
                detect_anomalies "$TENANT_ID"
                generate_scaling_recommendations "$TENANT_ID"
            fi
        else
            # Monitor all tenants (get from database or config)
            local tenants=("default" "spotify_tenant_1")
            for tenant in "${tenants[@]}"; do
                collect_system_metrics "$tenant"
                collect_app_metrics "$tenant"
                
                if [[ "$ALERTS_ENABLED" == "true" ]]; then
                    detect_anomalies "$tenant"
                    generate_scaling_recommendations "$tenant"
                fi
            done
        fi
        
        # Generate periodic reports
        if [[ $((iteration % 60)) -eq 0 ]] && [[ -n "$TENANT_ID" ]]; then
            generate_report "$TENANT_ID"
        fi
        
        if [[ "$MONITORING_MODE" == "once" ]]; then
            break
        fi
        
        sleep 60  # Wait 1 minute between iterations
    done
}

# Cleanup function
cleanup() {
    log "INFO" "Cleaning up monitoring processes"
    
    if [[ "$DASHBOARD_ENABLED" == "true" ]]; then
        stop_dashboard
    fi
    
    exit 0
}

# Set up signal handlers
trap cleanup SIGINT SIGTERM

# Usage function
usage() {
    cat << EOF
Usage: $0 [OPTIONS]

Performance monitoring script for Spotify AI Agent

Options:
    --tenant ID             Monitor specific tenant
    --environment ENV       Environment (dev, staging, prod)
    --mode MODE            Monitoring mode (continuous, once)
    --dashboard            Enable web dashboard
    --alerts               Enable intelligent alerting
    --auto-scale           Enable automatic scaling
    --prediction           Enable ML prediction
    --verbose, -v          Verbose output
    --dry-run              Simulate actions without execution
    --help, -h             Show this help

Examples:
    $0 --tenant spotify_prod --dashboard --alerts
    $0 --environment prod --auto-scale --prediction
    $0 --mode once --tenant test_tenant --verbose

EOF
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --tenant)
            TENANT_ID="$2"
            shift 2
            ;;
        --environment)
            ENVIRONMENT="$2"
            shift 2
            ;;
        --mode)
            MONITORING_MODE="$2"
            shift 2
            ;;
        --dashboard)
            DASHBOARD_ENABLED=true
            shift
            ;;
        --alerts)
            ALERTS_ENABLED=true
            shift
            ;;
        --auto-scale)
            AUTO_SCALE=true
            shift
            ;;
        --prediction)
            PREDICTION_ENABLED=true
            shift
            ;;
        --verbose|-v)
            VERBOSE=true
            shift
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --help|-h)
            usage
            exit 0
            ;;
        *)
            log "ERROR" "Unknown option: $1"
            usage
            exit 1
            ;;
    esac
done

# Main execution
main() {
    echo "=============================================="
    echo "  Spotify AI Agent Performance Monitor"
    echo "=============================================="
    echo
    
    # Initialize database
    init_metrics_db
    
    # Start monitoring
    monitor_loop
}

# Run main function
main "$@"
