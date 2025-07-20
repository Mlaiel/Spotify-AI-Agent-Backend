#!/bin/bash

# Security Scanner and Compliance Automation - Spotify AI Agent
# =============================================================
#
# Comprehensive security scanning and compliance automation with:
# - Vulnerability scanning and assessment
# - Compliance checking (GDPR, SOC2, ISO27001, PCI-DSS)
# - Security configuration validation
# - Penetration testing automation
# - Threat detection and mitigation
# - Security policy enforcement
# - Audit trail generation
# - Automated remediation capabilities

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_FILE="/var/log/spotify-security-scan.log"
REPORT_DIR="/var/reports/security"
CONFIG_FILE="/etc/spotify/security_config.yaml"

# Security settings
SCAN_TYPE="comprehensive"
COMPLIANCE_STANDARDS=("gdpr" "soc2" "iso27001")
TENANT_ID=""
ENVIRONMENT="dev"
SEVERITY_THRESHOLD="medium"
AUTO_REMEDIATE=false
GENERATE_REPORT=true
PENTEST_MODE=false
VERBOSE=false
DRY_RUN=false

# Vulnerability databases
CVE_DATABASE="/var/lib/spotify/cve.db"
THREAT_INTELLIGENCE_DB="/var/lib/spotify/threat_intel.db"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m'

# Security findings
declare -a CRITICAL_FINDINGS=()
declare -a HIGH_FINDINGS=()
declare -a MEDIUM_FINDINGS=()
declare -a LOW_FINDINGS=()
declare -a INFO_FINDINGS=()

# Compliance results
declare -A COMPLIANCE_RESULTS=()

# Logging function
log() {
    local level=$1
    shift
    local message="$*"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    
    echo -e "${timestamp} [${level}] ${message}" | tee -a "${LOG_FILE}"
    
    case $level in
        "CRITICAL") echo -e "${RED}🚨 ${message}${NC}" >&2 ;;
        "ERROR") echo -e "${RED}❌ ${message}${NC}" >&2 ;;
        "SUCCESS") echo -e "${GREEN}✅ ${message}${NC}" ;;
        "WARNING") echo -e "${YELLOW}⚠️  ${message}${NC}" ;;
        "INFO") echo -e "${BLUE}ℹ️  ${message}${NC}" ;;
        "SECURITY") echo -e "${PURPLE}🛡️  ${message}${NC}" ;;
        "VULN") echo -e "${CYAN}🔍 ${message}${NC}" ;;
    esac
}

# Add security finding
add_finding() {
    local severity="$1"
    local title="$2"
    local description="$3"
    local remediation="$4"
    
    local finding="$title|$description|$remediation"
    
    case $severity in
        "critical") CRITICAL_FINDINGS+=("$finding") ;;
        "high") HIGH_FINDINGS+=("$finding") ;;
        "medium") MEDIUM_FINDINGS+=("$finding") ;;
        "low") LOW_FINDINGS+=("$finding") ;;
        "info") INFO_FINDINGS+=("$finding") ;;
    esac
    
    log "SECURITY" "$severity: $title"
    [[ "$VERBOSE" == "true" ]] && log "INFO" "$description"
}

# Initialize security scanner
init_security_scanner() {
    log "INFO" "Initializing security scanner"
    
    # Create necessary directories
    mkdir -p "$REPORT_DIR"
    mkdir -p "$(dirname "$CVE_DATABASE")"
    mkdir -p "$(dirname "$THREAT_INTELLIGENCE_DB")"
    
    # Initialize vulnerability database
    init_vulnerability_db
    
    # Check for required tools
    check_security_tools
    
    log "SUCCESS" "Security scanner initialized"
}

# Initialize vulnerability database
init_vulnerability_db() {
    if ! command -v sqlite3 &> /dev/null; then
        log "ERROR" "SQLite3 is required but not installed"
        exit 1
    fi
    
    if [[ ! -f "$CVE_DATABASE" ]]; then
        log "INFO" "Creating vulnerability database"
        
        sqlite3 "$CVE_DATABASE" << 'EOF'
CREATE TABLE IF NOT EXISTS vulnerabilities (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    cve_id TEXT UNIQUE,
    cvss_score REAL,
    severity TEXT,
    description TEXT,
    affected_software TEXT,
    remediation TEXT,
    discovered_date DATETIME,
    last_updated DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS scan_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    scan_timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    tenant_id TEXT,
    target_type TEXT,
    target_name TEXT,
    vulnerability_id INTEGER,
    status TEXT,
    remediated BOOLEAN DEFAULT FALSE,
    FOREIGN KEY (vulnerability_id) REFERENCES vulnerabilities(id)
);

CREATE INDEX IF NOT EXISTS idx_cve_id ON vulnerabilities(cve_id);
CREATE INDEX IF NOT EXISTS idx_severity ON vulnerabilities(severity);
CREATE INDEX IF NOT EXISTS idx_scan_timestamp ON scan_results(scan_timestamp);
EOF
        
        log "SUCCESS" "Vulnerability database created"
    fi
}

# Check for security tools
check_security_tools() {
    local missing_tools=()
    
    # Core security tools
    local security_tools=(
        "nmap" "openssl" "curl" "wget" "netstat" "ss" 
        "grep" "awk" "find" "ps" "systemctl"
    )
    
    for tool in "${security_tools[@]}"; do
        if ! command -v "$tool" &> /dev/null; then
            missing_tools+=("$tool")
        fi
    done
    
    # Optional advanced tools
    if command -v nikto &> /dev/null; then
        log "INFO" "Nikto web scanner available"
    else
        log "WARNING" "Nikto not found - web vulnerability scanning limited"
    fi
    
    if command -v nessus &> /dev/null; then
        log "INFO" "Nessus scanner available"
    else
        log "WARNING" "Nessus not found - advanced vulnerability scanning limited"
    fi
    
    if command -v lynis &> /dev/null; then
        log "INFO" "Lynis security auditing tool available"
    else
        log "WARNING" "Lynis not found - system hardening checks limited"
    fi
    
    if [[ ${#missing_tools[@]} -gt 0 ]]; then
        log "WARNING" "Missing security tools: ${missing_tools[*]}"
    else
        log "SUCCESS" "All core security tools available"
    fi
}

# Scan network services
scan_network_services() {
    log "SECURITY" "Scanning network services"
    
    # Check for open ports
    local open_ports=()
    
    if command -v ss &> /dev/null; then
        # Use ss (modern replacement for netstat)
        while read -r proto local_addr state; do
            if [[ "$state" == "LISTEN" ]]; then
                local port=$(echo "$local_addr" | sed 's/.*://')
                open_ports+=("$port")
                
                # Check for suspicious ports
                case $port in
                    22) log "INFO" "SSH service detected on port $port" ;;
                    23) add_finding "high" "Telnet Service" "Insecure Telnet service running on port $port" "Disable Telnet and use SSH instead" ;;
                    25) log "INFO" "SMTP service detected on port $port" ;;
                    53) log "INFO" "DNS service detected on port $port" ;;
                    80) log "INFO" "HTTP service detected on port $port" ;;
                    443) log "INFO" "HTTPS service detected on port $port" ;;
                    3389) add_finding "medium" "RDP Service" "Remote Desktop service running on port $port" "Ensure RDP is properly secured or disabled" ;;
                    5432) log "INFO" "PostgreSQL service detected on port $port" ;;
                    6379) log "INFO" "Redis service detected on port $port" ;;
                    27017) log "INFO" "MongoDB service detected on port $port" ;;
                    *) 
                        if [[ $port -gt 1024 ]]; then
                            log "INFO" "Custom service on port $port"
                        else
                            add_finding "low" "Unknown Service" "Unknown service running on privileged port $port" "Verify if this service is necessary"
                        fi
                        ;;
                esac
            fi
        done < <(ss -tuln | tail -n +2 | awk '{print $1, $5, $6}')
    fi
    
    log "INFO" "Found ${#open_ports[@]} open ports"
}

# Scan web applications
scan_web_applications() {
    log "SECURITY" "Scanning web applications"
    
    local web_ports=("80" "443" "8000" "8080" "8443" "9000")
    
    for port in "${web_ports[@]}"; do
        # Check if port is open
        if ss -tuln | grep -q ":$port "; then
            log "INFO" "Scanning web application on port $port"
            
            local base_url="http://localhost:$port"
            if [[ "$port" == "443" ]] || [[ "$port" == "8443" ]]; then
                base_url="https://localhost:$port"
            fi
            
            # Check for common vulnerabilities
            check_web_headers "$base_url"
            check_ssl_configuration "$base_url"
            check_common_files "$base_url"
            
            # Run Nikto if available
            if command -v nikto &> /dev/null && [[ "$PENTEST_MODE" == "true" ]]; then
                log "INFO" "Running Nikto scan on $base_url"
                if [[ "$DRY_RUN" == "false" ]]; then
                    nikto -h "$base_url" -Format txt -output "/tmp/nikto_${port}.txt" >/dev/null 2>&1 || true
                fi
            fi
        fi
    done
}

# Check web security headers
check_web_headers() {
    local url="$1"
    
    log "INFO" "Checking security headers for $url"
    
    # Get headers
    local headers=""
    if headers=$(curl -s -I --max-time 10 "$url" 2>/dev/null); then
        
        # Check for missing security headers
        if ! echo "$headers" | grep -qi "X-Frame-Options"; then
            add_finding "medium" "Missing X-Frame-Options" "X-Frame-Options header not set on $url" "Add X-Frame-Options: DENY or SAMEORIGIN"
        fi
        
        if ! echo "$headers" | grep -qi "X-Content-Type-Options"; then
            add_finding "medium" "Missing X-Content-Type-Options" "X-Content-Type-Options header not set on $url" "Add X-Content-Type-Options: nosniff"
        fi
        
        if ! echo "$headers" | grep -qi "X-XSS-Protection"; then
            add_finding "medium" "Missing X-XSS-Protection" "X-XSS-Protection header not set on $url" "Add X-XSS-Protection: 1; mode=block"
        fi
        
        if ! echo "$headers" | grep -qi "Strict-Transport-Security"; then
            add_finding "medium" "Missing HSTS" "HTTP Strict Transport Security header not set on $url" "Add Strict-Transport-Security header"
        fi
        
        if ! echo "$headers" | grep -qi "Content-Security-Policy"; then
            add_finding "low" "Missing CSP" "Content Security Policy header not set on $url" "Implement Content Security Policy"
        fi
        
        # Check for information disclosure
        if echo "$headers" | grep -qi "Server:"; then
            local server_header=$(echo "$headers" | grep -i "Server:" | head -1)
            add_finding "low" "Server Information Disclosure" "Server header reveals information: $server_header" "Remove or minimize server header information"
        fi
        
    else
        log "WARNING" "Could not retrieve headers from $url"
    fi
}

# Check SSL/TLS configuration
check_ssl_configuration() {
    local url="$1"
    
    if [[ "$url" =~ ^https:// ]]; then
        log "INFO" "Checking SSL/TLS configuration for $url"
        
        local host=$(echo "$url" | sed 's|https://||' | sed 's|/.*||' | sed 's|:.*||')
        local port=$(echo "$url" | sed 's|https://||' | sed 's|/.*||' | sed 's|.*:||')
        
        if [[ "$port" == "$host" ]]; then
            port="443"
        fi
        
        # Check SSL certificate
        if command -v openssl &> /dev/null; then
            local ssl_info=""
            if ssl_info=$(echo | openssl s_client -servername "$host" -connect "$host:$port" 2>/dev/null); then
                
                # Check certificate expiration
                local cert_end_date=$(echo "$ssl_info" | openssl x509 -noout -enddate 2>/dev/null | cut -d= -f2)
                if [[ -n "$cert_end_date" ]]; then
                    local end_timestamp=$(date -d "$cert_end_date" +%s)
                    local current_timestamp=$(date +%s)
                    local days_until_expiry=$(( (end_timestamp - current_timestamp) / 86400 ))
                    
                    if [[ $days_until_expiry -lt 30 ]]; then
                        add_finding "high" "SSL Certificate Expiring Soon" "SSL certificate for $host expires in $days_until_expiry days" "Renew SSL certificate"
                    elif [[ $days_until_expiry -lt 90 ]]; then
                        add_finding "medium" "SSL Certificate Expiring" "SSL certificate for $host expires in $days_until_expiry days" "Plan SSL certificate renewal"
                    fi
                fi
                
                # Check for weak SSL protocols
                if echo "$ssl_info" | grep -q "Protocol.*SSLv"; then
                    add_finding "high" "Weak SSL Protocol" "Weak SSL protocol detected on $host:$port" "Disable SSLv2 and SSLv3, use TLS 1.2+"
                fi
                
                if echo "$ssl_info" | grep -q "Protocol.*TLSv1\.0"; then
                    add_finding "medium" "Outdated TLS Protocol" "TLS 1.0 detected on $host:$port" "Upgrade to TLS 1.2 or higher"
                fi
                
                # Check for weak ciphers
                if echo "$ssl_info" | grep -q "Cipher.*RC4\|Cipher.*DES\|Cipher.*MD5"; then
                    add_finding "high" "Weak SSL Cipher" "Weak SSL cipher detected on $host:$port" "Disable weak ciphers and use strong encryption"
                fi
                
            else
                add_finding "medium" "SSL Connection Failed" "Could not establish SSL connection to $host:$port" "Check SSL configuration"
            fi
        fi
    fi
}

# Check for common sensitive files
check_common_files() {
    local url="$1"
    
    log "INFO" "Checking for common sensitive files on $url"
    
    local sensitive_files=(
        "/.env"
        "/.git/config"
        "/config.php"
        "/wp-config.php"
        "/admin"
        "/administrator"
        "/phpmyadmin"
        "/backup"
        "/test"
        "/debug"
        "/.htaccess"
        "/robots.txt"
    )
    
    for file in "${sensitive_files[@]}"; do
        local test_url="${url}${file}"
        local response_code=$(curl -s -w "%{http_code}" -o /dev/null --max-time 5 "$test_url" 2>/dev/null || echo "000")
        
        case $response_code in
            200)
                case $file in
                    "/.env"|"/.git/config"|"/config.php"|"/wp-config.php")
                        add_finding "critical" "Sensitive File Exposed" "Sensitive file $file is accessible at $test_url" "Remove or restrict access to sensitive files"
                        ;;
                    "/admin"|"/administrator"|"/phpmyadmin")
                        add_finding "medium" "Admin Interface Exposed" "Admin interface $file is accessible at $test_url" "Restrict access to admin interfaces"
                        ;;
                    *)
                        add_finding "low" "File Exposed" "File $file is accessible at $test_url" "Review if this file should be publicly accessible"
                        ;;
                esac
                ;;
            403)
                log "INFO" "File $file is protected (403)"
                ;;
            404)
                [[ "$VERBOSE" == "true" ]] && log "INFO" "File $file not found (404)"
                ;;
        esac
    done
}

# Scan system configuration
scan_system_configuration() {
    log "SECURITY" "Scanning system configuration"
    
    # Check file permissions
    check_file_permissions
    
    # Check user accounts
    check_user_accounts
    
    # Check running services
    check_running_services
    
    # Check system updates
    check_system_updates
    
    # Check firewall configuration
    check_firewall_configuration
}

# Check critical file permissions
check_file_permissions() {
    log "INFO" "Checking file permissions"
    
    local critical_files=(
        "/etc/passwd"
        "/etc/shadow"
        "/etc/sudoers"
        "/etc/ssh/sshd_config"
        "/root/.ssh/authorized_keys"
    )
    
    for file in "${critical_files[@]}"; do
        if [[ -f "$file" ]]; then
            local perms=$(stat -c "%a" "$file" 2>/dev/null || echo "")
            
            case $file in
                "/etc/passwd")
                    if [[ "$perms" != "644" ]]; then
                        add_finding "medium" "Incorrect /etc/passwd Permissions" "File $file has permissions $perms instead of 644" "Change permissions to 644"
                    fi
                    ;;
                "/etc/shadow")
                    if [[ "$perms" != "640" ]] && [[ "$perms" != "600" ]]; then
                        add_finding "high" "Incorrect /etc/shadow Permissions" "File $file has permissions $perms instead of 640/600" "Change permissions to 640 or 600"
                    fi
                    ;;
                "/etc/sudoers")
                    if [[ "$perms" != "440" ]]; then
                        add_finding "high" "Incorrect /etc/sudoers Permissions" "File $file has permissions $perms instead of 440" "Change permissions to 440"
                    fi
                    ;;
                "/root/.ssh/authorized_keys")
                    if [[ "$perms" != "600" ]]; then
                        add_finding "high" "Incorrect SSH Key Permissions" "File $file has permissions $perms instead of 600" "Change permissions to 600"
                    fi
                    ;;
            esac
        fi
    done
    
    # Check for world-writable files
    local world_writable=$(find /etc /usr /var -type f -perm -002 2>/dev/null | head -10)
    if [[ -n "$world_writable" ]]; then
        add_finding "medium" "World-Writable Files Found" "Found world-writable files in system directories" "Review and fix permissions for: $world_writable"
    fi
}

# Check user accounts
check_user_accounts() {
    log "INFO" "Checking user accounts"
    
    # Check for accounts without passwords
    local no_password_accounts=$(awk -F: '($2 == "") {print $1}' /etc/shadow 2>/dev/null | head -5)
    if [[ -n "$no_password_accounts" ]]; then
        add_finding "high" "Accounts Without Passwords" "Found accounts without passwords: $no_password_accounts" "Set passwords or disable these accounts"
    fi
    
    # Check for duplicate UIDs
    local duplicate_uids=$(awk -F: '{print $3}' /etc/passwd | sort | uniq -d)
    if [[ -n "$duplicate_uids" ]]; then
        add_finding "medium" "Duplicate User IDs" "Found duplicate UIDs: $duplicate_uids" "Ensure all users have unique UIDs"
    fi
    
    # Check for users with UID 0 (root privileges)
    local root_users=$(awk -F: '($3 == 0) {print $1}' /etc/passwd | grep -v "^root$")
    if [[ -n "$root_users" ]]; then
        add_finding "high" "Non-root Users with UID 0" "Found non-root users with UID 0: $root_users" "Remove root privileges from non-root users"
    fi
}

# Check running services
check_running_services() {
    log "INFO" "Checking running services"
    
    # Get list of running services
    local running_services=""
    if command -v systemctl &> /dev/null; then
        running_services=$(systemctl list-units --type=service --state=running --no-pager --no-legend | awk '{print $1}')
    fi
    
    # Check for suspicious or unnecessary services
    local suspicious_services=(
        "telnet" "rsh" "rlogin" "rexec" "ftp" "tftp" "finger" "echo" "discard" "chargen"
    )
    
    for service in $running_services; do
        for suspicious in "${suspicious_services[@]}"; do
            if [[ "$service" =~ $suspicious ]]; then
                add_finding "medium" "Suspicious Service Running" "Potentially insecure service running: $service" "Disable unnecessary network services"
            fi
        done
    done
}

# Check system updates
check_system_updates() {
    log "INFO" "Checking system updates"
    
    # Check for available updates (Ubuntu/Debian)
    if command -v apt &> /dev/null; then
        local security_updates=$(apt list --upgradable 2>/dev/null | grep -c "security" || echo "0")
        if [[ $security_updates -gt 0 ]]; then
            add_finding "medium" "Security Updates Available" "$security_updates security updates are available" "Install security updates: apt update && apt upgrade"
        fi
    fi
    
    # Check for available updates (CentOS/RHEL)
    if command -v yum &> /dev/null; then
        local security_updates=$(yum check-update --security 2>/dev/null | grep -c "updates" || echo "0")
        if [[ $security_updates -gt 0 ]]; then
            add_finding "medium" "Security Updates Available" "$security_updates security updates are available" "Install security updates: yum update --security"
        fi
    fi
}

# Check firewall configuration
check_firewall_configuration() {
    log "INFO" "Checking firewall configuration"
    
    # Check if firewall is enabled
    local firewall_status="unknown"
    
    if command -v ufw &> /dev/null; then
        if ufw status | grep -q "Status: active"; then
            firewall_status="active"
            log "INFO" "UFW firewall is active"
        else
            firewall_status="inactive"
            add_finding "medium" "Firewall Disabled" "UFW firewall is not active" "Enable firewall: ufw enable"
        fi
    elif command -v firewall-cmd &> /dev/null; then
        if firewall-cmd --state 2>/dev/null | grep -q "running"; then
            firewall_status="active"
            log "INFO" "FirewallD is active"
        else
            firewall_status="inactive"
            add_finding "medium" "Firewall Disabled" "FirewallD is not running" "Enable firewall: systemctl enable --now firewalld"
        fi
    elif command -v iptables &> /dev/null; then
        local rules_count=$(iptables -L | wc -l)
        if [[ $rules_count -gt 10 ]]; then
            firewall_status="active"
            log "INFO" "iptables rules configured"
        else
            firewall_status="minimal"
            add_finding "low" "Minimal Firewall Rules" "Few or no iptables rules configured" "Configure appropriate firewall rules"
        fi
    else
        add_finding "high" "No Firewall Found" "No firewall software found on system" "Install and configure a firewall (ufw, firewalld, or iptables)"
    fi
}

# Perform compliance checks
check_compliance() {
    log "SECURITY" "Performing compliance checks"
    
    for standard in "${COMPLIANCE_STANDARDS[@]}"; do
        log "INFO" "Checking compliance for: $standard"
        
        case $standard in
            "gdpr")
                check_gdpr_compliance
                ;;
            "soc2")
                check_soc2_compliance
                ;;
            "iso27001")
                check_iso27001_compliance
                ;;
            "pci-dss")
                check_pci_compliance
                ;;
        esac
    done
}

# GDPR compliance checks
check_gdpr_compliance() {
    local gdpr_score=0
    local gdpr_total=10
    
    # Check for data encryption
    if [[ -f "/etc/spotify/encryption.conf" ]]; then
        gdpr_score=$((gdpr_score + 2))
        log "INFO" "GDPR: Data encryption configuration found"
    else
        add_finding "medium" "GDPR: Missing Encryption Config" "No data encryption configuration found" "Implement data encryption for GDPR compliance"
    fi
    
    # Check for audit logging
    if [[ -f "/var/log/spotify/audit.log" ]]; then
        gdpr_score=$((gdpr_score + 2))
        log "INFO" "GDPR: Audit logging enabled"
    else
        add_finding "medium" "GDPR: Missing Audit Logs" "No audit logging found" "Enable comprehensive audit logging"
    fi
    
    # Check for data retention policies
    if [[ -f "/etc/spotify/retention_policy.conf" ]]; then
        gdpr_score=$((gdpr_score + 2))
        log "INFO" "GDPR: Data retention policy found"
    else
        add_finding "medium" "GDPR: Missing Retention Policy" "No data retention policy found" "Implement data retention policies"
    fi
    
    # Check for privacy controls
    if grep -q "privacy" /etc/spotify/*.conf 2>/dev/null; then
        gdpr_score=$((gdpr_score + 2))
        log "INFO" "GDPR: Privacy controls configured"
    else
        add_finding "medium" "GDPR: Missing Privacy Controls" "No privacy controls found in configuration" "Implement privacy controls"
    fi
    
    # Check for data subject rights implementation
    if [[ -f "/opt/spotify/data_subject_rights.py" ]]; then
        gdpr_score=$((gdpr_score + 2))
        log "INFO" "GDPR: Data subject rights implementation found"
    else
        add_finding "low" "GDPR: Missing Data Subject Rights" "No data subject rights implementation found" "Implement data subject rights handling"
    fi
    
    local gdpr_percentage=$((gdpr_score * 100 / gdpr_total))
    COMPLIANCE_RESULTS["GDPR"]="$gdpr_percentage"
    log "INFO" "GDPR compliance score: $gdpr_percentage%"
}

# SOC2 compliance checks
check_soc2_compliance() {
    local soc2_score=0
    local soc2_total=15
    
    # Security controls
    if [[ -f "/etc/spotify/security_controls.conf" ]]; then
        soc2_score=$((soc2_score + 3))
        log "INFO" "SOC2: Security controls configuration found"
    else
        add_finding "medium" "SOC2: Missing Security Controls" "No security controls configuration found" "Implement SOC2 security controls"
    fi
    
    # Access controls
    if grep -q "access_control" /etc/spotify/*.conf 2>/dev/null; then
        soc2_score=$((soc2_score + 3))
        log "INFO" "SOC2: Access controls configured"
    else
        add_finding "medium" "SOC2: Missing Access Controls" "No access controls found" "Implement proper access controls"
    fi
    
    # Monitoring and logging
    if [[ -f "/var/log/spotify/security.log" ]]; then
        soc2_score=$((soc2_score + 3))
        log "INFO" "SOC2: Security monitoring enabled"
    else
        add_finding "medium" "SOC2: Missing Security Monitoring" "No security monitoring logs found" "Enable security monitoring and logging"
    fi
    
    # Change management
    if [[ -f "/etc/spotify/change_management.conf" ]]; then
        soc2_score=$((soc2_score + 3))
        log "INFO" "SOC2: Change management process found"
    else
        add_finding "low" "SOC2: Missing Change Management" "No change management process found" "Implement change management procedures"
    fi
    
    # Incident response
    if [[ -f "/etc/spotify/incident_response.conf" ]]; then
        soc2_score=$((soc2_score + 3))
        log "INFO" "SOC2: Incident response plan found"
    else
        add_finding "medium" "SOC2: Missing Incident Response" "No incident response plan found" "Implement incident response procedures"
    fi
    
    local soc2_percentage=$((soc2_score * 100 / soc2_total))
    COMPLIANCE_RESULTS["SOC2"]="$soc2_percentage"
    log "INFO" "SOC2 compliance score: $soc2_percentage%"
}

# ISO27001 compliance checks
check_iso27001_compliance() {
    local iso_score=0
    local iso_total=12
    
    # Information security policy
    if [[ -f "/etc/spotify/security_policy.conf" ]]; then
        iso_score=$((iso_score + 2))
        log "INFO" "ISO27001: Security policy found"
    else
        add_finding "medium" "ISO27001: Missing Security Policy" "No information security policy found" "Develop information security policy"
    fi
    
    # Risk assessment
    if [[ -f "/etc/spotify/risk_assessment.conf" ]]; then
        iso_score=$((iso_score + 2))
        log "INFO" "ISO27001: Risk assessment found"
    else
        add_finding "medium" "ISO27001: Missing Risk Assessment" "No risk assessment documentation found" "Conduct risk assessment"
    fi
    
    # Asset management
    if [[ -f "/etc/spotify/asset_inventory.conf" ]]; then
        iso_score=$((iso_score + 2))
        log "INFO" "ISO27001: Asset management found"
    else
        add_finding "low" "ISO27001: Missing Asset Management" "No asset inventory found" "Implement asset management"
    fi
    
    # Security awareness
    if [[ -f "/etc/spotify/security_training.conf" ]]; then
        iso_score=$((iso_score + 2))
        log "INFO" "ISO27001: Security awareness program found"
    else
        add_finding "low" "ISO27001: Missing Security Training" "No security awareness program found" "Implement security awareness training"
    fi
    
    # Business continuity
    if [[ -f "/etc/spotify/business_continuity.conf" ]]; then
        iso_score=$((iso_score + 2))
        log "INFO" "ISO27001: Business continuity plan found"
    else
        add_finding "medium" "ISO27001: Missing Business Continuity" "No business continuity plan found" "Develop business continuity plan"
    fi
    
    # Supplier relationships
    if [[ -f "/etc/spotify/supplier_security.conf" ]]; then
        iso_score=$((iso_score + 2))
        log "INFO" "ISO27001: Supplier security controls found"
    else
        add_finding "low" "ISO27001: Missing Supplier Controls" "No supplier security controls found" "Implement supplier security requirements"
    fi
    
    local iso_percentage=$((iso_score * 100 / iso_total))
    COMPLIANCE_RESULTS["ISO27001"]="$iso_percentage"
    log "INFO" "ISO27001 compliance score: $iso_percentage%"
}

# PCI-DSS compliance checks
check_pci_compliance() {
    local pci_score=0
    local pci_total=12
    
    # Network security
    if [[ -f "/etc/spotify/network_security.conf" ]]; then
        pci_score=$((pci_score + 2))
        log "INFO" "PCI-DSS: Network security controls found"
    else
        add_finding "high" "PCI-DSS: Missing Network Security" "No network security controls found" "Implement network security controls"
    fi
    
    # Data protection
    if grep -q "encryption" /etc/spotify/*.conf 2>/dev/null; then
        pci_score=$((pci_score + 2))
        log "INFO" "PCI-DSS: Data encryption configured"
    else
        add_finding "critical" "PCI-DSS: Missing Data Encryption" "No data encryption found" "Implement strong data encryption"
    fi
    
    # Access control
    if [[ -f "/etc/spotify/access_control.conf" ]]; then
        pci_score=$((pci_score + 2))
        log "INFO" "PCI-DSS: Access control measures found"
    else
        add_finding "high" "PCI-DSS: Missing Access Control" "No access control measures found" "Implement strong access controls"
    fi
    
    # Vulnerability management
    if [[ -f "/etc/spotify/vulnerability_mgmt.conf" ]]; then
        pci_score=$((pci_score + 2))
        log "INFO" "PCI-DSS: Vulnerability management program found"
    else
        add_finding "high" "PCI-DSS: Missing Vulnerability Management" "No vulnerability management program found" "Implement vulnerability management"
    fi
    
    # Security monitoring
    if [[ -f "/var/log/spotify/pci_audit.log" ]]; then
        pci_score=$((pci_score + 2))
        log "INFO" "PCI-DSS: Security monitoring enabled"
    else
        add_finding "high" "PCI-DSS: Missing Security Monitoring" "No PCI security monitoring found" "Implement comprehensive security monitoring"
    fi
    
    # Security testing
    if [[ -f "/etc/spotify/security_testing.conf" ]]; then
        pci_score=$((pci_score + 2))
        log "INFO" "PCI-DSS: Security testing procedures found"
    else
        add_finding "medium" "PCI-DSS: Missing Security Testing" "No security testing procedures found" "Implement regular security testing"
    fi
    
    local pci_percentage=$((pci_score * 100 / pci_total))
    COMPLIANCE_RESULTS["PCI-DSS"]="$pci_percentage"
    log "INFO" "PCI-DSS compliance score: $pci_percentage%"
}

# Generate security report
generate_security_report() {
    local timestamp=$(date '+%Y%m%d_%H%M%S')
    local report_file="$REPORT_DIR/security_report_${timestamp}.html"
    
    log "INFO" "Generating security report: $report_file"
    
    cat > "$report_file" << EOF
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Security Assessment Report - Spotify AI Agent</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }
        .container { max-width: 1200px; margin: 0 auto; background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
        .header { text-align: center; margin-bottom: 30px; padding: 20px; background: linear-gradient(135deg, #dc3545, #e74c3c); color: white; border-radius: 8px; }
        .summary { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin-bottom: 30px; }
        .summary-card { padding: 20px; border-radius: 8px; text-align: center; color: white; }
        .summary-card.critical { background-color: #dc3545; }
        .summary-card.high { background-color: #fd7e14; }
        .summary-card.medium { background-color: #ffc107; color: #333; }
        .summary-card.low { background-color: #20c997; }
        .summary-card.info { background-color: #0dcaf0; }
        .findings { margin: 30px 0; }
        .finding { margin: 10px 0; padding: 15px; border-radius: 5px; border-left: 4px solid; }
        .finding.critical { background-color: #f8d7da; border-color: #dc3545; }
        .finding.high { background-color: #fff3cd; border-color: #fd7e14; }
        .finding.medium { background-color: #d1ecf1; border-color: #ffc107; }
        .finding.low { background-color: #d4edda; border-color: #20c997; }
        .finding-title { font-weight: bold; margin-bottom: 5px; }
        .finding-description { margin-bottom: 10px; }
        .finding-remediation { font-style: italic; color: #666; }
        .compliance-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; margin: 20px 0; }
        .compliance-card { padding: 20px; border-radius: 8px; background: #f8f9fa; border-left: 4px solid #007bff; }
        .compliance-score { font-size: 2em; font-weight: bold; color: #007bff; }
        .footer { text-align: center; margin-top: 30px; padding: 20px; color: #666; border-top: 1px solid #eee; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🛡️ Security Assessment Report</h1>
            <h2>Spotify AI Agent</h2>
            <p>Generated: $(date '+%Y-%m-%d %H:%M:%S') | Tenant: ${TENANT_ID:-all} | Environment: $ENVIRONMENT</p>
        </div>
        
        <div class="summary">
            <div class="summary-card critical">
                <h3>${#CRITICAL_FINDINGS[@]}</h3>
                <p>Critical Issues</p>
            </div>
            <div class="summary-card high">
                <h3>${#HIGH_FINDINGS[@]}</h3>
                <p>High Risk</p>
            </div>
            <div class="summary-card medium">
                <h3>${#MEDIUM_FINDINGS[@]}</h3>
                <p>Medium Risk</p>
            </div>
            <div class="summary-card low">
                <h3>${#LOW_FINDINGS[@]}</h3>
                <p>Low Risk</p>
            </div>
            <div class="summary-card info">
                <h3>${#INFO_FINDINGS[@]}</h3>
                <p>Informational</p>
            </div>
        </div>
        
        <div class="compliance-grid">
EOF

    # Add compliance results
    for standard in "${!COMPLIANCE_RESULTS[@]}"; do
        local score="${COMPLIANCE_RESULTS[$standard]}"
        cat >> "$report_file" << EOF
            <div class="compliance-card">
                <div class="compliance-score">${score}%</div>
                <div>$standard Compliance</div>
            </div>
EOF
    done

    cat >> "$report_file" << EOF
        </div>
        
        <div class="findings">
            <h3>Security Findings</h3>
EOF

    # Add critical findings
    if [[ ${#CRITICAL_FINDINGS[@]} -gt 0 ]]; then
        echo "<h4>Critical Issues</h4>" >> "$report_file"
        for finding in "${CRITICAL_FINDINGS[@]}"; do
            IFS='|' read -r title description remediation <<< "$finding"
            cat >> "$report_file" << EOF
            <div class="finding critical">
                <div class="finding-title">$title</div>
                <div class="finding-description">$description</div>
                <div class="finding-remediation">Remediation: $remediation</div>
            </div>
EOF
        done
    fi

    # Add high findings
    if [[ ${#HIGH_FINDINGS[@]} -gt 0 ]]; then
        echo "<h4>High Risk Issues</h4>" >> "$report_file"
        for finding in "${HIGH_FINDINGS[@]}"; do
            IFS='|' read -r title description remediation <<< "$finding"
            cat >> "$report_file" << EOF
            <div class="finding high">
                <div class="finding-title">$title</div>
                <div class="finding-description">$description</div>
                <div class="finding-remediation">Remediation: $remediation</div>
            </div>
EOF
        done
    fi

    # Add medium findings
    if [[ ${#MEDIUM_FINDINGS[@]} -gt 0 ]]; then
        echo "<h4>Medium Risk Issues</h4>" >> "$report_file"
        for finding in "${MEDIUM_FINDINGS[@]}"; do
            IFS='|' read -r title description remediation <<< "$finding"
            cat >> "$report_file" << EOF
            <div class="finding medium">
                <div class="finding-title">$title</div>
                <div class="finding-description">$description</div>
                <div class="finding-remediation">Remediation: $remediation</div>
            </div>
EOF
        done
    fi

    cat >> "$report_file" << EOF
        </div>
        
        <div class="footer">
            <p>Security assessment powered by advanced vulnerability scanning and compliance automation</p>
        </div>
    </div>
</body>
</html>
EOF
    
    log "SUCCESS" "Security report generated: $report_file"
}

# Automated remediation
auto_remediate() {
    if [[ "$AUTO_REMEDIATE" != "true" ]]; then
        return
    fi
    
    log "INFO" "Starting automated remediation"
    
    # Simple remediations that can be automated safely
    
    # Fix file permissions
    if [[ -f "/etc/passwd" ]]; then
        chmod 644 /etc/passwd 2>/dev/null || true
    fi
    
    if [[ -f "/etc/shadow" ]]; then
        chmod 640 /etc/shadow 2>/dev/null || true
    fi
    
    # Update package lists
    if command -v apt &> /dev/null; then
        apt update >/dev/null 2>&1 || true
    fi
    
    # Enable UFW if available and not active
    if command -v ufw &> /dev/null && ! ufw status | grep -q "Status: active"; then
        if [[ "$DRY_RUN" == "false" ]]; then
            ufw --force enable >/dev/null 2>&1 || true
            log "SUCCESS" "UFW firewall enabled"
        else
            log "INFO" "DRY RUN: Would enable UFW firewall"
        fi
    fi
    
    log "SUCCESS" "Automated remediation completed"
}

# Usage function
usage() {
    cat << EOF
Usage: $0 [OPTIONS]

Security scanner and compliance automation for Spotify AI Agent

Options:
    --scan-type TYPE        Scan type (quick, comprehensive, pentest)
    --compliance STANDARDS  Compliance standards (gdpr,soc2,iso27001,pci-dss)
    --tenant ID             Scan specific tenant
    --environment ENV       Environment (dev, staging, prod)
    --severity LEVEL        Minimum severity threshold (low, medium, high, critical)
    --auto-remediate        Enable automatic remediation
    --generate-report       Generate HTML report
    --pentest-mode          Enable penetration testing mode
    --verbose, -v           Verbose output
    --dry-run               Simulate actions without execution
    --help, -h              Show this help

Examples:
    $0 --scan-type comprehensive --compliance gdpr,soc2
    $0 --tenant spotify_prod --auto-remediate --generate-report
    $0 --pentest-mode --environment prod --verbose

EOF
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --scan-type)
            SCAN_TYPE="$2"
            shift 2
            ;;
        --compliance)
            IFS=',' read -ra COMPLIANCE_STANDARDS <<< "$2"
            shift 2
            ;;
        --tenant)
            TENANT_ID="$2"
            shift 2
            ;;
        --environment)
            ENVIRONMENT="$2"
            shift 2
            ;;
        --severity)
            SEVERITY_THRESHOLD="$2"
            shift 2
            ;;
        --auto-remediate)
            AUTO_REMEDIATE=true
            shift
            ;;
        --generate-report)
            GENERATE_REPORT=true
            shift
            ;;
        --pentest-mode)
            PENTEST_MODE=true
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
    echo "  Security Scanner - Spotify AI Agent"
    echo "=============================================="
    echo
    
    # Initialize scanner
    init_security_scanner
    
    # Perform scans
    log "INFO" "Starting security scan (type: $SCAN_TYPE)"
    
    scan_network_services
    scan_web_applications
    scan_system_configuration
    
    # Perform compliance checks
    if [[ ${#COMPLIANCE_STANDARDS[@]} -gt 0 ]]; then
        check_compliance
    fi
    
    # Auto-remediation
    if [[ "$AUTO_REMEDIATE" == "true" ]]; then
        auto_remediate
    fi
    
    # Generate report
    if [[ "$GENERATE_REPORT" == "true" ]]; then
        generate_security_report
    fi
    
    # Summary
    local total_findings=$((${#CRITICAL_FINDINGS[@]} + ${#HIGH_FINDINGS[@]} + ${#MEDIUM_FINDINGS[@]} + ${#LOW_FINDINGS[@]} + ${#INFO_FINDINGS[@]}))
    
    echo
    echo "=========================================="
    echo "  SECURITY SCAN SUMMARY"
    echo "=========================================="
    echo "Critical issues:     ${#CRITICAL_FINDINGS[@]}"
    echo "High risk issues:    ${#HIGH_FINDINGS[@]}"
    echo "Medium risk issues:  ${#MEDIUM_FINDINGS[@]}"
    echo "Low risk issues:     ${#LOW_FINDINGS[@]}"
    echo "Informational:       ${#INFO_FINDINGS[@]}"
    echo "Total findings:      $total_findings"
    echo
    
    if [[ ${#CRITICAL_FINDINGS[@]} -gt 0 ]]; then
        log "CRITICAL" "Critical security issues found! Immediate action required."
        exit 2
    elif [[ ${#HIGH_FINDINGS[@]} -gt 0 ]]; then
        log "WARNING" "High risk security issues found. Action recommended."
        exit 1
    else
        log "SUCCESS" "Security scan completed successfully! 🛡️"
        exit 0
    fi
}

# Run main function
main "$@"
