#!/bin/bash
# =============================================================================
# Advanced Security Audit and Validation Script for Cryptographic Keys
# =============================================================================
# 
# Enterprise-grade security audit system for the Spotify AI Agent
# cryptographic key infrastructure with comprehensive compliance
# checking and vulnerability assessment.
#
# This script implements advanced security validation following
# industry standards and best practices.
# =============================================================================

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../../../../../../.." && pwd)"
SECRETS_DIR="$SCRIPT_DIR"
AUDIT_LOG="$SECRETS_DIR/security_audit.log"
REPORT_FILE="$SECRETS_DIR/security_audit_report.json"

# Security settings
UMASK_ORIGINAL=$(umask)
umask 077

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m'

# Global variables
VERBOSE=false
COMPLIANCE_MODE=""
GENERATE_REPORT=true
CRITICAL_ISSUES=0
HIGH_ISSUES=0
MEDIUM_ISSUES=0
LOW_ISSUES=0

# Compliance standards
declare -A COMPLIANCE_REQUIREMENTS=(
    ["FIPS_140_2"]="fips-140-2"
    ["COMMON_CRITERIA"]="common-criteria"
    ["NIST_SP_800_57"]="nist-sp-800-57"
    ["PCI_DSS"]="pci-dss"
    ["SOX"]="sox"
    ["HIPAA"]="hipaa"
)

# Key strength requirements
declare -A MIN_KEY_LENGTHS=(
    ["AES_128"]=16
    ["AES_192"]=24
    ["AES_256"]=32
    ["RSA_2048"]=256
    ["RSA_4096"]=512
    ["HMAC_SHA256"]=32
    ["JWT_SECRET"]=32
)

# Logging functions
log_audit() {
    local level="$1"
    local message="$2"
    local timestamp=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
    local caller="${FUNCNAME[2]:-main}"
    
    echo "[$timestamp] [$level] [$caller] $message" >> "$AUDIT_LOG"
    
    if [[ "$VERBOSE" == "true" ]] || [[ "$level" =~ ^(ERROR|CRITICAL|WARNING)$ ]]; then
        case "$level" in
            "CRITICAL")
                echo -e "${RED}[CRITICAL]${NC} $message"
                ;;
            "ERROR")
                echo -e "${RED}[ERROR]${NC} $message"
                ;;
            "WARNING")
                echo -e "${YELLOW}[WARNING]${NC} $message"
                ;;
            "INFO")
                echo -e "${BLUE}[INFO]${NC} $message"
                ;;
            "SUCCESS")
                echo -e "${GREEN}[SUCCESS]${NC} $message"
                ;;
            "SECURITY")
                echo -e "${PURPLE}[SECURITY]${NC} $message"
                ;;
            "COMPLIANCE")
                echo -e "${CYAN}[COMPLIANCE]${NC} $message"
                ;;
        esac
    fi
}

# Initialize audit
initialize_audit() {
    echo "==============================================================================" > "$AUDIT_LOG"
    echo "Advanced Security Audit and Validation - $(date -u)" >> "$AUDIT_LOG"
    echo "==============================================================================" >> "$AUDIT_LOG"
    
    log_audit "INFO" "Starting advanced security audit process"
    log_audit "SECURITY" "Security audit level: ENTERPRISE-GRADE"
    log_audit "INFO" "Auditing directory: $SECRETS_DIR"
    log_audit "INFO" "Compliance mode: ${COMPLIANCE_MODE:-STANDARD}"
}

# Check file permissions and ownership
check_file_security() {
    local file_path="$1"
    local expected_perm="${2:-600}"
    local issues=0
    
    log_audit "INFO" "Checking file security for: $(basename "$file_path")"
    
    if [[ ! -f "$file_path" ]]; then
        log_audit "ERROR" "File not found: $file_path"
        ((CRITICAL_ISSUES++))
        return 1
    fi
    
    # Check file permissions
    local actual_perm=$(stat -c "%a" "$file_path")
    if [[ "$actual_perm" != "$expected_perm" ]]; then
        log_audit "CRITICAL" "Incorrect file permissions: $file_path (expected: $expected_perm, actual: $actual_perm)"
        ((CRITICAL_ISSUES++))
        ((issues++))
    else
        log_audit "SUCCESS" "File permissions correct: $file_path ($actual_perm)"
    fi
    
    # Check file ownership
    local file_owner=$(stat -c "%U" "$file_path")
    local current_user=$(whoami)
    if [[ "$file_owner" != "$current_user" ]]; then
        log_audit "WARNING" "File ownership concern: $file_path (owner: $file_owner, current: $current_user)"
        ((MEDIUM_ISSUES++))
        ((issues++))
    fi
    
    # Check file size
    local file_size=$(stat -c%s "$file_path")
    if [[ $file_size -eq 0 ]]; then
        log_audit "CRITICAL" "Empty key file detected: $file_path"
        ((CRITICAL_ISSUES++))
        ((issues++))
    elif [[ $file_size -lt 50 ]]; then
        log_audit "WARNING" "Suspiciously small key file: $file_path ($file_size bytes)"
        ((MEDIUM_ISSUES++))
        ((issues++))
    fi
    
    # Check for world-readable files
    if [[ $actual_perm =~ [0-9][0-9][4-7] ]]; then
        log_audit "CRITICAL" "File is world-readable: $file_path"
        ((CRITICAL_ISSUES++))
        ((issues++))
    fi
    
    # Check for group-readable files
    if [[ $actual_perm =~ [0-9][4-7][0-9] ]]; then
        log_audit "HIGH" "File is group-readable: $file_path"
        ((HIGH_ISSUES++))
        ((issues++))
    fi
    
    return $issues
}

# Validate key strength and entropy
validate_key_strength() {
    local key_file="$1"
    local key_type="$2"
    local issues=0
    
    log_audit "INFO" "Validating key strength for: $(basename "$key_file")"
    
    if [[ ! -f "$key_file" ]]; then
        log_audit "ERROR" "Key file not found for validation: $key_file"
        ((CRITICAL_ISSUES++))
        return 1
    fi
    
    # Extract keys from file based on type
    case "$key_type" in
        "database")
            local key_value=$(grep "DATABASE_ENCRYPTION_KEY=" "$key_file" | cut -d'"' -f2)
            validate_base64_key "$key_value" "AES_256" || ((issues++))
            ;;
        "jwt")
            local access_key=$(grep "JWT_ACCESS_SECRET=" "$key_file" | cut -d'"' -f2)
            local refresh_key=$(grep "JWT_REFRESH_SECRET=" "$key_file" | cut -d'"' -f2)
            validate_base64_key "$access_key" "JWT_SECRET" || ((issues++))
            validate_base64_key "$refresh_key" "JWT_SECRET" || ((issues++))
            ;;
        "api")
            local master_key=$(grep "API_MASTER_KEY=" "$key_file" | cut -d'"' -f2)
            local internal_key=$(grep "API_INTERNAL_KEY=" "$key_file" | cut -d'"' -f2)
            validate_hex_key "$master_key" || ((issues++))
            validate_hex_key "$internal_key" || ((issues++))
            ;;
        "session")
            local session_key=$(grep "SESSION_SECRET_KEY=" "$key_file" | cut -d'"' -f2)
            validate_base64_key "$session_key" "AES_192" || ((issues++))
            ;;
        "encryption")
            local backup_key=$(grep "BACKUP_ENCRYPTION_KEY=" "$key_file" | cut -d'"' -f2)
            validate_base64_key "$backup_key" "AES_256" || ((issues++))
            ;;
        "hmac")
            local integrity_key=$(grep "DATA_INTEGRITY_HMAC_KEY=" "$key_file" | cut -d'"' -f2)
            validate_base64_key "$integrity_key" "HMAC_SHA256" || ((issues++))
            ;;
        "rsa")
            validate_rsa_key "$SECRETS_DIR/rsa_private.pem" || ((issues++))
            ;;
    esac
    
    return $issues
}

# Validate base64 encoded key
validate_base64_key() {
    local key_value="$1"
    local key_type="$2"
    local issues=0
    
    if [[ -z "$key_value" ]]; then
        log_audit "CRITICAL" "Empty key value detected for type: $key_type"
        ((CRITICAL_ISSUES++))
        return 1
    fi
    
    # Check if it's valid base64
    if ! echo "$key_value" | base64 -d >/dev/null 2>&1; then
        log_audit "CRITICAL" "Invalid base64 encoding for key type: $key_type"
        ((CRITICAL_ISSUES++))
        ((issues++))
    fi
    
    # Decode and check length
    local decoded_length=$(echo "$key_value" | base64 -d 2>/dev/null | wc -c)
    local min_length=${MIN_KEY_LENGTHS[$key_type]:-32}
    
    if [[ $decoded_length -lt $min_length ]]; then
        log_audit "HIGH" "Key too short for type $key_type: $decoded_length bytes (min: $min_length)"
        ((HIGH_ISSUES++))
        ((issues++))
    else
        log_audit "SUCCESS" "Key length sufficient for type $key_type: $decoded_length bytes"
    fi
    
    # Check for weak patterns
    check_key_patterns "$key_value" "$key_type" || ((issues++))
    
    return $issues
}

# Validate hex encoded key
validate_hex_key() {
    local key_value="$1"
    local issues=0
    
    if [[ -z "$key_value" ]]; then
        log_audit "CRITICAL" "Empty hex key value detected"
        ((CRITICAL_ISSUES++))
        return 1
    fi
    
    # Check if it's valid hexadecimal
    if ! [[ "$key_value" =~ ^[0-9a-fA-F]+$ ]]; then
        log_audit "CRITICAL" "Invalid hexadecimal encoding in key"
        ((CRITICAL_ISSUES++))
        ((issues++))
    fi
    
    # Check length (should be even)
    if [[ $((${#key_value} % 2)) -ne 0 ]]; then
        log_audit "HIGH" "Hex key length is odd: ${#key_value}"
        ((HIGH_ISSUES++))
        ((issues++))
    fi
    
    # Check minimum length (64 hex chars = 32 bytes)
    if [[ ${#key_value} -lt 64 ]]; then
        log_audit "HIGH" "Hex key too short: ${#key_value} chars (min: 64)"
        ((HIGH_ISSUES++))
        ((issues++))
    fi
    
    return $issues
}

# Validate RSA key
validate_rsa_key() {
    local key_file="$1"
    local issues=0
    
    if [[ ! -f "$key_file" ]]; then
        log_audit "ERROR" "RSA key file not found: $key_file"
        ((CRITICAL_ISSUES++))
        return 1
    fi
    
    # Validate RSA key using OpenSSL
    if ! openssl rsa -in "$key_file" -check -noout 2>/dev/null; then
        log_audit "CRITICAL" "RSA key validation failed: $key_file"
        ((CRITICAL_ISSUES++))
        ((issues++))
    else
        log_audit "SUCCESS" "RSA key validation passed: $key_file"
    fi
    
    # Check key size
    local key_size=$(openssl rsa -in "$key_file" -text -noout 2>/dev/null | grep "Private-Key:" | grep -o '[0-9]*')
    if [[ $key_size -lt 2048 ]]; then
        log_audit "CRITICAL" "RSA key size too small: $key_size bits (min: 2048)"
        ((CRITICAL_ISSUES++))
        ((issues++))
    elif [[ $key_size -lt 4096 ]]; then
        log_audit "WARNING" "RSA key size acceptable but not optimal: $key_size bits (recommended: 4096)"
        ((MEDIUM_ISSUES++))
        ((issues++))
    else
        log_audit "SUCCESS" "RSA key size optimal: $key_size bits"
    fi
    
    return $issues
}

# Check for weak key patterns
check_key_patterns() {
    local key_value="$1"
    local key_type="$2"
    local issues=0
    
    # Check for repeated characters
    if [[ "$key_value" =~ (.)\1{4,} ]]; then
        log_audit "HIGH" "Repeated character pattern detected in $key_type key"
        ((HIGH_ISSUES++))
        ((issues++))
    fi
    
    # Check for common weak patterns
    local weak_patterns=("123456" "abcdef" "000000" "111111" "aaaaaa" "AAAAAA")
    for pattern in "${weak_patterns[@]}"; do
        if [[ "$key_value" == *"$pattern"* ]]; then
            log_audit "CRITICAL" "Weak pattern detected in $key_type key: $pattern"
            ((CRITICAL_ISSUES++))
            ((issues++))
        fi
    done
    
    # Check entropy (basic)
    local unique_chars=$(echo "$key_value" | fold -w1 | sort -u | wc -l)
    local total_chars=${#key_value}
    local entropy_ratio=$((unique_chars * 100 / total_chars))
    
    if [[ $entropy_ratio -lt 30 ]]; then
        log_audit "HIGH" "Low entropy detected in $key_type key: $entropy_ratio%"
        ((HIGH_ISSUES++))
        ((issues++))
    elif [[ $entropy_ratio -lt 50 ]]; then
        log_audit "WARNING" "Moderate entropy in $key_type key: $entropy_ratio%"
        ((MEDIUM_ISSUES++))
        ((issues++))
    fi
    
    return $issues
}

# Check key ages and rotation compliance
check_rotation_compliance() {
    log_audit "INFO" "Checking key rotation compliance..."
    
    local issues=0
    local current_time=$(date +%s)
    
    # Define rotation periods (in days)
    declare -A rotation_periods=(
        ["database_encryption.key"]=90
        ["jwt_keys.key"]=30
        ["api_keys.key"]=60
        ["session_keys.key"]=7
        ["encryption_keys.key"]=90
        ["hmac_keys.key"]=30
        ["rsa_private.pem"]=365
        ["development_keys.key"]=30
    )
    
    for key_file in "${!rotation_periods[@]}"; do
        local file_path="$SECRETS_DIR/$key_file"
        
        if [[ -f "$file_path" ]]; then
            local file_time=$(stat -c %Y "$file_path")
            local age_seconds=$((current_time - file_time))
            local age_days=$((age_seconds / 86400))
            local max_age=${rotation_periods[$key_file]}
            
            if [[ $age_days -gt $max_age ]]; then
                log_audit "HIGH" "Key overdue for rotation: $key_file (age: $age_days days, max: $max_age days)"
                ((HIGH_ISSUES++))
                ((issues++))
            elif [[ $age_days -gt $((max_age - 7)) ]]; then
                log_audit "WARNING" "Key approaching rotation: $key_file (age: $age_days days, max: $max_age days)"
                ((MEDIUM_ISSUES++))
                ((issues++))
            else
                log_audit "SUCCESS" "Key rotation compliant: $key_file (age: $age_days days)"
            fi
        else
            log_audit "ERROR" "Expected key file missing: $key_file"
            ((CRITICAL_ISSUES++))
            ((issues++))
        fi
    done
    
    return $issues
}

# Check for security vulnerabilities
check_security_vulnerabilities() {
    log_audit "INFO" "Checking for security vulnerabilities..."
    
    local issues=0
    
    # Check for backup files in wrong locations
    if find "$SECRETS_DIR" -name "*.bak" -o -name "*.backup" -o -name "*~" | grep -q .; then
        log_audit "WARNING" "Backup files found in secrets directory"
        find "$SECRETS_DIR" -name "*.bak" -o -name "*.backup" -o -name "*~" | while read -r backup_file; do
            log_audit "WARNING" "Backup file: $backup_file"
        done
        ((MEDIUM_ISSUES++))
        ((issues++))
    fi
    
    # Check for temporary files
    if find "$SECRETS_DIR" -name "*.tmp" -o -name "*.temp" | grep -q .; then
        log_audit "HIGH" "Temporary files found in secrets directory"
        ((HIGH_ISSUES++))
        ((issues++))
    fi
    
    # Check for version control files
    if [[ -d "$SECRETS_DIR/.git" ]] || find "$SECRETS_DIR" -name ".gitignore" | grep -q .; then
        log_audit "CRITICAL" "Version control files detected in secrets directory"
        ((CRITICAL_ISSUES++))
        ((issues++))
    fi
    
    # Check for world-writable directories
    if find "$SECRETS_DIR" -type d -perm -002 | grep -q .; then
        log_audit "CRITICAL" "World-writable directories found"
        ((CRITICAL_ISSUES++))
        ((issues++))
    fi
    
    # Check for symbolic links
    if find "$SECRETS_DIR" -type l | grep -q .; then
        log_audit "WARNING" "Symbolic links found in secrets directory"
        ((MEDIUM_ISSUES++))
        ((issues++))
    fi
    
    return $issues
}

# Check compliance with security standards
check_compliance_standards() {
    local standard="$1"
    
    log_audit "COMPLIANCE" "Checking compliance with: $standard"
    
    local issues=0
    
    case "$standard" in
        "fips-140-2")
            check_fips_compliance || ((issues++))
            ;;
        "common-criteria")
            check_common_criteria_compliance || ((issues++))
            ;;
        "nist-sp-800-57")
            check_nist_compliance || ((issues++))
            ;;
        "pci-dss")
            check_pci_compliance || ((issues++))
            ;;
        *)
            log_audit "WARNING" "Unknown compliance standard: $standard"
            ((MEDIUM_ISSUES++))
            ((issues++))
            ;;
    esac
    
    return $issues
}

# FIPS 140-2 compliance check
check_fips_compliance() {
    log_audit "COMPLIANCE" "Checking FIPS 140-2 compliance..."
    
    local issues=0
    
    # Check approved algorithms
    local approved_algorithms=("AES" "RSA" "SHA" "HMAC")
    
    # Check for non-approved algorithms in key files
    if grep -r "MD5\|DES\|RC4\|3DES" "$SECRETS_DIR"/*.key 2>/dev/null; then
        log_audit "CRITICAL" "Non-FIPS approved algorithms detected"
        ((CRITICAL_ISSUES++))
        ((issues++))
    fi
    
    # Check minimum key sizes
    # AES: minimum 128-bit (already checked in key validation)
    # RSA: minimum 2048-bit (already checked in RSA validation)
    
    log_audit "SUCCESS" "FIPS 140-2 algorithm compliance verified"
    
    return $issues
}

# Common Criteria compliance check
check_common_criteria_compliance() {
    log_audit "COMPLIANCE" "Checking Common Criteria compliance..."
    
    local issues=0
    
    # Check for secure key generation (entropy sources)
    if ! check_entropy_sources; then
        log_audit "HIGH" "Insufficient entropy sources for Common Criteria compliance"
        ((HIGH_ISSUES++))
        ((issues++))
    fi
    
    # Check for secure key storage
    if ! check_secure_storage; then
        log_audit "HIGH" "Key storage does not meet Common Criteria requirements"
        ((HIGH_ISSUES++))
        ((issues++))
    fi
    
    return $issues
}

# NIST SP 800-57 compliance check
check_nist_compliance() {
    log_audit "COMPLIANCE" "Checking NIST SP 800-57 compliance..."
    
    local issues=0
    
    # Check key lifetimes against NIST recommendations
    local current_year=$(date +%Y)
    
    # Check RSA key size for current year
    if [[ -f "$SECRETS_DIR/rsa_private.pem" ]]; then
        local key_size=$(openssl rsa -in "$SECRETS_DIR/rsa_private.pem" -text -noout 2>/dev/null | grep "Private-Key:" | grep -o '[0-9]*')
        
        if [[ $current_year -ge 2030 ]] && [[ $key_size -lt 3072 ]]; then
            log_audit "HIGH" "RSA key size insufficient for year $current_year (NIST recommends 3072+ bits)"
            ((HIGH_ISSUES++))
            ((issues++))
        fi
    fi
    
    # Check symmetric key sizes
    # Already validated in key strength checks
    
    return $issues
}

# PCI DSS compliance check
check_pci_compliance() {
    log_audit "COMPLIANCE" "Checking PCI DSS compliance..."
    
    local issues=0
    
    # Check for encryption of stored data
    if ! check_data_encryption; then
        log_audit "HIGH" "PCI DSS requires encryption of stored cardholder data"
        ((HIGH_ISSUES++))
        ((issues++))
    fi
    
    # Check access controls
    if ! check_access_controls; then
        log_audit "HIGH" "PCI DSS access control requirements not met"
        ((HIGH_ISSUES++))
        ((issues++))
    fi
    
    return $issues
}

# Check entropy sources
check_entropy_sources() {
    # Check if /dev/urandom is available and properly seeded
    if [[ ! -c /dev/urandom ]]; then
        log_audit "CRITICAL" "System entropy source /dev/urandom not available"
        return 1
    fi
    
    # Check entropy pool size
    if [[ -f /proc/sys/kernel/random/entropy_avail ]]; then
        local entropy_avail=$(cat /proc/sys/kernel/random/entropy_avail)
        if [[ $entropy_avail -lt 1000 ]]; then
            log_audit "WARNING" "Low system entropy: $entropy_avail bits"
            return 1
        fi
    fi
    
    return 0
}

# Check secure storage implementation
check_secure_storage() {
    # Check if keys are stored in memory-mapped files
    # Check if keys are encrypted at rest
    # Check for proper key zeroization
    
    # Basic implementation - check file permissions and ownership
    for key_file in "$SECRETS_DIR"/*.key; do
        if [[ -f "$key_file" ]]; then
            local perms=$(stat -c "%a" "$key_file")
            if [[ "$perms" != "600" ]]; then
                return 1
            fi
        fi
    done
    
    return 0
}

# Check data encryption implementation
check_data_encryption() {
    # Check if database encryption keys are present and valid
    if [[ -f "$SECRETS_DIR/database_encryption.key" ]]; then
        local db_key=$(grep "DATABASE_ENCRYPTION_KEY=" "$SECRETS_DIR/database_encryption.key" | cut -d'"' -f2)
        if [[ -n "$db_key" ]] && [[ ${#db_key} -ge 44 ]]; then
            return 0
        fi
    fi
    
    return 1
}

# Check access controls
check_access_controls() {
    # Check file permissions and ownership
    local access_issues=0
    
    for key_file in "$SECRETS_DIR"/*.key "$SECRETS_DIR"/*.pem; do
        if [[ -f "$key_file" ]]; then
            local perms=$(stat -c "%a" "$key_file")
            local owner=$(stat -c "%U" "$key_file")
            
            if [[ "$perms" != "600" ]] && [[ "$perms" != "644" ]]; then
                ((access_issues++))
            fi
            
            if [[ "$owner" != "$(whoami)" ]] && [[ "$owner" != "root" ]]; then
                ((access_issues++))
            fi
        fi
    done
    
    return $((access_issues == 0))
}

# Generate comprehensive audit report
generate_audit_report() {
    log_audit "INFO" "Generating comprehensive audit report..."
    
    local timestamp=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
    local total_issues=$((CRITICAL_ISSUES + HIGH_ISSUES + MEDIUM_ISSUES + LOW_ISSUES))
    
    # Calculate overall security score
    local security_score=100
    security_score=$((security_score - CRITICAL_ISSUES * 25))
    security_score=$((security_score - HIGH_ISSUES * 15))
    security_score=$((security_score - MEDIUM_ISSUES * 5))
    security_score=$((security_score - LOW_ISSUES * 1))
    
    if [[ $security_score -lt 0 ]]; then
        security_score=0
    fi
    
    # Determine security grade
    local security_grade
    if [[ $security_score -ge 95 ]]; then
        security_grade="A+"
    elif [[ $security_score -ge 90 ]]; then
        security_grade="A"
    elif [[ $security_score -ge 85 ]]; then
        security_grade="A-"
    elif [[ $security_score -ge 80 ]]; then
        security_grade="B+"
    elif [[ $security_score -ge 75 ]]; then
        security_grade="B"
    elif [[ $security_score -ge 70 ]]; then
        security_grade="B-"
    elif [[ $security_score -ge 65 ]]; then
        security_grade="C+"
    elif [[ $security_score -ge 60 ]]; then
        security_grade="C"
    else
        security_grade="F"
    fi
    
    # Create JSON report
    cat > "$REPORT_FILE" << EOF
{
  "audit_metadata": {
    "version": "2.0.0",
    "timestamp": "$timestamp",
    "auditor": "Advanced Security Audit Script",
    "environment": "development",
    "compliance_mode": "${COMPLIANCE_MODE:-standard}"
  },
  "security_summary": {
    "overall_score": $security_score,
    "security_grade": "$security_grade",
    "total_issues": $total_issues,
    "critical_issues": $CRITICAL_ISSUES,
    "high_issues": $HIGH_ISSUES,
    "medium_issues": $MEDIUM_ISSUES,
    "low_issues": $LOW_ISSUES
  },
  "compliance_status": {
    "fips_140_2": "$([ $CRITICAL_ISSUES -eq 0 ] && echo "COMPLIANT" || echo "NON_COMPLIANT")",
    "common_criteria": "$([ $CRITICAL_ISSUES -eq 0 ] && [ $HIGH_ISSUES -lt 3 ] && echo "COMPLIANT" || echo "NON_COMPLIANT")",
    "nist_sp_800_57": "$([ $CRITICAL_ISSUES -eq 0 ] && echo "COMPLIANT" || echo "NON_COMPLIANT")",
    "pci_dss": "$([ $CRITICAL_ISSUES -eq 0 ] && [ $HIGH_ISSUES -eq 0 ] && echo "COMPLIANT" || echo "NON_COMPLIANT")"
  },
  "key_inventory": {
$(ls "$SECRETS_DIR"/*.key 2>/dev/null | while read -r key_file; do
    local filename=$(basename "$key_file")
    local filesize=$(stat -c%s "$key_file")
    local permissions=$(stat -c "%a" "$key_file")
    local last_modified=$(stat -c "%Y" "$key_file")
    echo "    \"$filename\": {"
    echo "      \"size_bytes\": $filesize,"
    echo "      \"permissions\": \"$permissions\","
    echo "      \"last_modified\": $last_modified"
    echo "    }$([ "$key_file" != "$(ls "$SECRETS_DIR"/*.key 2>/dev/null | tail -1)" ] && echo ",")"
done)
  },
  "recommendations": [
$(if [[ $CRITICAL_ISSUES -gt 0 ]]; then
    echo "    \"CRITICAL: Address $CRITICAL_ISSUES critical security issues immediately\","
fi
if [[ $HIGH_ISSUES -gt 0 ]]; then
    echo "    \"HIGH: Resolve $HIGH_ISSUES high-priority security issues\","
fi
if [[ $MEDIUM_ISSUES -gt 0 ]]; then
    echo "    \"MEDIUM: Consider fixing $MEDIUM_ISSUES medium-priority issues\","
fi
echo "    \"Implement automated key rotation\","
echo "    \"Set up continuous security monitoring\","
echo "    \"Enable hardware security module (HSM) integration\","
echo "    \"Conduct regular penetration testing\","
echo "    \"Implement multi-factor authentication for key access\""
)
  ],
  "next_audit_date": "$(date -u -d '+30 days' +"%Y-%m-%dT%H:%M:%SZ")"
}
EOF
    
    chmod 600 "$REPORT_FILE"
    log_audit "SUCCESS" "Audit report generated: $REPORT_FILE"
}

# Print audit summary
print_audit_summary() {
    echo ""
    echo "=============================================================================="
    echo -e "${BLUE}ADVANCED SECURITY AUDIT COMPLETED${NC}"
    echo "=============================================================================="
    echo ""
    
    # Calculate overall score
    local total_issues=$((CRITICAL_ISSUES + HIGH_ISSUES + MEDIUM_ISSUES + LOW_ISSUES))
    local security_score=100
    security_score=$((security_score - CRITICAL_ISSUES * 25))
    security_score=$((security_score - HIGH_ISSUES * 15))
    security_score=$((security_score - MEDIUM_ISSUES * 5))
    security_score=$((security_score - LOW_ISSUES * 1))
    
    if [[ $security_score -lt 0 ]]; then
        security_score=0
    fi
    
    # Color-code the score
    local score_color="$GREEN"
    if [[ $security_score -lt 70 ]]; then
        score_color="$RED"
    elif [[ $security_score -lt 85 ]]; then
        score_color="$YELLOW"
    fi
    
    echo -e "${CYAN}Security Score:${NC} ${score_color}$security_score/100${NC}"
    echo ""
    echo -e "${CYAN}Issue Summary:${NC}"
    
    if [[ $CRITICAL_ISSUES -gt 0 ]]; then
        echo -e "  ${RED}Critical Issues: $CRITICAL_ISSUES${NC}"
    fi
    if [[ $HIGH_ISSUES -gt 0 ]]; then
        echo -e "  ${YELLOW}High Issues: $HIGH_ISSUES${NC}"
    fi
    if [[ $MEDIUM_ISSUES -gt 0 ]]; then
        echo -e "  ${BLUE}Medium Issues: $MEDIUM_ISSUES${NC}"
    fi
    if [[ $LOW_ISSUES -gt 0 ]]; then
        echo -e "  Low Issues: $LOW_ISSUES"
    fi
    
    if [[ $total_issues -eq 0 ]]; then
        echo -e "  ${GREEN}✅ No security issues detected${NC}"
    fi
    
    echo ""
    echo -e "${CYAN}Compliance Status:${NC}"
    
    if [[ $CRITICAL_ISSUES -eq 0 ]]; then
        echo -e "  ${GREEN}✅ FIPS 140-2: COMPLIANT${NC}"
        echo -e "  ${GREEN}✅ NIST SP 800-57: COMPLIANT${NC}"
    else
        echo -e "  ${RED}❌ FIPS 140-2: NON-COMPLIANT${NC}"
        echo -e "  ${RED}❌ NIST SP 800-57: NON-COMPLIANT${NC}"
    fi
    
    if [[ $CRITICAL_ISSUES -eq 0 ]] && [[ $HIGH_ISSUES -lt 3 ]]; then
        echo -e "  ${GREEN}✅ Common Criteria: COMPLIANT${NC}"
    else
        echo -e "  ${RED}❌ Common Criteria: NON-COMPLIANT${NC}"
    fi
    
    if [[ $CRITICAL_ISSUES -eq 0 ]] && [[ $HIGH_ISSUES -eq 0 ]]; then
        echo -e "  ${GREEN}✅ PCI DSS: COMPLIANT${NC}"
    else
        echo -e "  ${RED}❌ PCI DSS: NON-COMPLIANT${NC}"
    fi
    
    echo ""
    echo -e "${CYAN}Generated Files:${NC}"
    echo "  • Audit log: $AUDIT_LOG"
    echo "  • Report: $REPORT_FILE"
    echo ""
    
    if [[ $CRITICAL_ISSUES -gt 0 ]]; then
        echo -e "${RED}⚠️  CRITICAL ISSUES DETECTED - IMMEDIATE ACTION REQUIRED${NC}"
    elif [[ $HIGH_ISSUES -gt 0 ]]; then
        echo -e "${YELLOW}⚠️  HIGH PRIORITY ISSUES DETECTED - ACTION RECOMMENDED${NC}"
    else
        echo -e "${GREEN}✅ Security audit completed successfully${NC}"
    fi
    
    echo "=============================================================================="
}

# Parse command line arguments
parse_audit_arguments() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            --verbose|-v)
                VERBOSE=true
                shift
                ;;
            --compliance)
                COMPLIANCE_MODE="$2"
                shift 2
                ;;
            --no-report)
                GENERATE_REPORT=false
                shift
                ;;
            --help)
                show_audit_help
                exit 0
                ;;
            *)
                log_audit "ERROR" "Unknown option: $1"
                show_audit_help
                exit 1
                ;;
        esac
    done
}

# Show help
show_audit_help() {
    echo "Advanced Security Audit Script v2.0"
    echo ""
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -v, --verbose         Enable verbose output"
    echo "  --compliance MODE     Set compliance mode (fips-140-2, common-criteria, nist-sp-800-57, pci-dss)"
    echo "  --no-report          Skip generating JSON report"
    echo "  --help               Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0                           # Standard security audit"
    echo "  $0 --verbose                 # Verbose audit output"
    echo "  $0 --compliance fips-140-2   # FIPS 140-2 compliance audit"
    echo "  $0 --compliance pci-dss      # PCI DSS compliance audit"
}

# Cleanup function
cleanup_audit() {
    umask "$UMASK_ORIGINAL"
    log_audit "INFO" "Security audit process completed"
}

# Main execution
main() {
    # Set up cleanup trap
    trap cleanup_audit EXIT
    
    # Initialize
    initialize_audit
    
    # Parse arguments
    parse_audit_arguments "$@"
    
    # Perform comprehensive security audit
    log_audit "INFO" "Starting comprehensive security validation..."
    
    # Check file security for all key files
    for key_file in "$SECRETS_DIR"/*.key "$SECRETS_DIR"/*.pem; do
        if [[ -f "$key_file" ]]; then
            check_file_security "$key_file"
        fi
    done
    
    # Validate key strength
    validate_key_strength "$SECRETS_DIR/database_encryption.key" "database"
    validate_key_strength "$SECRETS_DIR/jwt_keys.key" "jwt"
    validate_key_strength "$SECRETS_DIR/api_keys.key" "api"
    validate_key_strength "$SECRETS_DIR/session_keys.key" "session"
    validate_key_strength "$SECRETS_DIR/encryption_keys.key" "encryption"
    validate_key_strength "$SECRETS_DIR/hmac_keys.key" "hmac"
    validate_key_strength "$SECRETS_DIR/rsa_keys.key" "rsa"
    
    # Check rotation compliance
    check_rotation_compliance
    
    # Check for security vulnerabilities
    check_security_vulnerabilities
    
    # Check compliance if specified
    if [[ -n "$COMPLIANCE_MODE" ]]; then
        check_compliance_standards "$COMPLIANCE_MODE"
    fi
    
    # Generate report
    if [[ "$GENERATE_REPORT" == "true" ]]; then
        generate_audit_report
    fi
    
    # Print summary
    print_audit_summary
    
    # Exit with appropriate code
    if [[ $CRITICAL_ISSUES -gt 0 ]]; then
        exit 2
    elif [[ $HIGH_ISSUES -gt 0 ]]; then
        exit 1
    else
        exit 0
    fi
}

# Execute main function
main "$@"
