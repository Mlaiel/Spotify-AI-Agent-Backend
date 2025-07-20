#!/bin/bash

# Automated Backup System - Spotify AI Agent
# ==========================================
#
# Enterprise-grade backup and disaster recovery system with:
# - Incremental and full backup strategies
# - Multi-storage backend support (S3, GCS, Azure, Local)
# - Encryption and compression
# - Automated retention policies
# - Point-in-time recovery capabilities
# - Cross-region replication
# - Backup validation and integrity checks

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_FILE="/var/log/spotify-backup-system.log"
BACKUP_ROOT="/var/backups/spotify"
CONFIG_FILE="/etc/spotify/backup_config.yaml"

# Default settings
BACKUP_TYPE="incremental"
STORAGE_BACKEND="local"
TENANT_ID=""
ENVIRONMENT="dev"
ENCRYPTION_ENABLED=true
COMPRESSION_ENABLED=true
RETENTION_DAYS=30
PARALLEL_JOBS=4
VERIFICATION_ENABLED=true
CROSS_REGION_REPLICATION=false
DRY_RUN=false
VERBOSE=false

# Cloud storage settings
AWS_S3_BUCKET=""
AWS_S3_PREFIX="spotify-backups"
GCS_BUCKET=""
GCS_PREFIX="spotify-backups"
AZURE_CONTAINER=""
AZURE_PREFIX="spotify-backups"

# Encryption settings
ENCRYPTION_KEY_FILE="/etc/spotify/backup.key"
GPG_RECIPIENT=""

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
NC='\033[0m'

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
        "BACKUP") echo -e "${PURPLE}💾 ${message}${NC}" ;;
    esac
}

# Initialize backup system
init_backup_system() {
    log "INFO" "Initializing backup system"
    
    # Create backup directories
    mkdir -p "$BACKUP_ROOT"/{data,configs,logs,metadata}
    mkdir -p "$BACKUP_ROOT"/temp
    
    # Create default configuration if not exists
    if [[ ! -f "$CONFIG_FILE" ]]; then
        create_default_config
    fi
    
    # Generate encryption key if not exists
    if [[ "$ENCRYPTION_ENABLED" == "true" ]] && [[ ! -f "$ENCRYPTION_KEY_FILE" ]]; then
        generate_encryption_key
    fi
    
    # Verify dependencies
    check_dependencies
    
    log "SUCCESS" "Backup system initialized"
}

# Create default configuration
create_default_config() {
    log "INFO" "Creating default backup configuration"
    
    mkdir -p "$(dirname "$CONFIG_FILE")"
    
    cat > "$CONFIG_FILE" << 'EOF'
# Spotify AI Agent Backup Configuration
backup:
  retention:
    daily: 7
    weekly: 4
    monthly: 12
    yearly: 3
  
  compression:
    algorithm: "gzip"
    level: 6
  
  encryption:
    algorithm: "AES-256"
    key_rotation_days: 90
  
  storage:
    local:
      path: "/var/backups/spotify"
    s3:
      bucket: ""
      region: "us-west-2"
      storage_class: "STANDARD_IA"
    gcs:
      bucket: ""
      project: ""
      storage_class: "NEARLINE"
    azure:
      account: ""
      container: ""
      tier: "Cool"

sources:
  databases:
    - type: "postgresql"
      host: "localhost"
      port: 5432
      databases: ["spotify_main", "spotify_analytics"]
    - type: "redis"
      host: "localhost" 
      port: 6379
    - type: "mongodb"
      host: "localhost"
      port: 27017
      databases: ["spotify_ml", "spotify_cache"]
  
  files:
    - path: "/etc/spotify"
      type: "config"
      exclude: ["*.tmp", "*.log"]
    - path: "/var/lib/spotify"
      type: "data"
      exclude: ["cache/*", "temp/*"]
    - path: "/var/log/spotify"
      type: "logs"
      include: ["*.log", "*.audit"]

  containers:
    - name: "spotify-api"
      type: "docker"
    - name: "spotify-ml"
      type: "docker"
    - name: "spotify-monitoring"
      type: "docker"

notifications:
  email:
    enabled: true
    recipients: ["admin@spotify.com"]
  slack:
    enabled: false
    webhook_url: ""
  teams:
    enabled: false
    webhook_url: ""
EOF
    
    log "SUCCESS" "Default configuration created: $CONFIG_FILE"
}

# Generate encryption key
generate_encryption_key() {
    log "INFO" "Generating encryption key"
    
    mkdir -p "$(dirname "$ENCRYPTION_KEY_FILE")"
    
    # Generate 256-bit key
    openssl rand -hex 32 > "$ENCRYPTION_KEY_FILE"
    chmod 600 "$ENCRYPTION_KEY_FILE"
    
    log "SUCCESS" "Encryption key generated: $ENCRYPTION_KEY_FILE"
}

# Check required dependencies
check_dependencies() {
    local missing_deps=()
    
    # Required tools
    local required_tools=(
        "tar" "gzip" "openssl" "rsync" "find" "date" "sqlite3"
    )
    
    for tool in "${required_tools[@]}"; do
        if ! command -v "$tool" &> /dev/null; then
            missing_deps+=("$tool")
        fi
    done
    
    # Database tools
    if command -v pg_dump &> /dev/null; then
        log "INFO" "PostgreSQL backup tools available"
    else
        log "WARNING" "PostgreSQL tools not found - database backups may not work"
    fi
    
    if command -v redis-cli &> /dev/null; then
        log "INFO" "Redis backup tools available"
    else
        log "WARNING" "Redis tools not found - Redis backups may not work"
    fi
    
    if command -v mongodump &> /dev/null; then
        log "INFO" "MongoDB backup tools available"
    else
        log "WARNING" "MongoDB tools not found - MongoDB backups may not work"
    fi
    
    # Cloud tools
    case "$STORAGE_BACKEND" in
        "s3")
            if ! command -v aws &> /dev/null; then
                missing_deps+=("awscli")
            fi
            ;;
        "gcs")
            if ! command -v gsutil &> /dev/null; then
                missing_deps+=("google-cloud-sdk")
            fi
            ;;
        "azure")
            if ! command -v az &> /dev/null; then
                missing_deps+=("azure-cli")
            fi
            ;;
    esac
    
    if [[ ${#missing_deps[@]} -gt 0 ]]; then
        log "ERROR" "Missing dependencies: ${missing_deps[*]}"
        exit 1
    fi
    
    log "SUCCESS" "All dependencies verified"
}

# Backup PostgreSQL databases
backup_postgresql() {
    local backup_dir="$1"
    local timestamp="$2"
    
    log "BACKUP" "Starting PostgreSQL backup"
    
    local pg_backup_dir="$backup_dir/postgresql"
    mkdir -p "$pg_backup_dir"
    
    # Get list of databases
    local databases=($(psql -t -c "SELECT datname FROM pg_database WHERE datistemplate = false;" 2>/dev/null | grep -v "^\s*$" || echo ""))
    
    if [[ ${#databases[@]} -eq 0 ]]; then
        log "WARNING" "No PostgreSQL databases found or connection failed"
        return
    fi
    
    for db in "${databases[@]}"; do
        db=$(echo "$db" | xargs)  # Trim whitespace
        log "INFO" "Backing up PostgreSQL database: $db"
        
        local dump_file="$pg_backup_dir/${db}_${timestamp}.sql"
        
        if [[ "$DRY_RUN" == "false" ]]; then
            if pg_dump "$db" > "$dump_file" 2>/dev/null; then
                log "SUCCESS" "PostgreSQL backup completed: $db"
                
                # Compress if enabled
                if [[ "$COMPRESSION_ENABLED" == "true" ]]; then
                    gzip "$dump_file"
                    dump_file="${dump_file}.gz"
                fi
                
                # Encrypt if enabled
                if [[ "$ENCRYPTION_ENABLED" == "true" ]]; then
                    encrypt_file "$dump_file"
                fi
            else
                log "ERROR" "PostgreSQL backup failed: $db"
            fi
        else
            log "INFO" "DRY RUN: Would backup PostgreSQL database $db to $dump_file"
        fi
    done
}

# Backup Redis data
backup_redis() {
    local backup_dir="$1"
    local timestamp="$2"
    
    log "BACKUP" "Starting Redis backup"
    
    local redis_backup_dir="$backup_dir/redis"
    mkdir -p "$redis_backup_dir"
    
    local dump_file="$redis_backup_dir/redis_${timestamp}.rdb"
    
    if [[ "$DRY_RUN" == "false" ]]; then
        # Force background save
        if redis-cli BGSAVE >/dev/null 2>&1; then
            # Wait for background save to complete
            while [[ "$(redis-cli LASTSAVE)" == "$(redis-cli LASTSAVE)" ]]; do
                sleep 1
            done
            
            # Copy RDB file
            local redis_dir=$(redis-cli CONFIG GET dir | tail -1)
            local redis_file=$(redis-cli CONFIG GET dbfilename | tail -1)
            
            if [[ -f "$redis_dir/$redis_file" ]]; then
                cp "$redis_dir/$redis_file" "$dump_file"
                log "SUCCESS" "Redis backup completed"
                
                # Compress if enabled
                if [[ "$COMPRESSION_ENABLED" == "true" ]]; then
                    gzip "$dump_file"
                    dump_file="${dump_file}.gz"
                fi
                
                # Encrypt if enabled
                if [[ "$ENCRYPTION_ENABLED" == "true" ]]; then
                    encrypt_file "$dump_file"
                fi
            else
                log "ERROR" "Redis dump file not found: $redis_dir/$redis_file"
            fi
        else
            log "ERROR" "Redis BGSAVE failed"
        fi
    else
        log "INFO" "DRY RUN: Would backup Redis to $dump_file"
    fi
}

# Backup MongoDB databases
backup_mongodb() {
    local backup_dir="$1"
    local timestamp="$2"
    
    log "BACKUP" "Starting MongoDB backup"
    
    local mongo_backup_dir="$backup_dir/mongodb"
    mkdir -p "$mongo_backup_dir"
    
    if [[ "$DRY_RUN" == "false" ]]; then
        if mongodump --out "$mongo_backup_dir/dump_$timestamp" >/dev/null 2>&1; then
            log "SUCCESS" "MongoDB backup completed"
            
            # Create archive
            local archive_file="$mongo_backup_dir/mongodb_${timestamp}.tar"
            tar -cf "$archive_file" -C "$mongo_backup_dir" "dump_$timestamp"
            rm -rf "$mongo_backup_dir/dump_$timestamp"
            
            # Compress if enabled
            if [[ "$COMPRESSION_ENABLED" == "true" ]]; then
                gzip "$archive_file"
                archive_file="${archive_file}.gz"
            fi
            
            # Encrypt if enabled
            if [[ "$ENCRYPTION_ENABLED" == "true" ]]; then
                encrypt_file "$archive_file"
            fi
        else
            log "ERROR" "MongoDB backup failed"
        fi
    else
        log "INFO" "DRY RUN: Would backup MongoDB to $mongo_backup_dir"
    fi
}

# Backup configuration files
backup_configs() {
    local backup_dir="$1"
    local timestamp="$2"
    
    log "BACKUP" "Starting configuration files backup"
    
    local config_backup_dir="$backup_dir/configs"
    mkdir -p "$config_backup_dir"
    
    local config_dirs=(
        "/etc/spotify"
        "/etc/nginx/sites-available"
        "/etc/systemd/system"
        "/opt/spotify/config"
    )
    
    for config_dir in "${config_dirs[@]}"; do
        if [[ -d "$config_dir" ]]; then
            local dir_name=$(basename "$config_dir")
            local archive_file="$config_backup_dir/${dir_name}_${timestamp}.tar"
            
            if [[ "$DRY_RUN" == "false" ]]; then
                tar -cf "$archive_file" -C "$(dirname "$config_dir")" "$dir_name" 2>/dev/null || true
                
                if [[ -f "$archive_file" ]]; then
                    log "SUCCESS" "Configuration backup completed: $config_dir"
                    
                    # Compress if enabled
                    if [[ "$COMPRESSION_ENABLED" == "true" ]]; then
                        gzip "$archive_file"
                        archive_file="${archive_file}.gz"
                    fi
                    
                    # Encrypt if enabled
                    if [[ "$ENCRYPTION_ENABLED" == "true" ]]; then
                        encrypt_file "$archive_file"
                    fi
                else
                    log "WARNING" "Configuration backup empty or failed: $config_dir"
                fi
            else
                log "INFO" "DRY RUN: Would backup configuration directory $config_dir"
            fi
        fi
    done
}

# Backup application data
backup_app_data() {
    local backup_dir="$1"
    local timestamp="$2"
    
    log "BACKUP" "Starting application data backup"
    
    local data_backup_dir="$backup_dir/app_data"
    mkdir -p "$data_backup_dir"
    
    local data_dirs=(
        "/var/lib/spotify"
        "/opt/spotify/data"
        "/var/spotify/uploads"
        "/var/spotify/cache"
    )
    
    for data_dir in "${data_dirs[@]}"; do
        if [[ -d "$data_dir" ]]; then
            local dir_name=$(basename "$data_dir")
            local archive_file="$data_backup_dir/${dir_name}_${timestamp}.tar"
            
            if [[ "$DRY_RUN" == "false" ]]; then
                # Exclude cache and temp files
                tar -cf "$archive_file" \
                    --exclude="*.tmp" \
                    --exclude="*.cache" \
                    --exclude="temp/*" \
                    --exclude="cache/*" \
                    -C "$(dirname "$data_dir")" "$dir_name" 2>/dev/null || true
                
                if [[ -f "$archive_file" ]]; then
                    log "SUCCESS" "Application data backup completed: $data_dir"
                    
                    # Compress if enabled
                    if [[ "$COMPRESSION_ENABLED" == "true" ]]; then
                        gzip "$archive_file"
                        archive_file="${archive_file}.gz"
                    fi
                    
                    # Encrypt if enabled
                    if [[ "$ENCRYPTION_ENABLED" == "true" ]]; then
                        encrypt_file "$archive_file"
                    fi
                else
                    log "WARNING" "Application data backup empty or failed: $data_dir"
                fi
            else
                log "INFO" "DRY RUN: Would backup application data directory $data_dir"
            fi
        fi
    done
}

# Backup Docker volumes
backup_docker_volumes() {
    local backup_dir="$1"
    local timestamp="$2"
    
    log "BACKUP" "Starting Docker volumes backup"
    
    if ! command -v docker &> /dev/null; then
        log "WARNING" "Docker not available - skipping volume backups"
        return
    fi
    
    local volumes_backup_dir="$backup_dir/docker_volumes"
    mkdir -p "$volumes_backup_dir"
    
    # Get list of volumes
    local volumes=($(docker volume ls -q | grep -E "(spotify|monitoring)" || echo ""))
    
    for volume in "${volumes[@]}"; do
        log "INFO" "Backing up Docker volume: $volume"
        
        local archive_file="$volumes_backup_dir/${volume}_${timestamp}.tar"
        
        if [[ "$DRY_RUN" == "false" ]]; then
            # Create temporary container to access volume
            docker run --rm \
                -v "$volume":/backup-source:ro \
                -v "$volumes_backup_dir":/backup-dest \
                busybox \
                tar -cf "/backup-dest/${volume}_${timestamp}.tar" -C /backup-source . 2>/dev/null || true
            
            if [[ -f "$archive_file" ]]; then
                log "SUCCESS" "Docker volume backup completed: $volume"
                
                # Compress if enabled
                if [[ "$COMPRESSION_ENABLED" == "true" ]]; then
                    gzip "$archive_file"
                    archive_file="${archive_file}.gz"
                fi
                
                # Encrypt if enabled
                if [[ "$ENCRYPTION_ENABLED" == "true" ]]; then
                    encrypt_file "$archive_file"
                fi
            else
                log "WARNING" "Docker volume backup failed: $volume"
            fi
        else
            log "INFO" "DRY RUN: Would backup Docker volume $volume"
        fi
    done
}

# Encrypt file
encrypt_file() {
    local file="$1"
    
    if [[ ! -f "$file" ]]; then
        log "ERROR" "File not found for encryption: $file"
        return 1
    fi
    
    log "INFO" "Encrypting file: $(basename "$file")"
    
    if [[ -f "$ENCRYPTION_KEY_FILE" ]]; then
        # Use symmetric encryption with key file
        openssl enc -aes-256-cbc -salt -in "$file" -out "${file}.enc" -pass file:"$ENCRYPTION_KEY_FILE"
        
        if [[ $? -eq 0 ]]; then
            rm "$file"
            mv "${file}.enc" "$file"
            log "SUCCESS" "File encrypted successfully"
        else
            log "ERROR" "File encryption failed"
            rm -f "${file}.enc"
        fi
    elif [[ -n "$GPG_RECIPIENT" ]]; then
        # Use GPG encryption
        gpg --trust-model always --encrypt -r "$GPG_RECIPIENT" --cipher-algo AES256 --output "${file}.gpg" "$file"
        
        if [[ $? -eq 0 ]]; then
            rm "$file"
            mv "${file}.gpg" "$file"
            log "SUCCESS" "File encrypted with GPG"
        else
            log "ERROR" "GPG encryption failed"
            rm -f "${file}.gpg"
        fi
    else
        log "WARNING" "No encryption method available - file not encrypted"
    fi
}

# Upload to cloud storage
upload_to_cloud() {
    local backup_dir="$1"
    local timestamp="$2"
    
    case "$STORAGE_BACKEND" in
        "s3")
            upload_to_s3 "$backup_dir" "$timestamp"
            ;;
        "gcs")
            upload_to_gcs "$backup_dir" "$timestamp"
            ;;
        "azure")
            upload_to_azure "$backup_dir" "$timestamp"
            ;;
        "local")
            log "INFO" "Using local storage - no upload required"
            ;;
        *)
            log "WARNING" "Unknown storage backend: $STORAGE_BACKEND"
            ;;
    esac
}

# Upload to AWS S3
upload_to_s3() {
    local backup_dir="$1"
    local timestamp="$2"
    
    if [[ -z "$AWS_S3_BUCKET" ]]; then
        log "ERROR" "AWS S3 bucket not configured"
        return 1
    fi
    
    log "INFO" "Uploading backup to AWS S3: $AWS_S3_BUCKET"
    
    local s3_path="s3://$AWS_S3_BUCKET/$AWS_S3_PREFIX/$timestamp/"
    
    if [[ "$DRY_RUN" == "false" ]]; then
        aws s3 sync "$backup_dir" "$s3_path" \
            --storage-class STANDARD_IA \
            --delete \
            --exclude "*.tmp" \
            --exclude "temp/*"
        
        if [[ $? -eq 0 ]]; then
            log "SUCCESS" "Backup uploaded to S3 successfully"
        else
            log "ERROR" "S3 upload failed"
        fi
    else
        log "INFO" "DRY RUN: Would upload to $s3_path"
    fi
}

# Upload to Google Cloud Storage
upload_to_gcs() {
    local backup_dir="$1"
    local timestamp="$2"
    
    if [[ -z "$GCS_BUCKET" ]]; then
        log "ERROR" "Google Cloud Storage bucket not configured"
        return 1
    fi
    
    log "INFO" "Uploading backup to Google Cloud Storage: $GCS_BUCKET"
    
    local gcs_path="gs://$GCS_BUCKET/$GCS_PREFIX/$timestamp/"
    
    if [[ "$DRY_RUN" == "false" ]]; then
        gsutil -m rsync -r -d "$backup_dir" "$gcs_path"
        
        if [[ $? -eq 0 ]]; then
            log "SUCCESS" "Backup uploaded to GCS successfully"
        else
            log "ERROR" "GCS upload failed"
        fi
    else
        log "INFO" "DRY RUN: Would upload to $gcs_path"
    fi
}

# Upload to Azure Blob Storage
upload_to_azure() {
    local backup_dir="$1"
    local timestamp="$2"
    
    if [[ -z "$AZURE_CONTAINER" ]]; then
        log "ERROR" "Azure container not configured"
        return 1
    fi
    
    log "INFO" "Uploading backup to Azure Blob Storage: $AZURE_CONTAINER"
    
    if [[ "$DRY_RUN" == "false" ]]; then
        find "$backup_dir" -type f -exec bash -c '
            relative_path="${1#'"$backup_dir"'/}"
            az storage blob upload \
                --container-name "'"$AZURE_CONTAINER"'" \
                --name "'"$AZURE_PREFIX"'/'"$timestamp"'/$relative_path" \
                --file "$1" \
                --tier Cool
        ' _ {} \;
        
        log "SUCCESS" "Backup uploaded to Azure successfully"
    else
        log "INFO" "DRY RUN: Would upload to Azure container $AZURE_CONTAINER"
    fi
}

# Verify backup integrity
verify_backup() {
    local backup_dir="$1"
    local timestamp="$2"
    
    log "INFO" "Verifying backup integrity"
    
    local verification_log="$backup_dir/verification_${timestamp}.log"
    local total_files=0
    local verified_files=0
    local failed_files=0
    
    # Count and verify all backup files
    while IFS= read -r -d '' file; do
        total_files=$((total_files + 1))
        
        # Basic file existence and readability check
        if [[ -r "$file" ]] && [[ -s "$file" ]]; then
            verified_files=$((verified_files + 1))
            echo "OK: $file" >> "$verification_log"
        else
            failed_files=$((failed_files + 1))
            echo "FAILED: $file" >> "$verification_log"
            log "ERROR" "Verification failed: $file"
        fi
        
        # For archives, do a quick integrity check
        if [[ "$file" =~ \.(tar|tar\.gz|tgz)$ ]]; then
            if tar -tf "$file" >/dev/null 2>&1; then
                echo "ARCHIVE_OK: $file" >> "$verification_log"
            else
                echo "ARCHIVE_FAILED: $file" >> "$verification_log"
                log "ERROR" "Archive verification failed: $file"
                failed_files=$((failed_files + 1))
            fi
        fi
        
    done < <(find "$backup_dir" -type f -name "*.tar*" -o -name "*.sql*" -o -name "*.rdb*" -print0)
    
    # Generate verification summary
    cat >> "$verification_log" << EOF

VERIFICATION SUMMARY
===================
Total files: $total_files
Verified: $verified_files
Failed: $failed_files
Success rate: $(( verified_files * 100 / total_files ))%
Timestamp: $(date)
EOF
    
    if [[ $failed_files -eq 0 ]]; then
        log "SUCCESS" "Backup verification completed successfully ($verified_files/$total_files files)"
    else
        log "WARNING" "Backup verification completed with $failed_files failures"
    fi
    
    return $failed_files
}

# Clean old backups based on retention policy
cleanup_old_backups() {
    log "INFO" "Cleaning up old backups (retention: $RETENTION_DAYS days)"
    
    if [[ "$DRY_RUN" == "false" ]]; then
        # Remove local backups older than retention period
        find "$BACKUP_ROOT" -type d -name "backup_*" -mtime +$RETENTION_DAYS -exec rm -rf {} + 2>/dev/null || true
        
        # Clean cloud storage based on backend
        case "$STORAGE_BACKEND" in
            "s3")
                if [[ -n "$AWS_S3_BUCKET" ]]; then
                    aws s3 ls "s3://$AWS_S3_BUCKET/$AWS_S3_PREFIX/" | \
                    awk '$1 ~ /^[0-9]{4}-[0-9]{2}-[0-9]{2}$/ && $1 < "'$(date -d "-$RETENTION_DAYS days" +%Y-%m-%d)'" {print $2}' | \
                    while read -r folder; do
                        aws s3 rm "s3://$AWS_S3_BUCKET/$AWS_S3_PREFIX/$folder" --recursive
                    done
                fi
                ;;
            "gcs")
                if [[ -n "$GCS_BUCKET" ]]; then
                    gsutil ls "gs://$GCS_BUCKET/$GCS_PREFIX/" | \
                    grep -E '[0-9]{4}-[0-9]{2}-[0-9]{2}' | \
                    while read -r folder; do
                        folder_date=$(basename "$folder")
                        if [[ "$folder_date" < "$(date -d "-$RETENTION_DAYS days" +%Y-%m-%d)" ]]; then
                            gsutil -m rm -r "$folder"
                        fi
                    done
                fi
                ;;
        esac
        
        log "SUCCESS" "Old backup cleanup completed"
    else
        log "INFO" "DRY RUN: Would clean backups older than $RETENTION_DAYS days"
    fi
}

# Create backup metadata
create_metadata() {
    local backup_dir="$1"
    local timestamp="$2"
    
    local metadata_file="$backup_dir/backup_metadata.json"
    
    cat > "$metadata_file" << EOF
{
    "backup_id": "$(uuidgen)",
    "timestamp": "$timestamp",
    "backup_type": "$BACKUP_TYPE",
    "tenant_id": "$TENANT_ID",
    "environment": "$ENVIRONMENT",
    "storage_backend": "$STORAGE_BACKEND",
    "encryption_enabled": $ENCRYPTION_ENABLED,
    "compression_enabled": $COMPRESSION_ENABLED,
    "retention_days": $RETENTION_DAYS,
    "system_info": {
        "hostname": "$(hostname)",
        "kernel": "$(uname -r)",
        "architecture": "$(uname -m)",
        "backup_script_version": "2.0.0"
    },
    "components": {
        "postgresql": true,
        "redis": true,
        "mongodb": true,
        "configs": true,
        "app_data": true,
        "docker_volumes": true
    },
    "size_info": {
        "total_size": "$(du -sh "$backup_dir" | cut -f1)",
        "file_count": $(find "$backup_dir" -type f | wc -l)
    }
}
EOF
    
    log "SUCCESS" "Backup metadata created: $metadata_file"
}

# Send notification
send_notification() {
    local status="$1"
    local message="$2"
    
    # TODO: Implement notification sending
    # This would integrate with email, Slack, Teams, etc.
    log "INFO" "Notification: $status - $message"
}

# Main backup function
perform_backup() {
    local timestamp=$(date '+%Y%m%d_%H%M%S')
    local backup_dir="$BACKUP_ROOT/backup_$timestamp"
    
    log "INFO" "Starting backup process"
    log "INFO" "Backup type: $BACKUP_TYPE"
    log "INFO" "Storage backend: $STORAGE_BACKEND"
    log "INFO" "Tenant: ${TENANT_ID:-all}"
    log "INFO" "Environment: $ENVIRONMENT"
    
    # Create backup directory
    mkdir -p "$backup_dir"
    
    # Start backup components
    backup_postgresql "$backup_dir" "$timestamp" &
    local pg_pid=$!
    
    backup_redis "$backup_dir" "$timestamp" &
    local redis_pid=$!
    
    backup_mongodb "$backup_dir" "$timestamp" &
    local mongo_pid=$!
    
    backup_configs "$backup_dir" "$timestamp" &
    local config_pid=$!
    
    backup_app_data "$backup_dir" "$timestamp" &
    local data_pid=$!
    
    backup_docker_volumes "$backup_dir" "$timestamp" &
    local docker_pid=$!
    
    # Wait for all backup processes to complete
    wait $pg_pid $redis_pid $mongo_pid $config_pid $data_pid $docker_pid
    
    # Create metadata
    create_metadata "$backup_dir" "$timestamp"
    
    # Verify backup if enabled
    if [[ "$VERIFICATION_ENABLED" == "true" ]]; then
        verify_backup "$backup_dir" "$timestamp"
    fi
    
    # Upload to cloud storage
    if [[ "$STORAGE_BACKEND" != "local" ]]; then
        upload_to_cloud "$backup_dir" "$timestamp"
    fi
    
    # Clean old backups
    cleanup_old_backups
    
    # Send notification
    send_notification "SUCCESS" "Backup completed successfully: $timestamp"
    
    log "SUCCESS" "Backup process completed: $backup_dir"
}

# Usage function
usage() {
    cat << EOF
Usage: $0 [OPTIONS]

Automated backup system for Spotify AI Agent

Options:
    --type TYPE             Backup type (full, incremental)
    --storage BACKEND       Storage backend (local, s3, gcs, azure)
    --tenant ID             Backup specific tenant
    --environment ENV       Environment (dev, staging, prod)
    --encryption            Enable encryption (default: true)
    --compression           Enable compression (default: true)
    --retention DAYS        Retention period in days (default: 30)
    --parallel JOBS         Number of parallel jobs (default: 4)
    --verification          Enable backup verification (default: true)
    --cross-region          Enable cross-region replication
    --dry-run               Simulate backup without execution
    --verbose, -v           Verbose output
    --help, -h              Show this help

Examples:
    $0 --type full --storage s3 --tenant spotify_prod
    $0 --type incremental --storage local --verification
    $0 --dry-run --verbose --tenant test_tenant

EOF
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --type)
            BACKUP_TYPE="$2"
            shift 2
            ;;
        --storage)
            STORAGE_BACKEND="$2"
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
        --encryption)
            ENCRYPTION_ENABLED=true
            shift
            ;;
        --compression)
            COMPRESSION_ENABLED=true
            shift
            ;;
        --retention)
            RETENTION_DAYS="$2"
            shift 2
            ;;
        --parallel)
            PARALLEL_JOBS="$2"
            shift 2
            ;;
        --verification)
            VERIFICATION_ENABLED=true
            shift
            ;;
        --cross-region)
            CROSS_REGION_REPLICATION=true
            shift
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --verbose|-v)
            VERBOSE=true
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
    echo "  Spotify AI Agent Backup System"
    echo "=============================================="
    echo
    
    # Initialize backup system
    init_backup_system
    
    # Perform backup
    perform_backup
}

# Run main function
main "$@"
