#!/bin/bash
# =============================================================================
# Script de Gestion des Logs pour l'Environnement de Développement
# =============================================================================
# 
# Ce script gère la rotation, l'archivage et l'analyse des logs
# pour l'environnement de développement
#
# Développé par l'équipe d'experts dirigée par Fahed Mlaiel
# =============================================================================

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../../../../../.." && pwd)"
ENV_DIR="$(dirname "$SCRIPT_DIR")"

# Couleurs pour l'affichage
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Configuration par défaut
LOG_RETENTION_DAYS="${LOG_RETENTION_DAYS:-30}"
LOG_ARCHIVE_DAYS="${LOG_ARCHIVE_DAYS:-7}"
LOG_MAX_SIZE_MB="${LOG_MAX_SIZE_MB:-100}"
LOG_ANALYSIS_DAYS="${LOG_ANALYSIS_DAYS:-1}"

# Répertoires de logs
LOGS_DIR="$PROJECT_ROOT/logs"
ARCHIVE_DIR="$LOGS_DIR/archive"
MONITORING_DIR="$LOGS_DIR/monitoring"
REPORTS_DIR="$LOGS_DIR/reports"
ANALYTICS_DIR="$LOGS_DIR/analytics"

# Fonctions utilitaires
log_info() {
    echo -e "${BLUE}[INFO]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1"
}

log_analytics() {
    echo -e "${PURPLE}[ANALYTICS]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1"
}

# Initialisation des répertoires
init_log_directories() {
    log_info "Initialisation des répertoires de logs..."
    
    mkdir -p "$LOGS_DIR" "$ARCHIVE_DIR" "$MONITORING_DIR" "$REPORTS_DIR" "$ANALYTICS_DIR"
    
    # Création des fichiers de logs s'ils n'existent pas
    touch "$LOGS_DIR/application.log"
    touch "$LOGS_DIR/error.log"
    touch "$LOGS_DIR/access.log"
    touch "$LOGS_DIR/api_service.log"
    touch "$MONITORING_DIR/health.log"
    touch "$MONITORING_DIR/performance.log"
    touch "$MONITORING_DIR/alerts.log"
    touch "$MONITORING_DIR/metrics.log"
    
    log_success "Répertoires de logs initialisés"
}

# Fonction de rotation des logs
rotate_logs() {
    log_info "Rotation des logs en cours..."
    
    local current_date=$(date +%Y%m%d_%H%M%S)
    local rotated_count=0
    
    # Recherche des fichiers de logs dépassant la taille limite
    for log_file in "$LOGS_DIR"/*.log "$MONITORING_DIR"/*.log; do
        if [ -f "$log_file" ]; then
            local file_size_mb=$(du -m "$log_file" | cut -f1)
            
            if [ "$file_size_mb" -gt "$LOG_MAX_SIZE_MB" ]; then
                local base_name=$(basename "$log_file" .log)
                local archive_file="$ARCHIVE_DIR/${base_name}_${current_date}.log"
                
                # Rotation du fichier
                mv "$log_file" "$archive_file"
                touch "$log_file"
                
                # Compression du fichier archivé
                if command -v gzip &> /dev/null; then
                    gzip "$archive_file"
                    archive_file="${archive_file}.gz"
                fi
                
                log_success "Log roté: $(basename "$log_file") -> $(basename "$archive_file")"
                rotated_count=$((rotated_count + 1))
            fi
        fi
    done
    
    if [ "$rotated_count" -eq 0 ]; then
        log_info "Aucun log ne nécessite de rotation"
    else
        log_success "$rotated_count fichiers de logs rotés"
    fi
}

# Fonction d'archivage des anciens logs
archive_old_logs() {
    log_info "Archivage des anciens logs (>${LOG_ARCHIVE_DAYS} jours)..."
    
    local archived_count=0
    local current_date=$(date +%Y%m%d)
    
    # Archivage des logs anciens du répertoire principal
    find "$LOGS_DIR" -maxdepth 1 -name "*.log" -mtime +$LOG_ARCHIVE_DAYS -type f | while read -r log_file; do
        if [ -f "$log_file" ]; then
            local base_name=$(basename "$log_file" .log)
            local archive_file="$ARCHIVE_DIR/${base_name}_archived_${current_date}.log"
            
            cp "$log_file" "$archive_file"
            
            # Compression
            if command -v gzip &> /dev/null; then
                gzip "$archive_file"
                archive_file="${archive_file}.gz"
            fi
            
            # Vidage du fichier original (garde la structure)
            > "$log_file"
            
            log_success "Log archivé: $(basename "$log_file")"
            archived_count=$((archived_count + 1))
        fi
    done
    
    # Archivage des logs de monitoring
    find "$MONITORING_DIR" -name "*.log" -mtime +$LOG_ARCHIVE_DAYS -type f | while read -r log_file; do
        if [ -f "$log_file" ]; then
            local base_name=$(basename "$log_file" .log)
            local archive_file="$ARCHIVE_DIR/monitoring_${base_name}_${current_date}.log"
            
            cp "$log_file" "$archive_file"
            
            # Compression
            if command -v gzip &> /dev/null; then
                gzip "$archive_file"
            fi
            
            > "$log_file"
            
            log_success "Log de monitoring archivé: $(basename "$log_file")"
        fi
    done
    
    log_success "Archivage terminé"
}

# Fonction de nettoyage des anciennes archives
cleanup_old_archives() {
    log_info "Nettoyage des anciennes archives (>${LOG_RETENTION_DAYS} jours)..."
    
    local deleted_count=0
    
    if [ -d "$ARCHIVE_DIR" ]; then
        # Suppression des archives anciennes
        find "$ARCHIVE_DIR" -type f \( -name "*.log" -o -name "*.log.gz" \) -mtime +$LOG_RETENTION_DAYS | while read -r archive_file; do
            if [ -f "$archive_file" ]; then
                rm -f "$archive_file"
                log_info "Archive supprimée: $(basename "$archive_file")"
                deleted_count=$((deleted_count + 1))
            fi
        done
        
        # Suppression des rapports anciens
        find "$REPORTS_DIR" -name "daily_*.json" -mtime +$LOG_RETENTION_DAYS -delete 2>/dev/null || true
        find "$ANALYTICS_DIR" -name "*.json" -mtime +$LOG_RETENTION_DAYS -delete 2>/dev/null || true
    fi
    
    log_success "Nettoyage des archives terminé"
}

# Fonction d'analyse des logs d'erreurs
analyze_error_logs() {
    log_analytics "Analyse des logs d'erreurs..."
    
    local error_log="$LOGS_DIR/error.log"
    local analysis_file="$ANALYTICS_DIR/error_analysis_$(date +%Y%m%d).json"
    local start_date=$(date -d "$LOG_ANALYSIS_DAYS days ago" +%Y-%m-%d)
    
    if [ ! -f "$error_log" ]; then
        log_warning "Fichier error.log non trouvé"
        return 1
    fi
    
    local analysis=$(cat <<EOF
{
    "analysis_date": "$(date -Iseconds)",
    "period": {
        "start_date": "$start_date",
        "days": $LOG_ANALYSIS_DAYS
    },
    "error_summary": {
        "total_errors": 0,
        "critical_errors": 0,
        "warning_errors": 0,
        "info_errors": 0
    },
    "error_patterns": {},
    "top_errors": [],
    "recommendations": []
}
EOF
)
    
    # Analyse des erreurs par niveau
    local total_errors=$(grep -c "ERROR\|CRITICAL\|WARNING" "$error_log" 2>/dev/null || echo "0")
    local critical_errors=$(grep -c "CRITICAL" "$error_log" 2>/dev/null || echo "0")
    local warning_errors=$(grep -c "WARNING" "$error_log" 2>/dev/null || echo "0")
    local info_errors=$(grep -c "INFO" "$error_log" 2>/dev/null || echo "0")
    
    analysis=$(echo "$analysis" | jq --arg total "$total_errors" --arg critical "$critical_errors" --arg warning "$warning_errors" --arg info "$info_errors" '.error_summary.total_errors = ($total | tonumber) | .error_summary.critical_errors = ($critical | tonumber) | .error_summary.warning_errors = ($warning | tonumber) | .error_summary.info_errors = ($info | tonumber)')
    
    # Analyse des patterns d'erreurs fréquents
    local error_patterns=""
    if [ -f "$error_log" ] && [ -s "$error_log" ]; then
        # Top 5 des erreurs les plus fréquentes
        error_patterns=$(grep -o "ERROR.*" "$error_log" 2>/dev/null | head -100 | sort | uniq -c | sort -nr | head -5 | while read count error; do
            echo "{\"count\": $count, \"message\": \"$(echo "$error" | cut -c1-80)...\"}"
        done | jq -s '.')
    else
        error_patterns="[]"
    fi
    
    analysis=$(echo "$analysis" | jq --argjson patterns "$error_patterns" '.top_errors = $patterns')
    
    # Génération de recommendations basées sur l'analyse
    local recommendations="[]"
    
    if [ "$critical_errors" -gt 0 ]; then
        recommendations=$(echo "$recommendations" | jq '. += ["Erreurs critiques détectées - Investigation immédiate requise"]')
    fi
    
    if [ "$total_errors" -gt 100 ]; then
        recommendations=$(echo "$recommendations" | jq '. += ["Niveau d'"'"'erreurs élevé - Révision du code recommandée"]')
    fi
    
    if [ "$warning_errors" -gt 50 ]; then
        recommendations=$(echo "$recommendations" | jq '. += ["Nombreux warnings - Optimisation possible du code"]')
    fi
    
    if [ "$total_errors" -eq 0 ]; then
        recommendations=$(echo "$recommendations" | jq '. += ["Aucune erreur détectée - Système stable"]')
    fi
    
    analysis=$(echo "$analysis" | jq --argjson recs "$recommendations" '.recommendations = $recs')
    
    # Sauvegarde de l'analyse
    echo "$analysis" | jq '.' > "$analysis_file"
    
    log_analytics "Analyse des erreurs terminée"
    log_analytics "Erreurs totales: $total_errors, Critiques: $critical_errors, Warnings: $warning_errors"
    log_analytics "Rapport sauvegardé: $analysis_file"
}

# Fonction d'analyse des logs de performance
analyze_performance_logs() {
    log_analytics "Analyse des logs de performance..."
    
    local perf_log="$MONITORING_DIR/performance.log"
    local analysis_file="$ANALYTICS_DIR/performance_analysis_$(date +%Y%m%d).json"
    
    if [ ! -f "$perf_log" ]; then
        log_warning "Fichier performance.log non trouvé"
        return 1
    fi
    
    local analysis=$(cat <<EOF
{
    "analysis_date": "$(date -Iseconds)",
    "performance_metrics": {
        "avg_response_time": 0,
        "max_response_time": 0,
        "min_response_time": 0,
        "total_requests": 0,
        "success_rate": 0
    },
    "trends": {},
    "alerts": [],
    "recommendations": []
}
EOF
)
    
    # Extraction des métriques de performance du fichier JSON log
    if [ -s "$perf_log" ]; then
        # Calcul des moyennes de temps de réponse
        local response_times=$(grep "response_time" "$perf_log" 2>/dev/null | jq -r '.api_tests.load_test.avg_response_time_ms // empty' 2>/dev/null | grep -v null || echo "")
        
        if [ -n "$response_times" ]; then
            local avg_response=$(echo "$response_times" | awk '{sum+=$1; count++} END {if(count>0) print sum/count; else print 0}')
            local max_response=$(echo "$response_times" | sort -n | tail -1)
            local min_response=$(echo "$response_times" | sort -n | head -1)
            local total_tests=$(echo "$response_times" | wc -l)
            
            analysis=$(echo "$analysis" | jq --arg avg "$avg_response" --arg max "$max_response" --arg min "$min_response" --arg total "$total_tests" '.performance_metrics.avg_response_time = ($avg | tonumber) | .performance_metrics.max_response_time = ($max | tonumber) | .performance_metrics.min_response_time = ($min | tonumber) | .performance_metrics.total_requests = ($total | tonumber)')
        fi
        
        # Calcul du taux de succès moyen
        local success_rates=$(grep "success_rate" "$perf_log" 2>/dev/null | jq -r '.api_tests.load_test.success_rate_percent // empty' 2>/dev/null | grep -v null || echo "")
        
        if [ -n "$success_rates" ]; then
            local avg_success=$(echo "$success_rates" | awk '{sum+=$1; count++} END {if(count>0) print sum/count; else print 100}')
            analysis=$(echo "$analysis" | jq --arg success "$avg_success" '.performance_metrics.success_rate = ($success | tonumber)')
        fi
    fi
    
    # Génération de recommendations
    local recommendations="[]"
    local avg_response=$(echo "$analysis" | jq -r '.performance_metrics.avg_response_time')
    local success_rate=$(echo "$analysis" | jq -r '.performance_metrics.success_rate')
    
    if (( $(echo "$avg_response > 2000" | bc -l 2>/dev/null || echo "0") )); then
        recommendations=$(echo "$recommendations" | jq '. += ["Temps de réponse élevé - Optimisation nécessaire"]')
    fi
    
    if (( $(echo "$success_rate < 95" | bc -l 2>/dev/null || echo "0") )); then
        recommendations=$(echo "$recommendations" | jq '. += ["Taux de succès faible - Investigation des erreurs requise"]')
    fi
    
    if (( $(echo "$avg_response < 500" | bc -l 2>/dev/null || echo "1") )) && (( $(echo "$success_rate > 99" | bc -l 2>/dev/null || echo "0") )); then
        recommendations=$(echo "$recommendations" | jq '. += ["Performances excellentes - Maintenir les optimisations actuelles"]')
    fi
    
    analysis=$(echo "$analysis" | jq --argjson recs "$recommendations" '.recommendations = $recs')
    
    # Sauvegarde de l'analyse
    echo "$analysis" | jq '.' > "$analysis_file"
    
    log_analytics "Analyse de performance terminée"
    log_analytics "Temps de réponse moyen: ${avg_response}ms, Taux de succès: ${success_rate}%"
    log_analytics "Rapport sauvegardé: $analysis_file"
}

# Fonction d'analyse des logs d'accès
analyze_access_logs() {
    log_analytics "Analyse des logs d'accès..."
    
    local access_log="$LOGS_DIR/access.log"
    local analysis_file="$ANALYTICS_DIR/access_analysis_$(date +%Y%m%d).json"
    
    if [ ! -f "$access_log" ]; then
        log_warning "Fichier access.log non trouvé"
        return 1
    fi
    
    local analysis=$(cat <<EOF
{
    "analysis_date": "$(date -Iseconds)",
    "access_summary": {
        "total_requests": 0,
        "unique_ips": 0,
        "top_endpoints": [],
        "status_codes": {},
        "hourly_distribution": {}
    },
    "security_insights": {
        "suspicious_ips": [],
        "failed_requests": 0,
        "potential_attacks": []
    },
    "recommendations": []
}
EOF
)
    
    if [ -s "$access_log" ]; then
        # Nombre total de requêtes
        local total_requests=$(wc -l < "$access_log")
        analysis=$(echo "$analysis" | jq --arg total "$total_requests" '.access_summary.total_requests = ($total | tonumber)')
        
        # Nombre d'IPs uniques (approximation basée sur le format de log)
        local unique_ips=$(awk '{print $1}' "$access_log" 2>/dev/null | sort | uniq | wc -l || echo "0")
        analysis=$(echo "$analysis" | jq --arg ips "$unique_ips" '.access_summary.unique_ips = ($ips | tonumber)')
        
        # Top 5 des endpoints les plus consultés (approximation)
        local top_endpoints=$(awk '{print $7}' "$access_log" 2>/dev/null | sort | uniq -c | sort -nr | head -5 | while read count endpoint; do
            echo "{\"endpoint\": \"$endpoint\", \"count\": $count}"
        done | jq -s '.' 2>/dev/null || echo "[]")
        
        analysis=$(echo "$analysis" | jq --argjson endpoints "$top_endpoints" '.access_summary.top_endpoints = $endpoints')
        
        # Distribution des codes de statut
        local status_codes=$(awk '{print $9}' "$access_log" 2>/dev/null | sort | uniq -c | while read count code; do
            echo "\"$code\": $count"
        done | paste -sd, | sed 's/^/{/' | sed 's/$/}/' || echo "{}")
        
        analysis=$(echo "$analysis" | jq --argjson codes "$status_codes" '.access_summary.status_codes = $codes')
        
        # Détection de requêtes échouées (codes 4xx et 5xx)
        local failed_requests=$(awk '$9 ~ /^[45]/ {count++} END {print count+0}' "$access_log" 2>/dev/null)
        analysis=$(echo "$analysis" | jq --arg failed "$failed_requests" '.security_insights.failed_requests = ($failed | tonumber)')
    fi
    
    # Génération de recommendations
    local recommendations="[]"
    local total_requests=$(echo "$analysis" | jq -r '.access_summary.total_requests')
    local failed_requests=$(echo "$analysis" | jq -r '.security_insights.failed_requests')
    
    if [ "$total_requests" -gt 10000 ]; then
        recommendations=$(echo "$recommendations" | jq '. += ["Trafic élevé détecté - Surveiller les performances"]')
    fi
    
    if [ "$failed_requests" -gt 100 ]; then
        recommendations=$(echo "$recommendations" | jq '. += ["Nombreuses requêtes échouées - Vérifier la configuration"]')
    fi
    
    if [ "$total_requests" -eq 0 ]; then
        recommendations=$(echo "$recommendations" | jq '. += ["Aucune requête détectée - Vérifier la configuration des logs"]')
    fi
    
    analysis=$(echo "$analysis" | jq --argjson recs "$recommendations" '.recommendations = $recs')
    
    # Sauvegarde de l'analyse
    echo "$analysis" | jq '.' > "$analysis_file"
    
    log_analytics "Analyse des accès terminée"
    log_analytics "Requêtes totales: $total_requests, IPs uniques: $unique_ips, Requêtes échouées: $failed_requests"
    log_analytics "Rapport sauvegardé: $analysis_file"
}

# Fonction de génération de rapport consolidé
generate_consolidated_report() {
    log_info "Génération du rapport consolidé..."
    
    local report_date=$(date +%Y%m%d)
    local consolidated_report="$REPORTS_DIR/consolidated_report_$report_date.json"
    
    # Initialisation du rapport
    local report=$(cat <<EOF
{
    "report_date": "$(date -Iseconds)",
    "period": "$report_date",
    "summary": {
        "log_files_analyzed": 0,
        "total_log_size_mb": 0,
        "archived_files": 0,
        "alerts_generated": 0
    },
    "error_analysis": {},
    "performance_analysis": {},
    "access_analysis": {},
    "recommendations": [],
    "next_actions": []
}
EOF
)
    
    # Calcul de la taille totale des logs
    local total_size_bytes=$(find "$LOGS_DIR" -name "*.log" -type f -exec du -b {} \; 2>/dev/null | awk '{sum+=$1} END {print sum+0}')
    local total_size_mb=$((total_size_bytes / 1024 / 1024))
    
    # Comptage des fichiers analysés
    local log_files_count=$(find "$LOGS_DIR" "$MONITORING_DIR" -name "*.log" -type f | wc -l)
    
    # Comptage des archives
    local archived_count=$(find "$ARCHIVE_DIR" -type f | wc -l)
    
    report=$(echo "$report" | jq --arg files "$log_files_count" --arg size "$total_size_mb" --arg archived "$archived_count" '.summary.log_files_analyzed = ($files | tonumber) | .summary.total_log_size_mb = ($size | tonumber) | .summary.archived_files = ($archived | tonumber)')
    
    # Intégration des analyses existantes
    local error_analysis_file="$ANALYTICS_DIR/error_analysis_$report_date.json"
    local perf_analysis_file="$ANALYTICS_DIR/performance_analysis_$report_date.json"
    local access_analysis_file="$ANALYTICS_DIR/access_analysis_$report_date.json"
    
    if [ -f "$error_analysis_file" ]; then
        local error_data=$(cat "$error_analysis_file")
        report=$(echo "$report" | jq --argjson error "$error_data" '.error_analysis = $error')
    fi
    
    if [ -f "$perf_analysis_file" ]; then
        local perf_data=$(cat "$perf_analysis_file")
        report=$(echo "$report" | jq --argjson perf "$perf_data" '.performance_analysis = $perf')
    fi
    
    if [ -f "$access_analysis_file" ]; then
        local access_data=$(cat "$access_analysis_file")
        report=$(echo "$report" | jq --argjson access "$access_data" '.access_analysis = $access')
    fi
    
    # Consolidation des recommendations
    local all_recommendations="[]"
    
    # Recommendations basées sur la taille des logs
    if [ "$total_size_mb" -gt 500 ]; then
        all_recommendations=$(echo "$all_recommendations" | jq '. += ["Logs volumineux - Considérer une rotation plus fréquente"]')
    fi
    
    if [ "$archived_count" -gt 100 ]; then
        all_recommendations=$(echo "$all_recommendations" | jq '. += ["Nombreuses archives - Réviser la politique de rétention"]')
    fi
    
    # Actions suivantes recommandées
    local next_actions='["Surveiller la croissance des logs", "Vérifier l'"'"'efficacité de la rotation", "Optimiser les performances selon les analyses"]'
    
    report=$(echo "$report" | jq --argjson recs "$all_recommendations" --argjson actions "$next_actions" '.recommendations = $recs | .next_actions = $actions')
    
    # Sauvegarde du rapport consolidé
    echo "$report" | jq '.' > "$consolidated_report"
    
    log_success "Rapport consolidé généré: $consolidated_report"
    log_info "Taille totale des logs: ${total_size_mb}MB"
    log_info "Fichiers analysés: $log_files_count"
    log_info "Archives créées: $archived_count"
}

# Fonction d'affichage du statut des logs
show_log_status() {
    echo -e "${BLUE}==================================================================${NC}"
    echo -e "${BLUE}               STATUT DES LOGS - ENVIRONNEMENT DEV${NC}"
    echo -e "${BLUE}==================================================================${NC}"
    echo
    
    # Statut des répertoires
    echo -e "${CYAN}┌─ RÉPERTOIRES DE LOGS ──────────────────────────────────────────┐${NC}"
    echo -e "│ Logs principaux:    $(du -sh "$LOGS_DIR" 2>/dev/null | cut -f1 || echo "N/A") │"
    echo -e "│ Archives:           $(du -sh "$ARCHIVE_DIR" 2>/dev/null | cut -f1 || echo "N/A") │"
    echo -e "│ Monitoring:         $(du -sh "$MONITORING_DIR" 2>/dev/null | cut -f1 || echo "N/A") │"
    echo -e "│ Rapports:           $(du -sh "$REPORTS_DIR" 2>/dev/null | cut -f1 || echo "N/A") │"
    echo -e "${CYAN}└────────────────────────────────────────────────────────────────┘${NC}"
    echo
    
    # Statut des fichiers principaux
    echo -e "${CYAN}┌─ FICHIERS DE LOGS PRINCIPAUX ──────────────────────────────────┐${NC}"
    
    for log_file in application.log error.log access.log api_service.log; do
        local full_path="$LOGS_DIR/$log_file"
        if [ -f "$full_path" ]; then
            local size=$(du -sh "$full_path" | cut -f1)
            local lines=$(wc -l < "$full_path" 2>/dev/null || echo "0")
            local status="${GREEN}●${NC}"
        else
            local size="N/A"
            local lines="0"
            local status="${RED}●${NC}"
        fi
        
        printf "│ %s %-20s %8s %10s lignes │\n" "$status" "$log_file" "$size" "$lines"
    done
    
    echo -e "${CYAN}└────────────────────────────────────────────────────────────────┘${NC}"
    echo
    
    # Statut du monitoring
    echo -e "${CYAN}┌─ LOGS DE MONITORING ───────────────────────────────────────────┐${NC}"
    
    for log_file in health.log performance.log alerts.log metrics.log; do
        local full_path="$MONITORING_DIR/$log_file"
        if [ -f "$full_path" ]; then
            local size=$(du -sh "$full_path" | cut -f1)
            local lines=$(wc -l < "$full_path" 2>/dev/null || echo "0")
            local status="${GREEN}●${NC}"
        else
            local size="N/A"
            local lines="0"
            local status="${RED}●${NC}"
        fi
        
        printf "│ %s %-20s %8s %10s lignes │\n" "$status" "$log_file" "$size" "$lines"
    done
    
    echo -e "${CYAN}└────────────────────────────────────────────────────────────────┘${NC}"
    echo
    
    # Dernières analyses
    echo -e "${CYAN}┌─ DERNIÈRES ANALYSES ───────────────────────────────────────────┐${NC}"
    
    local latest_reports=$(find "$ANALYTICS_DIR" -name "*_$(date +%Y%m%d).json" 2>/dev/null | wc -l)
    echo -e "│ Analyses du jour:   $latest_reports                                          │"
    
    local latest_consolidated=$(find "$REPORTS_DIR" -name "consolidated_report_*.json" | tail -1)
    if [ -n "$latest_consolidated" ]; then
        local report_date=$(basename "$latest_consolidated" | sed 's/consolidated_report_\(.*\)\.json/\1/')
        echo -e "│ Dernier rapport:    $report_date                                        │"
    else
        echo -e "│ Dernier rapport:    Aucun                                              │"
    fi
    
    echo -e "${CYAN}└────────────────────────────────────────────────────────────────┘${NC}"
    echo
}

# Fonction d'affichage de l'aide
show_help() {
    echo "Usage: $0 [OPTIONS]"
    echo
    echo "Gère la rotation, l'archivage et l'analyse des logs de développement"
    echo
    echo "OPTIONS:"
    echo "  --rotate             Effectue la rotation des gros fichiers de logs"
    echo "  --archive            Archive les anciens logs"
    echo "  --cleanup            Nettoie les anciennes archives"
    echo "  --analyze-errors     Analyse les logs d'erreurs"
    echo "  --analyze-performance Analyse les logs de performance"
    echo "  --analyze-access     Analyse les logs d'accès"
    echo "  --analyze-all        Effectue toutes les analyses"
    echo "  --report             Génère le rapport consolidé"
    echo "  --status             Affiche le statut des logs"
    echo "  --full               Effectue toutes les opérations"
    echo "  --help               Affiche cette aide"
    echo
    echo "VARIABLES D'ENVIRONNEMENT:"
    echo "  LOG_RETENTION_DAYS    Rétention des archives en jours (défaut: 30)"
    echo "  LOG_ARCHIVE_DAYS      Seuil d'archivage en jours (défaut: 7)"
    echo "  LOG_MAX_SIZE_MB       Taille max avant rotation MB (défaut: 100)"
    echo "  LOG_ANALYSIS_DAYS     Période d'analyse en jours (défaut: 1)"
    echo
    echo "EXEMPLES:"
    echo "  $0 --status              # Affiche le statut actuel"
    echo "  $0 --rotate              # Rotation des gros logs"
    echo "  $0 --analyze-all         # Toutes les analyses"
    echo "  $0 --full                # Maintenance complète"
    echo
}

# Fonction principale
main() {
    local rotate=false
    local archive=false
    local cleanup=false
    local analyze_errors=false
    local analyze_performance=false
    local analyze_access=false
    local analyze_all=false
    local report=false
    local status=false
    local full=false
    
    # Analyse des arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --rotate)
                rotate=true
                shift
                ;;
            --archive)
                archive=true
                shift
                ;;
            --cleanup)
                cleanup=true
                shift
                ;;
            --analyze-errors)
                analyze_errors=true
                shift
                ;;
            --analyze-performance)
                analyze_performance=true
                shift
                ;;
            --analyze-access)
                analyze_access=true
                shift
                ;;
            --analyze-all)
                analyze_all=true
                shift
                ;;
            --report)
                report=true
                shift
                ;;
            --status)
                status=true
                shift
                ;;
            --full)
                full=true
                shift
                ;;
            --help)
                show_help
                exit 0
                ;;
            *)
                log_error "Option inconnue: $1"
                show_help
                exit 1
                ;;
        esac
    done
    
    # Initialisation
    init_log_directories
    
    # Mode statut seulement
    if [ "$status" = true ]; then
        show_log_status
        exit 0
    fi
    
    # Mode complet
    if [ "$full" = true ]; then
        rotate=true
        archive=true
        cleanup=true
        analyze_all=true
        report=true
    fi
    
    # Mode analyse complète
    if [ "$analyze_all" = true ]; then
        analyze_errors=true
        analyze_performance=true
        analyze_access=true
    fi
    
    # Exécution des opérations demandées
    if [ "$rotate" = true ]; then
        rotate_logs
    fi
    
    if [ "$archive" = true ]; then
        archive_old_logs
    fi
    
    if [ "$cleanup" = true ]; then
        cleanup_old_archives
    fi
    
    if [ "$analyze_errors" = true ]; then
        analyze_error_logs
    fi
    
    if [ "$analyze_performance" = true ]; then
        analyze_performance_logs
    fi
    
    if [ "$analyze_access" = true ]; then
        analyze_access_logs
    fi
    
    if [ "$report" = true ]; then
        generate_consolidated_report
    fi
    
    # Si aucune option spécifique, afficher le statut
    if [ "$rotate" = false ] && [ "$archive" = false ] && [ "$cleanup" = false ] && [ "$analyze_errors" = false ] && [ "$analyze_performance" = false ] && [ "$analyze_access" = false ] && [ "$report" = false ]; then
        show_log_status
    fi
    
    log_success "Gestion des logs terminée"
}

# Exécution du script principal
main "$@"
