#!/bin/bash
# =============================================================================
# Script de gestion avancé du système i18n Slack
# =============================================================================
# 
# Ce script fournit une interface complète pour la gestion du système
# d'internationalisation des alertes Slack, incluant:
# 
# - Synchronisation automatique des traductions
# - Validation et vérification de cohérence
# - Génération de rapports de qualité
# - Mise à jour des caches distribués
# - Surveillance des performances
# - Backup et restauration
# - Intégration continue des traductions
# 
# Auteur: Expert Team
# Version: 2.0.0
# =============================================================================

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$(dirname "$SCRIPT_DIR")"
TRANSLATIONS_DIR="$BASE_DIR"
LOGS_DIR="$BASE_DIR/logs"
BACKUP_DIR="$BASE_DIR/backups"
CONFIG_FILE="$BASE_DIR/i18n_config.yaml"

# Couleurs pour les messages
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Configuration Redis
REDIS_HOST="${REDIS_HOST:-localhost}"
REDIS_PORT="${REDIS_PORT:-6379}"
REDIS_DB="${REDIS_DB:-2}"

# Configuration logging
LOG_LEVEL="${LOG_LEVEL:-INFO}"
LOG_FILE="$LOGS_DIR/i18n_manager_$(date +%Y%m%d).log"

# =============================================================================
# FONCTIONS UTILITAIRES
# =============================================================================

log() {
    local level="$1"
    shift
    local message="$*"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    
    echo -e "[$timestamp] [$level] $message" | tee -a "$LOG_FILE"
}

log_info() {
    log "INFO" "${BLUE}$*${NC}"
}

log_warn() {
    log "WARN" "${YELLOW}$*${NC}"
}

log_error() {
    log "ERROR" "${RED}$*${NC}"
}

log_success() {
    log "SUCCESS" "${GREEN}$*${NC}"
}

check_dependencies() {
    local deps=("python3" "redis-cli" "jq" "yq")
    local missing_deps=()
    
    for dep in "${deps[@]}"; do
        if ! command -v "$dep" >/dev/null 2>&1; then
            missing_deps+=("$dep")
        fi
    done
    
    if [ ${#missing_deps[@]} -ne 0 ]; then
        log_error "Dépendances manquantes: ${missing_deps[*]}"
        log_info "Installation automatique..."
        
        for dep in "${missing_deps[@]}"; do
            case "$dep" in
                "jq")
                    sudo apt-get update && sudo apt-get install -y jq
                    ;;
                "yq")
                    sudo wget -qO /usr/local/bin/yq https://github.com/mikefarah/yq/releases/latest/download/yq_linux_amd64
                    sudo chmod +x /usr/local/bin/yq
                    ;;
                "redis-cli")
                    sudo apt-get update && sudo apt-get install -y redis-tools
                    ;;
                *)
                    log_error "Installation automatique non supportée pour: $dep"
                    exit 1
                    ;;
            esac
        done
    fi
}

check_redis_connection() {
    if ! redis-cli -h "$REDIS_HOST" -p "$REDIS_PORT" -n "$REDIS_DB" ping >/dev/null 2>&1; then
        log_error "Impossible de se connecter à Redis ($REDIS_HOST:$REDIS_PORT/$REDIS_DB)"
        return 1
    fi
    log_success "Connexion Redis OK"
    return 0
}

create_directories() {
    mkdir -p "$LOGS_DIR" "$BACKUP_DIR"
}

# =============================================================================
# FONCTIONS DE VALIDATION
# =============================================================================

validate_translations() {
    log_info "🔍 Validation des traductions..."
    
    local validation_script="$SCRIPT_DIR/validate_translations.py"
    local report_file="$LOGS_DIR/validation_report_$(date +%Y%m%d_%H%M%S).json"
    
    cat > "$validation_script" << 'EOF'
#!/usr/bin/env python3
import yaml
import json
import sys
from pathlib import Path

def validate_translations():
    translations_file = Path(__file__).parent.parent / "translations.yaml"
    
    if not translations_file.exists():
        return {"status": "error", "message": "Fichier translations.yaml non trouvé"}
    
    try:
        with open(translations_file, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        
        results = {
            "status": "success",
            "languages": [],
            "missing_keys": {},
            "empty_values": {},
            "statistics": {}
        }
        
        # Langues détectées
        languages = [k for k in data.keys() if k not in ['meta', 'rtl_config', 'pluralization_rules', 'cultural_emojis', 'contextual_templates', 'number_formatting', 'date_formatting', 'ai_enhancement', 'monitoring', 'cache_config', 'api_config', 'fallback_strategy', 'security', 'compliance']]
        results["languages"] = languages
        
        # Langue de référence (première trouvée)
        if not languages:
            return {"status": "error", "message": "Aucune langue trouvée"}
        
        ref_lang = languages[0]
        ref_keys = get_all_keys(data[ref_lang])
        
        # Validation par langue
        for lang in languages:
            if lang == ref_lang:
                continue
                
            lang_keys = get_all_keys(data[lang])
            missing = ref_keys - lang_keys
            
            if missing:
                results["missing_keys"][lang] = list(missing)
            
            # Valeurs vides
            empty = find_empty_values(data[lang])
            if empty:
                results["empty_values"][lang] = empty
        
        # Statistiques
        for lang in languages:
            lang_data = data[lang]
            results["statistics"][lang] = {
                "total_keys": len(get_all_keys(lang_data)),
                "completeness": calculate_completeness(ref_keys, get_all_keys(lang_data))
            }
        
        return results
        
    except Exception as e:
        return {"status": "error", "message": str(e)}

def get_all_keys(data, prefix=""):
    keys = set()
    for k, v in data.items():
        if isinstance(v, dict):
            keys.update(get_all_keys(v, f"{prefix}.{k}" if prefix else k))
        else:
            keys.add(f"{prefix}.{k}" if prefix else k)
    return keys

def find_empty_values(data, prefix=""):
    empty = []
    for k, v in data.items():
        if isinstance(v, dict):
            empty.extend(find_empty_values(v, f"{prefix}.{k}" if prefix else k))
        elif not v or (isinstance(v, str) and not v.strip()):
            empty.append(f"{prefix}.{k}" if prefix else k)
    return empty

def calculate_completeness(ref_keys, lang_keys):
    if not ref_keys:
        return 100.0
    return (len(lang_keys & ref_keys) / len(ref_keys)) * 100

if __name__ == "__main__":
    result = validate_translations()
    print(json.dumps(result, indent=2, ensure_ascii=False))
EOF

    python3 "$validation_script" > "$report_file"
    
    local status=$(jq -r '.status' "$report_file")
    
    if [ "$status" = "success" ]; then
        log_success "✅ Validation réussie"
        
        # Affichage des statistiques
        log_info "📊 Statistiques des traductions:"
        jq -r '.statistics | to_entries[] | "  \(.key): \(.value.total_keys) clés (\(.value.completeness | floor)% complet)"' "$report_file"
        
        # Clés manquantes
        local missing=$(jq -r '.missing_keys | length' "$report_file")
        if [ "$missing" -gt 0 ]; then
            log_warn "⚠️  Clés manquantes détectées dans $missing langue(s)"
            jq -r '.missing_keys | to_entries[] | "  \(.key): \(.value | length) clés manquantes"' "$report_file"
        fi
        
    else
        local message=$(jq -r '.message' "$report_file")
        log_error "❌ Échec de la validation: $message"
        return 1
    fi
    
    rm -f "$validation_script"
    log_info "📄 Rapport détaillé: $report_file"
}

check_translation_quality() {
    log_info "🎯 Vérification de la qualité des traductions..."
    
    local quality_script="$SCRIPT_DIR/quality_check.py"
    
    cat > "$quality_script" << 'EOF'
#!/usr/bin/env python3
import yaml
import re
import json
from pathlib import Path
from collections import defaultdict

def check_quality():
    translations_file = Path(__file__).parent.parent / "translations.yaml"
    
    with open(translations_file, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)
    
    issues = defaultdict(list)
    
    # Langues à vérifier
    languages = [k for k in data.keys() if k not in ['meta', 'rtl_config', 'pluralization_rules', 'cultural_emojis', 'contextual_templates', 'number_formatting', 'date_formatting', 'ai_enhancement', 'monitoring', 'cache_config', 'api_config', 'fallback_strategy', 'security', 'compliance']]
    
    for lang in languages:
        lang_data = data[lang]
        
        # Vérification des variables non traduites
        untranslated = find_untranslated_variables(lang_data)
        if untranslated:
            issues[lang].extend([f"Variable non traduite: {var}" for var in untranslated])
        
        # Vérification de la cohérence des placeholders
        placeholder_issues = check_placeholder_consistency(lang_data)
        if placeholder_issues:
            issues[lang].extend(placeholder_issues)
        
        # Vérification de la longueur des textes
        length_issues = check_text_lengths(lang_data)
        if length_issues:
            issues[lang].extend(length_issues)
        
        # Vérification des caractères spéciaux
        special_char_issues = check_special_characters(lang_data, lang)
        if special_char_issues:
            issues[lang].extend(special_char_issues)
    
    return dict(issues)

def find_untranslated_variables(data, prefix=""):
    untranslated = []
    for k, v in data.items():
        if isinstance(v, dict):
            untranslated.extend(find_untranslated_variables(v, f"{prefix}.{k}" if prefix else k))
        elif isinstance(v, str):
            # Détection de variables non traduites (contenant encore des mots anglais communs)
            english_patterns = [
                r'\b(error|warning|alert|system|service|application|user|admin)\b',
                r'\b(failed|success|pending|running|stopped)\b'
            ]
            
            for pattern in english_patterns:
                if re.search(pattern, v.lower()) and prefix != "en":
                    untranslated.append(f"{prefix}.{k}" if prefix else k)
                    break
    
    return untranslated

def check_placeholder_consistency(data, prefix=""):
    issues = []
    for k, v in data.items():
        if isinstance(v, dict):
            issues.extend(check_placeholder_consistency(v, f"{prefix}.{k}" if prefix else k))
        elif isinstance(v, str):
            # Vérification des placeholders non fermés
            open_braces = v.count('{')
            close_braces = v.count('}')
            
            if open_braces != close_braces:
                issues.append(f"Placeholders non équilibrés dans {prefix}.{k}: {v}")
    
    return issues

def check_text_lengths(data, prefix=""):
    issues = []
    for k, v in data.items():
        if isinstance(v, dict):
            issues.extend(check_text_lengths(v, f"{prefix}.{k}" if prefix else k))
        elif isinstance(v, str):
            # Textes trop longs pour Slack (>4000 caractères)
            if len(v) > 4000:
                issues.append(f"Texte trop long dans {prefix}.{k}: {len(v)} caractères")
            
            # Textes suspicieusement courts
            if len(v.strip()) < 2 and not v.strip() in ['', '✅', '❌', '⚠️']:
                issues.append(f"Texte suspicieusement court dans {prefix}.{k}: '{v}'")
    
    return issues

def check_special_characters(data, lang, prefix=""):
    issues = []
    
    # Caractères attendus par langue
    expected_chars = {
        'fr': 'àâäçéèêëïîôùûüÿ',
        'de': 'äöüß',
        'es': 'áéíóúüñ¿¡',
        'pt': 'ãáàâçéêíóôõúü',
        'it': 'àèéìíîòóù',
        'ru': 'абвгдеёжзийклмнопрстуфхцчшщъыьэюя',
        'ar': 'ابتثجحخدذرزسشصضطظعغفقكلمنهوي',
        'zh': '一二三四五六七八九十',
        'ja': 'あいうえおかきくけこさしすせそたちつてとなにぬねのはひふへほまみむめもやゆよらりるれろわをん'
    }
    
    if lang in expected_chars:
        for k, v in data.items():
            if isinstance(v, dict):
                issues.extend(check_special_characters(v, lang, f"{prefix}.{k}" if prefix else k))
            elif isinstance(v, str):
                # Vérification de la présence de caractères de la langue
                has_expected = any(char in v for char in expected_chars[lang])
                has_latin = re.search(r'[a-zA-Z]', v)
                
                if has_latin and not has_expected and len(v) > 10:
                    issues.append(f"Texte possiblement non traduit dans {prefix}.{k} ({lang}): {v[:50]}...")
    
    return issues

if __name__ == "__main__":
    issues = check_quality()
    print(json.dumps(issues, indent=2, ensure_ascii=False))
EOF

    local quality_report="$LOGS_DIR/quality_report_$(date +%Y%m%d_%H%M%S).json"
    python3 "$quality_script" > "$quality_report"
    
    local total_issues=$(jq '[.[]] | add | length' "$quality_report")
    
    if [ "$total_issues" -eq 0 ]; then
        log_success "✅ Aucun problème de qualité détecté"
    else
        log_warn "⚠️  $total_issues problème(s) de qualité détecté(s)"
        jq -r 'to_entries[] | "  \(.key): \(.value | length) problème(s)"' "$quality_report"
    fi
    
    rm -f "$quality_script"
    log_info "📄 Rapport qualité: $quality_report"
}

# =============================================================================
# FONCTIONS DE SYNCHRONISATION
# =============================================================================

sync_translations() {
    log_info "🔄 Synchronisation des traductions..."
    
    # Backup avant synchronisation
    backup_translations
    
    # Validation des traductions
    if ! validate_translations; then
        log_error "❌ Synchronisation annulée - Validation échouée"
        return 1
    fi
    
    # Rechargement du cache Redis
    if check_redis_connection; then
        log_info "🔄 Rechargement du cache Redis..."
        
        # Suppression des anciens caches
        redis-cli -h "$REDIS_HOST" -p "$REDIS_PORT" -n "$REDIS_DB" \
                  --scan --pattern "trans:*" | \
        xargs -r redis-cli -h "$REDIS_HOST" -p "$REDIS_PORT" -n "$REDIS_DB" del
        
        redis-cli -h "$REDIS_HOST" -p "$REDIS_PORT" -n "$REDIS_DB" \
                  --scan --pattern "detection:*" | \
        xargs -r redis-cli -h "$REDIS_HOST" -p "$REDIS_PORT" -n "$REDIS_DB" del
        
        log_success "✅ Cache Redis nettoyé"
    fi
    
    # Notification de rechargement aux services
    notify_services_reload
    
    log_success "✅ Synchronisation terminée"
}

backup_translations() {
    log_info "💾 Sauvegarde des traductions..."
    
    local backup_file="$BACKUP_DIR/translations_backup_$(date +%Y%m%d_%H%M%S).tar.gz"
    
    tar -czf "$backup_file" -C "$BASE_DIR" \
        translations.yaml \
        translation_manager.py \
        language_detector.py \
        cultural_formatter.py \
        2>/dev/null || true
    
    if [ -f "$backup_file" ]; then
        log_success "✅ Sauvegarde créée: $backup_file"
        
        # Nettoyage des anciennes sauvegardes (garde les 10 dernières)
        find "$BACKUP_DIR" -name "translations_backup_*.tar.gz" -type f | \
        sort -r | tail -n +11 | xargs -r rm -f
        
    else
        log_warn "⚠️  Échec de la sauvegarde"
    fi
}

restore_translations() {
    local backup_file="$1"
    
    if [ ! -f "$backup_file" ]; then
        log_error "❌ Fichier de sauvegarde non trouvé: $backup_file"
        return 1
    fi
    
    log_info "🔄 Restauration depuis: $backup_file"
    
    # Sauvegarde de sécurité avant restauration
    backup_translations
    
    # Restauration
    tar -xzf "$backup_file" -C "$BASE_DIR"
    
    if [ $? -eq 0 ]; then
        log_success "✅ Restauration réussie"
        sync_translations
    else
        log_error "❌ Échec de la restauration"
        return 1
    fi
}

notify_services_reload() {
    log_info "📢 Notification aux services..."
    
    # Notification via Redis pub/sub
    if check_redis_connection; then
        redis-cli -h "$REDIS_HOST" -p "$REDIS_PORT" -n "$REDIS_DB" \
                  publish "i18n:reload" "translations_updated" >/dev/null 2>&1
        log_success "✅ Notification Redis envoyée"
    fi
    
    # Notification via webhook (si configuré)
    if [ -n "${WEBHOOK_URL:-}" ]; then
        curl -s -X POST "$WEBHOOK_URL" \
             -H "Content-Type: application/json" \
             -d '{"event": "translations_updated", "timestamp": "'$(date -u +%Y-%m-%dT%H:%M:%SZ)'"}' \
             >/dev/null 2>&1 && log_success "✅ Webhook notifié"
    fi
}

# =============================================================================
# FONCTIONS DE MONITORING
# =============================================================================

monitor_performance() {
    log_info "📊 Monitoring des performances i18n..."
    
    if ! check_redis_connection; then
        log_warn "⚠️  Monitoring limité - Redis indisponible"
        return 1
    fi
    
    local stats_script="$SCRIPT_DIR/performance_stats.py"
    
    cat > "$stats_script" << 'EOF'
#!/usr/bin/env python3
import redis
import json
import time
from datetime import datetime, timedelta

def get_performance_stats():
    try:
        r = redis.Redis(host='localhost', port=6379, db=2, decode_responses=True)
        
        # Test de connexion
        r.ping()
        
        stats = {
            "timestamp": datetime.utcnow().isoformat(),
            "redis_info": {},
            "cache_stats": {},
            "performance_metrics": {}
        }
        
        # Informations Redis
        redis_info = r.info()
        stats["redis_info"] = {
            "connected_clients": redis_info.get("connected_clients", 0),
            "used_memory_human": redis_info.get("used_memory_human", "0B"),
            "keyspace_hits": redis_info.get("keyspace_hits", 0),
            "keyspace_misses": redis_info.get("keyspace_misses", 0),
            "ops_per_sec": redis_info.get("instantaneous_ops_per_sec", 0)
        }
        
        # Statistiques de cache
        trans_keys = r.keys("trans:*")
        detection_keys = r.keys("detection:*")
        user_profile_keys = r.keys("user_profile:*")
        
        stats["cache_stats"] = {
            "translation_cache_size": len(trans_keys),
            "detection_cache_size": len(detection_keys),
            "user_profiles_count": len(user_profile_keys),
            "total_cache_keys": len(trans_keys) + len(detection_keys) + len(user_profile_keys)
        }
        
        # Calcul du hit ratio
        hits = redis_info.get("keyspace_hits", 0)
        misses = redis_info.get("keyspace_misses", 0)
        total = hits + misses
        
        if total > 0:
            hit_ratio = (hits / total) * 100
        else:
            hit_ratio = 0
        
        stats["performance_metrics"] = {
            "cache_hit_ratio": round(hit_ratio, 2),
            "avg_response_time_ms": measure_response_time(r),
            "memory_efficiency": calculate_memory_efficiency(r, stats["cache_stats"]["total_cache_keys"])
        }
        
        return stats
        
    except Exception as e:
        return {"error": str(e)}

def measure_response_time(redis_client):
    """Mesure le temps de réponse moyen"""
    times = []
    
    for _ in range(10):
        start = time.time()
        redis_client.ping()
        end = time.time()
        times.append((end - start) * 1000)
    
    return round(sum(times) / len(times), 2)

def calculate_memory_efficiency(redis_client, key_count):
    """Calcule l'efficacité mémoire"""
    try:
        info = redis_client.info()
        used_memory = info.get("used_memory", 0)
        
        if key_count > 0:
            bytes_per_key = used_memory / key_count
            return round(bytes_per_key, 2)
        else:
            return 0
    except:
        return 0

if __name__ == "__main__":
    stats = get_performance_stats()
    print(json.dumps(stats, indent=2))
EOF

    local stats_file="$LOGS_DIR/performance_stats_$(date +%Y%m%d_%H%M%S).json"
    python3 "$stats_script" > "$stats_file"
    
    local error=$(jq -r '.error // empty' "$stats_file")
    
    if [ -n "$error" ]; then
        log_error "❌ Erreur monitoring: $error"
        rm -f "$stats_script"
        return 1
    fi
    
    # Affichage des métriques
    log_info "📊 Métriques de performance:"
    echo "  Cache Hit Ratio: $(jq -r '.performance_metrics.cache_hit_ratio' "$stats_file")%"
    echo "  Temps de réponse moyen: $(jq -r '.performance_metrics.avg_response_time_ms' "$stats_file")ms"
    echo "  Clés en cache: $(jq -r '.cache_stats.total_cache_keys' "$stats_file")"
    echo "  Mémoire utilisée: $(jq -r '.redis_info.used_memory_human' "$stats_file")"
    echo "  Opérations/sec: $(jq -r '.redis_info.ops_per_sec' "$stats_file")"
    
    # Alertes de performance
    local hit_ratio=$(jq -r '.performance_metrics.cache_hit_ratio' "$stats_file")
    local response_time=$(jq -r '.performance_metrics.avg_response_time_ms' "$stats_file")
    
    if (( $(echo "$hit_ratio < 80" | bc -l) )); then
        log_warn "⚠️  Hit ratio cache faible: ${hit_ratio}%"
    fi
    
    if (( $(echo "$response_time > 100" | bc -l) )); then
        log_warn "⚠️  Temps de réponse élevé: ${response_time}ms"
    fi
    
    rm -f "$stats_script"
    log_info "📄 Rapport détaillé: $stats_file"
}

generate_usage_report() {
    log_info "📈 Génération du rapport d'utilisation..."
    
    local report_script="$SCRIPT_DIR/usage_report.py"
    
    cat > "$report_script" << 'EOF'
#!/usr/bin/env python3
import redis
import json
from datetime import datetime, timedelta
from collections import defaultdict, Counter

def generate_report():
    try:
        r = redis.Redis(host='localhost', port=6379, db=2, decode_responses=True)
        r.ping()
        
        report = {
            "timestamp": datetime.utcnow().isoformat(),
            "language_usage": analyze_language_usage(r),
            "user_activity": analyze_user_activity(r),
            "cache_efficiency": analyze_cache_efficiency(r),
            "error_rates": analyze_error_rates(r),
            "recommendations": []
        }
        
        # Génération de recommandations
        report["recommendations"] = generate_recommendations(report)
        
        return report
        
    except Exception as e:
        return {"error": str(e)}

def analyze_language_usage(redis_client):
    """Analyse l'utilisation des langues"""
    language_stats = defaultdict(int)
    
    # Analyse des profils utilisateur
    user_keys = redis_client.keys("user_profile:*")
    
    for key in user_keys:
        try:
            profile_data = redis_client.get(key)
            if profile_data:
                profile = json.loads(profile_data)
                detected_languages = profile.get("detected_languages", {})
                
                for lang, count in detected_languages.items():
                    language_stats[lang] += count
        except:
            continue
    
    return dict(language_stats)

def analyze_user_activity(redis_client):
    """Analyse l'activité utilisateur"""
    user_keys = redis_client.keys("user_profile:*")
    
    active_users = 0
    total_detections = 0
    
    for key in user_keys:
        try:
            profile_data = redis_client.get(key)
            if profile_data:
                profile = json.loads(profile_data)
                last_updated = profile.get("last_updated", "")
                
                if last_updated:
                    last_update = datetime.fromisoformat(last_updated.replace('Z', '+00:00'))
                    if datetime.utcnow() - last_update.replace(tzinfo=None) < timedelta(days=7):
                        active_users += 1
                
                detected_languages = profile.get("detected_languages", {})
                total_detections += sum(detected_languages.values())
        except:
            continue
    
    return {
        "total_users": len(user_keys),
        "active_users_7d": active_users,
        "total_detections": total_detections,
        "avg_detections_per_user": round(total_detections / max(1, len(user_keys)), 2)
    }

def analyze_cache_efficiency(redis_client):
    """Analyse l'efficacité du cache"""
    info = redis_client.info()
    
    hits = info.get("keyspace_hits", 0)
    misses = info.get("keyspace_misses", 0)
    total = hits + misses
    
    if total > 0:
        hit_ratio = (hits / total) * 100
        miss_ratio = (misses / total) * 100
    else:
        hit_ratio = miss_ratio = 0
    
    return {
        "cache_hits": hits,
        "cache_misses": misses,
        "hit_ratio_percent": round(hit_ratio, 2),
        "miss_ratio_percent": round(miss_ratio, 2),
        "total_requests": total
    }

def analyze_error_rates(redis_client):
    """Analyse des taux d'erreur (simulé)"""
    # En réalité, ceci devrait analyser les logs d'erreur
    return {
        "translation_errors": 0,
        "detection_errors": 0,
        "cache_errors": 0,
        "total_errors": 0,
        "error_rate_percent": 0.0
    }

def generate_recommendations(report):
    """Génère des recommandations basées sur l'analyse"""
    recommendations = []
    
    # Recommandations cache
    cache_hit_ratio = report["cache_efficiency"]["hit_ratio_percent"]
    if cache_hit_ratio < 80:
        recommendations.append({
            "type": "performance",
            "priority": "high",
            "message": f"Hit ratio cache faible ({cache_hit_ratio}%). Considérez augmenter le TTL ou optimiser les clés de cache."
        })
    
    # Recommandations langues
    language_usage = report["language_usage"]
    if language_usage:
        most_used = max(language_usage, key=language_usage.get)
        least_used = min(language_usage, key=language_usage.get)
        
        if language_usage[least_used] < language_usage[most_used] * 0.1:
            recommendations.append({
                "type": "optimization",
                "priority": "medium",
                "message": f"Langue '{least_used}' peu utilisée. Considérez la suppression ou l'optimisation."
            })
    
    # Recommandations utilisateurs
    user_activity = report["user_activity"]
    if user_activity["total_users"] > 0:
        active_ratio = user_activity["active_users_7d"] / user_activity["total_users"]
        if active_ratio < 0.5:
            recommendations.append({
                "type": "engagement",
                "priority": "medium",
                "message": f"Seulement {active_ratio*100:.1f}% des utilisateurs actifs sur 7 jours. Revoyez la stratégie d'engagement."
            })
    
    return recommendations

if __name__ == "__main__":
    report = generate_report()
    print(json.dumps(report, indent=2, ensure_ascii=False))
EOF

    local report_file="$LOGS_DIR/usage_report_$(date +%Y%m%d_%H%M%S).json"
    python3 "$report_script" > "$report_file"
    
    local error=$(jq -r '.error // empty' "$report_file")
    
    if [ -n "$error" ]; then
        log_error "❌ Erreur génération rapport: $error"
        rm -f "$report_script"
        return 1
    fi
    
    # Affichage du résumé
    log_info "📈 Rapport d'utilisation:"
    echo "  Langues utilisées: $(jq -r '.language_usage | keys | length' "$report_file")"
    echo "  Utilisateurs totaux: $(jq -r '.user_activity.total_users' "$report_file")"
    echo "  Utilisateurs actifs (7j): $(jq -r '.user_activity.active_users_7d' "$report_file")"
    echo "  Hit ratio cache: $(jq -r '.cache_efficiency.hit_ratio_percent' "$report_file")%"
    
    local rec_count=$(jq -r '.recommendations | length' "$report_file")
    if [ "$rec_count" -gt 0 ]; then
        log_info "💡 $rec_count recommandation(s) générée(s)"
        jq -r '.recommendations[] | "  - [\(.priority)] \(.message)"' "$report_file"
    fi
    
    rm -f "$report_script"
    log_info "📄 Rapport complet: $report_file"
}

# =============================================================================
# FONCTIONS DE MAINTENANCE
# =============================================================================

cleanup_old_data() {
    log_info "🧹 Nettoyage des anciennes données..."
    
    # Nettoyage des logs (garde 30 jours)
    find "$LOGS_DIR" -name "*.log" -type f -mtime +30 -delete 2>/dev/null || true
    find "$LOGS_DIR" -name "*.json" -type f -mtime +30 -delete 2>/dev/null || true
    
    # Nettoyage des sauvegardes (garde 90 jours)
    find "$BACKUP_DIR" -name "*.tar.gz" -type f -mtime +90 -delete 2>/dev/null || true
    
    # Nettoyage du cache Redis (profils inactifs > 90 jours)
    if check_redis_connection; then
        local cleanup_script="$SCRIPT_DIR/redis_cleanup.py"
        
        cat > "$cleanup_script" << 'EOF'
#!/usr/bin/env python3
import redis
import json
from datetime import datetime, timedelta

def cleanup_redis():
    r = redis.Redis(host='localhost', port=6379, db=2, decode_responses=True)
    
    cleaned_count = 0
    cutoff_date = datetime.utcnow() - timedelta(days=90)
    
    user_keys = r.keys("user_profile:*")
    
    for key in user_keys:
        try:
            profile_data = r.get(key)
            if profile_data:
                profile = json.loads(profile_data)
                last_updated = profile.get("last_updated", "")
                
                if last_updated:
                    last_update = datetime.fromisoformat(last_updated.replace('Z', '+00:00'))
                    if last_update.replace(tzinfo=None) < cutoff_date:
                        r.delete(key)
                        cleaned_count += 1
        except:
            # Suppression des profils corrompus
            r.delete(key)
            cleaned_count += 1
    
    print(f"Profils supprimés: {cleaned_count}")

if __name__ == "__main__":
    cleanup_redis()
EOF

        local cleaned=$(python3 "$cleanup_script")
        log_success "✅ Nettoyage Redis: $cleaned"
        rm -f "$cleanup_script"
    fi
    
    log_success "✅ Nettoyage terminé"
}

optimize_cache() {
    log_info "⚡ Optimisation du cache..."
    
    if ! check_redis_connection; then
        log_warn "⚠️  Optimisation impossible - Redis indisponible"
        return 1
    fi
    
    # Analyse de l'utilisation du cache
    local optimization_script="$SCRIPT_DIR/cache_optimization.py"
    
    cat > "$optimization_script" << 'EOF'
#!/usr/bin/env python3
import redis
import json
import time
from collections import defaultdict

def optimize_cache():
    r = redis.Redis(host='localhost', port=6379, db=2, decode_responses=True)
    
    optimizations = {
        "expired_keys_removed": 0,
        "redundant_keys_merged": 0,
        "memory_saved_bytes": 0
    }
    
    # Suppression des clés expirées manuellement
    all_keys = r.keys("*")
    
    for key in all_keys:
        ttl = r.ttl(key)
        if ttl == -1:  # Pas d'expiration
            # Ajouter une expiration par défaut selon le type
            if key.startswith("trans:"):
                r.expire(key, 3600)  # 1h pour les traductions
            elif key.startswith("detection:"):
                r.expire(key, 1800)  # 30min pour les détections
            elif key.startswith("user_profile:"):
                r.expire(key, 86400 * 30)  # 30j pour les profils
    
    # Analyse des doublons potentiels
    detection_patterns = defaultdict(list)
    
    detection_keys = r.keys("detection:*")
    for key in detection_keys:
        # Extraction du pattern (hash du texte)
        parts = key.split(":")
        if len(parts) >= 2:
            hash_part = parts[1]
            detection_patterns[hash_part].append(key)
    
    # Suppression des doublons
    for hash_val, keys in detection_patterns.items():
        if len(keys) > 1:
            # Garder le plus récent, supprimer les autres
            keys_with_ttl = [(key, r.ttl(key)) for key in keys]
            keys_with_ttl.sort(key=lambda x: x[1], reverse=True)
            
            for key, _ in keys_with_ttl[1:]:
                r.delete(key)
                optimizations["redundant_keys_removed"] += 1
    
    # Compactage mémoire Redis
    try:
        memory_before = r.info()["used_memory"]
        r.execute_command("MEMORY", "PURGE")
        memory_after = r.info()["used_memory"]
        optimizations["memory_saved_bytes"] = memory_before - memory_after
    except:
        pass
    
    return optimizations

if __name__ == "__main__":
    result = optimize_cache()
    print(json.dumps(result))
EOF

    local optimization_result=$(python3 "$optimization_script")
    local saved_bytes=$(echo "$optimization_result" | jq -r '.memory_saved_bytes')
    local removed_keys=$(echo "$optimization_result" | jq -r '.redundant_keys_removed')
    
    log_success "✅ Optimisation terminée"
    echo "  Clés redondantes supprimées: $removed_keys"
    echo "  Mémoire libérée: $saved_bytes bytes"
    
    rm -f "$optimization_script"
}

# =============================================================================
# FONCTION PRINCIPALE
# =============================================================================

show_help() {
    cat << EOF
🌍 Gestionnaire avancé du système i18n Slack v2.0.0

UTILISATION:
    $0 <commande> [options]

COMMANDES:
    validate           Valide la cohérence des traductions
    quality           Vérifie la qualité des traductions
    sync              Synchronise les traductions (validation + cache)
    backup            Crée une sauvegarde des traductions
    restore <file>    Restaure depuis une sauvegarde
    monitor           Affiche les métriques de performance
    report            Génère un rapport d'utilisation
    cleanup           Nettoie les anciennes données
    optimize          Optimise les performances du cache
    status            Affiche le statut général du système
    
EXEMPLES:
    $0 validate                                    # Validation complète
    $0 sync                                        # Synchronisation
    $0 backup                                      # Sauvegarde
    $0 restore backups/translations_backup_*.tar.gz  # Restauration
    $0 monitor                                     # Monitoring
    $0 cleanup                                     # Nettoyage

VARIABLES D'ENVIRONNEMENT:
    REDIS_HOST        Host Redis (défaut: localhost)
    REDIS_PORT        Port Redis (défaut: 6379)
    REDIS_DB          DB Redis (défaut: 2)
    LOG_LEVEL         Niveau de log (défaut: INFO)
    WEBHOOK_URL       URL webhook pour notifications

AUTEUR:
    Expert Team - Système i18n industrialisé pour Spotify AI Agent
EOF
}

show_status() {
    log_info "📊 Statut du système i18n"
    
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "🌍 SYSTÈME D'INTERNATIONALISATION SLACK"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    
    # Vérification des fichiers
    echo "📁 FICHIERS:"
    if [ -f "$TRANSLATIONS_DIR/translations.yaml" ]; then
        echo "  ✅ translations.yaml"
        local file_size=$(du -h "$TRANSLATIONS_DIR/translations.yaml" | cut -f1)
        echo "     Taille: $file_size"
        local lang_count=$(yq eval 'keys | length' "$TRANSLATIONS_DIR/translations.yaml" 2>/dev/null || echo "0")
        echo "     Langues: $lang_count"
    else
        echo "  ❌ translations.yaml manquant"
    fi
    
    for file in translation_manager.py language_detector.py cultural_formatter.py; do
        if [ -f "$TRANSLATIONS_DIR/$file" ]; then
            echo "  ✅ $file"
        else
            echo "  ❌ $file manquant"
        fi
    done
    
    echo ""
    echo "🔌 CONNEXIONS:"
    
    # Test Redis
    if check_redis_connection >/dev/null 2>&1; then
        echo "  ✅ Redis ($REDIS_HOST:$REDIS_PORT/$REDIS_DB)"
        local redis_memory=$(redis-cli -h "$REDIS_HOST" -p "$REDIS_PORT" info memory | grep used_memory_human | cut -d: -f2 | tr -d '\r')
        echo "     Mémoire: $redis_memory"
    else
        echo "  ❌ Redis indisponible"
    fi
    
    echo ""
    echo "📊 CACHE:"
    if check_redis_connection >/dev/null 2>&1; then
        local trans_keys=$(redis-cli -h "$REDIS_HOST" -p "$REDIS_PORT" -n "$REDIS_DB" --scan --pattern "trans:*" | wc -l)
        local detection_keys=$(redis-cli -h "$REDIS_HOST" -p "$REDIS_PORT" -n "$REDIS_DB" --scan --pattern "detection:*" | wc -l)
        local user_keys=$(redis-cli -h "$REDIS_HOST" -p "$REDIS_PORT" -n "$REDIS_DB" --scan --pattern "user_profile:*" | wc -l)
        
        echo "  Traductions en cache: $trans_keys"
        echo "  Détections en cache: $detection_keys"
        echo "  Profils utilisateur: $user_keys"
    else
        echo "  Cache indisponible"
    fi
    
    echo ""
    echo "📈 STATISTIQUES:"
    echo "  Logs: $(find "$LOGS_DIR" -name "*.log" 2>/dev/null | wc -l) fichier(s)"
    echo "  Sauvegardes: $(find "$BACKUP_DIR" -name "*.tar.gz" 2>/dev/null | wc -l) fichier(s)"
    
    local latest_backup=$(find "$BACKUP_DIR" -name "translations_backup_*.tar.gz" -type f 2>/dev/null | sort -r | head -n1)
    if [ -n "$latest_backup" ]; then
        local backup_date=$(basename "$latest_backup" .tar.gz | sed 's/translations_backup_//')
        echo "  Dernière sauvegarde: $backup_date"
    fi
    
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
}

main() {
    local command="${1:-}"
    
    # Création des répertoires nécessaires
    create_directories
    
    # Vérification des dépendances
    check_dependencies
    
    case "$command" in
        "validate")
            validate_translations
            ;;
        "quality")
            check_translation_quality
            ;;
        "sync")
            sync_translations
            ;;
        "backup")
            backup_translations
            ;;
        "restore")
            local backup_file="${2:-}"
            if [ -z "$backup_file" ]; then
                log_error "❌ Fichier de sauvegarde requis"
                echo "Usage: $0 restore <fichier_sauvegarde>"
                exit 1
            fi
            restore_translations "$backup_file"
            ;;
        "monitor")
            monitor_performance
            ;;
        "report")
            generate_usage_report
            ;;
        "cleanup")
            cleanup_old_data
            ;;
        "optimize")
            optimize_cache
            ;;
        "status")
            show_status
            ;;
        "help"|"--help"|"-h"|"")
            show_help
            ;;
        *)
            log_error "❌ Commande inconnue: $command"
            echo ""
            show_help
            exit 1
            ;;
    esac
}

# Exécution du script principal
main "$@"
