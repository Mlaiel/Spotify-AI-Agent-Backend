#!/bin/bash

# Script de validation finale - Ultra simplifié
# Développé par Fahed Mlaiel

set -euo pipefail

readonly GREEN='\033[0;32m'
readonly RED='\033[0;31m'
readonly CYAN='\033[0;36m'
readonly BOLD='\033[1m'
readonly NC='\033[0m'

echo -e "${BOLD}${CYAN}╔══════════════════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BOLD}${CYAN}║               VALIDATION FINALE - MODULE JOBS KUBERNETES                    ║${NC}"
echo -e "${BOLD}${CYAN}║                  Développé par Fahed Mlaiel - DevOps Expert                 ║${NC}"
echo -e "${BOLD}${CYAN}╚══════════════════════════════════════════════════════════════════════════════╝${NC}"
echo

# Validation YAML
echo -e "${BOLD}🔍 Validation des fichiers YAML...${NC}"
yaml_errors=0
for file in *.yaml; do
    if [[ -f "$file" ]]; then
        if yq eval . "$file" >/dev/null 2>&1; then
            echo -e "${GREEN}✅ $file - Syntaxe YAML valide${NC}"
        else
            echo -e "${RED}❌ $file - Erreur de syntaxe${NC}"
            ((yaml_errors++))
        fi
    fi
done

# Validation Python
echo -e "\n${BOLD}🐍 Validation du code Python...${NC}"
python_errors=0
for file in *.py; do
    if [[ -f "$file" ]]; then
        if python3 -m py_compile "$file" 2>/dev/null; then
            echo -e "${GREEN}✅ $file - Syntaxe Python valide${NC}"
        else
            echo -e "${RED}❌ $file - Erreur de syntaxe${NC}"
            ((python_errors++))
        fi
    fi
done

# Validation Shell
echo -e "\n${BOLD}🔧 Validation des scripts Shell...${NC}"
shell_errors=0
for file in *.sh; do
    if [[ -f "$file" ]]; then
        if [[ -x "$file" ]]; then
            echo -e "${GREEN}✅ $file - Permissions exécutables OK${NC}"
        else
            echo -e "${RED}❌ $file - Permissions manquantes${NC}"
            ((shell_errors++))
        fi
    fi
done

# Validation Makefile
echo -e "\n${BOLD}📋 Validation Makefile...${NC}"
makefile_errors=0
if [[ -f "Makefile" ]]; then
    if make -f Makefile --dry-run help >/dev/null 2>&1; then
        echo -e "${GREEN}✅ Makefile - Syntaxe valide${NC}"
    else
        echo -e "${RED}❌ Makefile - Erreur de syntaxe${NC}"
        ((makefile_errors++))
    fi
else
    echo -e "${RED}❌ Makefile manquant${NC}"
    ((makefile_errors++))
fi

# Validation complétude
echo -e "\n${BOLD}📁 Validation de la complétude...${NC}"
completeness_errors=0
expected_files=(
    "__init__.py"
    "manage-jobs.sh"
    "Makefile"
    "ml-training-job.yaml"
    "data-etl-job.yaml"
    "tenant-backup-job.yaml"
    "security-scan-job.yaml"
    "billing-reporting-job.yaml"
    "README.md"
    "README.de.md"
    "README.fr.md"
)

for file in "${expected_files[@]}"; do
    if [[ -f "$file" ]]; then
        size=$(stat -c%s "$file")
        if [[ $size -gt 500 ]]; then
            echo -e "${GREEN}✅ $file présent et substantiel (${size} bytes)${NC}"
        else
            echo -e "${RED}❌ $file présent mais trop petit (${size} bytes)${NC}"
            ((completeness_errors++))
        fi
    else
        echo -e "${RED}❌ $file manquant${NC}"
        ((completeness_errors++))
    fi
done

# Résultat final
total_errors=$((yaml_errors + python_errors + shell_errors + makefile_errors + completeness_errors))

echo
echo -e "${BOLD}📊 RÉSUMÉ DE LA VALIDATION :${NC}"
echo -e "   • Fichiers YAML : $(ls *.yaml 2>/dev/null | wc -l) fichiers - ${yaml_errors} erreur(s)"
echo -e "   • Code Python : $(ls *.py 2>/dev/null | wc -l) fichier(s) - ${python_errors} erreur(s)"
echo -e "   • Scripts Shell : $(ls *.sh 2>/dev/null | wc -l) fichier(s) - ${shell_errors} erreur(s)"
echo -e "   • Makefile : 1 fichier - ${makefile_errors} erreur(s)"
echo -e "   • Complétude : ${#expected_files[@]} fichiers attendus - ${completeness_errors} manquant(s)"
echo

if [[ $total_errors -eq 0 ]]; then
    echo -e "${GREEN}${BOLD}🎉 VALIDATION RÉUSSIE ! Module Jobs Ultra-Avancé prêt pour la production${NC}"
    echo -e "${GREEN}${BOLD}👨‍💻 Développé par Fahed Mlaiel - Excellence DevOps${NC}"
    exit 0
else
    echo -e "${RED}${BOLD}❌ VALIDATION ÉCHOUÉE - ${total_errors} erreur(s) détectée(s)${NC}"
    exit 1
fi
