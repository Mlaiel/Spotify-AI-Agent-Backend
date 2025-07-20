#!/bin/bash
# Script professionnel pour créer un environnement virtuel Python 3.10 dédié à Spleeter
# Usage : bash setup_spleeter_venv.sh

set -e

# Vérifier la présence de python3.10
if ! command -v python3.10 &> /dev/null; then
  echo "Erreur : python3.10 n'est pas installé. Installez Python 3.10 avant de continuer." >&2
  exit 1
fi

# Créer le venv
python3.10 -m venv venv-spleeter
source venv-spleeter/bin/activate

# Mettre à jour pip
pip install --upgrade pip

# Installer Spleeter (et numpy compatible)
pip install spleeter

# Vérification
spleeter separate -h

echo "\n✅ Environnement virtuel Spleeter prêt (Python 3.10, venv-spleeter)"
echo "Activez-le avec : source venv-spleeter/bin/activate"
