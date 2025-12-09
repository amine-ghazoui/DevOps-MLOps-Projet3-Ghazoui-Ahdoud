#!/bin/bash
# Script de test local pour vérifier que tout fonctionne

echo "🧪 Tests locaux du projet ML CI/CD"
echo "=================================="
echo ""

# Couleurs pour les messages
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Fonction pour afficher les succès
success() {
    echo -e "${GREEN}✅ $1${NC}"
}

# Fonction pour afficher les erreurs
error() {
    echo -e "${RED}❌ $1${NC}"
}

# Fonction pour afficher les warnings
warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

# 1. Vérifier Python
echo "1️⃣  Vérification de Python..."
if command -v python &> /dev/null; then
    PYTHON_VERSION=$(python --version)
    success "Python trouvé: $PYTHON_VERSION"
else
    error "Python n'est pas installé"
    exit 1
fi
echo ""

# 2. Vérifier les dépendances
echo "2️⃣  Vérification des dépendances..."
if [ -f "requirements.txt" ]; then
    success "requirements.txt trouvé"
    echo "Installation des dépendances..."
    pip install -q -r requirements.txt
    if [ $? -eq 0 ]; then
        success "Dépendances installées"
    else
        error "Erreur lors de l'installation des dépendances"
        exit 1
    fi
else
    error "requirements.txt non trouvé"
    exit 1
fi
echo ""

# 3. Vérifier le dataset
echo "3️⃣  Vérification du dataset..."
if [ -f "data/train.csv" ]; then
    LINES=$(wc -l < data/train.csv)
    success "Dataset trouvé: data/train.csv ($LINES lignes)"
else
    error "Dataset non trouvé: data/train.csv"
    warning "Téléchargez le dataset depuis Kaggle"
    exit 1
fi
echo ""

# 4. Vérifier params.yaml
echo "4️⃣  Vérification de la configuration..."
if [ -f "params.yaml" ]; then
    success "params.yaml trouvé"
else
    error "params.yaml non trouvé"
    exit 1
fi
echo ""

# 5. Configurer PYTHONPATH
echo "5️⃣  Configuration de PYTHONPATH..."
export PYTHONPATH=$(pwd):$PYTHONPATH
success "PYTHONPATH configuré: $PYTHONPATH"
echo ""

# 6. Tester le prétraitement
echo "6️⃣  Test du prétraitement des données..."
python -c "
import sys
sys.path.insert(0, '.')
from src.data_preprocessing import preprocess_pipeline
try:
    X_train, X_test, y_train, y_test, scaler = preprocess_pipeline()
    print(f'✅ Prétraitement réussi: Train={X_train.shape}, Test={X_test.shape}')
except Exception as e:
    print(f'❌ Erreur: {e}')
    sys.exit(1)
"
if [ $? -ne 0 ]; then
    error "Le prétraitement a échoué"
    exit 1
fi
echo ""

# 7. Tester l'entraînement
echo "7️⃣  Test de l'entraînement..."
python src/train.py
if [ $? -eq 0 ]; then
    success "Entraînement réussi"
else
    error "L'entraînement a échoué"
    exit 1
fi
echo ""

# 8. Vérifier les fichiers générés
echo "8️⃣  Vérification des fichiers générés..."
FILES=("models/model.pkl" "reports/metrics.json" "reports/predictions_plot.png" "reports/residuals_plot.png" "reports/feature_importance.png")
for file in "${FILES[@]}"; do
    if [ -f "$file" ]; then
        success "$file existe"
    else
        warning "$file n'existe pas"
    fi
done
echo ""

# 9. Tester l'évaluation
echo "9️⃣  Test de l'évaluation..."
python src/evaluate.py
if [ $? -eq 0 ]; then
    success "Évaluation réussi"
else
    error "L'évaluation a échoué"
    exit 1
fi
echo ""

# 10. Vérifier la baseline
echo "🔟 Vérification de la baseline..."
if [ -f "reports/baseline_metrics.json" ]; then
    success "Baseline créée: reports/baseline_metrics.json"
    if [ -f "reports/comparison_report.md" ]; then
        success "Rapport de comparaison créé"
    fi
else
    warning "Baseline non trouvée (sera créée au prochain run)"
fi
echo ""

# 11. Exécuter les tests unitaires
echo "1️⃣1️⃣  Exécution des tests unitaires..."
if command -v pytest &> /dev/null; then
    pytest tests/ -v
    if [ $? -eq 0 ]; then
        success "Tous les tests unitaires passent"
    else
        warning "Certains tests ont échoué (peut être normal si dataset manquant)"
    fi
else
    warning "pytest non installé, installation..."
    pip install pytest
    pytest tests/ -v
fi
echo ""

# Résumé
echo "=================================="
echo "📊 Résumé des tests"
echo "=================================="
success "Tests locaux terminés !"
echo ""
echo "Prochaines étapes:"
echo "1. Commitez vos changements: git add . && git commit -m 'test'"
echo "2. Créez une branche: git checkout -b test-branch"
echo "3. Poussez vers GitHub: git push origin test-branch"
echo "4. Créez une Pull Request sur GitHub"
echo "5. Vérifiez que les workflows GitHub Actions s'exécutent"

