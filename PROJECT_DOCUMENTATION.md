# 📚 Documentation Complète du Projet ML CI/CD

## 📋 Table des matières

1. [Vue d'ensemble](#vue-densemble)
2. [Structure du projet](#structure-du-projet)
3. [Fichiers et leurs rôles](#fichiers-et-leurs-rôles)
4. [Workflows GitHub Actions](#workflows-github-actions)
5. [Comment tout fonctionne ensemble](#comment-tout-fonctionne-ensemble)
6. [Guide d'utilisation](#guide-dutilisation)

---

## 🎯 Vue d'ensemble

Ce projet implémente un système CI/CD (Continuous Integration/Continuous Deployment) pour l'apprentissage automatique (Machine Learning). Il automatise :

- ✅ L'entraînement de modèles ML pour prédire les prix immobiliers
- ✅ L'évaluation et la comparaison avec une baseline
- ✅ La génération de rapports automatiques dans les Pull Requests
- ✅ Les tests de plusieurs modèles/hyperparamètres (matrice d'expériences)

**Technologies utilisées :**
- Python, scikit-learn, pandas, matplotlib
- GitHub Actions (CI/CD)
- CML (Continuous Machine Learning) pour les rapports
- Docker (optionnel)

---

## 📁 Structure du projet

```
ml-cicd-project/
│
├── 📂 .github/
│   └── 📂 workflows/
│       ├── train.yml              # Workflow principal d'entraînement
│       └── experiment-matrix.yml  # Workflow de tests multiples
│
├── 📂 src/                        # Code source Python
│   ├── __init__.py               # Fichier d'initialisation Python
│   ├── data_preprocessing.py     # Prétraitement des données
│   ├── train.py                  # Entraînement du modèle
│   └── evaluate.py               # Évaluation et comparaison
│
├── 📂 tests/                      # Tests unitaires
│   └── test_model.py             # Tests du pipeline ML
│
├── 📂 data/                       # Données (dataset Kaggle House Prices)
│   ├── train.csv                 # Dataset d'entraînement (OBLIGATOIRE)
│   ├── test.csv                  # Dataset de test (optionnel)
│   └── sample_submission.csv     # Exemple de soumission (optionnel)
│
├── 📂 models/                     # Modèles sauvegardés (généré automatiquement)
│   └── model.pkl                 # Modèle entraîné sauvegardé
│
├── 📂 reports/                    # Rapports et graphiques (généré automatiquement)
│   ├── metrics.json              # Métriques du modèle
│   ├── baseline_metrics.json     # Métriques de référence
│   ├── comparison_report.md      # Rapport de comparaison
│   ├── predictions_plot.png     # Graphique prédictions vs réelles
│   ├── residuals_plot.png       # Graphique des résidus
│   ├── feature_importance.png   # Importance des features
│   └── comparison_plot.png      # Comparaison baseline vs actuel
│
├── 📄 params.yaml                # Configuration du projet
├── 📄 requirements.txt           # Dépendances Python
├── 📄 Dockerfile                 # Configuration Docker (optionnel)
├── 📄 .gitignore                 # Fichiers ignorés par Git
└── 📄 PROJECT_DOCUMENTATION.md   # Ce fichier
```

---

## 📄 Fichiers et leurs rôles

### 🔧 Fichiers de configuration

#### `params.yaml`
**Rôle :** Fichier de configuration central du projet.

**Contenu :**
```yaml
data:
  path: 'data/train.csv'      # Chemin vers le dataset
  test_size: 0.2              # 20% pour le test, 80% pour l'entraînement
  random_state: 42            # Graine aléatoire pour reproductibilité
  target: 'SalePrice'         # Variable à prédire

model:
  type: 'random_forest'       # Type de modèle (random_forest, gradient_boosting, ridge, lasso)
  params:
    n_estimators: 100         # Nombre d'arbres (pour Random Forest)
    max_depth: 15             # Profondeur maximale
    min_samples_split: 5     # Échantillons minimum pour diviser
    random_state: 42          # Graine aléatoire

training:
  cv_folds: 5                # Nombre de folds pour validation croisée

evaluation:
  metrics:                    # Métriques à calculer
    - rmse                   # Root Mean Squared Error
    - mae                    # Mean Absolute Error
    - r2                     # Coefficient de détermination R²
  generate_plots: true       # Générer les graphiques
```

**Utilisation :** Modifiez ce fichier pour changer le modèle, les hyperparamètres, etc.

---

#### `requirements.txt`
**Rôle :** Liste toutes les dépendances Python nécessaires.

**Contenu :**
```
pandas==2.0.3          # Manipulation de données
numpy==1.24.3          # Calculs numériques
scikit-learn==1.3.0    # Machine Learning
matplotlib==3.7.2      # Visualisation
seaborn==0.12.2        # Graphiques statistiques
pyyaml==6.0.1          # Lecture des fichiers YAML
```

**Utilisation :** Installez avec `pip install -r requirements.txt`

---

#### `.gitignore`
**Rôle :** Indique à Git quels fichiers ne pas versionner.

**Contenu principal :**
- `__pycache__/` - Fichiers Python compilés
- `models/*.pkl` - Modèles (trop volumineux)
- `reports/*.png` - Graphiques générés
- `*.csv` - Fichiers CSV (sauf `data/*.csv`)
- `venv/` - Environnements virtuels

**Important :** `data/train.csv` est inclus (pas ignoré) car nécessaire pour le workflow.

---

#### `Dockerfile`
**Rôle :** Configuration Docker pour containeriser l'application (optionnel, non utilisé dans les workflows actuellement).

**Contenu :**
- Image de base : Python 3.10
- Installation des dépendances système (git)
- Installation des dépendances Python
- Configuration de l'environnement

---

### 🐍 Code source Python (`src/`)

#### `src/__init__.py`
**Rôle :** Fichier d'initialisation Python qui transforme `src/` en package Python.

**Contenu :** Vide (fichier minimal)

---

#### `src/data_preprocessing.py`
**Rôle :** Prétraitement complet des données avant l'entraînement.

**Fonctions principales :**

1. **`load_data(data_path)`**
   - Charge le fichier CSV
   - Vérifie que le fichier existe
   - Retourne un DataFrame pandas

2. **`clean_data(df)`**
   - Supprime la colonne `Id` (non utile)
   - Gère les valeurs manquantes :
     - Numériques → remplissage par la médiane
     - Catégorielles → remplissage par le mode
   - Supprime les doublons
   - Retire les outliers extrêmes dans `SalePrice`

3. **`encode_categorical_features(df)`**
   - Convertit les variables catégorielles en numériques
   - Utilise LabelEncoder de scikit-learn

4. **`create_features(df)`**
   - Crée de nouvelles features (feature engineering) :
     - `QualityArea = GrLivArea × OverallQual`
     - `TotalSF = TotalBsmtSF + GrLivArea`
     - `Age = 2024 - YearBuilt`
     - `TotalBath` (combinaison de plusieurs colonnes)

5. **`select_features(df)`**
   - Sélectionne les features les plus importantes
   - Liste prédéfinie de 20 features principales

6. **`split_data(df)`**
   - Sépare les données en train (80%) et test (20%)
   - Utilise `train_test_split` de scikit-learn

7. **`scale_features(X_train, X_test)`**
   - Normalise les features avec StandardScaler
   - Important pour certains modèles (Ridge, Lasso)

8. **`preprocess_pipeline()`** ⭐ **Fonction principale**
   - Exécute toutes les étapes ci-dessus dans l'ordre
   - Retourne : `X_train, X_test, y_train, y_test, scaler`

**Utilisation :**
```python
from src.data_preprocessing import preprocess_pipeline
X_train, X_test, y_train, y_test, scaler = preprocess_pipeline()
```

---

#### `src/train.py`
**Rôle :** Entraîne le modèle ML et génère les métriques et graphiques.

**Fonctions principales :**

1. **`load_params(params_file)`**
   - Charge la configuration depuis `params.yaml`

2. **`get_model(model_type, params)`**
   - Crée le modèle selon le type :
     - `random_forest` → RandomForestRegressor
     - `gradient_boosting` → GradientBoostingRegressor
     - `ridge` → Ridge
     - `lasso` → Lasso

3. **`train_model(X_train, y_train, model_type, params)`**
   - Entraîne le modèle sur les données d'entraînement
   - Retourne le modèle entraîné

4. **`calculate_metrics(y_true, y_pred)`**
   - Calcule les métriques :
     - RMSE (Root Mean Squared Error)
     - MAE (Mean Absolute Error)
     - R² (Coefficient de détermination)

5. **`save_model(model, scaler, model_path)`**
   - Sauvegarde le modèle et le scaler dans `models/model.pkl`
   - Format : pickle

6. **`save_metrics(metrics, metrics_path)`**
   - Sauvegarde les métriques dans `reports/metrics.json`
   - Format JSON

7. **`plot_predictions(y_true, y_pred)`**
   - Crée un graphique : Prédictions vs Valeurs réelles
   - Sauvegarde dans `reports/predictions_plot.png`

8. **`plot_residuals(y_true, y_pred)`**
   - Crée deux graphiques des résidus :
     - Résidus vs Prédictions
     - Distribution des résidus
   - Sauvegarde dans `reports/residuals_plot.png`

9. **`plot_feature_importance(model, feature_names)`**
   - Crée un graphique de l'importance des features (top 20)
   - Fonctionne uniquement pour les modèles avec `feature_importances_`
   - Sauvegarde dans `reports/feature_importance.png`

10. **`main()`** ⭐ **Fonction principale**
    - Orchestre tout le processus :
      1. Charge les paramètres
      2. Prétraite les données
      3. Entraîne le modèle
      4. Calcule les métriques
      5. Sauvegarde le modèle et les métriques
      6. Génère les graphiques

**Utilisation :**
```bash
python src/train.py
```

**Fichiers générés :**
- `models/model.pkl`
- `reports/metrics.json`
- `reports/predictions_plot.png`
- `reports/residuals_plot.png`
- `reports/feature_importance.png`

---

#### `src/evaluate.py`
**Rôle :** Évalue le modèle et le compare avec une baseline.

**Fonctions principales :**

1. **`load_model(model_path)`**
   - Charge le modèle sauvegardé depuis `models/model.pkl`
   - Retourne le modèle et le scaler

2. **`load_metrics(metrics_path)`**
   - Charge les métriques depuis `reports/metrics.json`

3. **`compare_with_baseline(current_metrics, baseline_path)`** ⭐
   - Compare les métriques actuelles avec la baseline
   - Si pas de baseline → crée la première baseline
   - Si baseline existe → calcule les améliorations/dégradations
   - Retourne un dictionnaire de comparaison

4. **`generate_comparison_report(comparison)`**
   - Génère un rapport Markdown de comparaison
   - Affiche un tableau avec les métriques
   - Indique les améliorations avec ✅ ou dégradations avec ❌
   - Sauvegarde dans `reports/comparison_report.md`

5. **`plot_comparison(comparison)`**
   - Crée un graphique comparant baseline vs actuel
   - Graphique en barres avec les métriques
   - Sauvegarde dans `reports/comparison_plot.png`

6. **`main()`** ⭐ **Fonction principale**
   - Charge les métriques actuelles
   - Compare avec la baseline
   - Génère le rapport et le graphique

**Utilisation :**
```bash
python src/evaluate.py
```

**Fichiers générés :**
- `reports/baseline_metrics.json` (première fois)
- `reports/comparison_report.md`
- `reports/comparison_plot.png`

**Exemple de rapport généré :**
```markdown
# Rapport de comparaison des modèles

## Métriques

| Métrique | Baseline | Actuel | Amélioration |
|----------|----------|--------|--------------|
| RMSE     | 30000.00 | 25000.00 | ✅ +16.67% |
| MAE      | 25000.00 | 20000.00 | ✅ +20.00% |
| R²       | 0.80     | 0.85     | ✅ +6.25%  |
```

---

### 🧪 Tests (`tests/`)

#### `tests/test_model.py`
**Rôle :** Tests unitaires pour vérifier que le code fonctionne correctement.

**Classes de tests :**

1. **`TestDataPreprocessing`**
   - `test_load_data()` - Vérifie le chargement des données
   - `test_clean_data()` - Vérifie le nettoyage
   - `test_create_features()` - Vérifie la création de features

2. **`TestModelTraining`**
   - `test_get_model()` - Vérifie la création de modèles
   - `test_calculate_metrics()` - Vérifie le calcul des métriques

3. **`TestModelEvaluation`**
   - `test_load_metrics()` - Vérifie le chargement des métriques

4. **`test_end_to_end_pipeline()`**
   - Test complet du pipeline de bout en bout

**Utilisation :**
```bash
pytest tests/ -v
```

---

### 🔄 Workflows GitHub Actions (`.github/workflows/`)

#### `.github/workflows/train.yml`
**Rôle :** Workflow principal qui s'exécute automatiquement à chaque commit/PR.

**Déclencheurs :**
- `push` sur les branches `main` ou `develop`
- `pull_request` vers `main`
- `workflow_dispatch` (déclenchement manuel)

**Étapes du workflow :**

1. **Checkout code**
   - Récupère le code du dépôt

2. **Set up Python**
   - Installe Python 3.9
   - Configure le cache pip

3. **Install dependencies**
   - Installe les dépendances depuis `requirements.txt`
   - Installe pytest pour les tests

4. **Set PYTHONPATH**
   - Configure le chemin Python pour les imports

5. **Check if data exists**
   - Vérifie que `data/train.csv` existe
   - Affiche le nombre de lignes

6. **Train model**
   - Exécute `python src/train.py`
   - Entraîne le modèle

7. **Evaluate model**
   - Exécute `python src/evaluate.py`
   - Compare avec la baseline

8. **Run tests**
   - Exécute `pytest tests/ -v`
   - Continue même en cas d'erreur (`continue-on-error: true`)

9. **Setup CML** (si PR)
   - Configure CML pour publier des rapports

10. **Create CML report** (si PR)
    - Crée un rapport Markdown avec :
      - Métriques JSON
      - Rapport de comparaison
      - Graphiques (via `cml-publish`)
    - Publie le rapport dans la PR (via `cml comment create`)

11. **Upload artifacts**
    - Sauvegarde `models/` et `reports/` comme artifacts
    - Disponible pour téléchargement pendant 30 jours

12. **Summary**
    - Affiche un résumé dans GitHub Actions avec les métriques

**Durée typique :** 30-60 secondes

---

#### `.github/workflows/experiment-matrix.yml`
**Rôle :** Teste automatiquement plusieurs combinaisons de modèles/hyperparamètres.

**Déclencheurs :**
- `workflow_dispatch` (déclenchement manuel)
- `pull_request` vers `main`
- `schedule` (tous les dimanches à 2h du matin)

**Stratégie de matrice :**
```yaml
matrix:
  model_type: ['random_forest', 'gradient_boosting', 'ridge']
  n_estimators: [50, 100, 200]
  max_depth: [10, 15, 20]
```

**Résultat :** Teste 18 combinaisons différentes (excluant les combinaisons invalides pour Ridge)

**Jobs :**

1. **`matrix-experiments`** (exécuté en parallèle)
   - Pour chaque combinaison :
     - Met à jour `params.yaml` avec les paramètres de la matrice
     - Entraîne le modèle
     - Sauvegarde les résultats dans `experiments/`
     - Upload les artifacts

2. **`summarize-experiments`** (après tous les jobs)
   - Télécharge tous les artifacts
   - Agrège les résultats dans un tableau
   - Identifie le meilleur modèle
   - Génère un résumé Markdown
   - Publie un rapport CML si c'est une PR

**Durée typique :** 5-15 minutes (selon le nombre de combinaisons)

---

## 🔄 Comment tout fonctionne ensemble

### Flux complet d'exécution

```
1. Déclenchement (commit/PR)
   ↓
2. GitHub Actions démarre
   ↓
3. Checkout du code
   ↓
4. Installation des dépendances
   ↓
5. Vérification du dataset
   ↓
6. data_preprocessing.py
   ├── Charge data/train.csv
   ├── Nettoie les données
   ├── Encode les variables catégorielles
   ├── Crée de nouvelles features
   ├── Sélectionne les features importantes
   ├── Normalise les données
   └── Sépare en train/test
   ↓
7. train.py
   ├── Charge params.yaml
   ├── Crée le modèle (selon params.yaml)
   ├── Entraîne le modèle
   ├── Fait des prédictions
   ├── Calcule les métriques
   ├── Sauvegarde le modèle (models/model.pkl)
   ├── Sauvegarde les métriques (reports/metrics.json)
   └── Génère les graphiques (reports/*.png)
   ↓
8. evaluate.py
   ├── Charge les métriques actuelles
   ├── Compare avec baseline
   │   ├── Si pas de baseline → crée baseline_metrics.json
   │   └── Si baseline existe → calcule les différences
   ├── Génère comparison_report.md
   └── Génère comparison_plot.png
   ↓
9. Tests unitaires (pytest)
   ↓
10. CML (si PR)
    ├── Crée un rapport Markdown
    ├── Publie les graphiques
    └── Poste le commentaire dans la PR
    ↓
11. Upload des artifacts
    ↓
12. Résumé dans GitHub Actions
```

### Exemple concret

**Scénario :** Vous modifiez `params.yaml` pour changer `n_estimators` de 100 à 150.

1. Vous commitez et poussez vers GitHub
2. GitHub Actions détecte le changement
3. Le workflow s'exécute :
   - Entraîne un Random Forest avec 150 arbres
   - Calcule les métriques
   - Compare avec la baseline précédente (100 arbres)
4. Si c'est une PR :
   - Un commentaire apparaît avec :
     - Les nouvelles métriques
     - La comparaison (amélioration ou dégradation)
     - Les graphiques
5. Vous décidez :
   - ✅ Si amélioration → Merge la PR
   - ❌ Si dégradation → Fermer la PR ou ajuster

---

## 📖 Guide d'utilisation

### Utilisation locale

1. **Installer les dépendances**
   ```bash
   pip install -r requirements.txt
   ```

2. **Configurer PYTHONPATH**
   ```bash
   # Windows PowerShell
   $env:PYTHONPATH = "$PWD;$env:PYTHONPATH"
   
   # Linux/Mac
   export PYTHONPATH=$(pwd):$PYTHONPATH
   ```

3. **Vérifier le dataset**
   ```bash
   # Le fichier data/train.csv doit exister
   ls data/train.csv
   ```

4. **Entraîner le modèle**
   ```bash
   python src/train.py
   ```

5. **Évaluer le modèle**
   ```bash
   python src/evaluate.py
   ```

6. **Voir les résultats**
   - Métriques : `reports/metrics.json`
   - Graphiques : `reports/*.png`
   - Modèle : `models/model.pkl`

### Utilisation avec GitHub Actions

1. **Modifier la configuration**
   - Éditez `params.yaml` pour changer le modèle/hyperparamètres

2. **Commiter et pousser**
   ```bash
   git add params.yaml
   git commit -m "test: changement des hyperparamètres"
   git push origin main
   ```

3. **Vérifier les résultats**
   - Allez sur GitHub → Actions
   - Cliquez sur l'exécution réussie
   - Téléchargez les artifacts ou consultez le résumé

### Tester avec une Pull Request

1. **Créer une branche**
   ```bash
   git checkout -b test-nouveau-modele
   ```

2. **Modifier params.yaml**
   ```yaml
   model:
     type: 'gradient_boosting'  # Changer le modèle
     params:
       n_estimators: 200
   ```

3. **Commiter et pousser**
   ```bash
   git add params.yaml
   git commit -m "test: essai avec Gradient Boosting"
   git push origin test-nouveau-modele
   ```

4. **Créer une Pull Request**
   - Sur GitHub, créez une PR depuis `test-nouveau-modele` vers `main`
   - Le workflow s'exécutera automatiquement
   - Un commentaire CML apparaîtra avec les résultats

### Tester la matrice d'expériences

1. **Aller sur GitHub Actions**
   - Onglet "Actions" → "Experiment Matrix"

2. **Déclencher manuellement**
   - Cliquez sur "Run workflow"
   - Sélectionnez la branche
   - Cliquez sur "Run workflow"

3. **Attendre les résultats**
   - Le workflow créera plusieurs jobs en parallèle
   - Un job de résumé agrègera tous les résultats
   - Les résultats seront dans les artifacts

---

## 🎯 Concepts clés

### Baseline
La **baseline** est le premier modèle entraîné, utilisé comme référence. Tous les modèles suivants sont comparés à cette baseline pour mesurer les améliorations ou dégradations.

**Fichier :** `reports/baseline_metrics.json`

### Pull Request (PR)
Une **Pull Request** est une demande de fusion de changements. Dans ce projet, les PRs déclenchent automatiquement le workflow et génèrent des rapports CML.

### CML (Continuous Machine Learning)
**CML** est un outil qui publie automatiquement des rapports ML dans les PRs GitHub. Il affiche les métriques et graphiques directement dans les commentaires.

### Artifacts
Les **artifacts** sont des fichiers sauvegardés par GitHub Actions. Ils sont disponibles pour téléchargement pendant 30 jours (configurable).

---

## 📊 Métriques expliquées

### RMSE (Root Mean Squared Error)
- **Signification :** Erreur quadratique moyenne racine
- **Interprétation :** Plus c'est bas, mieux c'est
- **Unité :** Même unité que la variable cible (dollars pour les prix)

### MAE (Mean Absolute Error)
- **Signification :** Erreur absolue moyenne
- **Interprétation :** Plus c'est bas, mieux c'est
- **Avantage :** Moins sensible aux outliers que RMSE

### R² (Coefficient de détermination)
- **Signification :** Proportion de variance expliquée
- **Interprétation :** Entre 0 et 1, plus c'est haut, mieux c'est
- **Exemple :** R² = 0.85 signifie que le modèle explique 85% de la variance

---

## 🔧 Personnalisation

### Changer le modèle
Éditez `params.yaml` :
```yaml
model:
  type: 'gradient_boosting'  # Au lieu de 'random_forest'
```

### Changer les hyperparamètres
Éditez `params.yaml` :
```yaml
model:
  params:
    n_estimators: 200      # Plus d'arbres
    max_depth: 20         # Plus profond
```

### Ajouter de nouvelles métriques
Modifiez `src/train.py` dans la fonction `calculate_metrics()`.

### Modifier le workflow
Éditez `.github/workflows/train.yml` pour ajouter des étapes.

---

## 🐛 Dépannage

### Erreur : "ModuleNotFoundError"
**Solution :** Configurez PYTHONPATH
```bash
export PYTHONPATH=$(pwd):$PYTHONPATH
```

### Erreur : "FileNotFoundError: data/train.csv"
**Solution :** Vérifiez que le fichier existe et est commité dans Git

### Le workflow GitHub Actions échoue
**Solution :** 
1. Vérifiez les logs dans GitHub Actions
2. Vérifiez que toutes les dépendances sont dans `requirements.txt`
3. Vérifiez la syntaxe YAML des workflows

### CML ne publie pas de rapport
**Solution :**
1. Vérifiez que c'est bien une Pull Request
2. Vérifiez que les fichiers `reports/*.png` sont générés
3. Vérifiez les permissions dans le workflow

---

## 📚 Ressources

- **GitHub Actions :** https://docs.github.com/en/actions
- **CML :** https://cml.dev/
- **scikit-learn :** https://scikit-learn.org/
- **Kaggle House Prices :** https://www.kaggle.com/c/house-prices-advanced-regression-techniques

---

## ✅ Checklist de vérification

Avant de considérer le projet comme fonctionnel :

- [ ] Le dataset `data/train.csv` est présent
- [ ] Les dépendances sont installées (`pip install -r requirements.txt`)
- [ ] `train.py` s'exécute sans erreur localement
- [ ] `evaluate.py` s'exécute sans erreur localement
- [ ] Les fichiers sont générés (`models/`, `reports/`)
- [ ] Le workflow GitHub Actions s'exécute avec succès
- [ ] Les artifacts sont disponibles dans GitHub Actions
- [ ] Le rapport CML apparaît dans une PR (si testé)

---

**Dernière mise à jour :** 2024
**Version du projet :** 1.0

