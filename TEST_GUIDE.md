# 🧪 Guide de Test du Projet ML CI/CD

Ce guide vous explique comment tester votre projet pour vérifier que tout fonctionne correctement.

## 📋 Table des matières

1. [Tests locaux](#tests-locaux)
2. [Tests sur GitHub Actions](#tests-sur-github-actions)
3. [Tests manuels](#tests-manuels)
4. [Dépannage](#dépannage)

---

## 🖥️ Tests locaux

### Option 1 : Script automatique (Recommandé)

#### Sur Windows :
```bash
test_local.bat
```

#### Sur Linux/Mac :
```bash
chmod +x test_local.sh
./test_local.sh
```

### Option 2 : Tests manuels étape par étape

#### 1. Installer les dépendances
```bash
pip install -r requirements.txt
pip install pytest  # Pour les tests unitaires
```

#### 2. Configurer PYTHONPATH
```bash
# Windows (PowerShell)
$env:PYTHONPATH = "$PWD;$env:PYTHONPATH"

# Linux/Mac
export PYTHONPATH=$(pwd):$PYTHONPATH
```

#### 3. Vérifier le dataset
```bash
# Vérifier que le fichier existe
ls data/train.csv
```

#### 4. Tester le prétraitement
```bash
python -c "from src.data_preprocessing import preprocess_pipeline; X_train, X_test, y_train, y_test, scaler = preprocess_pipeline(); print('✅ OK')"
```

#### 5. Tester l'entraînement
```bash
python src/train.py
```

**Résultat attendu :**
- ✅ Modèle sauvegardé dans `models/model.pkl`
- ✅ Métriques sauvegardées dans `reports/metrics.json`
- ✅ Graphiques créés dans `reports/`

#### 6. Tester l'évaluation
```bash
python src/evaluate.py
```

**Résultat attendu :**
- ✅ Baseline créée dans `reports/baseline_metrics.json`
- ✅ Rapport de comparaison dans `reports/comparison_report.md`
- ✅ Graphique de comparaison dans `reports/comparison_plot.png`

#### 7. Exécuter les tests unitaires
```bash
pytest tests/ -v
```

---

## ☁️ Tests sur GitHub Actions

### Méthode 1 : Via Pull Request (Recommandé)

1. **Créer une branche**
   ```bash
   git checkout -b test-workflow
   ```

2. **Faire un petit changement** (pour déclencher le workflow)
   ```bash
   # Modifier params.yaml par exemple
   # Changer n_estimators de 100 à 150
   ```

3. **Commiter et pousser**
   ```bash
   git add .
   git commit -m "test: vérification du workflow CI/CD"
   git push origin test-workflow
   ```

4. **Créer une Pull Request sur GitHub**
   - Allez sur votre dépôt GitHub
   - Cliquez sur "Pull requests" → "New pull request"
   - Sélectionnez votre branche `test-workflow`
   - Créez la PR

5. **Vérifier l'exécution du workflow**
   - Dans la PR, allez dans l'onglet "Checks" ou "Actions"
   - Vous devriez voir le workflow "Train ML Model" s'exécuter
   - Attendez la fin de l'exécution (environ 2-5 minutes)

6. **Vérifier le rapport CML**
   - À la fin du workflow, un commentaire automatique devrait apparaître dans la PR
   - Ce commentaire contient les métriques et graphiques générés par CML

### Méthode 2 : Via workflow_dispatch (Test manuel)

1. **Aller sur GitHub Actions**
   - Allez sur votre dépôt → onglet "Actions"

2. **Sélectionner le workflow**
   - Cliquez sur "Train ML Model" dans la liste

3. **Déclencher manuellement**
   - Cliquez sur "Run workflow"
   - Sélectionnez la branche (ex: `main`)
   - Cliquez sur "Run workflow"

4. **Suivre l'exécution**
   - Cliquez sur le workflow en cours d'exécution
   - Suivez les logs en temps réel

### Méthode 3 : Test de la matrice d'expériences

1. **Aller sur GitHub Actions**
   - Onglet "Actions" → "Experiment Matrix"

2. **Déclencher le workflow**
   - Cliquez sur "Run workflow"
   - Ce workflow teste plusieurs combinaisons de modèles/hyperparamètres

3. **Vérifier les résultats**
   - Le workflow crée plusieurs jobs en parallèle
   - Un job de résumé agrège tous les résultats
   - Les résultats sont sauvegardés dans les artifacts

---

## 🔍 Tests manuels

### Vérifier les fichiers générés

Après l'exécution de `train.py`, vérifiez que ces fichiers existent :

```bash
# Modèle
models/model.pkl

# Métriques
reports/metrics.json

# Graphiques
reports/predictions_plot.png
reports/residuals_plot.png
reports/feature_importance.png
```

### Vérifier le contenu des métriques

```bash
# Afficher les métriques
cat reports/metrics.json

# Ou sur Windows
type reports\metrics.json
```

**Format attendu :**
```json
{
    "model_type": "random_forest",
    "model_params": {...},
    "train": {
        "rmse": 25000.0,
        "mae": 20000.0,
        "r2": 0.85
    },
    "test": {
        "rmse": 30000.0,
        "mae": 25000.0,
        "r2": 0.80
    }
}
```

### Tester avec différents modèles

Modifiez `params.yaml` pour tester différents modèles :

```yaml
# Test 1: Random Forest
model:
  type: 'random_forest'

# Test 2: Gradient Boosting
model:
  type: 'gradient_boosting'

# Test 3: Ridge
model:
  type: 'ridge'
```

Exécutez `train.py` et `evaluate.py` pour chaque configuration.

---

## 🐛 Dépannage

### Problème : "ModuleNotFoundError: No module named 'src'"

**Solution :**
```bash
# Configurer PYTHONPATH
export PYTHONPATH=$(pwd):$PYTHONPATH  # Linux/Mac
# ou
set PYTHONPATH=%CD%;%PYTHONPATH%  # Windows CMD
```

### Problème : "FileNotFoundError: data/train.csv"

**Solution :**
- Vérifiez que le fichier existe : `ls data/train.csv`
- Si absent, téléchargez-le depuis Kaggle
- Placez-le dans le dossier `data/`

### Problème : "Le workflow GitHub Actions ne s'exécute pas"

**Solutions :**
1. Vérifiez que les fichiers `.github/workflows/*.yml` sont bien commités
2. Vérifiez la syntaxe YAML (pas d'erreurs de formatage)
3. Vérifiez que vous avez les permissions sur le dépôt
4. Regardez l'onglet "Actions" pour voir les erreurs

### Problème : "CML ne publie pas de rapport dans la PR"

**Solutions :**
1. Vérifiez que le workflow s'exécute bien sur une PR (`if: github.event_name == 'pull_request'`)
2. Vérifiez que le token `GITHUB_TOKEN` est disponible
3. Vérifiez que les fichiers `reports/*.png` sont bien générés
4. Regardez les logs du workflow pour voir les erreurs

### Problème : "Les tests unitaires échouent"

**Solutions :**
1. Vérifiez que le dataset est présent
2. Certains tests peuvent être ignorés si le dataset n'est pas disponible :
   ```python
   pytest tests/ -v --ignore=tests/test_model.py::TestDataPreprocessing::test_load_data
   ```

---

## ✅ Checklist de vérification

Avant de considérer que tout fonctionne :

- [ ] Les tests locaux passent (`test_local.sh` ou `test_local.bat`)
- [ ] `train.py` s'exécute sans erreur
- [ ] `evaluate.py` s'exécute sans erreur
- [ ] Les fichiers sont générés correctement (`models/`, `reports/`)
- [ ] Les tests unitaires passent (`pytest tests/`)
- [ ] Le workflow GitHub Actions s'exécute sur une PR
- [ ] Le rapport CML apparaît dans la PR
- [ ] La comparaison avec la baseline fonctionne

---

## 📞 Besoin d'aide ?

Si vous rencontrez des problèmes :

1. Vérifiez les logs d'erreur
2. Consultez la section Dépannage ci-dessus
3. Vérifiez que toutes les dépendances sont installées
4. Vérifiez que le dataset est présent et valide

