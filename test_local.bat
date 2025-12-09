@echo off
REM Script de test local pour Windows

echo 🧪 Tests locaux du projet ML CI/CD
echo ==================================
echo.

REM 1. Vérifier Python
echo 1️⃣  Vérification de Python...
python --version >nul 2>&1
if %errorlevel% equ 0 (
    python --version
    echo ✅ Python trouvé
) else (
    echo ❌ Python n'est pas installé
    exit /b 1
)
echo.

REM 2. Vérifier les dépendances
echo 2️⃣  Vérification des dépendances...
if exist requirements.txt (
    echo ✅ requirements.txt trouvé
    echo Installation des dépendances...
    pip install -q -r requirements.txt
    if %errorlevel% equ 0 (
        echo ✅ Dépendances installées
    ) else (
        echo ❌ Erreur lors de l'installation
        exit /b 1
    )
) else (
    echo ❌ requirements.txt non trouvé
    exit /b 1
)
echo.

REM 3. Vérifier le dataset
echo 3️⃣  Vérification du dataset...
if exist data\train.csv (
    echo ✅ Dataset trouvé: data\train.csv
) else (
    echo ❌ Dataset non trouvé: data\train.csv
    echo ⚠️  Téléchargez le dataset depuis Kaggle
    exit /b 1
)
echo.

REM 4. Configurer PYTHONPATH
echo 4️⃣  Configuration de PYTHONPATH...
set PYTHONPATH=%CD%;%PYTHONPATH%
echo ✅ PYTHONPATH configuré
echo.

REM 5. Tester le prétraitement
echo 5️⃣  Test du prétraitement des données...
python -c "import sys; sys.path.insert(0, '.'); from src.data_preprocessing import preprocess_pipeline; X_train, X_test, y_train, y_test, scaler = preprocess_pipeline(); print(f'✅ Prétraitement réussi: Train={X_train.shape}, Test={X_test.shape}')"
if %errorlevel% neq 0 (
    echo ❌ Le prétraitement a échoué
    exit /b 1
)
echo.

REM 6. Tester l'entraînement
echo 6️⃣  Test de l'entraînement...
python src\train.py
if %errorlevel% equ 0 (
    echo ✅ Entraînement réussi
) else (
    echo ❌ L'entraînement a échoué
    exit /b 1
)
echo.

REM 7. Vérifier les fichiers générés
echo 7️⃣  Vérification des fichiers générés...
if exist models\model.pkl echo ✅ models\model.pkl existe
if exist reports\metrics.json echo ✅ reports\metrics.json existe
if exist reports\predictions_plot.png echo ✅ reports\predictions_plot.png existe
echo.

REM 8. Tester l'évaluation
echo 8️⃣  Test de l'évaluation...
python src\evaluate.py
if %errorlevel% equ 0 (
    echo ✅ Évaluation réussi
) else (
    echo ❌ L'évaluation a échoué
    exit /b 1
)
echo.

REM 9. Exécuter les tests unitaires
echo 9️⃣  Exécution des tests unitaires...
pytest tests\ -v
if %errorlevel% equ 0 (
    echo ✅ Tous les tests unitaires passent
) else (
    echo ⚠️  Certains tests ont échoué
)
echo.

echo ==================================
echo 📊 Résumé des tests
echo ==================================
echo ✅ Tests locaux terminés !
echo.
echo Prochaines étapes:
echo 1. Commitez vos changements: git add . ^&^& git commit -m "test"
echo 2. Créez une branche: git checkout -b test-branch
echo 3. Poussez vers GitHub: git push origin test-branch
echo 4. Créez une Pull Request sur GitHub
echo 5. Vérifiez que les workflows GitHub Actions s'exécutent

