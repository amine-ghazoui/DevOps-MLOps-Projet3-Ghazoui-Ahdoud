"""
Module d'entraînement du modèle
"""

import numpy as np
import pandas as pd
import yaml
import json
import os
import pickle
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, Lasso
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib
matplotlib.use('Agg')  # Backend non-interactif pour CI/CD
import matplotlib.pyplot as plt
import seaborn as sns
import sys

# Ajouter le répertoire parent au path pour les imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data_preprocessing import preprocess_pipeline


def load_params(params_file='params.yaml'):
    """Charge les paramètres depuis le fichier YAML"""
    with open(params_file, 'r') as f:
        params = yaml.safe_load(f)
    return params


def get_model(model_type, params):
    """Retourne le modèle selon le type spécifié"""
    models = {
        'random_forest': RandomForestRegressor(
            n_estimators=params.get('n_estimators', 100),
            max_depth=params.get('max_depth', None),
            min_samples_split=params.get('min_samples_split', 2),
            random_state=params.get('random_state', 42)
        ),
        'gradient_boosting': GradientBoostingRegressor(
            n_estimators=params.get('n_estimators', 100),
            max_depth=params.get('max_depth', 3),
            learning_rate=params.get('learning_rate', 0.1),
            min_samples_split=params.get('min_samples_split', 2),
            min_samples_leaf=params.get('min_samples_leaf', 1),
            subsample=params.get('subsample', 1.0),
            random_state=params.get('random_state', 42)
        ),
        'ridge': Ridge(
            alpha=params.get('alpha', 1.0),
            random_state=params.get('random_state', 42)
        ),
        'lasso': Lasso(
            alpha=params.get('alpha', 1.0),
            random_state=params.get('random_state', 42)
        )
    }
    
    return models.get(model_type, models['random_forest'])


def train_model(X_train, y_train, model_type, params):
    """Entraîne le modèle"""
    model = get_model(model_type, params)
    
    print(f"Entraînement du modèle {model_type}...")
    model.fit(X_train, y_train)
    
    return model


def save_model(model, scaler, model_path='models/model.pkl'):
    """Sauvegarde le modèle"""
    os.makedirs('models', exist_ok=True)
    
    with open(model_path, 'wb') as f:
        pickle.dump({'model': model, 'scaler': scaler}, f)
    
    print(f"Modèle sauvegardé: {model_path}")


def calculate_metrics(y_true, y_pred):
    """Calcule les métriques de performance"""
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    return {
        'mse': float(mse),
        'rmse': float(rmse),
        'mae': float(mae),
        'r2': float(r2)
    }


def save_metrics(metrics, metrics_path='reports/metrics.json'):
    """Sauvegarde les métriques"""
    os.makedirs('reports', exist_ok=True)
    
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    
    print(f"Métriques sauvegardées: {metrics_path}")


def plot_predictions(y_true, y_pred, output_path='reports/predictions_plot.png'):
    """Crée un graphique des prédictions vs valeurs réelles"""
    plt.figure(figsize=(10, 6))
    
    plt.scatter(y_true, y_pred, alpha=0.5)
    plt.plot([y_true.min(), y_true.max()], 
             [y_true.min(), y_true.max()], 
             'r--', lw=2)
    
    plt.xlabel('Valeurs réelles')
    plt.ylabel('Prédictions')
    plt.title('Prédictions vs Valeurs réelles')
    plt.tight_layout()
    
    os.makedirs('reports', exist_ok=True)
    plt.savefig(output_path, dpi=120)
    plt.close()
    
    print(f"Graphique sauvegardé: {output_path}")


def plot_residuals(y_true, y_pred, output_path='reports/residuals_plot.png'):
    """Crée un graphique des résidus"""
    residuals = y_true - y_pred
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Résidus vs prédictions
    axes[0].scatter(y_pred, residuals, alpha=0.5)
    axes[0].axhline(y=0, color='r', linestyle='--')
    axes[0].set_xlabel('Prédictions')
    axes[0].set_ylabel('Résidus')
    axes[0].set_title('Résidus vs Prédictions')
    
    # Distribution des résidus
    axes[1].hist(residuals, bins=30, edgecolor='black')
    axes[1].set_xlabel('Résidus')
    axes[1].set_ylabel('Fréquence')
    axes[1].set_title('Distribution des résidus')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=120)
    plt.close()
    
    print(f"Graphique des résidus sauvegardé: {output_path}")


def plot_feature_importance(model, feature_names, output_path='reports/feature_importance.png'):
    """Crée un graphique de l'importance des features"""
    if not hasattr(model, 'feature_importances_'):
        print("Le modèle ne supporte pas feature_importances_")
        return
    
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    # Limiter au top 20 features pour la lisibilité
    top_n = min(20, len(indices))
    top_indices = indices[:top_n]
    top_importances = importances[top_indices]
    top_names = [feature_names[i] for i in top_indices]
    
    plt.figure(figsize=(12, 6))
    plt.bar(range(len(top_importances)), top_importances)
    plt.xticks(range(len(top_importances)), 
               top_names, 
               rotation=45, 
               ha='right')
    plt.xlabel('Features')
    plt.ylabel('Importance')
    plt.title(f'Importance des {len(top_importances)} features principales')
    plt.tight_layout()
    
    plt.savefig(output_path, dpi=120)
    plt.close()
    
    print(f"Graphique d'importance sauvegardé: {output_path}")


def main():
    """Fonction principale d'entraînement"""
    # Charger les paramètres
    params = load_params()
    
    # Prétraiter les données
    print("Prétraitement des données...")
    result = preprocess_pipeline()
    X_train, X_test, y_train, y_test, scaler = result[:5]
    log_transform = result[5] if len(result) > 5 else False
    y_train_orig = result[6] if len(result) > 6 else y_train
    y_test_orig = result[7] if len(result) > 7 else y_test
    
    # Récupérer les paramètres du modèle
    model_type = params.get('model', {}).get('type', 'random_forest')
    model_params = params.get('model', {}).get('params', {})
    
    # Entraîner le modèle
    model = train_model(X_train, y_train, model_type, model_params)
    
    # Prédictions (sur l'échelle log si transformation appliquée)
    print("Calcul des prédictions...")
    y_pred_train_log = model.predict(X_train)
    y_pred_test_log = model.predict(X_test)
    
    # Transformer les prédictions en arrière (expm1 = exp(x) - 1, inverse de log1p)
    if log_transform:
        y_pred_train = np.expm1(y_pred_train_log)
        y_pred_test = np.expm1(y_pred_test_log)
        # Utiliser les valeurs originales pour les métriques
        y_train_for_metrics = y_train_orig
        y_test_for_metrics = y_test_orig
    else:
        y_pred_train = y_pred_train_log
        y_pred_test = y_pred_test_log
        y_train_for_metrics = y_train
        y_test_for_metrics = y_test
    
    # Calculer les métriques sur l'échelle originale
    train_metrics = calculate_metrics(y_train_for_metrics, y_pred_train)
    test_metrics = calculate_metrics(y_test_for_metrics, y_pred_test)
    
    metrics = {
        'model_type': model_type,
        'model_params': model_params,
        'log_transform': log_transform,
        'train': train_metrics,
        'test': test_metrics
    }
    
    print(f"\nMétriques Train - RMSE: {train_metrics['rmse']:.2f}, R2: {train_metrics['r2']:.4f}")
    print(f"Métriques Test  - RMSE: {test_metrics['rmse']:.2f}, R2: {test_metrics['r2']:.4f}")
    if log_transform:
        print("(Métriques calculées après transformation inverse log)")
    
    # Sauvegarder
    save_metrics(metrics)
    save_model(model, scaler)
    
    # Créer les visualisations
    print("\nCréation des visualisations...")
    plot_predictions(y_test_for_metrics, y_pred_test)
    plot_residuals(y_test_for_metrics, y_pred_test)
    plot_feature_importance(model, X_train.columns.tolist())
    
    print("\n✓ Entraînement terminé avec succès!")


if __name__ == "__main__":
    main()