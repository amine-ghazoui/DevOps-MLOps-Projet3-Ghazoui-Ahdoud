"""
Module d'évaluation et de comparaison des modèles
"""

import json
import os
import pickle
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Backend non-interactif pour CI/CD
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import sys

# Ajouter le répertoire parent au path pour les imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data_preprocessing import preprocess_pipeline


def load_model(model_path='models/model.pkl'):
    """Charge le modèle sauvegardé"""
    with open(model_path, 'rb') as f:
        saved_data = pickle.load(f)
    
    return saved_data['model'], saved_data['scaler']


def load_metrics(metrics_path='reports/metrics.json'):
    """Charge les métriques sauvegardées"""
    if not os.path.exists(metrics_path):
        return None
    
    with open(metrics_path, 'r') as f:
        metrics = json.load(f)
    
    return metrics


def compare_with_baseline(current_metrics, baseline_path='reports/baseline_metrics.json'):
    """Compare les métriques actuelles avec la baseline"""
    os.makedirs(os.path.dirname(baseline_path), exist_ok=True)
    if not os.path.exists(baseline_path):
        print("Pas de baseline trouvée. Création de la baseline...")
        with open(baseline_path, 'w') as f:
            json.dump(current_metrics, f, indent=4)
        return None
    
    with open(baseline_path, 'r') as f:
        baseline_metrics = json.load(f)
    
    # Calculer les différences
    comparison = {
        'baseline': baseline_metrics['test'],
        'current': current_metrics['test'],
        'improvements': {}
    }
    
    for metric in ['rmse', 'mae', 'r2']:
        baseline_val = baseline_metrics['test'][metric]
        current_val = current_metrics['test'][metric]
        
        if metric == 'r2':
            # Pour R2, plus c'est haut, mieux c'est
            improvement = ((current_val - baseline_val) / abs(baseline_val)) * 100
        else:
            # Pour RMSE et MAE, plus c'est bas, mieux c'est
            improvement = ((baseline_val - current_val) / baseline_val) * 100
        
        comparison['improvements'][metric] = {
            'baseline': baseline_val,
            'current': current_val,
            'improvement_pct': improvement
        }
    
    return comparison


def generate_comparison_report(comparison, output_path='reports/comparison_report.md'):
    """Génère un rapport de comparaison en Markdown"""
    if comparison is None:
        report = "# Rapport d'évaluation\n\n"
        report += "Première exécution - baseline créée.\n"
    else:
        report = "# Rapport de comparaison des modèles\n\n"
        report += "## Métriques\n\n"
        report += "| Métrique | Baseline | Actuel | Amélioration |\n"
        report += "|----------|----------|--------|---------------|\n"
        
        for metric, values in comparison['improvements'].items():
            report += f"| {metric.upper()} | {values['baseline']:.4f} | "
            report += f"{values['current']:.4f} | "
            
            improvement = values['improvement_pct']
            symbol = "✅" if improvement > 0 else "❌"
            report += f"{symbol} {improvement:+.2f}% |\n"
        
        report += "\n## Analyse\n\n"
        
        # Analyse globale
        avg_improvement = np.mean([v['improvement_pct'] for v in comparison['improvements'].values()])
        if avg_improvement > 0:
            report += f"🎉 Le modèle actuel montre une amélioration moyenne de **{avg_improvement:.2f}%**.\n\n"
        else:
            report += f"⚠️ Le modèle actuel montre une dégradation moyenne de **{avg_improvement:.2f}%**.\n\n"
        
        # Détails par métrique
        for metric, values in comparison['improvements'].items():
            improvement = values['improvement_pct']
            if improvement > 5:
                report += f"- **{metric.upper()}**: Amélioration significative ({improvement:+.2f}%)\n"
            elif improvement > 0:
                report += f"- **{metric.upper()}**: Légère amélioration ({improvement:+.2f}%)\n"
            elif improvement > -5:
                report += f"- **{metric.upper()}**: Légère dégradation ({improvement:+.2f}%)\n"
            else:
                report += f"- **{metric.upper()}**: Dégradation significative ({improvement:+.2f}%)\n"
    
    with open(output_path, 'w') as f:
        f.write(report)
    
    print(f"Rapport de comparaison généré: {output_path}")
    return report


def plot_comparison(comparison, output_path='reports/comparison_plot.png'):
    """Crée un graphique de comparaison"""
    if comparison is None:
        return
    
    metrics = list(comparison['improvements'].keys())
    baseline_values = [comparison['improvements'][m]['baseline'] for m in metrics]
    current_values = [comparison['improvements'][m]['current'] for m in metrics]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    bars1 = ax.bar(x - width/2, baseline_values, width, label='Baseline', alpha=0.8)
    bars2 = ax.bar(x + width/2, current_values, width, label='Actuel', alpha=0.8)
    
    ax.set_xlabel('Métriques')
    ax.set_ylabel('Valeurs')
    ax.set_title('Comparaison des modèles')
    ax.set_xticks(x)
    ax.set_xticklabels([m.upper() for m in metrics])
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=120)
    plt.close()
    
    print(f"Graphique de comparaison sauvegardé: {output_path}")


def evaluate_model_on_test_set():
    """Évalue le modèle sur l'ensemble de test"""
    print("Chargement du modèle et des données...")
    
    # Charger le modèle
    model, scaler = load_model()
    
    # Charger et préparer les données
    _, X_test, _, y_test, _ = preprocess_pipeline()
    
    # Prédictions
    print("Calcul des prédictions...")
    y_pred = model.predict(X_test)
    
    # Métriques
    metrics = {
        'rmse': float(np.sqrt(mean_squared_error(y_test, y_pred))),
        'mae': float(mean_absolute_error(y_test, y_pred)),
        'r2': float(r2_score(y_test, y_pred))
    }
    
    return metrics


def main():
    """Fonction principale d'évaluation"""
    print("=== Évaluation du modèle ===\n")
    
    # Charger les métriques actuelles
    current_metrics = load_metrics()
    
    if current_metrics is None:
        print("Aucune métrique trouvée. Exécutez d'abord train.py")
        return
    
    print(f"Modèle: {current_metrics['model_type']}")
    print(f"RMSE Test: {current_metrics['test']['rmse']:.2f}")
    print(f"R2 Test: {current_metrics['test']['r2']:.4f}\n")
    
    # Comparer avec la baseline
    comparison = compare_with_baseline(current_metrics)
    
    # Générer le rapport
    report = generate_comparison_report(comparison)
    print("\n" + report)
    
    # Créer le graphique de comparaison
    if comparison:
        plot_comparison(comparison)
    
    print("\n✓ Évaluation terminée avec succès!")


if __name__ == "__main__":
    main()