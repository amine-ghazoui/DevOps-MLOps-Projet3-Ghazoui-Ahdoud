"""
Tests pour le pipeline ML
"""

import pytest
import pandas as pd
import numpy as np
import os
import sys

# Ajouter le répertoire racine au path pour les imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_preprocessing import (
    load_data, clean_data, encode_categorical_features,
    create_features, preprocess_pipeline
)
from src.train import get_model, calculate_metrics
from src.evaluate import load_model, load_metrics


class TestDataPreprocessing:
    """Tests pour le prétraitement des données"""
    
    def test_load_data(self):
        """Test du chargement des données"""
        # Ce test nécessite que le fichier existe
        if os.path.exists('data/train.csv'):
            df = load_data('data/train.csv')
            assert isinstance(df, pd.DataFrame)
            assert len(df) > 0
            assert 'SalePrice' in df.columns
    
    def test_clean_data(self):
        """Test du nettoyage des données"""
        # Créer des données de test avec des valeurs manquantes
        df = pd.DataFrame({
            'SalePrice': [100000, 200000, np.nan, 300000],
            'GrLivArea': [1500, np.nan, 2000, 2500],
            'OverallQual': [7, 8, 9, 10]
        })
        
        df_clean = clean_data(df)
        
        # Vérifier qu'il n'y a plus de valeurs manquantes
        assert df_clean.isnull().sum().sum() == 0
        
        # Vérifier qu'il n'y a plus de prix négatifs
        if 'SalePrice' in df_clean.columns:
            assert (df_clean['SalePrice'] > 0).all()
    
    def test_create_features(self):
        """Test de la création de features"""
        df = pd.DataFrame({
            'GrLivArea': [1500, 2000, 2500],
            'OverallQual': [7, 8, 9],
            'YearBuilt': [2000, 2010, 2015],
            'YearRemodAdd': [2005, 2012, 2018]
        })
        
        df_features = create_features(df)
        
        # Vérifier que de nouvelles colonnes ont été créées
        assert 'Age' in df_features.columns
        assert 'RemodAge' in df_features.columns


class TestModelTraining:
    """Tests pour l'entraînement du modèle"""
    
    def test_get_model(self):
        """Test de la création de modèles"""
        params = {'n_estimators': 10, 'random_state': 42}
        
        model = get_model('random_forest', params)
        assert model is not None
        assert hasattr(model, 'fit')
        assert hasattr(model, 'predict')
    
    def test_calculate_metrics(self):
        """Test du calcul des métriques"""
        y_true = np.array([100000, 200000, 300000, 400000])
        y_pred = np.array([110000, 190000, 310000, 390000])
        
        metrics = calculate_metrics(y_true, y_pred)
        
        assert 'mae' in metrics
        assert 'r2' in metrics
        assert metrics['mae'] > 0
        assert 0 <= metrics['r2'] <= 1


class TestModelEvaluation:
    """Tests pour l'évaluation du modèle"""
    
    def test_load_metrics(self):
        """Test du chargement des métriques"""
        if os.path.exists('reports/metrics.json'):
            metrics = load_metrics()
            assert metrics is not None
            assert 'test' in metrics
            assert 'mae' in metrics['test']
            assert 'r2' in metrics['test']


def test_end_to_end_pipeline():
    """Test du pipeline complet"""
    if not os.path.exists('data/train.csv'):
        pytest.skip("Dataset non disponible")
    
    try:
        # Exécuter le pipeline
        X_train, X_test, y_train, y_test, scaler = preprocess_pipeline()
        
        # Vérifications
        assert X_train.shape[0] > 0
        assert X_test.shape[0] > 0
        assert len(y_train) == X_train.shape[0]
        assert len(y_test) == X_test.shape[0]
        
        # Vérifier qu'il n'y a pas de valeurs manquantes
        assert X_train.isnull().sum().sum() == 0
        assert X_test.isnull().sum().sum() == 0
        
        print(f"✓ Pipeline test réussi: Train={X_train.shape}, Test={X_test.shape}")
        
    except Exception as e:
        pytest.fail(f"Pipeline a échoué: {str(e)}")


if __name__ == "__main__":
    pytest.main([__file__, '-v'])