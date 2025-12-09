"""
Module de prétraitement des données - Dataset Kaggle House Prices
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import yaml
import os


def load_params(params_file='params.yaml'):
    """Charge les paramètres depuis le fichier YAML"""
    with open(params_file, 'r') as f:
        params = yaml.safe_load(f)
    return params


def load_data(data_path='data/train.csv'):
    """Charge les données depuis le fichier CSV Kaggle"""
    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"Fichier {data_path} non trouvé. "
            "Téléchargez le dataset depuis Kaggle: "
            "https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data"
        )
    
    df = pd.read_csv(data_path)
    print(f"Dataset chargé: {df.shape}")
    print(f"Colonnes: {list(df.columns)}")
    return df


def clean_data(df):
    """Nettoie les données"""
    df = df.copy()
    
    # Supprimer la colonne Id (pas utile pour la prédiction)
    if 'Id' in df.columns:
        df = df.drop('Id', axis=1)
    
    # Gérer les valeurs manquantes
    # Pour les colonnes numériques: remplir avec la médiane
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df[col].isnull().sum() > 0:
            df[col].fillna(df[col].median(), inplace=True)
    
    # Pour les colonnes catégorielles: remplir avec 'None' ou le mode
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        if df[col].isnull().sum() > 0:
            df[col].fillna(df[col].mode()[0] if len(df[col].mode()) > 0 else 'None', inplace=True)
    
    # Supprimer les doublons
    df = df.drop_duplicates()
    
    # Supprimer les outliers extrêmes dans SalePrice (si présent)
    if 'SalePrice' in df.columns:
        df = df[df['SalePrice'] > 0]
        # Retirer les valeurs extrêmes (au-delà de 3 écarts-types)
        mean_price = df['SalePrice'].mean()
        std_price = df['SalePrice'].std()
        df = df[df['SalePrice'] <= mean_price + 3 * std_price]
    
    return df


def encode_categorical_features(df):
    """Encode les features catégorielles"""
    df = df.copy()
    
    # Identifier les colonnes catégorielles
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    # Retirer SalePrice si présente
    if 'SalePrice' in categorical_cols:
        categorical_cols.remove('SalePrice')
    
    # Encoder les variables catégorielles
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le
    
    return df, label_encoders


def create_features(df):
    """Crée de nouvelles features"""
    df = df.copy()
    
    # Features d'interaction si les colonnes existent
    if 'GrLivArea' in df.columns and 'OverallQual' in df.columns:
        df['QualityArea'] = df['GrLivArea'] * df['OverallQual']
    
    if 'TotalBsmtSF' in df.columns and 'GrLivArea' in df.columns:
        df['TotalSF'] = df['TotalBsmtSF'] + df['GrLivArea']
    
    if 'YearBuilt' in df.columns and 'YearRemodAdd' in df.columns:
        df['Age'] = 2024 - df['YearBuilt']
        df['RemodAge'] = 2024 - df['YearRemodAdd']
    
    if 'BsmtFullBath' in df.columns and 'BsmtHalfBath' in df.columns:
        if 'FullBath' in df.columns and 'HalfBath' in df.columns:
            df['TotalBath'] = (df['BsmtFullBath'] + df['FullBath'] + 
                               0.5 * (df['BsmtHalfBath'] + df['HalfBath']))
    
    if 'GarageArea' in df.columns and 'GarageCars' in df.columns:
        # Éviter la division par zéro
        df['AvgGarageCarSpace'] = np.where(
            df['GarageCars'] > 0,
            df['GarageArea'] / df['GarageCars'],
            0
        )
    
    return df


def select_features(df, target='SalePrice'):
    """Sélectionne les features les plus importantes"""
    # Liste des features à utiliser (ajustez selon vos besoins)
    important_features = [
        'OverallQual', 'GrLivArea', 'GarageCars', 'GarageArea', 'TotalBsmtSF',
        '1stFlrSF', 'FullBath', 'TotRmsAbvGrd', 'YearBuilt', 'YearRemodAdd',
        'Fireplaces', 'BsmtFinSF1', 'LotArea', 'OpenPorchSF', 'WoodDeckSF',
        'MSSubClass', 'LotFrontage', 'BsmtUnfSF', 'BedroomAbvGr', 'KitchenAbvGr'
    ]
    
    # Ajouter les features créées si elles existent
    engineered_features = ['QualityArea', 'TotalSF', 'Age', 'RemodAge', 'TotalBath', 'AvgGarageCarSpace']
    
    # Sélectionner uniquement les colonnes qui existent dans le dataframe
    available_features = [f for f in important_features + engineered_features if f in df.columns]
    
    # Inclure toutes les colonnes numériques si la liste est trop petite
    if len(available_features) < 10:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if target in numeric_cols:
            numeric_cols.remove(target)
        available_features = numeric_cols
    
    print(f"Features sélectionnées: {len(available_features)}")
    
    return df[available_features + ([target] if target in df.columns else [])]


def split_data(df, target='SalePrice', test_size=0.2, random_state=42):
    """Sépare les données en ensembles train/test"""
    if target not in df.columns:
        raise ValueError(f"Colonne cible '{target}' non trouvée dans le dataset")
    
    X = df.drop(target, axis=1)
    y = df[target]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=test_size, 
        random_state=random_state
    )
    
    return X_train, X_test, y_train, y_test


def scale_features(X_train, X_test):
    """Normalise les features"""
    scaler = StandardScaler()
    
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Convertir en DataFrame pour conserver les noms de colonnes
    X_train_scaled = pd.DataFrame(
        X_train_scaled, 
        columns=X_train.columns,
        index=X_train.index
    )
    X_test_scaled = pd.DataFrame(
        X_test_scaled, 
        columns=X_test.columns,
        index=X_test.index
    )
    
    return X_train_scaled, X_test_scaled, scaler


def preprocess_pipeline(data_path='data/train.csv', params_file='params.yaml'):
    """Pipeline complet de prétraitement pour le dataset Kaggle"""
    # Charger les paramètres
    params = load_params(params_file)
    
    # Charger les données
    print("\n=== Chargement des données ===")
    df = load_data(data_path)
    
    # Nettoyer
    print("\n=== Nettoyage des données ===")
    df = clean_data(df)
    print(f"Après nettoyage: {df.shape}")
    
    # Encoder les variables catégorielles
    print("\n=== Encodage des variables catégorielles ===")
    df, label_encoders = encode_categorical_features(df)
    
    # Créer features
    print("\n=== Création de features ===")
    df = create_features(df)
    
    # Sélectionner features
    print("\n=== Sélection des features ===")
    df = select_features(df, target='SalePrice')
    print(f"Shape après sélection: {df.shape}")
    
    # Séparer
    test_size = params.get('data', {}).get('test_size', 0.2)
    random_state = params.get('data', {}).get('random_state', 42)
    
    print("\n=== Séparation train/test ===")
    X_train, X_test, y_train, y_test = split_data(
        df, 
        target='SalePrice',
        test_size=test_size,
        random_state=random_state
    )
    
    # Normaliser
    print("\n=== Normalisation ===")
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)
    
    print(f"\nTrain set: {X_train_scaled.shape}")
    print(f"Test set: {X_test_scaled.shape}")
    print(f"Features: {list(X_train_scaled.columns)}\n")
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler


if __name__ == "__main__":
    # Test du pipeline
    try:
        X_train, X_test, y_train, y_test, scaler = preprocess_pipeline()
        print("\n✓ Prétraitement réussi!")
        print(f"\nStatistiques de SalePrice:")
        print(f"  Train - Min: ${y_train.min():,.0f}, Max: ${y_train.max():,.0f}, Mean: ${y_train.mean():,.0f}")
        print(f"  Test  - Min: ${y_test.min():,.0f}, Max: ${y_test.max():,.0f}, Mean: ${y_test.mean():,.0f}")
    except Exception as e:
        print(f"\n❌ Erreur: {e}")
        print("\nAssurez-vous d'avoir téléchargé le dataset depuis:")
        print("https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data")
        print("Et placez le fichier 'train.csv' dans le dossier 'data/'")