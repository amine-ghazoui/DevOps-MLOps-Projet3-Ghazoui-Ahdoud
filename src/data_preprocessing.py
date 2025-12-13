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
    """Nettoie les données avec une gestion améliorée des valeurs manquantes"""
    df = df.copy()
    
    # Supprimer la colonne Id (pas utile pour la prédiction)
    if 'Id' in df.columns:
        df = df.drop('Id', axis=1)
    
    # Gérer les valeurs manquantes de manière plus sophistiquée
    # Colonnes où NaN signifie "pas de X" -> remplacer par 0 ou 'None'
    missing_as_none = ['Alley', 'PoolQC', 'Fence', 'MiscFeature', 'FireplaceQu',
                       'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond',
                       'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',
                       'MasVnrType']
    
    for col in missing_as_none:
        if col in df.columns:
            if df[col].dtype == 'object':
                df[col].fillna('None', inplace=True)
            else:
                df[col].fillna(0, inplace=True)
    
    # Pour les colonnes numériques: remplir avec la médiane (plus robuste que la moyenne)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df[col].isnull().sum() > 0:
            # Utiliser la médiane pour les valeurs manquantes
            df[col].fillna(df[col].median(), inplace=True)
    
    # Pour les colonnes catégorielles restantes: remplir avec le mode
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        if df[col].isnull().sum() > 0:
            mode_value = df[col].mode()[0] if len(df[col].mode()) > 0 else 'None'
            df[col].fillna(mode_value, inplace=True)
    
    # Supprimer les doublons
    df = df.drop_duplicates()
    
    # Gérer les outliers dans SalePrice (si présent)
    if 'SalePrice' in df.columns:
        df = df[df['SalePrice'] > 0]
        # Utiliser l'IQR (Interquartile Range) pour une détection d'outliers plus robuste
        Q1 = df['SalePrice'].quantile(0.25)
        Q3 = df['SalePrice'].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 3 * IQR  # 3 IQR pour être moins agressif
        upper_bound = Q3 + 3 * IQR
        df = df[(df['SalePrice'] >= lower_bound) & (df['SalePrice'] <= upper_bound)]
    
    return df


def encode_categorical_features(df, target='SalePrice', use_target_encoding=True):
    """Encode les features catégorielles avec Target Encoding (plus performant) ou Label Encoding"""
    df = df.copy()
    
    # Identifier les colonnes catégorielles
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    # Retirer SalePrice si présente
    if target in categorical_cols:
        categorical_cols.remove(target)
    
    encoders = {}
    
    if use_target_encoding and target in df.columns:
        # Target Encoding (mean encoding) - meilleur que Label Encoding
        for col in categorical_cols:
            # Calculer la moyenne du target par catégorie
            target_mean = df.groupby(col)[target].mean()
            # Remplacer les catégories par leur moyenne de target
            df[col] = df[col].map(target_mean)
            # Pour les valeurs manquantes (nouvelles catégories), utiliser la moyenne globale
            df[col].fillna(df[target].mean(), inplace=True)
            encoders[col] = target_mean
    else:
        # Label Encoding (fallback)
        for col in categorical_cols:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            encoders[col] = le
    
    return df, encoders


def create_features(df):
    """Crée de nouvelles features améliorées"""
    df = df.copy()
    
    # Features d'interaction si les colonnes existent
    if 'GrLivArea' in df.columns and 'OverallQual' in df.columns:
        df['QualityArea'] = df['GrLivArea'] * df['OverallQual']
        df['QualityArea2'] = df['GrLivArea'] * (df['OverallQual'] ** 2)  # Interaction non-linéaire
    
    if 'TotalBsmtSF' in df.columns and 'GrLivArea' in df.columns:
        df['TotalSF'] = df['TotalBsmtSF'] + df['GrLivArea']
        df['SF_per_Room'] = np.where(
            df['TotRmsAbvGrd'] > 0,
            df['GrLivArea'] / df['TotRmsAbvGrd'],
            0
        ) if 'TotRmsAbvGrd' in df.columns else df['TotalSF']
    
    if 'YearBuilt' in df.columns and 'YearRemodAdd' in df.columns:
        current_year = 2024
        df['Age'] = current_year - df['YearBuilt']
        df['RemodAge'] = current_year - df['YearRemodAdd']
        df['YearsSinceRemod'] = df['YearRemodAdd'] - df['YearBuilt']
        df['RemodIndicator'] = (df['YearRemodAdd'] != df['YearBuilt']).astype(int)
    
    if 'BsmtFullBath' in df.columns and 'BsmtHalfBath' in df.columns:
        if 'FullBath' in df.columns and 'HalfBath' in df.columns:
            df['TotalBath'] = (df['BsmtFullBath'] + df['FullBath'] + 
                               0.5 * (df['BsmtHalfBath'] + df['HalfBath']))
            df['HasHalfBath'] = ((df['BsmtHalfBath'] + df['HalfBath']) > 0).astype(int)
    
    if 'GarageArea' in df.columns and 'GarageCars' in df.columns:
        # Éviter la division par zéro
        df['AvgGarageCarSpace'] = np.where(
            df['GarageCars'] > 0,
            df['GarageArea'] / df['GarageCars'],
            0
        )
        df['HasGarage'] = (df['GarageArea'] > 0).astype(int)
    
    # Features de ratio
    if 'LotArea' in df.columns and 'GrLivArea' in df.columns:
        df['LotAreaRatio'] = np.where(
            df['LotArea'] > 0,
            df['GrLivArea'] / df['LotArea'],
            0
        )
    
    if 'TotalBsmtSF' in df.columns and '1stFlrSF' in df.columns:
        df['BsmtRatio'] = np.where(
            df['1stFlrSF'] > 0,
            df['TotalBsmtSF'] / df['1stFlrSF'],
            0
        )
    
    # Features de qualité combinées
    if 'OverallQual' in df.columns and 'OverallCond' in df.columns:
        df['OverallScore'] = df['OverallQual'] * df['OverallCond']
        df['QualityCondDiff'] = df['OverallQual'] - df['OverallCond']
    
    # Features d'extérieur
    if 'OpenPorchSF' in df.columns and 'EnclosedPorch' in df.columns:
        if '3SsnPorch' in df.columns and 'ScreenPorch' in df.columns:
            df['TotalPorchSF'] = (df['OpenPorchSF'] + df['EnclosedPorch'] + 
                                  df.get('3SsnPorch', 0) + df.get('ScreenPorch', 0))
    
    # Features de décennie
    if 'YearBuilt' in df.columns:
        df['DecadeBuilt'] = (df['YearBuilt'] // 10) * 10
    
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


def split_data(df, target='SalePrice', test_size=0.2, random_state=42, log_transform_target=True):
    """Sépare les données en ensembles train/test avec transformation log optionnelle"""
    if target not in df.columns:
        raise ValueError(f"Colonne cible '{target}' non trouvée dans le dataset")
    
    X = df.drop(target, axis=1)
    y = df[target].copy()
    
    # Séparer d'abord
    X_train, X_test, y_train_orig, y_test_orig = train_test_split(
        X, y, 
        test_size=test_size, 
        random_state=random_state
    )
    
    # Transformation log après séparation (sur train et test séparément)
    if log_transform_target:
        y_train = np.log1p(y_train_orig)  # log1p = log(1+x) pour éviter log(0)
        y_test = np.log1p(y_test_orig)
    else:
        y_train = y_train_orig
        y_test = y_test_orig
    
    return X_train, X_test, y_train, y_test, y_train_orig, y_test_orig, log_transform_target


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
    
    # Encoder les variables catégorielles (AVANT la séparation train/test pour target encoding)
    print("\n=== Encodage des variables catégorielles ===")
    df, encoders = encode_categorical_features(df, target='SalePrice', use_target_encoding=True)
    
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
    X_train, X_test, y_train, y_test, y_train_orig, y_test_orig, log_transform = split_data(
        df, 
        target='SalePrice',
        test_size=test_size,
        random_state=random_state,
        log_transform_target=True
    )
    
    # Normaliser
    print("\n=== Normalisation ===")
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)
    
    print(f"\nTrain set: {X_train_scaled.shape}")
    print(f"Test set: {X_test_scaled.shape}")
    print(f"Features: {list(X_train_scaled.columns)}")
    print(f"Transformation log du target: {log_transform}\n")
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, log_transform, y_train_orig, y_test_orig


if __name__ == "__main__":
    # Test du pipeline
    try:
        result = preprocess_pipeline()
        X_train, X_test, y_train, y_test, scaler = result[:5]
        log_transform = result[5] if len(result) > 5 else False
        y_train_orig = result[6] if len(result) > 6 else y_train
        y_test_orig = result[7] if len(result) > 7 else y_test
        
        print("\n✓ Prétraitement réussi!")
        print(f"\nStatistiques de SalePrice:")
        print(f"  Train - Min: ${y_train_orig.min():,.0f}, Max: ${y_train_orig.max():,.0f}, Mean: ${y_train_orig.mean():,.0f}")
        print(f"  Test  - Min: ${y_test_orig.min():,.0f}, Max: ${y_test_orig.max():,.0f}, Mean: ${y_test_orig.mean():,.0f}")
        if log_transform:
            print(f"  (Transformation log appliquée)")
    except Exception as e:
        print(f"\n❌ Erreur: {e}")
        import traceback
        traceback.print_exc()
        print("\nAssurez-vous d'avoir téléchargé le dataset depuis:")
        print("https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data")
        print("Et placez le fichier 'train.csv' dans le dossier 'data/'")