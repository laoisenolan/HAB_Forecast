"""
Spatial-Temporal HAB Model Trainer

Advanced training pipeline for spatial-temporal HAB prediction models that addresses
class imbalance through spatial grid-based modeling approach.

Key Features:
- Grid-based spatial-temporal modeling
- Location-aware feature engineering  
- Species-specific spatial models
- Proper handling of spatial autocorrelation
- Balanced sampling strategies for spatial data
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Tuple, Optional, Any, Union
from datetime import datetime, timedelta
from pathlib import Path
import json
import pickle
import warnings
from dataclasses import dataclass
from sklearn.model_selection import train_test_split, TimeSeriesSplit, GroupKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.utils.class_weight import compute_class_weight
import xgboost as xgb
from scipy.stats import chi2_contingency
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore', category=FutureWarning)
logger = logging.getLogger(__name__)

@dataclass
class SpatialModelConfig:
    """Configuration for spatial-temporal model training"""
    model_type: str = 'xgboost'
    spatial_cv_method: str = 'spatial_blocks'  # 'spatial_blocks', 'temporal', 'group_kfold'
    test_size: float = 0.2
    validation_size: float = 0.1
    max_features: int = 50
    balance_strategy: str = 'spatial_sampling'  # 'spatial_sampling', 'class_weight', 'smote'
    min_positive_samples: int = 50
    spatial_buffer_km: float = 20.0  # Distance for spatial independence
    random_state: int = 42

@dataclass
class SpatialTrainingResults:
    """Results from spatial-temporal model training"""
    model_name: str
    grid_cell_models: Dict[str, Any]
    regional_models: Dict[str, Any]
    global_model: Any
    feature_importance: Dict[str, float]
    spatial_performance: Dict[str, Dict[str, float]]
    temporal_performance: Dict[str, Dict[str, float]]
    metadata: Dict[str, Any]

class SpatialTemporalHABTrainer:
    """
    Advanced trainer for spatial-temporal HAB prediction models
    
    This trainer implements sophisticated approaches for spatial-temporal modeling:
    1. Grid-cell specific models
    2. Regional hierarchical models  
    3. Spatial cross-validation
    4. Location-aware feature selection
    5. Balanced spatial sampling
    """
    
    def __init__(self, 
                 config: Optional[SpatialModelConfig] = None,
                 output_dir: str = "cache/spatial_temporal_models"):
        """
        Initialize spatial-temporal trainer
        
        Args:
            config: Training configuration
            output_dir: Directory for saving models and results
        """
        self.config = config or SpatialModelConfig()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Model storage
        self.trained_models = {}
        self.feature_selectors = {}
        self.scalers = {}
        self.label_encoders = {}
        
        # Training data and metadata
        self.training_data = None
        self.spatial_grid_info = None
        self.feature_categories = None
        
        # Performance tracking
        self.training_history = []
        
        logger.info("🎯 Spatial-Temporal HAB Trainer initialized")
        logger.info(f"   Output directory: {self.output_dir}")
        logger.info(f"   Configuration: {self.config}")
    
    def load_spatial_temporal_data(self, dataset_path: str) -> pd.DataFrame:
        """
        Load spatial-temporal training dataset
        
        Args:
            dataset_path: Path to spatial-temporal dataset CSV
            
        Returns:
            Loaded and validated dataset
        """
        logger.info(f"📊 Loading spatial-temporal dataset from {dataset_path}")
        
        try:
            # Load dataset (parquet preferred, fallback to csv)
            if dataset_path.endswith('.parquet'):
                df = pd.read_parquet(dataset_path)
            else:
                df = pd.read_csv(dataset_path)
            logger.info(f"Loaded {len(df):,} records")
            
            # Validate required columns
            required_cols = ['date', 'grid_cell_id', 'lat', 'lon']
            # Accept either 'hab_occurrence' or 'has_hab' as label
            if 'hab_occurrence' not in df.columns and 'has_hab' not in df.columns:
                raise ValueError("Missing required label column: 'hab_occurrence' or 'has_hab'")
            if 'hab_occurrence' not in df.columns and 'has_hab' in df.columns:
                df['hab_occurrence'] = df['has_hab'].astype('uint8')
            missing_cols = [col for col in required_cols if col not in df.columns]
            
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")
            
            # Convert date column
            df['date'] = pd.to_datetime(df['date'])
            
            # Validate data quality
            self._validate_spatial_data(df)
            
            # Store dataset
            self.training_data = df
            
            # Extract spatial grid information
            self._extract_spatial_grid_info(df)
            
            logger.info("✅ Spatial-temporal dataset loaded successfully")
            return df
            
        except Exception as e:
            logger.error(f"❌ Error loading spatial-temporal dataset: {e}")
            raise
    
    def _validate_spatial_data(self, df: pd.DataFrame):
        """Validate spatial-temporal dataset quality"""
        logger.info("🔍 Validating spatial-temporal data quality...")
        
        # Check for missing critical data
        critical_missing = df[['date', 'grid_cell_id', 'hab_occurrence']].isna().sum()
        if critical_missing.sum() > 0:
            logger.warning(f"Missing critical data: {critical_missing.to_dict()}")
        
        # Check class distribution
        class_dist = df['hab_occurrence'].value_counts()
        positive_ratio = class_dist.get(1, 0) / len(df)
        
        logger.info(f"Overall class distribution:")
        logger.info(f"   Positive (HAB): {class_dist.get(1, 0):,} ({positive_ratio*100:.1f}%)")
        logger.info(f"   Negative (No HAB): {class_dist.get(0, 0):,} ({(1-positive_ratio)*100:.1f}%)")
        
        # Check spatial distribution
        spatial_dist = df.groupby('grid_cell_id')['hab_occurrence'].agg(['count', 'sum', 'mean'])
        spatial_dist['positive_ratio'] = spatial_dist['sum'] / spatial_dist['count']
        
        logger.info(f"Spatial distribution:")
        logger.info(f"   Grid cells: {len(spatial_dist)}")
        logger.info(f"   Cells with HAB events: {(spatial_dist['sum'] > 0).sum()}")
        logger.info(f"   Average positive ratio: {spatial_dist['positive_ratio'].mean():.3f}")
        
        # Check temporal distribution
        temporal_dist = df.groupby('date')['hab_occurrence'].agg(['count', 'sum', 'mean'])
        
        logger.info(f"Temporal distribution:")
        logger.info(f"   Date range: {df['date'].min()} to {df['date'].max()}")
        logger.info(f"   Days with HAB events: {(temporal_dist['sum'] > 0).sum()}")
        logger.info(f"   Total days: {len(temporal_dist)}")
    
    def _extract_spatial_grid_info(self, df: pd.DataFrame):
        """Extract spatial grid information from dataset"""
        # Get unique grid cells and their properties
        grid_info = df[['grid_cell_id', 'lat', 'lon', 'bay_region']].drop_duplicates()
        
        self.spatial_grid_info = {
            'grid_cells': grid_info.set_index('grid_cell_id').to_dict('index'),
            'n_cells': len(grid_info),
            'bay_regions': grid_info['bay_region'].value_counts().to_dict(),
            'spatial_bounds': {
                'lat_min': grid_info['lat'].min(),
                'lat_max': grid_info['lat'].max(),
                'lon_min': grid_info['lon'].min(),
                'lon_max': grid_info['lon'].max()
            }
        }
        
        logger.info(f"Spatial grid info extracted: {self.spatial_grid_info['n_cells']} cells")
    
    def train_spatial_temporal_models(self, 
                                    target_species: Optional[List[str]] = None,
                                    modeling_approaches: Optional[List[str]] = None) -> SpatialTrainingResults:
        """
        Train spatial-temporal HAB prediction models
        
        Args:
            target_species: List of species to train models for (None for general HAB occurrence)
            modeling_approaches: List of approaches ['global', 'regional', 'cell_specific']
            
        Returns:
            Training results with all models and performance metrics
        """
        if self.training_data is None:
            raise ValueError("No training data loaded. Call load_spatial_temporal_data() first.")
        
        if modeling_approaches is None:
            modeling_approaches = ['global', 'regional']
        
        logger.info("🚀 Training spatial-temporal HAB models...")
        logger.info(f"   Modeling approaches: {modeling_approaches}")
        logger.info(f"   Target species: {target_species or 'General HAB occurrence'}")
        
        results = SpatialTrainingResults(
            model_name=f"spatial_temporal_hab_{datetime.now().strftime('%Y%m%d_%H%M')}",
            grid_cell_models={},
            regional_models={},
            global_model=None,
            feature_importance={},
            spatial_performance={},
            temporal_performance={},
            metadata={}
        )
        
        # Prepare features
        feature_data = self._prepare_features(self.training_data)
        
        # Train different modeling approaches
        if 'global' in modeling_approaches:
            logger.info("1️⃣ Training global spatial-temporal model...")
            results.global_model = self._train_global_model(feature_data)
        
        if 'regional' in modeling_approaches:
            logger.info("2️⃣ Training regional models...")
            results.regional_models = self._train_regional_models(feature_data)
        
        if 'cell_specific' in modeling_approaches:
            logger.info("3️⃣ Training cell-specific models...")
            results.grid_cell_models = self._train_cell_specific_models(feature_data)
        
        # Evaluate models
        logger.info("4️⃣ Evaluating model performance...")
        results.spatial_performance = self._evaluate_spatial_performance(results, feature_data)
        results.temporal_performance = self._evaluate_temporal_performance(results, feature_data)
        
        # Calculate feature importance
        results.feature_importance = self._calculate_global_feature_importance(results)
        
        # Save results
        self._save_training_results(results)
        
        logger.info("✅ Spatial-temporal model training completed!")
        return results
    
    def _prepare_features(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Prepare features for model training"""
        logger.info("🧮 Preparing features for spatial-temporal modeling...")
        
        # Identify feature columns (exclude metadata, target, and categorical columns)
        exclude_cols = ['date', 'grid_cell_id', 'lat', 'lon', 'hab_occurrence', 'has_hab',
                        'hab_species', 'hab_cell_count', 'hab_severity', 'hab_duration_days', 'event_count',
                        'bay_region']
        
        # Get environmental and spatial feature columns
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        # Handle categorical variables (bay_region)
        df_features = df.copy()
        if 'bay_region' in df_features.columns:
            le_bay = LabelEncoder()
            df_features['bay_region_encoded'] = le_bay.fit_transform(df_features['bay_region'].fillna('Unknown'))
            feature_cols.append('bay_region_encoded')
            self.label_encoders['bay_region'] = le_bay
        
        # Prepare features matrix
        X_full = df_features[feature_cols].copy()
        # Keep only numeric columns to avoid median/selector errors
        X = X_full.select_dtypes(include=['number'])
        numeric_feature_cols = X.columns.tolist()
        y = df_features['hab_occurrence'].copy()

        # Handle missing values
        # 1) median per column
        X = X.fillna(X.median(numeric_only=True))
        # 2) any columns still entirely NaN -> fill with 0
        X = X.fillna(0)
        # 3) if any columns are constant zeros, keep (XGBoost can handle)
        
        # Feature selection for global model
        logger.info(f"📊 Feature selection from {len(numeric_feature_cols)} numeric candidates...")
        
        # Guard: if too few features, skip selection
        if len(numeric_feature_cols) == 0:
            raise ValueError("No numeric features available after preprocessing")
        k = min(self.config.max_features, len(numeric_feature_cols))
        selector = SelectKBest(f_classif, k=k)
        X_selected = selector.fit_transform(X, y)
        selected_features = [numeric_feature_cols[i] for i in selector.get_support(indices=True)]
        
        self.feature_selectors['global'] = selector
        
        logger.info(f"✅ Selected {len(selected_features)} features for modeling")
        
        # Feature scaling
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_selected)
        self.scalers['global'] = scaler
        
        return {
            'X': X_scaled,
            'y': y.values,
            'feature_names': selected_features,
            'original_features': X,
            'spatial_coords': df_features[['lat', 'lon']].values,
            'temporal_coords': df_features['date'].values,
            'grid_cell_ids': df_features['grid_cell_id'].values,
            'bay_regions': df_features['bay_region'].values
        }
    
    def _train_global_model(self, feature_data: Dict[str, Any]) -> Dict[str, Any]:
        """Train global spatial-temporal model"""
        logger.info("🌍 Training global spatial-temporal model...")
        
        X, y = feature_data['X'], feature_data['y']
        grid_cell_ids = feature_data['grid_cell_ids']
        
        # Spatial cross-validation to handle spatial autocorrelation
        cv_results = self._spatial_cross_validation(X, y, grid_cell_ids)
        
        # Handle class imbalance
        if self.config.balance_strategy == 'class_weight':
            class_weights = compute_class_weight('balanced', classes=np.unique(y), y=y)
            scale_pos_weight = class_weights[1] / class_weights[0]
        else:
            scale_pos_weight = 1.0
        
        # Train XGBoost model
        model = xgb.XGBClassifier(
            objective='binary:logistic',
            eval_metric='logloss',
            max_depth=6,
            min_child_weight=3,
            subsample=0.8,
            colsample_bytree=0.8,
            learning_rate=0.1,
            n_estimators=200,
            reg_alpha=0.1,
            reg_lambda=1.0,
            scale_pos_weight=scale_pos_weight,
            random_state=self.config.random_state,
            n_jobs=-1
        )
        
        # Train on full dataset
        model.fit(X, y)
        
        # Calculate feature importance
        feature_importance = dict(zip(feature_data['feature_names'], model.feature_importances_))
        
        global_model = {
            'model': model,
            'cv_results': cv_results,
            'feature_importance': feature_importance,
            'training_samples': len(X),
            'positive_samples': int(y.sum()),
            'scale_pos_weight': scale_pos_weight
        }
        
        logger.info(f"✅ Global model trained: {len(X):,} samples, CV score: {cv_results['mean_score']:.3f}")
        
        return global_model
    
    def _train_regional_models(self, feature_data: Dict[str, Any]) -> Dict[str, Any]:
        """Train separate models for each bay region"""
        logger.info("🗺️ Training regional models...")
        
        X, y = feature_data['X'], feature_data['y']
        bay_regions = feature_data['bay_regions']
        
        regional_models = {}
        
        # Get unique bay regions
        unique_regions = pd.Series(bay_regions).value_counts()
        logger.info(f"Training models for {len(unique_regions)} bay regions")
        
        for region in unique_regions.index:
            if pd.isna(region) or region == 'Unknown':
                continue
                
            logger.info(f"   Training model for {region}...")
            
            # Get data for this region
            region_mask = bay_regions == region
            X_region = X[region_mask]
            y_region = y[region_mask]
            
            # Check minimum sample requirements
            positive_samples = int(y_region.sum())
            total_samples = len(y_region)
            
            if positive_samples < self.config.min_positive_samples:
                logger.warning(f"   Skipping {region}: insufficient positive samples ({positive_samples})")
                continue
            
            if total_samples < 100:
                logger.warning(f"   Skipping {region}: insufficient total samples ({total_samples})")
                continue
            
            # Handle class imbalance for this region
            if self.config.balance_strategy == 'class_weight':
                class_weights = compute_class_weight('balanced', classes=np.unique(y_region), y=y_region)
                if len(class_weights) == 2:
                    scale_pos_weight = class_weights[1] / class_weights[0]
                else:
                    scale_pos_weight = 1.0
            else:
                scale_pos_weight = 1.0
            
            # Train regional model
            regional_model = xgb.XGBClassifier(
                objective='binary:logistic',
                eval_metric='logloss',
                max_depth=4,  # Smaller depth for regional models
                min_child_weight=2,
                subsample=0.8,
                colsample_bytree=0.8,
                learning_rate=0.1,
                n_estimators=100,
                reg_alpha=0.1,
                reg_lambda=1.0,
                scale_pos_weight=scale_pos_weight,
                random_state=self.config.random_state,
                n_jobs=-1
            )
            
            # Split data for training and validation
            X_train, X_val, y_train, y_val = train_test_split(
                X_region, y_region, test_size=0.2, 
                stratify=y_region, random_state=self.config.random_state
            )
            
            # Train model
            regional_model.fit(X_train, y_train)
            
            # Evaluate on validation set
            y_pred = regional_model.predict(X_val)
            y_pred_proba = regional_model.predict_proba(X_val)[:, 1]
            
            # Calculate metrics
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
            
            metrics = {
                'accuracy': accuracy_score(y_val, y_pred),
                'precision': precision_score(y_val, y_pred),
                'recall': recall_score(y_val, y_pred),
                'f1': f1_score(y_val, y_pred),
                'auc': roc_auc_score(y_val, y_pred_proba) if len(np.unique(y_val)) > 1 else 0.5
            }
            
            regional_models[region] = {
                'model': regional_model,
                'metrics': metrics,
                'training_samples': len(X_train),
                'positive_samples': int(y_train.sum()),
                'scale_pos_weight': scale_pos_weight,
                'feature_importance': dict(zip(feature_data['feature_names'], regional_model.feature_importances_))
            }
            
            logger.info(f"   ✅ {region}: F1={metrics['f1']:.3f}, AUC={metrics['auc']:.3f}")
        
        logger.info(f"✅ Regional models trained: {len(regional_models)} regions")
        return regional_models
    
    def _train_cell_specific_models(self, feature_data: Dict[str, Any]) -> Dict[str, Any]:
        """Train models for individual grid cells (for cells with sufficient data)"""
        logger.info("🔬 Training cell-specific models...")
        
        X, y = feature_data['X'], feature_data['y']
        grid_cell_ids = feature_data['grid_cell_ids']
        
        cell_models = {}
        
        # Get data distribution by cell
        cell_data_counts = pd.Series(grid_cell_ids).value_counts()
        
        # Only train models for cells with sufficient data
        min_samples_per_cell = 200
        eligible_cells = cell_data_counts[cell_data_counts >= min_samples_per_cell]
        
        logger.info(f"Training models for {len(eligible_cells)} cells (minimum {min_samples_per_cell} samples)")
        
        for cell_id in eligible_cells.index:
            # Get data for this cell
            cell_mask = grid_cell_ids == cell_id
            X_cell = X[cell_mask]
            y_cell = y[cell_mask]
            
            positive_samples = int(y_cell.sum())
            
            # Check minimum positive samples
            if positive_samples < self.config.min_positive_samples:
                continue
            
            logger.debug(f"   Training model for {cell_id}...")
            
            try:
                # Train cell-specific model
                cell_model = xgb.XGBClassifier(
                    objective='binary:logistic',
                    eval_metric='logloss',
                    max_depth=3,  # Shallow for cell-specific models
                    min_child_weight=1,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    learning_rate=0.1,
                    n_estimators=50,
                    reg_alpha=0.1,
                    reg_lambda=1.0,
                    random_state=self.config.random_state,
                    n_jobs=-1
                )
                
                # Simple train/validation split
                X_train, X_val, y_train, y_val = train_test_split(
                    X_cell, y_cell, test_size=0.2, 
                    stratify=y_cell, random_state=self.config.random_state
                )
                
                cell_model.fit(X_train, y_train)
                
                # Evaluate
                y_pred = cell_model.predict(X_val)
                y_pred_proba = cell_model.predict_proba(X_val)[:, 1]
                
                from sklearn.metrics import f1_score
                f1 = f1_score(y_val, y_pred)
                auc = roc_auc_score(y_val, y_pred_proba) if len(np.unique(y_val)) > 1 else 0.5
                
                cell_models[cell_id] = {
                    'model': cell_model,
                    'f1_score': f1,
                    'auc_score': auc,
                    'training_samples': len(X_train),
                    'positive_samples': int(y_train.sum())
                }
                
            except Exception as e:
                logger.debug(f"   Failed to train model for {cell_id}: {e}")
        
        logger.info(f"✅ Cell-specific models trained: {len(cell_models)} cells")
        return cell_models
    
    def _spatial_cross_validation(self, X: np.ndarray, y: np.ndarray, 
                                grid_cell_ids: np.ndarray, n_splits: int = 5) -> Dict[str, Any]:
        """Perform spatial cross-validation to handle spatial autocorrelation"""
        logger.info("🔄 Performing spatial cross-validation...")
        
        # Group by grid cells for spatial independence
        unique_cells = np.unique(grid_cell_ids)
        np.random.shuffle(unique_cells)
        
        # Split cells into folds
        fold_size = len(unique_cells) // n_splits
        folds = []
        
        for i in range(n_splits):
            start_idx = i * fold_size
            end_idx = (i + 1) * fold_size if i < n_splits - 1 else len(unique_cells)
            fold_cells = unique_cells[start_idx:end_idx]
            folds.append(fold_cells)
        
        cv_scores = []
        
        for fold_idx, test_cells in enumerate(folds):
            # Create spatial train/test split
            test_mask = np.isin(grid_cell_ids, test_cells)
            train_mask = ~test_mask
            
            X_train, X_test = X[train_mask], X[test_mask]
            y_train, y_test = y[train_mask], y[test_mask]
            
            # Check if both classes are present
            if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
                logger.warning(f"Fold {fold_idx}: Skipping due to missing class")
                continue
            
            # Train model on this fold
            fold_model = xgb.XGBClassifier(
                objective='binary:logistic',
                eval_metric='logloss',
                max_depth=6,
                learning_rate=0.1,
                n_estimators=100,
                random_state=self.config.random_state,
                n_jobs=-1
            )
            
            fold_model.fit(X_train, y_train)
            
            # Evaluate
            y_pred_proba = fold_model.predict_proba(X_test)[:, 1]
            score = roc_auc_score(y_test, y_pred_proba)
            cv_scores.append(score)
            
            logger.debug(f"   Fold {fold_idx}: AUC = {score:.3f}")
        
        cv_results = {
            'scores': cv_scores,
            'mean_score': np.mean(cv_scores),
            'std_score': np.std(cv_scores),
            'n_folds': len(cv_scores)
        }
        
        logger.info(f"✅ Spatial CV completed: {cv_results['mean_score']:.3f} ± {cv_results['std_score']:.3f}")
        return cv_results
    
    def _evaluate_spatial_performance(self, results: SpatialTrainingResults, 
                                    feature_data: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
        """Evaluate model performance across different spatial regions"""
        logger.info("📊 Evaluating spatial performance...")
        
        spatial_performance = {}
        
        # Evaluate by bay region if regional models exist
        if results.regional_models:
            for region, regional_model_info in results.regional_models.items():
                spatial_performance[region] = regional_model_info['metrics']
        
        return spatial_performance
    
    def _evaluate_temporal_performance(self, results: SpatialTrainingResults,
                                     feature_data: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
        """Evaluate model performance across different time periods"""
        logger.info("📅 Evaluating temporal performance...")
        
        # For now, return empty dict - can be expanded for temporal analysis
        return {}
    
    def _calculate_global_feature_importance(self, results: SpatialTrainingResults) -> Dict[str, float]:
        """Calculate aggregated feature importance across all models"""
        if results.global_model:
            return results.global_model['feature_importance']
        elif results.regional_models:
            # Average feature importance across regional models
            all_importances = {}
            for region_model in results.regional_models.values():
                for feature, importance in region_model['feature_importance'].items():
                    if feature not in all_importances:
                        all_importances[feature] = []
                    all_importances[feature].append(importance)
            
            return {feature: np.mean(importances) for feature, importances in all_importances.items()}
        else:
            return {}
    
    def _save_training_results(self, results: SpatialTrainingResults):
        """Save training results and models"""
        logger.info("💾 Saving training results...")
        
        # Create model directory
        model_dir = self.output_dir / results.model_name
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # Save models
        if results.global_model:
            with open(model_dir / "global_model.pkl", 'wb') as f:
                pickle.dump(results.global_model, f)
        
        if results.regional_models:
            regional_dir = model_dir / "regional_models"
            regional_dir.mkdir(exist_ok=True)
            for region, model_info in results.regional_models.items():
                with open(regional_dir / f"{region}_model.pkl", 'wb') as f:
                    pickle.dump(model_info, f)
        
        if results.grid_cell_models:
            cell_dir = model_dir / "cell_models"
            cell_dir.mkdir(exist_ok=True)
            for cell_id, model_info in results.grid_cell_models.items():
                with open(cell_dir / f"{cell_id}_model.pkl", 'wb') as f:
                    pickle.dump(model_info, f)
        
        # Save feature selectors and scalers
        with open(model_dir / "feature_selectors.pkl", 'wb') as f:
            pickle.dump(self.feature_selectors, f)
        
        with open(model_dir / "scalers.pkl", 'wb') as f:
            pickle.dump(self.scalers, f)
        
        with open(model_dir / "label_encoders.pkl", 'wb') as f:
            pickle.dump(self.label_encoders, f)
        
        # Save metadata and results summary
        metadata = {
            'model_name': results.model_name,
            'creation_date': datetime.now().isoformat(),
            'config': self.config.__dict__,
            'spatial_grid_info': self.spatial_grid_info,
            'feature_importance': results.feature_importance,
            'spatial_performance': results.spatial_performance,
            'temporal_performance': results.temporal_performance,
            'model_counts': {
                'global': 1 if results.global_model else 0,
                'regional': len(results.regional_models),
                'cell_specific': len(results.grid_cell_models)
            }
        }
        
        # JSON-safe dump (handle numpy/pandas types)
        def _json_default(o):
            try:
                import numpy as np
                import pandas as pd
                if isinstance(o, (np.integer,)):
                    return int(o)
                if isinstance(o, (np.floating,)):
                    return float(o)
                if isinstance(o, (pd.Timestamp,)):
                    return o.isoformat()
            except Exception:
                pass
            return str(o)

        with open(model_dir / "training_metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2, default=_json_default)
        
        logger.info(f"✅ Training results saved to {model_dir}")
    
    def predict_spatial_temporal(self, 
                                environmental_data: pd.DataFrame,
                                model_name: str,
                                prediction_approach: str = 'regional') -> pd.DataFrame:
        """
        Make spatial-temporal predictions using trained models
        
        Args:
            environmental_data: Environmental data with spatial-temporal grid structure
            model_name: Name of trained model to use
            prediction_approach: 'global', 'regional', or 'cell_specific'
            
        Returns:
            DataFrame with HAB predictions for each grid cell and date
        """
        logger.info(f"🔮 Making spatial-temporal HAB predictions...")
        logger.info(f"   Model: {model_name}")
        logger.info(f"   Approach: {prediction_approach}")
        
        # Load model
        model_dir = self.output_dir / model_name
        if not model_dir.exists():
            raise ValueError(f"Model not found: {model_name}")
        
        # Implementation would load models and make predictions
        # This is a placeholder for the prediction pipeline
        
        predictions_df = environmental_data.copy()
        predictions_df['hab_probability'] = 0.5  # Placeholder
        predictions_df['hab_prediction'] = 0     # Placeholder
        
        logger.info("✅ Spatial-temporal predictions completed")
        return predictions_df


def train_spatial_temporal_hab_models(dataset_path: str,
                                    output_dir: str,
                                    config: Optional[SpatialModelConfig] = None) -> str:
    """
    Convenience function to train spatial-temporal HAB models
    
    Args:
        dataset_path: Path to spatial-temporal training dataset
        output_dir: Output directory for models
        config: Training configuration
        
    Returns:
        Path to trained models directory
    """
    logger.info("🚀 Starting spatial-temporal HAB model training pipeline...")
    
    # Initialize trainer
    trainer = SpatialTemporalHABTrainer(config=config, output_dir=output_dir)
    
    # Load data
    trainer.load_spatial_temporal_data(dataset_path)
    
    # Train models
    results = trainer.train_spatial_temporal_models(
        modeling_approaches=['global', 'regional']
    )
    
    logger.info(f"✅ Spatial-temporal HAB models trained successfully!")
    logger.info(f"   Models saved to: {trainer.output_dir / results.model_name}")
    
    return str(trainer.output_dir / results.model_name)


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Example training configuration
    config = SpatialModelConfig(
        spatial_cv_method='spatial_blocks',
        balance_strategy='class_weight',
        max_features=40,
        min_positive_samples=30
    )
    
    # Example training call
    """
    model_path = train_spatial_temporal_hab_models(
        dataset_path="cache/spatial_temporal_training/spatial_temporal_training_dataset.csv",
        output_dir="cache/spatial_temporal_models",
        config=config
    )
    print(f"Models trained: {model_path}")
    """
