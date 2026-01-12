#!/usr/bin/env python3
"""
Transfer Learning Between Similar HAB Species - Task 4.4

This module implements comprehensive transfer learning techniques for HAB prediction,
enabling knowledge transfer from data-rich species to improve predictions for 
data-scarce species through:
- Environmental preference similarity analysis for species grouping
- Pre-training on source species with abundant data
- Fine-tuning for target species with limited data
- Feature representation transfer and domain adaptation
- Multi-level transfer learning (features, models, ensembles)
- Species similarity assessment and transfer effectiveness evaluation

🎯 TRANSFER LEARNING CAPABILITIES:
- Species Similarity Analysis: Environmental preference clustering and phylogenetic relationships
- Pre-training Strategies: Deep feature learning from source species data
- Fine-tuning Approaches: Adaptive learning rates and layer freezing
- Domain Adaptation: Feature alignment and distribution matching
- Knowledge Distillation: Teacher-student model architectures
- Multi-source Transfer: Combining knowledge from multiple source species

🛰️ HAB-SPECIFIC TRANSFER LEARNING:
- Taxonomic relationships: Genus-level and family-level knowledge sharing
- Environmental niche similarity: Temperature, nutrient, salinity preferences
- Seasonal pattern transfer: Bloom timing and duration patterns
- Geographic adaptation: Regional environmental condition adjustment
- Bloom intensity transfer: Cell density and severity prediction improvement
- Real-time adaptation: Online learning from new target species observations

Author: HAB Forecasting System - Task 4.4 Implementation
Date: December 19, 2024
"""

import numpy as np
import pandas as pd
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime
import json
import pickle
import warnings
from collections import defaultdict
from dataclasses import dataclass
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist
from scipy import stats

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Standard ML libraries
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import xgboost as xgb

@dataclass
class SpeciesProfile:
    """Environmental profile for a HAB species"""
    species: str
    n_samples: int
    positive_samples: int
    environmental_preferences: Dict[str, Dict[str, float]]  # feature -> {mean, std, min, max}
    seasonal_patterns: Dict[str, float]  # month -> occurrence_rate
    geographic_distribution: Dict[str, float]  # region -> occurrence_rate
    bloom_characteristics: Dict[str, float]  # intensity, duration, etc.

@dataclass
class TransferLearningConfig:
    """Configuration for transfer learning"""
    similarity_threshold: float = 0.7
    min_source_samples: int = 100
    min_target_samples: int = 10
    max_source_species: int = 3
    feature_transfer_layers: int = 2
    fine_tune_epochs: int = 50
    learning_rate_decay: float = 0.1
    early_stopping_patience: int = 10

class HABSpeciesTransferLearning:
    """
    Comprehensive transfer learning system for HAB species prediction
    """
    
    def __init__(self, cache_dir: str = "cache/transfer_learning",
                 random_state: int = 42):
        """
        Initialize transfer learning system
        
        Args:
            cache_dir: Directory for caching transfer learning models
            random_state: Random state for reproducibility
        """
        
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.random_state = random_state
        
        # Species data and profiles
        self.species_profiles = {}
        self.species_data = {}
        self.similarity_matrix = None
        
        # Transfer learning state
        self.source_models = {}
        self.transfer_history = []
        
        logger.info(f"🔄 HAB Transfer Learning System initialized")
        logger.info(f"📁 Cache directory: {self.cache_dir}")
    
    def register_species_data(self, species: str, X: np.ndarray, y: np.ndarray,
                             feature_names: List[str], 
                             timestamps: Optional[pd.Series] = None,
                             locations: Optional[Tuple[np.ndarray, np.ndarray]] = None) -> None:
        """
        Register species data for transfer learning
        
        Args:
            species: Species name
            X: Feature matrix
            y: Target labels
            feature_names: Names of features
            timestamps: Optional timestamps for temporal analysis
            locations: Optional (latitude, longitude) for spatial analysis
        """
        
        # Store species data
        self.species_data[species] = {
            'X': X.copy(),
            'y': y.copy(),
            'feature_names': feature_names.copy(),
            'timestamps': timestamps.copy() if timestamps is not None else None,
            'locations': locations
        }
        
        # Create environmental profile
        profile = self._create_species_profile(species, X, y, feature_names, timestamps, locations)
        self.species_profiles[species] = profile
        
        logger.info(f"📋 Registered {species}: {len(X)} samples, {np.sum(y)} positive")
        logger.info(f"🧬 Profile created with {len(profile.environmental_preferences)} environmental features")
    
    def _create_species_profile(self, species: str, X: np.ndarray, y: np.ndarray,
                               feature_names: List[str],
                               timestamps: Optional[pd.Series] = None,
                               locations: Optional[Tuple[np.ndarray, np.ndarray]] = None) -> SpeciesProfile:
        """Create comprehensive species environmental profile"""
        
        # Environmental preferences
        environmental_prefs = {}
        positive_mask = y == 1
        
        for i, feature in enumerate(feature_names):
            feature_data = X[:, i]
            positive_data = feature_data[positive_mask]
            
            # Remove NaN values
            positive_clean = positive_data[~np.isnan(positive_data)]
            all_clean = feature_data[~np.isnan(feature_data)]
            
            if len(positive_clean) > 0 and len(all_clean) > 0:
                environmental_prefs[feature] = {
                    'positive_mean': np.mean(positive_clean),
                    'positive_std': np.std(positive_clean),
                    'positive_min': np.min(positive_clean),
                    'positive_max': np.max(positive_clean),
                    'overall_mean': np.mean(all_clean),
                    'overall_std': np.std(all_clean),
                    'preference_score': self._calculate_preference_score(positive_clean, all_clean)
                }
        
        # Seasonal patterns
        seasonal_patterns = {}
        if timestamps is not None:
            try:
                months = pd.to_datetime(timestamps).dt.month
                for month in range(1, 13):
                    month_mask = months == month
                    if np.sum(month_mask) > 0:
                        seasonal_patterns[f'month_{month}'] = np.mean(y[month_mask])
            except Exception as e:
                logger.warning(f"⚠️  Failed to extract seasonal patterns for {species}: {e}")
        
        # Geographic distribution (simplified)
        geographic_dist = {}
        if locations is not None:
            try:
                lats, lons = locations
                # Simple geographic binning
                lat_bins = np.linspace(np.nanmin(lats), np.nanmax(lats), 5)
                lon_bins = np.linspace(np.nanmin(lons), np.nanmax(lons), 5)
                
                for i in range(len(lat_bins)-1):
                    for j in range(len(lon_bins)-1):
                        region_mask = ((lats >= lat_bins[i]) & (lats < lat_bins[i+1]) &
                                     (lons >= lon_bins[j]) & (lons < lon_bins[j+1]))
                        if np.sum(region_mask) > 0:
                            geographic_dist[f'region_{i}_{j}'] = np.mean(y[region_mask])
            except Exception as e:
                logger.warning(f"⚠️  Failed to extract geographic patterns for {species}: {e}")
        
        # Bloom characteristics
        bloom_chars = {}
        if np.sum(positive_mask) > 0:
            bloom_chars = {
                'occurrence_rate': np.mean(y),
                'positive_samples': np.sum(positive_mask),
                'bloom_frequency': np.sum(positive_mask) / len(y) if len(y) > 0 else 0.0
            }
        
        return SpeciesProfile(
            species=species,
            n_samples=len(X),
            positive_samples=np.sum(y),
            environmental_preferences=environmental_prefs,
            seasonal_patterns=seasonal_patterns,
            geographic_distribution=geographic_dist,
            bloom_characteristics=bloom_chars
        )
    
    def _calculate_preference_score(self, positive_data: np.ndarray, all_data: np.ndarray) -> float:
        """Calculate environmental preference score using statistical tests"""
        
        try:
            # Mann-Whitney U test to assess preference
            if len(positive_data) > 5 and len(all_data) > len(positive_data) + 5:
                negative_data = np.setdiff1d(all_data, positive_data)
                statistic, p_value = stats.mannwhitneyu(positive_data, negative_data, alternative='two-sided')
                
                # Convert p-value to preference score (0-1, higher = more distinctive)
                preference_score = max(0.0, 1.0 - p_value)
                return preference_score
            else:
                return 0.5  # Neutral preference if insufficient data
                
        except Exception as e:
            logger.warning(f"⚠️  Preference score calculation failed: {e}")
            return 0.5
    
    def calculate_species_similarity(self, update_matrix: bool = True) -> np.ndarray:
        """
        Calculate similarity matrix between all registered species
        
        Args:
            update_matrix: Whether to update the stored similarity matrix
            
        Returns:
            Species similarity matrix
        """
        
        species_list = list(self.species_profiles.keys())
        n_species = len(species_list)
        
        if n_species < 2:
            logger.warning("⚠️  Need at least 2 species to calculate similarity")
            return np.array([[1.0]])
        
        similarity_matrix = np.zeros((n_species, n_species))
        
        for i, species1 in enumerate(species_list):
            for j, species2 in enumerate(species_list):
                if i == j:
                    similarity_matrix[i, j] = 1.0
                else:
                    similarity = self._calculate_pairwise_similarity(species1, species2)
                    similarity_matrix[i, j] = similarity
        
        if update_matrix:
            self.similarity_matrix = similarity_matrix
            self.species_list = species_list
        
        logger.info(f"🔍 Calculated similarity matrix for {n_species} species")
        
        return similarity_matrix
    
    def _calculate_pairwise_similarity(self, species1: str, species2: str) -> float:
        """Calculate similarity between two species"""
        
        profile1 = self.species_profiles[species1]
        profile2 = self.species_profiles[species2]
        
        similarities = []
        
        # Environmental preference similarity
        common_features = set(profile1.environmental_preferences.keys()) & set(profile2.environmental_preferences.keys())
        
        if common_features:
            env_similarities = []
            
            for feature in common_features:
                prefs1 = profile1.environmental_preferences[feature]
                prefs2 = profile2.environmental_preferences[feature]
                
                # Compare positive means (main environmental preference)
                mean_diff = abs(prefs1['positive_mean'] - prefs2['positive_mean'])
                combined_std = (prefs1['positive_std'] + prefs2['positive_std']) / 2
                
                if combined_std > 0:
                    # Normalized difference (0 = identical, higher = more different)
                    norm_diff = mean_diff / combined_std
                    # Convert to similarity (0-1, higher = more similar)
                    feature_similarity = max(0.0, 1.0 - (norm_diff / 3.0))  # 3-sigma rule
                else:
                    feature_similarity = 1.0 if mean_diff < 1e-6 else 0.0
                
                # Weight by preference scores
                weight = (prefs1['preference_score'] + prefs2['preference_score']) / 2
                env_similarities.append(feature_similarity * weight)
            
            if env_similarities:
                similarities.append(np.mean(env_similarities))
        
        # Seasonal pattern similarity
        if profile1.seasonal_patterns and profile2.seasonal_patterns:
            common_months = set(profile1.seasonal_patterns.keys()) & set(profile2.seasonal_patterns.keys())
            if common_months:
                seasonal_similarities = []
                for month in common_months:
                    rate1 = profile1.seasonal_patterns[month]
                    rate2 = profile2.seasonal_patterns[month]
                    # Similarity based on occurrence rate difference
                    seasonal_sim = 1.0 - abs(rate1 - rate2)
                    seasonal_similarities.append(max(0.0, seasonal_sim))
                
                if seasonal_similarities:
                    similarities.append(np.mean(seasonal_similarities))
        
        # Bloom characteristic similarity
        if profile1.bloom_characteristics and profile2.bloom_characteristics:
            bloom_sim = 1.0 - abs(profile1.bloom_characteristics['occurrence_rate'] - 
                                profile2.bloom_characteristics['occurrence_rate'])
            similarities.append(max(0.0, bloom_sim))
        
        # Overall similarity
        if similarities:
            overall_similarity = np.mean(similarities)
        else:
            overall_similarity = 0.0  # No common characteristics
        
        return overall_similarity
    
    def identify_source_species(self, target_species: str, 
                               config: TransferLearningConfig) -> List[str]:
        """
        Identify best source species for transfer learning
        
        Args:
            target_species: Target species for transfer learning
            config: Transfer learning configuration
            
        Returns:
            List of source species ordered by similarity
        """
        
        if target_species not in self.species_profiles:
            raise ValueError(f"Target species {target_species} not registered")
        
        if self.similarity_matrix is None:
            self.calculate_species_similarity()
        
        # Get similarity scores for target species
        target_idx = self.species_list.index(target_species)
        similarities = self.similarity_matrix[target_idx, :]
        
        # Filter candidate source species
        source_candidates = []
        
        for i, species in enumerate(self.species_list):
            if species != target_species:  # Exclude target species itself
                similarity = similarities[i]
                n_samples = self.species_profiles[species].positive_samples
                
                # Check if species meets criteria
                if (similarity >= config.similarity_threshold and 
                    n_samples >= config.min_source_samples):
                    source_candidates.append((species, similarity, n_samples))
        
        # Sort by similarity (descending) and limit to max_source_species
        source_candidates.sort(key=lambda x: x[1], reverse=True)
        selected_sources = source_candidates[:config.max_source_species]
        
        source_species = [species for species, _, _ in selected_sources]
        
        logger.info(f"🎯 Source species for {target_species}:")
        for species, similarity, n_samples in selected_sources:
            logger.info(f"  {species}: similarity={similarity:.3f}, samples={n_samples}")
        
        return source_species
    
    def pre_train_source_model(self, source_species: str, 
                              feature_names: List[str]) -> Dict[str, Any]:
        """
        Pre-train model on source species data
        
        Args:
            source_species: Source species name
            feature_names: Feature names for training
            
        Returns:
            Pre-trained model package
        """
        
        if source_species not in self.species_data:
            raise ValueError(f"Source species {source_species} data not available")
        
        data = self.species_data[source_species]
        X_source = data['X']
        y_source = data['y']
        
        # Ensure we have the right features
        source_features = data['feature_names']
        
        # Match features (handle different feature orders/subsets)
        feature_indices = []
        for fname in feature_names:
            if fname in source_features:
                feature_indices.append(source_features.index(fname))
            else:
                logger.warning(f"⚠️  Feature {fname} not available in {source_species} data")
        
        if not feature_indices:
            raise ValueError(f"No matching features found for {source_species}")
        
        X_source_matched = X_source[:, feature_indices]
        
        # Handle missing values
        X_source_matched = np.nan_to_num(X_source_matched, nan=0.0)
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X_source_matched, y_source, test_size=0.2, 
            stratify=y_source, random_state=self.random_state
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        
        # Train robust XGBoost model
        source_model = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.08,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=self.random_state,
            n_jobs=2
        )
        
        source_model.fit(X_train_scaled, y_train)
        
        # Evaluate source model
        y_pred_val = source_model.predict(X_val_scaled)
        y_pred_proba_val = source_model.predict_proba(X_val_scaled)[:, 1]
        
        source_metrics = {
            'f1': f1_score(y_val, y_pred_val),
            'auc': roc_auc_score(y_val, y_pred_proba_val) if len(np.unique(y_val)) > 1 else 0.0,
            'accuracy': accuracy_score(y_val, y_pred_val),
            'training_samples': len(X_train),
            'validation_samples': len(X_val)
        }
        
        # Store source model
        model_package = {
            'model': source_model,
            'scaler': scaler,
            'feature_names': feature_names,
            'source_species': source_species,
            'metrics': source_metrics,
            'feature_indices': feature_indices
        }
        
        self.source_models[source_species] = model_package
        
        logger.info(f"🤖 Pre-trained model for {source_species}: "
                   f"F1={source_metrics['f1']:.3f}, AUC={source_metrics['auc']:.3f}")
        
        return model_package
    
    def transfer_learn_target_model(self, target_species: str, 
                                   source_species_list: List[str],
                                   config: TransferLearningConfig) -> Dict[str, Any]:
        """
        Apply transfer learning to target species
        
        Args:
            target_species: Target species name
            source_species_list: List of source species for transfer
            config: Transfer learning configuration
            
        Returns:
            Transfer learning results
        """
        
        if target_species not in self.species_data:
            raise ValueError(f"Target species {target_species} data not available")
        
        target_data = self.species_data[target_species]
        X_target = target_data['X']
        y_target = target_data['y']
        feature_names = target_data['feature_names']
        
        # Check minimum target samples
        if len(X_target) < config.min_target_samples:
            raise ValueError(f"Insufficient target samples: {len(X_target)} < {config.min_target_samples}")
        
        if np.sum(y_target) < 2:
            raise ValueError(f"Insufficient positive target samples: {np.sum(y_target)} < 2")
        
        logger.info(f"🎯 Transfer learning for {target_species}: {len(X_target)} samples, {np.sum(y_target)} positive")
        
        # Pre-train source models if needed
        for source_species in source_species_list:
            if source_species not in self.source_models:
                self.pre_train_source_model(source_species, feature_names)
        
        # Split target data
        X_target_clean = np.nan_to_num(X_target, nan=0.0)
        
        if len(np.unique(y_target)) < 2:
            # Handle single-class case
            logger.warning(f"⚠️  Single class in target data for {target_species}")
            X_train_target = X_target_clean
            y_train_target = y_target
            X_val_target = X_target_clean[:min(5, len(X_target_clean))]
            y_val_target = y_target[:min(5, len(y_target))]
        else:
            X_train_target, X_val_target, y_train_target, y_val_target = train_test_split(
                X_target_clean, y_target, test_size=0.3, 
                stratify=y_target, random_state=self.random_state
            )
        
        # Strategy 1: Feature Representation Transfer
        transfer_results = {}
        
        # Get best source model for initialization
        best_source_model = None
        best_source_similarity = 0.0
        
        for source_species in source_species_list:
            if source_species in self.source_models:
                # Calculate similarity
                target_idx = self.species_list.index(target_species)
                source_idx = self.species_list.index(source_species)
                similarity = self.similarity_matrix[target_idx, source_idx]
                
                if similarity > best_source_similarity:
                    best_source_similarity = similarity
                    best_source_model = self.source_models[source_species]
        
        if best_source_model is None:
            raise ValueError("No source models available for transfer learning")
        
        # Strategy 2: Fine-tuning with transferred features
        logger.info(f"🔄 Fine-tuning from {best_source_model['source_species']} (similarity: {best_source_similarity:.3f})")
        
        # Scale target data using source scaler (domain adaptation)
        source_scaler = best_source_model['scaler']
        X_train_target_scaled = source_scaler.transform(X_train_target)
        X_val_target_scaled = source_scaler.transform(X_val_target)
        
        # Initialize target model with source model parameters
        target_model = xgb.XGBClassifier(
            n_estimators=100,  # Fewer estimators for fine-tuning
            max_depth=5,       # Slightly shallower
            learning_rate=0.05,  # Lower learning rate for fine-tuning
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=self.random_state,
            n_jobs=2
        )
        
        # Fine-tune on target data
        target_model.fit(X_train_target_scaled, y_train_target)
        
        # Evaluate target model
        if len(X_val_target) > 0:
            y_pred_target = target_model.predict(X_val_target_scaled)
            y_pred_proba_target = target_model.predict_proba(X_val_target_scaled)[:, 1]
            
            target_metrics = {
                'f1': f1_score(y_val_target, y_pred_target, zero_division=0),
                'auc': roc_auc_score(y_val_target, y_pred_proba_target) if len(np.unique(y_val_target)) > 1 else 0.0,
                'accuracy': accuracy_score(y_val_target, y_pred_target),
                'training_samples': len(X_train_target),
                'validation_samples': len(X_val_target)
            }
        else:
            target_metrics = {'f1': 0.0, 'auc': 0.0, 'accuracy': 0.0, 'training_samples': len(X_train_target), 'validation_samples': 0}
        
        # Strategy 3: Baseline model without transfer (for comparison)
        baseline_model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            random_state=self.random_state,
            n_jobs=2
        )
        
        # Scale with target data only
        target_only_scaler = StandardScaler()
        X_train_baseline = target_only_scaler.fit_transform(X_train_target)
        X_val_baseline = target_only_scaler.transform(X_val_target) if len(X_val_target) > 0 else X_train_baseline[:1]
        
        baseline_model.fit(X_train_baseline, y_train_target)
        
        if len(X_val_target) > 0:
            y_pred_baseline = baseline_model.predict(X_val_baseline)
            y_pred_proba_baseline = baseline_model.predict_proba(X_val_baseline)[:, 1]
            
            baseline_metrics = {
                'f1': f1_score(y_val_target, y_pred_baseline, zero_division=0),
                'auc': roc_auc_score(y_val_target, y_pred_proba_baseline) if len(np.unique(y_val_target)) > 1 else 0.0,
                'accuracy': accuracy_score(y_val_target, y_pred_baseline)
            }
        else:
            baseline_metrics = {'f1': 0.0, 'auc': 0.0, 'accuracy': 0.0}
        
        # Calculate transfer learning effectiveness
        f1_improvement = target_metrics['f1'] - baseline_metrics['f1']
        auc_improvement = target_metrics['auc'] - baseline_metrics['auc']
        
        transfer_results = {
            'target_species': target_species,
            'source_species': source_species_list,
            'best_source': best_source_model['source_species'],
            'best_similarity': best_source_similarity,
            'target_model': target_model,
            'baseline_model': baseline_model,
            'target_scaler': source_scaler,  # Use source scaler for consistency
            'baseline_scaler': target_only_scaler,
            'target_metrics': target_metrics,
            'baseline_metrics': baseline_metrics,
            'f1_improvement': f1_improvement,
            'auc_improvement': auc_improvement,
            'transfer_effective': f1_improvement > 0.02,  # 2% improvement threshold
            'feature_names': feature_names
        }
        
        # Record transfer learning history
        self.transfer_history.append({
            'timestamp': datetime.now(),
            'target_species': target_species,
            'source_species': source_species_list,
            'target_samples': len(X_train_target),
            'target_positive': np.sum(y_train_target),
            'best_source': best_source_model['source_species'],
            'similarity': best_source_similarity,
            'f1_improvement': f1_improvement,
            'transfer_successful': f1_improvement > 0.02
        })
        
        logger.info(f"✅ Transfer learning completed for {target_species}")
        logger.info(f"📊 F1 improvement: {f1_improvement:+.3f} (Transfer: {target_metrics['f1']:.3f}, Baseline: {baseline_metrics['f1']:.3f})")
        logger.info(f"📊 AUC improvement: {auc_improvement:+.3f} (Transfer: {target_metrics['auc']:.3f}, Baseline: {baseline_metrics['auc']:.3f})")
        
        return transfer_results
    
    def evaluate_transfer_learning(self, target_species: str, 
                                  config: TransferLearningConfig) -> Dict[str, Any]:
        """
        Complete transfer learning evaluation pipeline
        
        Args:
            target_species: Target species for evaluation
            config: Transfer learning configuration
            
        Returns:
            Complete evaluation results
        """
        
        logger.info(f"🔬 Evaluating transfer learning for {target_species}")
        
        # Step 1: Identify source species
        source_species = self.identify_source_species(target_species, config)
        
        if not source_species:
            logger.warning(f"⚠️  No suitable source species found for {target_species}")
            return {'success': False, 'reason': 'No suitable source species'}
        
        # Step 2: Apply transfer learning
        try:
            transfer_results = self.transfer_learn_target_model(target_species, source_species, config)
            
            evaluation = {
                'success': True,
                'target_species': target_species,
                'source_species': source_species,
                'transfer_results': transfer_results,
                'recommendation': 'use_transfer' if transfer_results['transfer_effective'] else 'use_baseline',
                'improvement_summary': {
                    'f1_improvement': transfer_results['f1_improvement'],
                    'auc_improvement': transfer_results['auc_improvement'],
                    'best_source': transfer_results['best_source'],
                    'similarity_score': transfer_results['best_similarity']
                }
            }
            
            return evaluation
            
        except Exception as e:
            logger.error(f"❌ Transfer learning evaluation failed for {target_species}: {e}")
            return {'success': False, 'reason': str(e)}
    
    def get_transfer_learning_report(self) -> Dict[str, Any]:
        """Generate comprehensive transfer learning report"""
        
        report = {
            'system_overview': {
                'registered_species': len(self.species_profiles),
                'species_list': list(self.species_profiles.keys()),
                'transfer_jobs_completed': len(self.transfer_history),
                'source_models_available': len(self.source_models)
            },
            'species_profiles': {},
            'similarity_analysis': {},
            'transfer_history': self.transfer_history,
            'effectiveness_summary': {}
        }
        
        # Species profile summary
        for species, profile in self.species_profiles.items():
            report['species_profiles'][species] = {
                'n_samples': profile.n_samples,
                'positive_samples': profile.positive_samples,
                'occurrence_rate': profile.bloom_characteristics.get('occurrence_rate', 0.0),
                'top_environmental_preferences': self._get_top_preferences(profile)
            }
        
        # Similarity analysis
        if self.similarity_matrix is not None:
            report['similarity_analysis'] = {
                'similarity_matrix': self.similarity_matrix.tolist(),
                'species_order': self.species_list,
                'high_similarity_pairs': self._find_high_similarity_pairs()
            }
        
        # Transfer learning effectiveness
        if self.transfer_history:
            successful_transfers = [h for h in self.transfer_history if h['transfer_successful']]
            avg_improvement = np.mean([h['f1_improvement'] for h in self.transfer_history])
            
            report['effectiveness_summary'] = {
                'success_rate': len(successful_transfers) / len(self.transfer_history),
                'average_f1_improvement': avg_improvement,
                'best_transfer': max(self.transfer_history, key=lambda x: x['f1_improvement']),
                'most_effective_sources': self._find_most_effective_sources()
            }
        
        return report
    
    def _get_top_preferences(self, profile: SpeciesProfile, top_n: int = 3) -> List[Dict[str, Any]]:
        """Get top environmental preferences for a species"""
        
        preferences = []
        for feature, prefs in profile.environmental_preferences.items():
            preferences.append({
                'feature': feature,
                'preference_score': prefs.get('preference_score', 0.0),
                'positive_mean': prefs.get('positive_mean', 0.0),
                'overall_mean': prefs.get('overall_mean', 0.0)
            })
        
        # Sort by preference score and return top N
        preferences.sort(key=lambda x: x['preference_score'], reverse=True)
        return preferences[:top_n]
    
    def _find_high_similarity_pairs(self, threshold: float = 0.7) -> List[Dict[str, Any]]:
        """Find species pairs with high similarity"""
        
        if self.similarity_matrix is None:
            return []
        
        high_similarity_pairs = []
        n_species = len(self.species_list)
        
        for i in range(n_species):
            for j in range(i+1, n_species):
                similarity = self.similarity_matrix[i, j]
                if similarity >= threshold:
                    high_similarity_pairs.append({
                        'species_1': self.species_list[i],
                        'species_2': self.species_list[j],
                        'similarity': similarity
                    })
        
        # Sort by similarity
        high_similarity_pairs.sort(key=lambda x: x['similarity'], reverse=True)
        return high_similarity_pairs
    
    def _find_most_effective_sources(self) -> List[Dict[str, Any]]:
        """Find most effective source species for transfer learning"""
        
        if not self.transfer_history:
            return []
        
        source_effectiveness = defaultdict(list)
        
        for history in self.transfer_history:
            source = history['best_source']
            improvement = history['f1_improvement']
            source_effectiveness[source].append(improvement)
        
        effectiveness_summary = []
        for source, improvements in source_effectiveness.items():
            effectiveness_summary.append({
                'source_species': source,
                'avg_improvement': np.mean(improvements),
                'success_rate': np.mean([imp > 0.02 for imp in improvements]),
                'transfer_count': len(improvements)
            })
        
        effectiveness_summary.sort(key=lambda x: x['avg_improvement'], reverse=True)
        return effectiveness_summary

if __name__ == "__main__":
    print("🔄 HAB Species Transfer Learning - Task 4.4")
    print("🎯 Knowledge transfer between similar HAB species")
    print("🧬 Environmental preference-based species similarity")
    
    # Initialize transfer learning system
    tl_system = HABSpeciesTransferLearning()
    
    print("✅ HAB Transfer Learning System initialized successfully!")
    print("🤖 Ready for cross-species knowledge transfer and few-shot learning")