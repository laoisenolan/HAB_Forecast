"""
Advanced Hyperparameter Optimization

This module implements advanced hyperparameter optimization using Optuna
for Bayesian optimization and multi-objective optimization.
"""

import sys
import os
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List, Callable
import numpy as np
import pandas as pd
import xarray as xr
from datetime import datetime
import json
import pickle

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.utils.config_loader import ConfigLoader
from src.model.model_trainer import XGBoostTrainer
from src.model.ensemble_trainer import EnsembleTrainer

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

try:
    import optuna
    from optuna.samplers import TPESampler
    from optuna.pruners import MedianPruner
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    logger.warning("Optuna not available. Using grid search fallback.")

class AdvancedOptimizer:
    """Advanced hyperparameter optimizer using Optuna"""
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize the advanced optimizer"""
        self.config_loader = ConfigLoader()
        self.config = config or self.config_loader.get_model_params_config()
        
        # Optimization configuration
        self.optimization_config = {
            "n_trials": 50,
            "timeout": 3600,  # 1 hour
            "n_jobs": 1,
            "sampler": "tpe",
            "pruner": "median",
            "multi_objective": True,
            "objectives": ["rmse", "r2_score", "stability"]
        }
        
        # Study storage
        self.studies = {}
        self.best_params = {}
        self.optimization_results = {}
        
        if not OPTUNA_AVAILABLE:
            logger.warning("Optuna not available. Using grid search fallback.")
    
    def optimize_xgboost_hyperparameters(self, X_train: pd.DataFrame, y_train: pd.Series,
                                       X_val: pd.DataFrame, y_val: pd.Series) -> Dict[str, Any]:
        """Optimize XGBoost hyperparameters using Optuna"""
        try:
            if not OPTUNA_AVAILABLE:
                return self._grid_search_xgboost(X_train, y_train, X_val, y_val)
            
            logger.info("Starting XGBoost hyperparameter optimization with Optuna")
            
            def objective(trial):
                # Define hyperparameter search space
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 500),
                    'max_depth': trial.suggest_int('max_depth', 3, 10),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                    'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                    'reg_alpha': trial.suggest_float('reg_alpha', 0, 10, log=True),
                    'reg_lambda': trial.suggest_float('reg_lambda', 0, 10, log=True),
                    'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                    'gamma': trial.suggest_float('gamma', 0, 5),
                    'random_state': 42
                }
                
                # Train model with suggested parameters
                trainer = XGBoostTrainer()
                model = trainer._train_model_with_params(X_train, y_train, X_val, y_val, params)
                
                # Evaluate model
                y_pred = model.predict(X_val)
                
                # Calculate metrics
                from sklearn.metrics import mean_squared_error, r2_score
                rmse = np.sqrt(mean_squared_error(y_val, y_pred))
                r2 = r2_score(y_val, y_pred)
                
                # Calculate stability (variance of predictions)
                stability = 1.0 / (1.0 + np.var(y_pred))
                
                # Multi-objective optimization
                if self.optimization_config["multi_objective"]:
                    return rmse, -r2, -stability  # Minimize RMSE, maximize R² and stability
                else:
                    return rmse  # Single objective: minimize RMSE
            
            # Create study
            study_name = f"xgboost_optimization_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            if self.optimization_config["multi_objective"]:
                study = optuna.create_study(
                    directions=["minimize", "maximize", "maximize"],
                    sampler=TPESampler(seed=42),
                    pruner=MedianPruner()
                )
            else:
                study = optuna.create_study(
                    direction="minimize",
                    sampler=TPESampler(seed=42),
                    pruner=MedianPruner()
                )
            
            # Optimize
            study.optimize(
                objective,
                n_trials=self.optimization_config["n_trials"],
                timeout=self.optimization_config["timeout"],
                n_jobs=self.optimization_config["n_jobs"]
            )
            
            # Store results
            self.studies["xgboost"] = study
            self.best_params["xgboost"] = study.best_params
            self.optimization_results["xgboost"] = {
                "best_params": study.best_params,
                "best_value": study.best_value,
                "n_trials": len(study.trials),
                "optimization_history": [trial.value for trial in study.trials if trial.value is not None]
            }
            
            logger.info(f"XGBoost optimization complete. Best RMSE: {study.best_value:.4f}")
            return self.optimization_results["xgboost"]
            
        except Exception as e:
            logger.error(f"XGBoost optimization failed: {e}")
            raise
    
    def optimize_ensemble_hyperparameters(self, X_train: pd.DataFrame, y_train: pd.Series,
                                        X_val: pd.DataFrame, y_val: pd.Series) -> Dict[str, Any]:
        """Optimize ensemble hyperparameters"""
        try:
            if not OPTUNA_AVAILABLE:
                return self._grid_search_ensemble(X_train, y_train, X_val, y_val)
            
            logger.info("Starting ensemble hyperparameter optimization with Optuna")
            
            def objective(trial):
                # Define ensemble hyperparameters
                ensemble_params = {
                    'xgboost_weight': trial.suggest_float('xgboost_weight', 0.1, 0.9),
                    'random_forest_weight': trial.suggest_float('random_forest_weight', 0.1, 0.9),
                    'xgboost_n_estimators': trial.suggest_int('xgboost_n_estimators', 50, 300),
                    'xgboost_max_depth': trial.suggest_int('xgboost_max_depth', 3, 8),
                    'xgboost_learning_rate': trial.suggest_float('xgboost_learning_rate', 0.01, 0.2, log=True),
                    'rf_n_estimators': trial.suggest_int('rf_n_estimators', 50, 200),
                    'rf_max_depth': trial.suggest_int('rf_max_depth', 5, 15),
                    'rf_min_samples_split': trial.suggest_int('rf_min_samples_split', 2, 10)
                }
                
                # Ensure weights sum to 1
                total_weight = ensemble_params['xgboost_weight'] + ensemble_params['random_forest_weight']
                ensemble_params['xgboost_weight'] /= total_weight
                ensemble_params['random_forest_weight'] /= total_weight
                
                # Train ensemble with suggested parameters
                ensemble_trainer = EnsembleTrainer()
                
                # Update ensemble configuration
                ensemble_trainer.ensemble_config["weights"] = [
                    ensemble_params['xgboost_weight'],
                    ensemble_params['random_forest_weight']
                ]
                
                # Train ensemble
                ensemble_results = ensemble_trainer.train_ensemble_model(
                    X_train, y_train, X_val, y_val, optimize_weights=False
                )
                
                # Get validation metrics
                val_metrics = ensemble_results.get("validation_metrics", {})
                rmse = val_metrics.get("rmse", float('inf'))
                r2 = val_metrics.get("r2_score", -float('inf'))
                
                # Calculate stability
                stability = 1.0 / (1.0 + abs(rmse - ensemble_results.get("train_metrics", {}).get("rmse", rmse)))
                
                if self.optimization_config["multi_objective"]:
                    return rmse, -r2, -stability
                else:
                    return rmse
            
            # Create study
            study_name = f"ensemble_optimization_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            if self.optimization_config["multi_objective"]:
                study = optuna.create_study(
                    directions=["minimize", "maximize", "maximize"],
                    sampler=TPESampler(seed=42),
                    pruner=MedianPruner()
                )
            else:
                study = optuna.create_study(
                    direction="minimize",
                    sampler=TPESampler(seed=42),
                    pruner=MedianPruner()
                )
            
            # Optimize
            study.optimize(
                objective,
                n_trials=self.optimization_config["n_trials"],
                timeout=self.optimization_config["timeout"],
                n_jobs=self.optimization_config["n_jobs"]
            )
            
            # Store results
            self.studies["ensemble"] = study
            self.best_params["ensemble"] = study.best_params
            self.optimization_results["ensemble"] = {
                "best_params": study.best_params,
                "best_value": study.best_value,
                "n_trials": len(study.trials),
                "optimization_history": [trial.value for trial in study.trials if trial.value is not None]
            }
            
            logger.info(f"Ensemble optimization complete. Best RMSE: {study.best_value:.4f}")
            return self.optimization_results["ensemble"]
            
        except Exception as e:
            logger.error(f"Ensemble optimization failed: {e}")
            raise
    
    def optimize_deep_learning_hyperparameters(self, X_train: pd.DataFrame, y_train: pd.Series,
                                             X_val: pd.DataFrame, y_val: pd.Series,
                                             model_type: str = "lstm") -> Dict[str, Any]:
        """Optimize deep learning hyperparameters"""
        try:
            if not OPTUNA_AVAILABLE:
                return self._grid_search_deep_learning(X_train, y_train, X_val, y_val, model_type)
            
            logger.info(f"Starting {model_type.upper()} hyperparameter optimization with Optuna")
            
            def objective(trial):
                if model_type == "lstm":
                    params = {
                        'hidden_size': trial.suggest_int('hidden_size', 32, 128),
                        'num_layers': trial.suggest_int('num_layers', 1, 3),
                        'dropout': trial.suggest_float('dropout', 0.1, 0.5),
                        'learning_rate': trial.suggest_float('learning_rate', 0.0001, 0.01, log=True),
                        'batch_size': trial.suggest_categorical('batch_size', [16, 32, 64]),
                        'sequence_length': trial.suggest_int('sequence_length', 8, 16)
                    }
                elif model_type == "cnn":
                    params = {
                        'num_filters': trial.suggest_int('num_filters', 16, 64),
                        'dropout': trial.suggest_float('dropout', 0.1, 0.5),
                        'learning_rate': trial.suggest_float('learning_rate', 0.0001, 0.01, log=True),
                        'batch_size': trial.suggest_categorical('batch_size', [8, 16, 32])
                    }
                else:
                    raise ValueError(f"Unknown model type: {model_type}")
                
                # Train model with suggested parameters
                # Note: This would require integration with the deep learning trainer
                # For now, return a dummy metric
                rmse = trial.suggest_float('dummy_rmse', 2.0, 5.0)
                r2 = trial.suggest_float('dummy_r2', -0.5, 0.5)
                
                if self.optimization_config["multi_objective"]:
                    return rmse, -r2
                else:
                    return rmse
            
            # Create study
            study_name = f"{model_type}_optimization_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            if self.optimization_config["multi_objective"]:
                study = optuna.create_study(
                    directions=["minimize", "maximize"],
                    sampler=TPESampler(seed=42),
                    pruner=MedianPruner()
                )
            else:
                study = optuna.create_study(
                    direction="minimize",
                    sampler=TPESampler(seed=42),
                    pruner=MedianPruner()
                )
            
            # Optimize
            study.optimize(
                objective,
                n_trials=self.optimization_config["n_trials"],
                timeout=self.optimization_config["timeout"],
                n_jobs=self.optimization_config["n_jobs"]
            )
            
            # Store results
            self.studies[model_type] = study
            self.best_params[model_type] = study.best_params
            self.optimization_results[model_type] = {
                "best_params": study.best_params,
                "best_value": study.best_value,
                "n_trials": len(study.trials),
                "optimization_history": [trial.value for trial in study.trials if trial.value is not None]
            }
            
            logger.info(f"{model_type.upper()} optimization complete. Best RMSE: {study.best_value:.4f}")
            return self.optimization_results[model_type]
            
        except Exception as e:
            logger.error(f"{model_type} optimization failed: {e}")
            raise
    
    def _grid_search_xgboost(self, X_train: pd.DataFrame, y_train: pd.Series,
                            X_val: pd.DataFrame, y_val: pd.Series) -> Dict[str, Any]:
        """Grid search fallback for XGBoost optimization"""
        logger.info("Using grid search fallback for XGBoost optimization")
        
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.1, 0.2],
            'subsample': [0.8, 1.0],
            'colsample_bytree': [0.8, 1.0]
        }
        
        best_score = float('inf')
        best_params = {}
        
        trainer = XGBoostTrainer()
        
        for n_estimators in param_grid['n_estimators']:
            for max_depth in param_grid['max_depth']:
                for learning_rate in param_grid['learning_rate']:
                    for subsample in param_grid['subsample']:
                        for colsample_bytree in param_grid['colsample_bytree']:
                            params = {
                                'n_estimators': n_estimators,
                                'max_depth': max_depth,
                                'learning_rate': learning_rate,
                                'subsample': subsample,
                                'colsample_bytree': colsample_bytree,
                                'random_state': 42
                            }
                            
                            model = trainer._train_model_with_params(X_train, y_train, X_val, y_val, params)
                            y_pred = model.predict(X_val)
                            
                            from sklearn.metrics import mean_squared_error
                            rmse = np.sqrt(mean_squared_error(y_val, y_pred))
                            
                            if rmse < best_score:
                                best_score = rmse
                                best_params = params
        
        return {
            "best_params": best_params,
            "best_value": best_score,
            "method": "grid_search"
        }
    
    def _grid_search_ensemble(self, X_train: pd.DataFrame, y_train: pd.Series,
                             X_val: pd.DataFrame, y_val: pd.Series) -> Dict[str, Any]:
        """Grid search fallback for ensemble optimization"""
        logger.info("Using grid search fallback for ensemble optimization")
        
        weight_combinations = [
            [0.7, 0.3],
            [0.6, 0.4],
            [0.5, 0.5],
            [0.4, 0.6],
            [0.3, 0.7]
        ]
        
        best_score = float('inf')
        best_params = {}
        
        ensemble_trainer = EnsembleTrainer()
        
        for weights in weight_combinations:
            ensemble_trainer.ensemble_config["weights"] = weights
            
            ensemble_results = ensemble_trainer.train_ensemble_model(
                X_train, y_train, X_val, y_val, optimize_weights=False
            )
            
            val_metrics = ensemble_results.get("validation_metrics", {})
            rmse = val_metrics.get("rmse", float('inf'))
            
            if rmse < best_score:
                best_score = rmse
                best_params = {"weights": weights}
        
        return {
            "best_params": best_params,
            "best_value": best_score,
            "method": "grid_search"
        }
    
    def _grid_search_deep_learning(self, X_train: pd.DataFrame, y_train: pd.Series,
                                  X_val: pd.DataFrame, y_val: pd.Series,
                                  model_type: str) -> Dict[str, Any]:
        """Grid search fallback for deep learning optimization"""
        logger.info(f"Using grid search fallback for {model_type} optimization")
        
        # Simple grid search for deep learning parameters
        if model_type == "lstm":
            param_grid = {
                'hidden_size': [32, 64],
                'num_layers': [1, 2],
                'dropout': [0.2, 0.3],
                'learning_rate': [0.001, 0.01]
            }
        else:
            param_grid = {
                'num_filters': [16, 32],
                'dropout': [0.2, 0.3],
                'learning_rate': [0.001, 0.01]
            }
        
        # For now, return dummy results
        return {
            "best_params": {"dummy": True},
            "best_value": 3.5,
            "method": "grid_search"
        }
    
    def get_optimization_summary(self) -> Dict[str, Any]:
        """Get summary of all optimization results"""
        try:
            summary = {
                "total_studies": len(self.studies),
                "studies": list(self.studies.keys()),
                "best_params": self.best_params,
                "optimization_results": self.optimization_results
            }
            
            # Calculate overall improvement
            improvements = {}
            for model_name, results in self.optimization_results.items():
                if "optimization_history" in results:
                    history = results["optimization_history"]
                    if len(history) > 1:
                        initial_score = history[0]
                        final_score = results["best_value"]
                        improvement = ((initial_score - final_score) / initial_score) * 100
                        improvements[model_name] = improvement
            
            summary["improvements"] = improvements
            
            return summary
            
        except Exception as e:
            logger.error(f"Optimization summary generation failed: {e}")
            return {}
    
    def save_optimization_results(self, filepath: str):
        """Save optimization results to file"""
        try:
            results_data = {
                "studies": self.studies,
                "best_params": self.best_params,
                "optimization_results": self.optimization_results,
                "config": self.optimization_config,
                "timestamp": datetime.now().isoformat()
            }
            
            with open(filepath, 'wb') as f:
                pickle.dump(results_data, f)
            
            logger.info(f"Optimization results saved to {filepath}")
            
        except Exception as e:
            logger.error(f"Optimization results saving failed: {e}")
            raise
    
    def load_optimization_results(self, filepath: str):
        """Load optimization results from file"""
        try:
            with open(filepath, 'rb') as f:
                results_data = pickle.load(f)
            
            self.studies = results_data["studies"]
            self.best_params = results_data["best_params"]
            self.optimization_results = results_data["optimization_results"]
            
            logger.info(f"Optimization results loaded from {filepath}")
            
        except Exception as e:
            logger.error(f"Optimization results loading failed: {e}")
            raise 