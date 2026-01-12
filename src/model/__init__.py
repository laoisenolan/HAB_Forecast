"""
Model Module for HAB Forecasting System

This module handles machine learning tasks:
- XGBoost model training
- SHAP interpretability
- Model evaluation
- Prediction pipeline
"""

from .xgboost_trainer import XGBoostTrainer
from .shap_analyzer import SHAPAnalyzer
from .predictor import HABPredictor

__all__ = [
    'XGBoostTrainer',
    'SHAPAnalyzer',
    'HABPredictor'
] 