#!/usr/bin/env python3
"""
ULTIMATE HAB Prediction Dashboard
Advanced visualization and prediction interface for the species-specific HAB prediction system
"""

import os
import sys
import json
import pickle
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.io as pio
from flask import Flask, render_template, jsonify, request, send_from_directory

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class UltimateHABDashboard:
    """
    Ultimate HAB Prediction Dashboard with advanced visualizations and insights
    """
    
    def __init__(self):
        self.app = Flask(__name__, 
                        template_folder=str(project_root / "src/web/templates"),
                        static_folder=str(project_root / "src/web/static"))
        
        # Coverage area coordinates
        self.coverage_bounds = {
            'lat_min': 33.575, 'lat_max': 34.425,
            'lon_min': 132.575, 'lon_max': 133.425
        }
        
        # Model directory
        self.model_dir = project_root / "cache/species_models_ultimate"
        
        # Load models and metadata
        self.models = {}
        self.species_metadata = {}
        self.load_models()
        
        # Load environmental data for visualizations
        self.env_data = None
        self.load_environmental_data()
        
        # Setup Flask routes
        self.setup_routes()
        
        logger.info("Ultimate HAB Dashboard initialized")
        logger.info(f"Loaded {len(self.models)} species models")
        logger.info(f"Coverage area: {self.coverage_bounds}")
    
    def load_models(self):
        """Load all trained species models"""
        logger.info("Loading species models...")
        
        if not self.model_dir.exists():
            logger.warning(f"Model directory not found: {self.model_dir}")
            return
        
        for species_dir in self.model_dir.iterdir():
            if species_dir.is_dir():
                try:
                    # Load metadata
                    metadata_file = species_dir / "metadata.json"
                    if metadata_file.exists():
                        with open(metadata_file, 'r') as f:
                            metadata = json.load(f)
                        
                        species_name = metadata['species']
                        self.species_metadata[species_name] = metadata
                        
                        # Load model components
                        model_file = species_dir / "model.pkl"
                        scaler_file = species_dir / "scaler.pkl"
                        
                        if model_file.exists() and scaler_file.exists():
                            with open(model_file, 'rb') as f:
                                model = pickle.load(f)
                            with open(scaler_file, 'rb') as f:
                                scaler = pickle.load(f)
                            
                            # Load selector if exists
                            selector = None
                            selector_file = species_dir / "selector.pkl"
                            if selector_file.exists():
                                with open(selector_file, 'rb') as f:
                                    selector = pickle.load(f)
                            
                            self.models[species_name] = {
                                'model': model,
                                'scaler': scaler,
                                'selector': selector,
                                'metadata': metadata
                            }
                            
                            logger.info(f"Loaded model for {species_name}")
                        
                except Exception as e:
                    logger.error(f"Error loading model from {species_dir}: {e}")
    
    def load_environmental_data(self):
        """Load environmental data for visualizations"""
        logger.info("Loading environmental data...")
        
        try:
            # Load the species-integrated dataset
            data_file = project_root / "cache/species_integrated/environmental_with_species.parquet"
            if data_file.exists():
                self.env_data = pd.read_parquet(data_file)
                self.env_data['date'] = pd.to_datetime(self.env_data['date'])
                
                # CRITICAL FIX: Clean the data immediately after loading
                self.clean_environmental_data()
                
                logger.info(f"Loaded environmental data: {len(self.env_data):,} samples")
            else:
                logger.warning(f"Environmental data file not found: {data_file}")
        
        except Exception as e:
            logger.error(f"Error loading environmental data: {e}")
    
    def clean_environmental_data(self):
        """Clean environmental data to remove NaN and infinite values"""
        if self.env_data is None:
            return
            
        logger.info("Cleaning environmental data...")
        
        # Get feature columns
        feature_cols = [col for col in self.env_data.columns if col not in 
                       ['date', 'grid_cell_id', 'lat', 'lon', 'bay_region', 'has_hab', 'hab_species', 'hab_occurrence']]
        
        # Replace infinite values
        for col in feature_cols:
            if self.env_data[col].dtype in ['float64', 'float32']:
                self.env_data[col] = self.env_data[col].replace([np.inf, -np.inf], 0)
        
        # Fill NaN values
        for col in feature_cols:
            if self.env_data[col].dtype in ['float64', 'float32', 'int64', 'int32']:
                if self.env_data[col].isna().all():
                    # All NaN - fill with 0
                    self.env_data[col] = self.env_data[col].fillna(0)
                else:
                    # Some valid data - use median, fallback to 0
                    median_val = self.env_data[col].median()
                    if pd.isna(median_val):
                        self.env_data[col] = self.env_data[col].fillna(0)
                    else:
                        self.env_data[col] = self.env_data[col].fillna(median_val)
            else:
                self.env_data[col] = self.env_data[col].fillna(0)
        
        # Final safety check
        self.env_data = self.env_data.replace([np.inf, -np.inf], 0)
        self.env_data = self.env_data.fillna(0)
        
        logger.info("Environmental data cleaned successfully")
    
    def setup_routes(self):
        """Setup Flask routes"""
        
        @self.app.route('/')
        def dashboard():
            """Main prediction dashboard page"""
            return render_template('prediction_dashboard.html')
        
        @self.app.route('/analysis')
        def analysis_dashboard():
            """Analysis dashboard page"""
            return render_template('ultimate_dashboard.html')
        
        @self.app.route('/api/models/overview')
        def models_overview():
            """API endpoint for models overview"""
            try:
                overview = []
                for species, metadata in self.species_metadata.items():
                    model_info = {
                        'species': species,
                        'auc': metadata['metrics']['auc'],
                        'f1': metadata['metrics']['f1'],
                        'precision': metadata['metrics']['precision'],
                        'recall': metadata['metrics']['recall'],
                        'training_samples': metadata['training_info']['train_samples'],
                        'positive_samples': metadata['training_info']['positive_samples'],
                        'feature_count': metadata['feature_count'],
                        'created_at': metadata['training_info']['created_at']
                    }
                    overview.append(model_info)
                
                # Sort by AUC descending
                overview.sort(key=lambda x: x['auc'], reverse=True)
                
                return jsonify({
                    'success': True,
                    'models': overview,
                    'total_models': len(overview),
                    'coverage_area': self.coverage_bounds
                })
            
            except Exception as e:
                logger.error(f"Error in models overview: {e}")
                return jsonify({'success': False, 'error': str(e)})
        
        @self.app.route('/api/species/<species_name>/details')
        def species_details(species_name):
            """API endpoint for detailed species information"""
            try:
                if species_name not in self.species_metadata:
                    return jsonify({'success': False, 'error': 'Species not found'})
                
                metadata = self.species_metadata[species_name]
                
                # Get feature importance if available
                feature_importance = metadata.get('feature_importance', [])
                
                # Get confusion matrix
                cm = metadata.get('confusion_matrix', {})
                
                details = {
                    'species': species_name,
                    'metrics': metadata['metrics'],
                    'feature_importance': feature_importance,
                    'confusion_matrix': cm,
                    'training_info': metadata['training_info'],
                    'model_performance_grade': self.get_performance_grade(metadata['metrics']['auc'])
                }
                
                return jsonify({'success': True, 'details': details})
            
            except Exception as e:
                logger.error(f"Error getting species details for {species_name}: {e}")
                return jsonify({'success': False, 'error': str(e)})
        
        @self.app.route('/api/environmental/current')
        def current_environmental():
            """API endpoint for current environmental conditions"""
            try:
                if self.env_data is None:
                    return jsonify({'success': False, 'error': 'Environmental data not available'})
                
                # Get latest data
                latest_date = self.env_data['date'].max()
                latest_data = self.env_data[self.env_data['date'] == latest_date]
                
                # Calculate summary statistics
                env_summary = {
                    'date': latest_date.isoformat(),
                    'total_grid_points': len(latest_data),
                    'hab_positive_points': int(latest_data['has_hab'].sum()),
                    'environmental_conditions': {
                        'sst_mean': float(latest_data['sst'].mean()) if 'sst' in latest_data.columns else None,
                        'sst_range': [float(latest_data['sst'].min()), float(latest_data['sst'].max())] if 'sst' in latest_data.columns else None,
                        'salinity_mean': float(latest_data['ocean_salinity'].mean()) if 'ocean_salinity' in latest_data.columns else None,
                        'chlor_a_mean': float(latest_data['chlor_a'].mean()) if 'chlor_a' in latest_data.columns else None
                    }
                }
                
                return jsonify({'success': True, 'environmental': env_summary})
            
            except Exception as e:
                logger.error(f"Error getting current environmental conditions: {e}")
                return jsonify({'success': False, 'error': str(e)})
        
        @self.app.route('/api/predictions/current')
        def current_predictions():
            """API endpoint for current HAB risk predictions across all grid points"""
            try:
                if self.env_data is None:
                    return jsonify({'success': False, 'error': 'Environmental data not available'})
                
                # Get latest environmental conditions (or simulate current conditions)
                latest_date = self.env_data['date'].max()
                current_conditions = self.env_data[self.env_data['date'] == latest_date].copy()
                
                # Generate predictions for all species and all grid points
                predictions_by_location = []
                
                for _, point in current_conditions.iterrows():
                    location_predictions = {
                        'lat': float(point['lat']),
                        'lon': float(point['lon']),
                        'environmental_conditions': {
                            'sst': float(point['sst']) if pd.notna(point['sst']) else None,
                            'salinity': float(point['ocean_salinity']) if pd.notna(point['ocean_salinity']) else None,
                            'chlor_a': float(point['chlor_a']) if pd.notna(point['chlor_a']) else None,
                            'turbidity': float(point['turbidity_index']) if pd.notna(point['turbidity_index']) else None
                        },
                        'species_predictions': {},
                        'max_risk_species': None,
                        'max_risk_probability': 0.0,
                        'overall_risk_level': 'low'
                    }
                    
                    # Get predictions for each species
                    for species_name in self.models.keys():
                        prediction = self.predict_single_point(point, species_name)
                        location_predictions['species_predictions'][species_name] = {
                            'probability': float(prediction) if prediction is not None else 0.0,
                            'risk_level': self.get_risk_level(prediction) if prediction is not None else 'low'
                        }
                        
                        # Track highest risk species
                        if prediction is not None and prediction > location_predictions['max_risk_probability']:
                            location_predictions['max_risk_probability'] = float(prediction)
                            location_predictions['max_risk_species'] = species_name
                    
                    # Determine overall risk level
                    location_predictions['overall_risk_level'] = self.get_risk_level(location_predictions['max_risk_probability'])
                    
                    predictions_by_location.append(location_predictions)
                
                return jsonify({
                    'success': True,
                    'predictions': predictions_by_location,
                    'total_locations': len(predictions_by_location),
                    'prediction_date': latest_date.isoformat(),
                    'coverage_bounds': self.coverage_bounds,
                    'species_count': len(self.models)
                })
            
            except Exception as e:
                logger.error(f"Error generating current predictions: {e}")
                return jsonify({'success': False, 'error': str(e)})
        
        @self.app.route('/api/predictions/location')
        def location_prediction():
            """API endpoint for predictions at a specific location"""
            try:
                lat = float(request.args.get('lat', 0))
                lon = float(request.args.get('lon', 0))
                
                if not (self.coverage_bounds['lat_min'] <= lat <= self.coverage_bounds['lat_max'] and
                        self.coverage_bounds['lon_min'] <= lon <= self.coverage_bounds['lon_max']):
                    return jsonify({'success': False, 'error': 'Location outside coverage area'})
                
                # Find nearest grid point
                if self.env_data is None:
                    return jsonify({'success': False, 'error': 'Environmental data not available'})
                
                latest_date = self.env_data['date'].max()
                current_data = self.env_data[self.env_data['date'] == latest_date].copy()
                
                # Calculate distances and find nearest point
                current_data['distance'] = np.sqrt((current_data['lat'] - lat)**2 + (current_data['lon'] - lon)**2)
                nearest_point = current_data.loc[current_data['distance'].idxmin()]
                
                # Generate predictions for all species at this location
                location_analysis = {
                    'requested_location': {'lat': lat, 'lon': lon},
                    'nearest_grid_point': {
                        'lat': float(nearest_point['lat']),
                        'lon': float(nearest_point['lon']),
                        'distance_km': float(nearest_point['distance'] * 111)  # Approximate km conversion
                    },
                    'environmental_conditions': {
                        'sea_surface_temperature': float(nearest_point['sst']) if pd.notna(nearest_point['sst']) else None,
                        'ocean_salinity': float(nearest_point['ocean_salinity']) if pd.notna(nearest_point['ocean_salinity']) else None,
                        'chlorophyll_a': float(nearest_point['chlor_a']) if pd.notna(nearest_point['chlor_a']) else None,
                        'turbidity_index': float(nearest_point['turbidity_index']) if pd.notna(nearest_point['turbidity_index']) else None,
                        'ocean_current_u': float(nearest_point['ocean_current_u']) if pd.notna(nearest_point['ocean_current_u']) else None,
                        'ocean_current_v': float(nearest_point['ocean_current_v']) if pd.notna(nearest_point['ocean_current_v']) else None
                    },
                    'species_risk_assessment': {},
                    'highest_risk': {
                        'species': None,
                        'probability': 0.0,
                        'risk_level': 'low',
                        'confidence': 'low'
                    },
                    'recommendations': []
                }
                
                # Predict for each species
                for species_name in self.models.keys():
                    prediction = self.predict_single_point(nearest_point, species_name)
                    if prediction is not None:
                        risk_assessment = {
                            'probability': float(prediction),
                            'risk_level': self.get_risk_level(prediction),
                            'confidence': self.get_confidence_level(prediction, species_name),
                            'interpretation': self.interpret_prediction(prediction, species_name)
                        }
                        location_analysis['species_risk_assessment'][species_name] = risk_assessment
                        
                        # Update highest risk
                        if prediction > location_analysis['highest_risk']['probability']:
                            location_analysis['highest_risk'] = {
                                'species': species_name,
                                'probability': float(prediction),
                                'risk_level': self.get_risk_level(prediction),
                                'confidence': self.get_confidence_level(prediction, species_name)
                            }
                
                # Generate recommendations
                location_analysis['recommendations'] = self.generate_recommendations(location_analysis)
                
                return jsonify({
                    'success': True,
                    'analysis': location_analysis,
                    'prediction_timestamp': datetime.now().isoformat()
                })
            
            except Exception as e:
                logger.error(f"Error generating location prediction: {e}")
                return jsonify({'success': False, 'error': str(e)})
        
        @self.app.route('/api/analysis/temporal')
        def temporal_analysis():
            """API endpoint for temporal analysis"""
            try:
                if self.env_data is None:
                    return jsonify({'success': False, 'error': 'Environmental data not available'})
                
                # Calculate daily HAB statistics
                daily_stats = self.env_data.groupby('date').agg({
                    'has_hab': ['sum', 'count'],
                    'sst': 'mean',
                    'ocean_salinity': 'mean',
                    'chlor_a': 'mean'
                }).reset_index()
                
                daily_stats.columns = ['date', 'hab_positive', 'total_points', 'sst_mean', 'salinity_mean', 'chlor_a_mean']
                daily_stats['hab_rate'] = daily_stats['hab_positive'] / daily_stats['total_points']
                daily_stats['date'] = daily_stats['date'].dt.strftime('%Y-%m-%d')
                
                # Species temporal distribution
                species_temporal = self.env_data[self.env_data['has_hab'] == 1].groupby(['date', 'hab_species']).size().reset_index(name='count')
                species_temporal['date'] = species_temporal['date'].dt.strftime('%Y-%m-%d')
                
                return jsonify({
                    'success': True,
                    'daily_statistics': daily_stats.to_dict('records'),
                    'species_temporal': species_temporal.to_dict('records')
                })
            
            except Exception as e:
                logger.error(f"Error in temporal analysis: {e}")
                return jsonify({'success': False, 'error': str(e)})
        
        @self.app.route('/api/performance/validation')
        def performance_validation():
            """API endpoint for model performance validation"""
            try:
                validation_results = []
                
                for species, metadata in self.species_metadata.items():
                    metrics = metadata['metrics']
                    
                    # Performance assessment
                    performance = {
                        'species': species,
                        'metrics': metrics,
                        'performance_grade': self.get_performance_grade(metrics['auc']),
                        'interpretation': self.interpret_performance(metrics),
                        'strengths': self.identify_strengths(metadata),
                        'considerations': self.identify_considerations(metadata)
                    }
                    
                    validation_results.append(performance)
                
                return jsonify({
                    'success': True,
                    'validation_results': validation_results,
                    'overall_summary': self.generate_overall_summary()
                })
            
            except Exception as e:
                logger.error(f"Error in performance validation: {e}")
                return jsonify({'success': False, 'error': str(e)})
    
    def predict_species(self, data: pd.DataFrame, species: str) -> Optional[np.ndarray]:
        """Generate predictions for a specific species"""
        try:
            if species not in self.models:
                return None
            
            model_info = self.models[species]
            model = model_info['model']
            scaler = model_info['scaler']
            selector = model_info['selector']
            
            # CRITICAL FIX: Align features with what selector was trained on
            if selector and hasattr(selector, 'feature_names_in_'):
                # Use only the features the selector was trained on
                selector_features = selector.feature_names_in_
                available_features = [f for f in selector_features if f in data.columns]
                
                if len(available_features) != len(selector_features):
                    logger.warning(f"Feature mismatch for {species}: {len(available_features)}/{len(selector_features)} features available")
                    # Try fallback to metadata features
                    metadata_features = model_info['metadata']['features']
                    if all(f in data.columns for f in metadata_features):
                        logger.info(f"Using metadata features as fallback for {species}")
                        X = data[metadata_features].copy()
                                    # Handle missing values - ROBUST NaN handling
            for col in X.columns:
                if X[col].dtype in ['float64', 'float32', 'int64', 'int32']:
                    # Use 0 if all values are NaN (median would fail)
                    if X[col].isna().all():
                        X[col] = X[col].fillna(0.0)
                    else:
                        X[col] = X[col].fillna(X[col].median())
                else:
                    X[col] = X[col].fillna(0)
            
            # Final safety check - replace any remaining NaN/inf values
            X = X.replace([np.inf, -np.inf], 0)
            X = X.fillna(0)
                        X_scaled = scaler.transform(X.values)
                        predictions = model.predict_proba(X_scaled)[:, 1]
                        return predictions
                    else:
                        logger.error(f"Cannot predict for {species}: missing required features")
                        return None
                
                # Prepare data with aligned features
                X = data[available_features].copy()
                
                            # Handle missing values - ROBUST NaN handling
            for col in X.columns:
                if X[col].dtype in ['float64', 'float32', 'int64', 'int32']:
                    # Use 0 if all values are NaN (median would fail)
                    if X[col].isna().all():
                        X[col] = X[col].fillna(0.0)
                    else:
                        X[col] = X[col].fillna(X[col].median())
                else:
                    X[col] = X[col].fillna(0)
            
            # Final safety check - replace any remaining NaN/inf values
            X = X.replace([np.inf, -np.inf], 0)
            X = X.fillna(0)
                
                # Apply selector
                X_selected = selector.transform(X)
                
            else:
                # No selector or no feature names - use metadata features directly
                metadata_features = model_info['metadata']['features']
                if not all(f in data.columns for f in metadata_features):
                    logger.error(f"Missing metadata features for {species}")
                    return None
                
                X = data[metadata_features].copy()
                
                            # Handle missing values - ROBUST NaN handling
            for col in X.columns:
                if X[col].dtype in ['float64', 'float32', 'int64', 'int32']:
                    # Use 0 if all values are NaN (median would fail)
                    if X[col].isna().all():
                        X[col] = X[col].fillna(0.0)
                    else:
                        X[col] = X[col].fillna(X[col].median())
                else:
                    X[col] = X[col].fillna(0)
            
            # Final safety check - replace any remaining NaN/inf values
            X = X.replace([np.inf, -np.inf], 0)
            X = X.fillna(0)
                
                X_selected = X.values
            
            # Scale features
            X_scaled = scaler.transform(X_selected)
            
            # Generate predictions
            predictions = model.predict_proba(X_scaled)[:, 1]
            
            return predictions
        
        except Exception as e:
            logger.error(f"Error predicting for species {species}: {e}")
            return None
    
    def predict_single_point(self, point: pd.Series, species: str) -> Optional[float]:
        """Generate prediction for a single point and species"""
        try:
            if species not in self.models:
                return None
            
            model_info = self.models[species]
            model = model_info['model']
            scaler = model_info['scaler']
            selector = model_info['selector']
            
            # CRITICAL FIX: Align features with what selector was trained on
            if selector and hasattr(selector, 'feature_names_in_'):
                # Use only the features the selector was trained on
                selector_features = selector.feature_names_in_
                available_features = [f for f in selector_features if f in point.index]
                
                if len(available_features) != len(selector_features):
                    logger.warning(f"Feature mismatch for {species}: {len(available_features)}/{len(selector_features)} features available")
                    # Try to use fallback to metadata features
                    metadata_features = model_info['metadata']['features']
                    if all(f in point.index for f in metadata_features):
                        logger.info(f"Using metadata features as fallback for {species}")
                        point_df = pd.DataFrame([point])
                        X = point_df[metadata_features].copy()
                                    # Handle missing values - ROBUST NaN handling
            for col in X.columns:
                if X[col].dtype in ['float64', 'float32', 'int64', 'int32']:
                    # Use 0 if all values are NaN (median would fail)
                    if X[col].isna().all():
                        X[col] = X[col].fillna(0.0)
                    else:
                        X[col] = X[col].fillna(X[col].median())
                else:
                    X[col] = X[col].fillna(0)
            
            # Final safety check - replace any remaining NaN/inf values
            X = X.replace([np.inf, -np.inf], 0)
            X = X.fillna(0)
                        X_scaled = scaler.transform(X.values)
                        prediction = model.predict_proba(X_scaled)[0, 1]
                        return float(prediction)
                    else:
                        logger.error(f"Cannot predict for {species}: missing required features")
                        return None
                
                # Prepare data with aligned features
                point_df = pd.DataFrame([point])
                X = point_df[available_features].copy()
                
                            # Handle missing values - ROBUST NaN handling
            for col in X.columns:
                if X[col].dtype in ['float64', 'float32', 'int64', 'int32']:
                    # Use 0 if all values are NaN (median would fail)
                    if X[col].isna().all():
                        X[col] = X[col].fillna(0.0)
                    else:
                        X[col] = X[col].fillna(X[col].median())
                else:
                    X[col] = X[col].fillna(0)
            
            # Final safety check - replace any remaining NaN/inf values
            X = X.replace([np.inf, -np.inf], 0)
            X = X.fillna(0)
                
                # Apply selector
                X_selected = selector.transform(X)
                
            else:
                # No selector or no feature names - use metadata features directly
                metadata_features = model_info['metadata']['features']
                if not all(f in point.index for f in metadata_features):
                    logger.error(f"Missing metadata features for {species}")
                    return None
                
                point_df = pd.DataFrame([point])
                X = point_df[metadata_features].copy()
                
                            # Handle missing values - ROBUST NaN handling
            for col in X.columns:
                if X[col].dtype in ['float64', 'float32', 'int64', 'int32']:
                    # Use 0 if all values are NaN (median would fail)
                    if X[col].isna().all():
                        X[col] = X[col].fillna(0.0)
                    else:
                        X[col] = X[col].fillna(X[col].median())
                else:
                    X[col] = X[col].fillna(0)
            
            # Final safety check - replace any remaining NaN/inf values
            X = X.replace([np.inf, -np.inf], 0)
            X = X.fillna(0)
                
                X_selected = X.values
            
            # Scale features
            X_scaled = scaler.transform(X_selected)
            
            # Generate prediction
            prediction = model.predict_proba(X_scaled)[0, 1]
            
            return float(prediction)
        
        except Exception as e:
            logger.error(f"Error predicting single point for species {species}: {e}")
            return None
    
    def get_risk_level(self, probability: float) -> str:
        """Convert probability to risk level"""
        if probability >= 0.7:
            return 'very_high'
        elif probability >= 0.5:
            return 'high'
        elif probability >= 0.3:
            return 'medium'
        elif probability >= 0.1:
            return 'low'
        else:
            return 'very_low'
    
    def get_confidence_level(self, probability: float, species: str) -> str:
        """Get confidence level based on model performance and prediction"""
        try:
            if species not in self.species_metadata:
                return 'low'
            
            model_auc = self.species_metadata[species]['metrics']['auc']
            
            # High confidence for excellent models with strong predictions
            if model_auc >= 0.95 and (probability >= 0.7 or probability <= 0.1):
                return 'high'
            elif model_auc >= 0.85 and (probability >= 0.6 or probability <= 0.2):
                return 'medium'
            else:
                return 'low'
        
        except Exception:
            return 'low'
    
    def interpret_prediction(self, probability: float, species: str) -> str:
        """Interpret the prediction probability"""
        risk_level = self.get_risk_level(probability)
        
        interpretations = {
            'very_high': f"Very high risk of {species} bloom. Immediate monitoring recommended.",
            'high': f"High risk of {species} bloom. Enhanced surveillance advised.",
            'medium': f"Moderate risk of {species} bloom. Regular monitoring sufficient.",
            'low': f"Low risk of {species} bloom. Routine monitoring adequate.",
            'very_low': f"Very low risk of {species} bloom. Standard monitoring protocol."
        }
        
        return interpretations.get(risk_level, "Risk assessment available.")
    
    def generate_recommendations(self, analysis: Dict) -> List[str]:
        """Generate actionable recommendations based on analysis"""
        recommendations = []
        highest_risk = analysis['highest_risk']
        
        if highest_risk['probability'] >= 0.7:
            recommendations.extend([
                "🚨 IMMEDIATE ACTION: Deploy additional monitoring buoys",
                "🐟 AQUACULTURE ALERT: Consider early harvesting of vulnerable species",
                "🔬 SAMPLING: Increase water sample collection frequency",
                "📢 NOTIFICATION: Alert local fishing cooperatives"
            ])
        elif highest_risk['probability'] >= 0.5:
            recommendations.extend([
                "⚠️ ENHANCED MONITORING: Increase observation frequency",
                "🐟 AQUACULTURE PREP: Prepare contingency plans",
                "🔬 TESTING: Schedule additional water quality tests"
            ])
        elif highest_risk['probability'] >= 0.3:
            recommendations.extend([
                "👀 REGULAR MONITORING: Maintain standard surveillance",
                "📊 DATA REVIEW: Monitor environmental trends"
            ])
        else:
            recommendations.extend([
                "✅ ROUTINE MONITORING: Standard protocol sufficient",
                "📈 TREND ANALYSIS: Review historical patterns"
            ])
        
        # Add species-specific recommendations
        if highest_risk['species']:
            species = highest_risk['species']
            if 'Karenia' in species:
                recommendations.append("🦠 KARENIA SPECIFIC: Monitor for fish kills and shellfish toxicity")
            elif 'Chattonella' in species:
                recommendations.append("🐟 CHATTONELLA SPECIFIC: High risk to finfish aquaculture")
            elif 'Cochlodinium' in species:
                recommendations.append("🦐 COCHLODINIUM SPECIFIC: Monitor crustacean health")
        
        return recommendations
    
    def calculate_overall_risk(self, point: pd.Series) -> str:
        """Calculate overall HAB risk for a point"""
        try:
            # Simple risk calculation based on environmental conditions
            risk_factors = 0
            
            # Temperature risk
            if 'sst' in point and not pd.isna(point['sst']):
                if 20 <= point['sst'] <= 28:  # Optimal HAB temperature range
                    risk_factors += 1
            
            # Salinity risk
            if 'ocean_salinity' in point and not pd.isna(point['ocean_salinity']):
                if 25 <= point['ocean_salinity'] <= 35:  # Optimal salinity range
                    risk_factors += 1
            
            # Chlorophyll risk
            if 'chlor_a' in point and not pd.isna(point['chlor_a']):
                if point['chlor_a'] > 2.0:  # High productivity
                    risk_factors += 1
            
            if risk_factors >= 2:
                return 'high'
            elif risk_factors == 1:
                return 'medium'
            else:
                return 'low'
        
        except Exception:
            return 'unknown'
    
    def get_performance_grade(self, auc: float) -> str:
        """Get performance grade based on AUC"""
        if auc >= 0.95:
            return 'Excellent'
        elif auc >= 0.85:
            return 'Very Good'
        elif auc >= 0.75:
            return 'Good'
        elif auc >= 0.65:
            return 'Fair'
        else:
            return 'Needs Improvement'
    
    def interpret_performance(self, metrics: Dict) -> str:
        """Interpret model performance metrics"""
        auc = metrics['auc']
        f1 = metrics['f1']
        
        if auc >= 0.95 and f1 < 0.1:
            return "Excellent discrimination capability with low F1 due to extreme event rarity. Perfect for early warning systems."
        elif auc >= 0.85:
            return "Strong predictive performance. Model reliably ranks HAB conditions correctly."
        elif auc >= 0.75:
            return "Good performance with room for improvement. Suitable for operational use with validation."
        else:
            return "Limited predictive capability. Consider additional features or more training data."
    
    def identify_strengths(self, metadata: Dict) -> List[str]:
        """Identify model strengths"""
        strengths = []
        
        metrics = metadata['metrics']
        feature_importance = metadata.get('feature_importance', [])
        
        if metrics['auc'] >= 0.95:
            strengths.append("Excellent discrimination between HAB and non-HAB conditions")
        
        if metrics['recall'] >= 0.7:
            strengths.append("High sensitivity - detects most actual HAB events")
        
        if len(feature_importance) > 0:
            top_feature = feature_importance[0][0]
            strengths.append(f"Clear environmental driver identified: {top_feature}")
        
        if metadata['training_info']['positive_samples'] >= 20:
            strengths.append("Sufficient training data for reliable predictions")
        
        return strengths
    
    def identify_considerations(self, metadata: Dict) -> List[str]:
        """Identify model considerations/limitations"""
        considerations = []
        
        metrics = metadata['metrics']
        
        if metrics['f1'] < 0.01:
            considerations.append("Low F1 score due to extreme event rarity - normal for HAB prediction")
        
        if metrics['precision'] < 0.1:
            considerations.append("High false positive rate - validate alerts with additional monitoring")
        
        if metadata['training_info']['positive_samples'] < 10:
            considerations.append("Limited training data - predictions should be interpreted cautiously")
        
        return considerations
    
    def generate_overall_summary(self) -> Dict:
        """Generate overall system performance summary"""
        try:
            total_models = len(self.species_metadata)
            excellent_models = sum(1 for m in self.species_metadata.values() if m['metrics']['auc'] >= 0.95)
            good_models = sum(1 for m in self.species_metadata.values() if m['metrics']['auc'] >= 0.85)
            
            avg_auc = np.mean([m['metrics']['auc'] for m in self.species_metadata.values()])
            
            return {
                'total_models': total_models,
                'excellent_models': excellent_models,
                'good_models': good_models,
                'average_auc': float(avg_auc),
                'deployment_ready': excellent_models + good_models,
                'system_grade': 'Production Ready' if excellent_models >= total_models * 0.6 else 'Development'
            }
        
        except Exception:
            return {'error': 'Unable to generate summary'}
    
    def run(self, host='127.0.0.1', port=5000, debug=True):
        """Run the dashboard application"""
        logger.info(f"Starting Ultimate HAB Dashboard on {host}:{port}")
        self.app.run(host=host, port=port, debug=debug)

def main():
    """Main function to run the dashboard"""
    dashboard = UltimateHABDashboard()
    dashboard.run()

if __name__ == '__main__':
    main()