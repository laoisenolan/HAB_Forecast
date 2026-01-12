"""
Main Pipeline Orchestrator: HAB Forecasting System

This module orchestrates the complete HAB forecasting pipeline:
1. Data Collection (Phase 2)
2. Data Processing (Phase 3)
3. Feature Engineering (Phase 3)
4. Model Training (Phase 4)
5. Risk Prediction (Phase 4)
6. Results Generation (Phase 5)
"""

import sys
import os
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, Tuple
import pandas as pd
import xarray as xr
import numpy as np

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

# Handle imports for both direct execution and module import
try:
    # Try relative imports first (when run as module)
    from ..feature_engineering.comprehensive_feature_pipeline import ComprehensiveHABFeaturePipeline as FeaturePipeline
    from ..model.species_specific_trainer import SpeciesSpecificXGBoostTrainer as SpeciesSpecificTrainer
    from ..model.ensemble_hab_trainer import EnsembleHABTrainer
    from ..model.predictor import HABPredictor
    from ..utils.config_loader import ConfigLoader
except ImportError:
    # Fall back to absolute imports (when run directly)
    # Add src to path
    current_dir = Path(__file__).parent
    src_dir = current_dir.parent
    project_root = src_dir.parent
    sys.path.insert(0, str(src_dir))
    sys.path.insert(0, str(project_root))
    
    from feature_engineering.comprehensive_feature_pipeline import ComprehensiveHABFeaturePipeline as FeaturePipeline
    from model.species_specific_trainer import SpeciesSpecificXGBoostTrainer as SpeciesSpecificTrainer
    from model.ensemble_hab_trainer import EnsembleHABTrainer
    from model.predictor import HABPredictor
    from utils.config_loader import ConfigLoader

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class HABForecastOrchestrator:
    """
    Main orchestrator for the HAB forecasting system
    """
    
    def __init__(self, config_path: str = "config"):
        """
        Initialize the orchestrator
        
        Args:
            config_path: Path to configuration files
        """
        self.config_path = config_path
        self.config_loader = ConfigLoader(config_path)
        
        # Load configurations
        self.data_sources_config = self.config_loader.get_data_sources_config()
        self.processing_config = self.config_loader.get_processing_config()
        self.features_config = self.config_loader.get_features_config()
        self.model_config = self.config_loader.get_model_params_config()
        
        # Initialize components
        self.data_orchestrator = None
        self.data_loader = None
        self.data_integrator = None
        self.spatial_processor = None
        self.temporal_processor = None
        self.quality_controller = None
        self.feature_pipeline = None
        self.model_trainer = None
        self.shap_analyzer = None
        
        # Pipeline state
        self.pipeline_state = {
            "data_collected": False,
            "data_processed": False,
            "features_engineered": False,
            "model_trained": False,
            "predictions_generated": False
        }
        
        logger.info("HAB Forecast Orchestrator initialized")
    
    def run_complete_pipeline(self, 
                            start_date: str = None,
                            end_date: str = None,
                            force_retrain: bool = False,
                            generate_predictions: bool = True) -> Dict[str, Any]:
        """
        Run the complete HAB forecasting pipeline
        
        Args:
            start_date: Start date for data collection (YYYY-MM-DD)
            end_date: End date for data collection (YYYY-MM-DD)
            force_retrain: Force model retraining
            generate_predictions: Generate new predictions
            
        Returns:
            Pipeline results dictionary
        """
        logger.info("Starting complete HAB forecasting pipeline")
        
        try:
            results = {}
            
            # Step 1: Data Collection
            logger.info("=" * 60)
            logger.info("STEP 1: DATA COLLECTION")
            logger.info("=" * 60)
            data_results = self._collect_data(start_date, end_date)
            results["data_collection"] = data_results
            
            # Step 2: Data Processing
            logger.info("=" * 60)
            logger.info("STEP 2: DATA PROCESSING")
            logger.info("=" * 60)
            processing_results = self._process_data()
            results["data_processing"] = processing_results
            
            # Step 3: Feature Engineering
            logger.info("=" * 60)
            logger.info("STEP 3: FEATURE ENGINEERING")
            logger.info("=" * 60)
            feature_results = self._engineer_features()
            results["feature_engineering"] = feature_results
            
            # Step 4: Model Training
            logger.info("=" * 60)
            logger.info("STEP 4: MODEL TRAINING")
            logger.info("=" * 60)
            model_results = self._train_model(force_retrain)
            results["model_training"] = model_results
            
            # Step 5: Risk Prediction
            if generate_predictions:
                logger.info("=" * 60)
                logger.info("STEP 5: RISK PREDICTION")
                logger.info("=" * 60)
                prediction_results = self._generate_predictions()
                results["risk_prediction"] = prediction_results
            
            # Step 6: Generate Reports
            logger.info("=" * 60)
            logger.info("STEP 6: GENERATE REPORTS")
            logger.info("=" * 60)
            report_results = self._generate_reports(results)
            results["reports"] = report_results
            
            logger.info("=" * 60)
            logger.info("PIPELINE COMPLETED SUCCESSFULLY")
            logger.info("=" * 60)
            
            return results
            
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            raise
    
    def _collect_data(self, start_date: str = None, end_date: str = None) -> Dict[str, Any]:
        """Collect environmental data"""
        logger.info("Collecting environmental data")
        
        try:
            # Initialize data orchestrator
            self.data_orchestrator = DataCollectionOrchestrator()
            
            # Set date range
            if start_date is None:
                start_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
            if end_date is None:
                end_date = datetime.now().strftime("%Y-%m-%d")
            
            # Collect data
            collection_results = self.data_orchestrator.collect_all_data(
                start_date=start_date,
                end_date=end_date
            )
            
            self.pipeline_state["data_collected"] = True
            logger.info("Data collection completed successfully")
            
            return collection_results
            
        except Exception as e:
            logger.error(f"Data collection failed: {e}")
            raise
    
    def _process_data(self) -> Dict[str, Any]:
        """Process collected data"""
        logger.info("Processing collected data")
        
        try:
            # Initialize data processing components
            self.data_loader = DataLoader()
            self.data_integrator = DataIntegrator()
            self.spatial_processor = SpatialProcessor()
            self.temporal_processor = TemporalProcessor()
            self.quality_controller = QualityController()
            
            # Load raw data
            raw_data = self.data_loader.load_all_raw_data()
            
            # Integrate data
            integrated_data = self.data_integrator.integrate_data(raw_data)
            
            # Spatial processing
            spatial_data = self.spatial_processor.process_spatial_data(integrated_data)
            
            # Temporal processing
            temporal_data = self.temporal_processor.process_temporal_data(spatial_data)
            
            # Quality control
            processed_data = self.quality_controller.apply_quality_control(temporal_data)
            
            # Save processed data
            output_path = Path("data/processed")
            output_path.mkdir(parents=True, exist_ok=True)
            processed_data.to_netcdf(output_path / "processed_data.nc")
            
            self.pipeline_state["data_processed"] = True
            logger.info("Data processing completed successfully")
            
            return {
                "raw_data_shape": raw_data.dims,
                "processed_data_shape": processed_data.dims,
                "output_path": str(output_path / "processed_data.nc")
            }
            
        except Exception as e:
            logger.error(f"Data processing failed: {e}")
            raise
    
    def _engineer_features(self) -> Dict[str, Any]:
        """Engineer features for model training"""
        logger.info("Engineering features")
        
        try:
            # Initialize feature pipeline
            self.feature_pipeline = FeaturePipeline()
            
            # Load processed data
            processed_data_path = Path("data/processed/processed_data.nc")
            if not processed_data_path.exists():
                raise FileNotFoundError("Processed data not found. Run data processing first.")
            
            processed_data = xr.open_dataset(processed_data_path)
            
            # Run feature engineering pipeline
            feature_results = self.feature_pipeline.run_complete_pipeline(processed_data)
            
            self.pipeline_state["features_engineered"] = True
            logger.info("Feature engineering completed successfully")
            
            return feature_results
            
        except Exception as e:
            logger.error(f"Feature engineering failed: {e}")
            raise
    
    def _train_model(self, force_retrain: bool = False) -> Dict[str, Any]:
        """Train the risk prediction model"""
        logger.info("Training risk prediction model")
        
        try:
            # Initialize model trainer
            self.model_trainer = XGBoostTrainer()
            
            # Check if model exists and force_retrain is False
            model_path = Path("outputs/models/risk_model.json")
            if model_path.exists() and not force_retrain:
                logger.info("Loading existing model")
                model = self.model_trainer.load_model()
                self.pipeline_state["model_trained"] = True
                return {"model_loaded": True, "model_path": str(model_path)}
            
            # Load feature data
            feature_data_path = Path("outputs/features/feature_dataset.nc")
            if not feature_data_path.exists():
                raise FileNotFoundError("Feature data not found. Run feature engineering first.")
            
            feature_data = xr.open_dataset(feature_data_path)
            
            # Prepare dataset
            features_df, target_series = self.model_trainer.prepare_dataset(
                feature_data, target_variable="hab_risk_percentage"
            )
            
            # Create splits
            X_train, X_val, X_test, y_train, y_val, y_test = self.model_trainer.create_temporal_split(
                features_df, target_series
            )
            
            # Scale features
            X_train_scaled, X_val_scaled, X_test_scaled = self.model_trainer.scale_features(
                X_train, X_val, X_test
            )
            
            # Select features
            X_train_selected, X_val_selected, X_test_selected = self.model_trainer.select_features(
                X_train_scaled, y_train, X_val_scaled, X_test_scaled
            )
            
            # Train model
            model = self.model_trainer.train_baseline_model(
                X_train_selected, y_train, X_val_selected, y_val
            )
            
            # Evaluate model
            evaluation_metrics = self.model_trainer.evaluate_model(X_test_selected, y_test)
            
            # Save model
            output_path = Path("outputs/models")
            output_path.mkdir(parents=True, exist_ok=True)
            self.model_trainer.save_model(output_path / "risk_model.json")
            
            self.pipeline_state["model_trained"] = True
            logger.info("Model training completed successfully")
            
            return {
                "model_trained": True,
                "evaluation_metrics": evaluation_metrics,
                "model_path": str(output_path / "risk_model.json")
            }
            
        except Exception as e:
            logger.error(f"Model training failed: {e}")
            raise
    
    def _generate_predictions(self) -> Dict[str, Any]:
        """Generate risk predictions"""
        logger.info("Generating risk predictions")
        
        try:
            # Load model
            if not self.pipeline_state["model_trained"]:
                raise RuntimeError("Model not trained. Run model training first.")
            
            model = self.model_trainer.load_model()
            
            # Load latest data for predictions
            processed_data_path = Path("data/processed/processed_data.nc")
            processed_data = xr.open_dataset(processed_data_path)
            
            # Prepare data for prediction
            features_df, _ = self.model_trainer.prepare_dataset(
                processed_data, target_variable="hab_risk_percentage"
            )
            
            # Scale and select features
            X_scaled = self.model_trainer.scaler.transform(features_df)
            X_selected = self.model_trainer.feature_selector.transform(X_scaled)
            
            # Generate predictions
            predictions = model.predict(X_selected)
            predictions = np.clip(predictions, 0, 100)
            
            # Create prediction dataset
            prediction_data = processed_data.copy()
            prediction_data["predicted_risk_percentage"] = (
                processed_data.dims, predictions.reshape(processed_data.dims)
            )
            
            # Save predictions
            output_path = Path("outputs/predictions")
            output_path.mkdir(parents=True, exist_ok=True)
            prediction_data.to_netcdf(output_path / f"risk_predictions_{datetime.now().strftime('%Y%m%d')}.nc")
            
            self.pipeline_state["predictions_generated"] = True
            logger.info("Risk predictions generated successfully")
            
            return {
                "predictions_generated": True,
                "prediction_range": f"{predictions.min():.1f}% - {predictions.max():.1f}%",
                "mean_risk": f"{predictions.mean():.1f}%",
                "output_path": str(output_path / f"risk_predictions_{datetime.now().strftime('%Y%m%d')}.nc")
            }
            
        except Exception as e:
            logger.error(f"Prediction generation failed: {e}")
            raise
    
    def _generate_reports(self, pipeline_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive reports"""
        logger.info("Generating comprehensive reports")
        
        try:
            # Initialize SHAP analyzer
            self.shap_analyzer = SHAPAnalyzer()
            
            # Generate model interpretability report
            if self.pipeline_state["model_trained"]:
                model = self.model_trainer.load_model()
                feature_data_path = Path("outputs/features/feature_dataset.nc")
                feature_data = xr.open_dataset(feature_data_path)
                
                features_df, target_series = self.model_trainer.prepare_dataset(
                    feature_data, target_variable="hab_risk_percentage"
                )
                
                # Create explainer and compute SHAP values
                explainer = self.shap_analyzer.create_explainer(model, features_df)
                shap_values = self.shap_analyzer.compute_shap_values(features_df)
                
                # Generate interpretability report
                interpretability_report = self.shap_analyzer.create_interpretability_report(
                    features_df, target_series
                )
            else:
                interpretability_report = None
            
            # Create pipeline summary report
            pipeline_summary = {
                "pipeline_completion_time": datetime.now().isoformat(),
                "pipeline_state": self.pipeline_state,
                "data_collection_summary": pipeline_results.get("data_collection", {}),
                "data_processing_summary": pipeline_results.get("data_processing", {}),
                "feature_engineering_summary": pipeline_results.get("feature_engineering", {}),
                "model_training_summary": pipeline_results.get("model_training", {}),
                "prediction_summary": pipeline_results.get("risk_prediction", {}),
                "interpretability_report": interpretability_report
            }
            
            # Save pipeline report
            output_path = Path("outputs/reports")
            output_path.mkdir(parents=True, exist_ok=True)
            
            report_file = output_path / f"pipeline_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            
            import json
            with open(report_file, 'w') as f:
                json.dump(pipeline_summary, f, indent=2, default=str)
            
            logger.info("Reports generated successfully")
            
            return {
                "reports_generated": True,
                "pipeline_summary": pipeline_summary,
                "report_path": str(report_file)
            }
            
        except Exception as e:
            logger.error(f"Report generation failed: {e}")
            raise
    
    def get_pipeline_status(self) -> Dict[str, Any]:
        """Get current pipeline status"""
        return {
            "pipeline_state": self.pipeline_state,
            "configurations_loaded": {
                "data_sources": self.data_sources_config is not None,
                "processing": self.processing_config is not None,
                "features": self.features_config is not None,
                "model": self.model_config is not None
            }
        } 