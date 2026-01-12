"""
Species-Specific HAB Predictor

This module uses real HAB event data to train models that can predict:
1. Which species is most likely to bloom
2. Risk percentage for each species
3. Environmental conditions that trigger specific blooms
"""

import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from pathlib import Path
import yaml
from typing import Dict, List, Tuple, Optional
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

logger = logging.getLogger(__name__)

class SpeciesPredictor:
    """
    Predicts HAB species based on environmental conditions.
    
    This class uses real HAB event data to train models that can predict:
    - Which species is most likely to bloom
    - Risk percentage for each species
    - Environmental triggers for specific blooms
    """
    
    def __init__(self, config_path: str = "config/features.yaml"):
        """Initialize the species predictor."""
        self.config_path = config_path
        self.config = self._load_config()
        self.hab_events = None
        self.species_models = {}
        self.species_encoders = {}
        self.environmental_data = None
        
    def _load_config(self) -> Dict:
        """Load configuration from YAML file."""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except UnicodeDecodeError as e:
            # Try with different encodings if UTF-8 fails
            for encoding in ['latin-1', 'cp1252', 'iso-8859-1']:
                try:
                    with open(self.config_path, 'r', encoding=encoding) as f:
                        return yaml.safe_load(f)
                except (UnicodeDecodeError, yaml.YAMLError):
                    continue
            logger.warning(f"Could not load config from {self.config_path}: {e}")
            return {}
        except Exception as e:
            logger.warning(f"Could not load config from {self.config_path}: {e}")
            return {}
    
    def load_hab_events(self, csv_path: str) -> pd.DataFrame:
        """
        Load and process HAB event data.
        
        Args:
            csv_path: Path to the CSV file containing HAB event data
            
        Returns:
            DataFrame with processed HAB event data
        """
        try:
            # Load the CSV file
            df = pd.read_csv(csv_path, sep=';', encoding='utf-8')
            logger.info(f"Loaded {len(df)} HAB events from {csv_path}")
            
            # Clean and process the data
            df = self._clean_hab_data(df)
            self.hab_events = df
            
            return df
            
        except Exception as e:
            logger.error(f"Failed to load HAB events from {csv_path}: {e}")
            raise
    
    def _clean_hab_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and standardize HAB event data.
        
        Args:
            df: Raw HAB event DataFrame
            
        Returns:
            Cleaned DataFrame
        """
        df_clean = df.copy()
        
        # Parse dates
        df_clean['date_occurrence_parsed'] = pd.to_datetime(
            df_clean['date_occurrence_parsed'], errors='coerce'
        )
        df_clean['date_end_parsed'] = pd.to_datetime(
            df_clean['date_end_parsed'], errors='coerce'
        )
        
        # Clean cell count data
        df_clean['highest_cell_count_cells_per_ml'] = pd.to_numeric(
            df_clean['highest_cell_count_cells_per_ml'].str.replace(',', '').str.replace('cells/ml', '').str.replace('cells/day', ''),
            errors='coerce'
        )
        
        # Extract species information
        df_clean['primary_species'] = df_clean['plankton_component'].str.split(';').str[0]
        df_clean['species_count'] = df_clean['plankton_component'].str.count(';') + 1
        
        # Calculate event duration
        df_clean['duration_days'] = (
            df_clean['date_end_parsed'] - df_clean['date_occurrence_parsed']
        ).dt.days
        
        # Create severity categories based on cell count
        df_clean['severity_category'] = self._categorize_severity(
            df_clean['highest_cell_count_cells_per_ml']
        )
        
        # Extract geographic information
        df_clean['region'] = df_clean['nada_name'].fillna('Unknown')
        df_clean['prefecture'] = df_clean['prefecture_name'].fillna('Unknown')
        
        logger.info(f"Cleaned {len(df_clean)} HAB events")
        return df_clean
    
    def _categorize_severity(self, cell_counts: pd.Series) -> pd.Series:
        """
        Categorize HAB events by severity based on cell counts.
        
        Args:
            cell_counts: Series of cell count values
            
        Returns:
            Series of severity categories
        """
        def categorize(count):
            if pd.isna(count):
                return 'Unknown'
            elif count < 1000:
                return 'Low'
            elif count < 10000:
                return 'Moderate'
            elif count < 100000:
                return 'High'
            else:
                return 'Extreme'
        
        return cell_counts.apply(categorize)
    
    def create_species_training_data(self, 
                                   start_date: str = "2023-01-01",
                                   end_date: str = "2023-12-31") -> pd.DataFrame:
        """
        Create training data for species prediction.
        
        Args:
            start_date: Start date for training data
            end_date: End date for training data
            
        Returns:
            DataFrame with daily species labels
        """
        if self.hab_events is None:
            raise ValueError("No HAB events loaded. Call load_hab_events() first.")
        
        # Filter events by date range
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)
        
        mask = (
            (self.hab_events['date_occurrence_parsed'] >= start_dt) &
            (self.hab_events['date_occurrence_parsed'] <= end_dt)
        )
        
        filtered_events = self.hab_events[mask].copy()
        logger.info(f"Filtered to {len(filtered_events)} events in date range")
        
        # Create daily labels
        date_range = pd.date_range(start=start_dt, end=end_dt, freq='D')
        labels = []
        
        for date in date_range:
            # Find events active on this date
            active_events = filtered_events[
                (filtered_events['date_occurrence_parsed'] <= date) &
                (filtered_events['date_end_parsed'] >= date)
            ]
            
            if len(active_events) > 0:
                # Multiple events possible, create multi-species label
                species_list = active_events['primary_species'].unique().tolist()
                
                # Handle NaN values in cell counts
                valid_cell_counts = active_events['highest_cell_count_cells_per_ml'].dropna()
                
                if len(valid_cell_counts) > 0:
                    max_cell_count = valid_cell_counts.max()
                    max_cell_count_idx = valid_cell_counts.idxmax()
                    dominant_species = active_events.loc[max_cell_count_idx, 'primary_species']
                else:
                    # No valid cell counts, use default values
                    max_cell_count = 1000  # Default moderate level
                    dominant_species = active_events.iloc[0]['primary_species']
                
                label = {
                    'date': date,
                    'hab_active': True,
                    'species_present': species_list,
                    'dominant_species': dominant_species,
                    'max_cell_count': max_cell_count,
                    'event_count': len(active_events)
                }
            else:
                # No HAB events on this date
                label = {
                    'date': date,
                    'hab_active': False,
                    'species_present': [],
                    'dominant_species': 'None',
                    'max_cell_count': 0,
                    'event_count': 0
                }
            
            labels.append(label)
        
        labels_df = pd.DataFrame(labels)
        logger.info(f"Created {len(labels_df)} daily species labels")
        
        return labels_df
    
    def generate_synthetic_environmental_data(self, labels_df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate synthetic environmental data for training.
        In production, this would be replaced with real environmental data.
        
        Args:
            labels_df: DataFrame with species labels
            
        Returns:
            DataFrame with environmental features
        """
        logger.info("Generating synthetic environmental data for training...")
        
        # Create environmental features based on species patterns
        environmental_data = []
        
        for _, row in labels_df.iterrows():
            date = row['date']
            species = row['dominant_species']
            hab_active = row['hab_active']
            
            # Generate environmental conditions based on species
            if hab_active and species != 'None':
                # Species-specific environmental conditions
                if 'Karenia' in species:
                    # Karenia prefers warm, calm, nutrient-rich conditions
                    sst = np.random.normal(25, 2)  # Warm water
                    chlorophyll = np.random.normal(3, 1.5)  # High chlorophyll
                    nitrate = np.random.normal(8, 2)  # High nutrients
                    wind_speed = np.random.normal(2, 1)  # Calm conditions
                    precipitation = np.random.exponential(4)  # More rain
                    
                elif 'Heterosigma' in species:
                    # Heterosigma prefers moderate conditions
                    sst = np.random.normal(22, 3)
                    chlorophyll = np.random.normal(2.5, 1)
                    nitrate = np.random.normal(6, 2)
                    wind_speed = np.random.normal(4, 2)
                    precipitation = np.random.exponential(3)
                    
                elif 'Noctiluca' in species:
                    # Noctiluca prefers moderate conditions
                    sst = np.random.normal(20, 4)
                    chlorophyll = np.random.normal(2, 1)
                    nitrate = np.random.normal(5, 1.5)
                    wind_speed = np.random.normal(5, 2)
                    precipitation = np.random.exponential(2)
                    
                elif 'Chattonella' in species:
                    # Chattonella prefers specific conditions
                    sst = np.random.normal(23, 2)
                    chlorophyll = np.random.normal(2.8, 1.2)
                    nitrate = np.random.normal(7, 1.8)
                    wind_speed = np.random.normal(3, 1.5)
                    precipitation = np.random.exponential(3.5)
                    
                else:
                    # Default conditions for other species
                    sst = np.random.normal(21, 3)
                    chlorophyll = np.random.normal(2.2, 1)
                    nitrate = np.random.normal(5.5, 1.5)
                    wind_speed = np.random.normal(4.5, 2)
                    precipitation = np.random.exponential(2.5)
                    
            else:
                # No HAB conditions
                sst = np.random.normal(18, 4)
                chlorophyll = np.random.normal(1.5, 0.8)
                nitrate = np.random.normal(4, 1.5)
                wind_speed = np.random.normal(6, 3)
                precipitation = np.random.exponential(1.5)
            
            # Add seasonal and temporal features
            month = date.month
            day_of_year = date.dayofyear
            
            # Seasonal adjustments
            seasonal_factor = np.sin(2 * np.pi * day_of_year / 365.25)
            
            environmental_data.append({
                'date': date,
                'sst': sst + seasonal_factor * 2,  # Seasonal variation
                'chlorophyll': chlorophyll,
                'nitrate': nitrate,
                'wind_speed': wind_speed,
                'precipitation': precipitation,
                'month': month,
                'day_of_year': day_of_year,
                'seasonal_factor': seasonal_factor
            })
        
        env_df = pd.DataFrame(environmental_data)
        logger.info(f"Generated environmental data for {len(env_df)} days")
        
        return env_df
    
    def train_species_prediction_models(self, 
                                      environmental_data: pd.DataFrame,
                                      species_labels: pd.DataFrame) -> Dict:
        """
        Train models to predict species blooms.
        
        Args:
            environmental_data: DataFrame with environmental features
            species_labels: DataFrame with species labels
            
        Returns:
            Dictionary containing trained models
        """
        logger.info("Training species prediction models...")
        
        # Merge environmental data with species labels
        training_data = environmental_data.merge(species_labels, on='date', how='inner')
        
        # Prepare features
        feature_cols = ['sst', 'chlorophyll', 'nitrate', 'wind_speed', 'precipitation', 
                       'month', 'day_of_year', 'seasonal_factor']
        X = training_data[feature_cols]
        
        # Get unique species
        all_species = []
        for species_list in training_data['species_present']:
            if isinstance(species_list, list):
                all_species.extend(species_list)
            else:
                all_species.append(species_list)
        
        # Count species occurrences
        species_counts = pd.Series(all_species).value_counts()
        
        # Filter to species that appear at least 3 times (for reliable training)
        common_species = species_counts[species_counts >= 3].index.tolist()
        logger.info(f"Found {len(common_species)} species with sufficient data: {common_species}")
        
        # Train models for each common species
        models = {}
        encoders = {}
        
        for species in common_species:
            logger.info(f"Training model for {species}...")
            
            # Create binary labels for this species
            y_species = training_data['species_present'].apply(
                lambda x: 1 if species in x else 0
            )
            
            # Check if we have enough samples for both classes
            class_counts = y_species.value_counts()
            if len(class_counts) < 2 or class_counts.min() < 2:
                logger.warning(f"Skipping {species} - insufficient data for both classes")
                continue
            
            # Split data (without stratification for rare species)
            try:
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y_species, test_size=0.2, random_state=42, stratify=y_species
                )
            except ValueError:
                # If stratification fails, use regular split
                logger.warning(f"Using regular split for {species} due to class imbalance")
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y_species, test_size=0.2, random_state=42
                )
            
            # Train XGBoost model
            model = xgb.XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42
            )
            
            model.fit(X_train, y_train)
            
            # Evaluate model
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            
            logger.info(f"  {species} model accuracy: {accuracy:.3f}")
            
            # Store model and encoder
            models[species] = model
            encoders[species] = None  # No encoding needed for binary classification
        
        self.species_models = models
        self.species_encoders = encoders
        
        logger.info(f"Trained {len(models)} species prediction models")
        
        return {
            'models': models,
            'encoders': encoders,
            'feature_names': feature_cols,
            'species_list': common_species
        }
    
    def predict_species_probabilities(self, environmental_features: pd.DataFrame) -> Dict:
        """
        Predict probabilities for each species based on environmental conditions.
        
        Args:
            environmental_features: DataFrame with environmental features
            
        Returns:
            Dictionary with species probabilities and predictions
        """
        if not self.species_models:
            raise ValueError("No models trained. Call train_species_prediction_models() first.")
        
        # Prepare features
        feature_cols = ['sst', 'chlorophyll', 'nitrate', 'wind_speed', 'precipitation', 
                       'month', 'day_of_year', 'seasonal_factor']
        
        # Ensure all required features are present
        missing_features = set(feature_cols) - set(environmental_features.columns)
        if missing_features:
            raise ValueError(f"Missing features: {missing_features}")
        
        X = environmental_features[feature_cols]
        
        # Get predictions from all models
        predictions = {}
        probabilities = {}
        
        for species, model in self.species_models.items():
            # Get probability of this species blooming
            prob = model.predict_proba(X)[:, 1]  # Probability of positive class
            pred = model.predict(X)
            
            predictions[species] = pred
            probabilities[species] = prob
        
        # Find the most likely species
        if probabilities:
            # Get the species with highest probability
            max_prob_species = max(probabilities.keys(), 
                                 key=lambda s: np.mean(probabilities[s]))
            max_prob = np.mean(probabilities[max_prob_species])
            
            # Calculate overall HAB risk
            overall_risk = np.mean([np.mean(probs) for probs in probabilities.values()])
            
            return {
                'species_probabilities': probabilities,
                'species_predictions': predictions,
                'most_likely_species': max_prob_species,
                'max_probability': max_prob,
                'overall_hab_risk': overall_risk,
                'feature_importance': self._get_feature_importance()
            }
        
        return {}
    
    def _get_feature_importance(self) -> Dict:
        """
        Get feature importance for all species models.
        
        Returns:
            Dictionary with feature importance for each species
        """
        importance_dict = {}
        
        for species, model in self.species_models.items():
            feature_names = ['sst', 'chlorophyll', 'nitrate', 'wind_speed', 'precipitation', 
                           'month', 'day_of_year', 'seasonal_factor']
            
            importance = model.feature_importances_
            importance_dict[species] = dict(zip(feature_names, importance))
        
        return importance_dict
    
    def generate_prediction_report(self, predictions: Dict) -> str:
        """
        Generate a human-readable prediction report.
        
        Args:
            predictions: Dictionary with prediction results
            
        Returns:
            String containing the prediction report
        """
        if not predictions:
            return "No predictions available."
        
        report = []
        report.append("=" * 60)
        report.append("HAB SPECIES PREDICTION REPORT")
        report.append("=" * 60)
        
        # Overall risk
        overall_risk = predictions.get('overall_hab_risk', 0)
        report.append(f"Overall HAB Risk: {overall_risk:.1%}")
        
        # Most likely species
        most_likely = predictions.get('most_likely_species', 'None')
        max_prob = predictions.get('max_probability', 0)
        report.append(f"Most Likely Species: {most_likely} ({max_prob:.1%} probability)")
        
        # All species probabilities
        species_probs = predictions.get('species_probabilities', {})
        if species_probs:
            report.append("\nSpecies Probabilities:")
            for species, probs in species_probs.items():
                avg_prob = np.mean(probs)
                report.append(f"  {species}: {avg_prob:.1%}")
        
        # Feature importance
        importance = predictions.get('feature_importance', {})
        if importance and most_likely in importance:
            report.append(f"\nKey Environmental Factors for {most_likely}:")
            factors = importance[most_likely]
            sorted_factors = sorted(factors.items(), key=lambda x: x[1], reverse=True)
            for factor, importance_val in sorted_factors[:5]:
                report.append(f"  {factor}: {importance_val:.3f}")
        
        report.append("=" * 60)
        
        return "\n".join(report)
    
    def save_models(self, output_dir: str = "models"):
        """
        Save trained models to disk.
        
        Args:
            output_dir: Directory to save models
        """
        import pickle
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        for species, model in self.species_models.items():
            # Clean species name for filename
            clean_name = species.replace(' ', '_').replace('/', '_')
            model_path = output_path / f"{clean_name}_model.pkl"
            
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            
            logger.info(f"Saved {species} model to {model_path}")
        
        # Save metadata
        metadata = {
            'species_list': list(self.species_models.keys()),
            'feature_names': ['sst', 'chlorophyll', 'nitrate', 'wind_speed', 'precipitation', 
                             'month', 'day_of_year', 'seasonal_factor']
        }
        
        metadata_path = output_path / "species_models_metadata.json"
        with open(metadata_path, 'w') as f:
            import json
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Saved metadata to {metadata_path}")
    
    def load_models(self, models_dir: str = "models"):
        """
        Load trained models from disk.
        
        Args:
            models_dir: Directory containing saved models
        """
        import pickle
        
        models_path = Path(models_dir)
        
        if not models_path.exists():
            raise ValueError(f"Models directory {models_dir} does not exist")
        
        # Load metadata
        metadata_path = models_path / "species_models_metadata.json"
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                import json
                metadata = json.load(f)
                species_list = metadata.get('species_list', [])
        else:
            # Try to load models directly
            species_list = []
            for model_file in models_path.glob("*_model.pkl"):
                species_name = model_file.stem.replace('_model', '').replace('_', ' ')
                species_list.append(species_name)
        
        # Load models
        for species in species_list:
            clean_name = species.replace(' ', '_').replace('/', '_')
            model_path = models_path / f"{clean_name}_model.pkl"
            
            if model_path.exists():
                with open(model_path, 'rb') as f:
                    model = pickle.load(f)
                
                self.species_models[species] = model
                logger.info(f"Loaded {species} model from {model_path}")
        
        logger.info(f"Loaded {len(self.species_models)} species models") 