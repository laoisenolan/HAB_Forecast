#!/usr/bin/env python3
"""
24-HOUR HAB PREDICTION MODEL
Uses existing environmental data + matched HAB events
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pickle
import json
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

class Rapid24hHABTrainer:
    """24-hour emergency HAB prediction system"""
    
    def __init__(self):
        self.our_coverage = {
            'lat_min': 33.575, 'lat_max': 34.425,
            'lon_min': 132.575, 'lon_max': 133.425
        }
        
        # Target bays in our coverage
        self.target_bays = [
            'Hiuchi-nada', 'Aki-nada', 'Iyo-nada', 
            'Hiroshima', 'Hiuchi', 'Iyo', 'Aki'
        ]
        
        self.core_features = [
            'sst', 'ocean_temperature_hycom', 'sea_surface_elevation',
            'ocean_current_u', 'ocean_current_v', 'wind_u', 'wind_v',
            'chlor_a', 'turbidity_index', 'ocean_salinity', 'precipitation'
        ]
    
    def load_and_match_data(self):
        """Load environmental data and match with HAB events"""
        print("🔥 LOADING DATA...")
        
        # Load environmental data
        env_df = pd.read_csv('cache/batch_processing/batches/2017-01/spatial_temporal_training_dataset.csv')
        print(f"Environmental data: {len(env_df):,} samples")
        
        # Load HAB events
        hab_df = pd.read_csv('test_data/HAB_2017_2023.csv')
        print(f"HAB events: {len(hab_df)} total")
        
        # Filter HAB events for our coverage area
        our_hab_events = []
        for pattern in self.target_bays:
            pattern_events = hab_df[hab_df['location'].str.contains(pattern, na=False, case=False)]
            our_hab_events.extend(pattern_events.index.tolist())
        
        our_hab_events = list(set(our_hab_events))
        our_habs = hab_df.loc[our_hab_events].copy()
        print(f"HAB events in our area: {len(our_habs)}")
        
        return env_df, our_habs
    
    def create_training_labels(self, env_df, hab_events):
        """Create HAB labels for environmental data"""
        print("🎯 CREATING TRAINING LABELS...")
        
        # Start with all negative labels
        env_df['has_hab'] = 0
        env_df['hab_species'] = 'none'
        env_df['date'] = pd.to_datetime(env_df['date'])
        
        # Process each HAB event
        hab_events['start_date'] = pd.to_datetime(hab_events['start_date'])
        hab_events['end_date'] = pd.to_datetime(hab_events['end_date'])
        hab_events['duration_days'] = pd.to_numeric(hab_events['duration_days'], errors='coerce')
        
        positive_samples = 0
        
        for idx, hab in hab_events.iterrows():
            if pd.isna(hab['duration_days']):
                continue
                
            # Create date range for this HAB event
            hab_dates = pd.date_range(
                start=hab['start_date'],
                end=hab['end_date'],
                freq='D'
            )
            
            # Label environmental samples during HAB period
            for hab_date in hab_dates:
                mask = (env_df['date'] == hab_date)
                if mask.sum() > 0:
                    env_df.loc[mask, 'has_hab'] = 1
                    env_df.loc[mask, 'hab_species'] = hab['plankton_species']
                    positive_samples += mask.sum()
        
        print(f"Positive samples created: {positive_samples:,}")
        print(f"Positive rate: {positive_samples/len(env_df)*100:.2f}%")
        
        return env_df
    
    def prepare_features(self, df):
        """Prepare features for training"""
        print("⚡ PREPARING FEATURES...")
        
        # Select available features
        available_features = [f for f in self.core_features if f in df.columns]
        print(f"Available features: {len(available_features)}/{len(self.core_features)}")
        
        X = df[available_features].copy()
        y = df['has_hab'].copy()
        
        # Handle missing values
        X = X.fillna(X.median())
        
        # Add simple derived features
        if 'sst' in X.columns:
            X['sst_squared'] = X['sst'] ** 2
        if 'chlor_a' in X.columns:
            X['log_chlor_a'] = np.log1p(X['chlor_a'])
        if 'wind_u' in X.columns and 'wind_v' in X.columns:
            X['wind_speed'] = np.sqrt(X['wind_u']**2 + X['wind_v']**2)
        
        print(f"Final features: {X.shape[1]}")
        print(f"Training samples: {len(X):,}")
        print(f"Positive rate: {y.mean()*100:.2f}%")
        
        return X, y, available_features
    
    def train_models(self, X, y, df):
        """Train multiple models quickly"""
        print("🚀 TRAINING MODELS...")
        
        # Quick temporal split for validation
        df_with_features = df.copy()
        df_with_features = df_with_features.reset_index(drop=True)
        X_indexed = X.reset_index(drop=True)
        y_indexed = y.reset_index(drop=True)
        
        # Split by time (first 80% for train, last 20% for test)
        split_idx = int(len(df_with_features) * 0.8)
        
        X_train = X_indexed.iloc[:split_idx]
        X_test = X_indexed.iloc[split_idx:]
        y_train = y_indexed.iloc[:split_idx]
        y_test = y_indexed.iloc[split_idx:]
        
        print(f"Train: {len(X_train):,} samples, {y_train.sum()} positive")
        print(f"Test:  {len(X_test):,} samples, {y_test.sum()} positive")
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        models = {}
        
        # Random Forest (fast training)
        print("\nTraining Random Forest...")
        rf = RandomForestClassifier(
            n_estimators=100, 
            max_depth=10,
            random_state=42,
            n_jobs=-1,
            class_weight='balanced'
        )
        rf.fit(X_train_scaled, y_train)
        
        rf_pred = rf.predict(X_test_scaled)
        rf_prob = rf.predict_proba(X_test_scaled)[:, 1]
        rf_auc = roc_auc_score(y_test, rf_prob) if y_test.sum() > 0 else 0.5
        
        print(f"Random Forest AUC: {rf_auc:.3f}")
        print(classification_report(y_test, rf_pred, zero_division=0))
        
        models['random_forest'] = {
            'model': rf,
            'scaler': scaler,
            'auc': rf_auc,
            'features': X.columns.tolist()
        }
        
        # XGBoost (if time permits)
        if y_train.sum() > 10:  # Only if enough positive samples
            print("\nTraining XGBoost...")
            xgb_model = xgb.XGBClassifier(
                n_estimators=50,
                max_depth=6,
                learning_rate=0.1,
                random_state=42,
                eval_metric='logloss',
                scale_pos_weight=len(y_train) / y_train.sum() if y_train.sum() > 0 else 1
            )
            xgb_model.fit(X_train_scaled, y_train)
            
            xgb_pred = xgb_model.predict(X_test_scaled)
            xgb_prob = xgb_model.predict_proba(X_test_scaled)[:, 1]
            xgb_auc = roc_auc_score(y_test, xgb_prob) if y_test.sum() > 0 else 0.5
            
            print(f"XGBoost AUC: {xgb_auc:.3f}")
            print(classification_report(y_test, xgb_pred, zero_division=0))
            
            models['xgboost'] = {
                'model': xgb_model,
                'scaler': scaler,
                'auc': xgb_auc,
                'features': X.columns.tolist()
            }
        
        return models
    
    def save_results(self, models, hab_events):
        """Save models and results"""
        print("💾 SAVING RESULTS...")
        
        # Save best model
        best_model_name = max(models.keys(), key=lambda k: models[k]['auc'])
        best_model = models[best_model_name]
        
        # Save model
        with open('rapid_24h_hab_model.pkl', 'wb') as f:
            pickle.dump(best_model, f)
        
        # Save results summary
        results = {
            'timestamp': datetime.now().isoformat(),
            'coverage_area': self.our_coverage,
            'hab_events_used': len(hab_events),
            'models_trained': len(models),
            'best_model': best_model_name,
            'best_auc': best_model['auc'],
            'features_used': best_model['features']
        }
        
        with open('rapid_24h_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"✅ Best model: {best_model_name} (AUC: {best_model['auc']:.3f})")
        print(f"✅ Model saved: rapid_24h_hab_model.pkl")
        print(f"✅ Results saved: rapid_24h_results.json")
        
        return results
    
    def run_24h_sprint(self):
        """Complete 24-hour sprint"""
        print("🚨 24-HOUR HAB PREDICTION SPRINT")
        print("=" * 50)
        
        start_time = datetime.now()
        
        try:
            # Step 1: Load data
            env_df, hab_events = self.load_and_match_data()
            
            # Step 2: Create labels
            labeled_df = self.create_training_labels(env_df, hab_events)
            
            # Step 3: Prepare features
            X, y, features = self.prepare_features(labeled_df)
            
            # Step 4: Train models
            models = self.train_models(X, y, labeled_df)
            
            # Step 5: Save results
            results = self.save_results(models, hab_events)
            
            duration = datetime.now() - start_time
            print(f"\n🏆 24-HOUR SPRINT COMPLETE!")
            print(f"Duration: {duration.total_seconds():.0f} seconds")
            print(f"Success: {len(models)} models trained")
            
            return results
            
        except Exception as e:
            print(f"❌ Sprint failed: {e}")
            import traceback
            traceback.print_exc()
            return None

def main():
    trainer = Rapid24hHABTrainer()
    results = trainer.run_24h_sprint()
    
    if results:
        print("\n🚀 READY FOR DEMO!")
    else:
        print("\n💥 Need to debug...")

if __name__ == "__main__":
    main()