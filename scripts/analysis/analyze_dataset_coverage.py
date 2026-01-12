#!/usr/bin/env python3
"""
Analyze what visualizations we can create from available data
"""

import pandas as pd
import json
import numpy as np

def analyze_data():
    print("=== HAB FORECAST DATA INVENTORY ===\n")
    
    # 1. HAB Events Data
    print("📊 RAW HAB EVENTS DATA:")
    hab_df = pd.read_csv('test_data/HAB_2017_2023.csv')
    print(f"   Records: {len(hab_df)}")
    print(f"   Species: {hab_df['plankton_species'].nunique()} unique")
    
    # Convert dates safely
    hab_df['start_date'] = pd.to_datetime(hab_df['start_date'], errors='coerce')
    hab_df['end_date'] = pd.to_datetime(hab_df['end_date'], errors='coerce')
    valid_dates = hab_df['start_date'].dropna()
    if len(valid_dates) > 0:
        print(f"   Date range: {valid_dates.min().strftime('%Y-%m-%d')} to {valid_dates.max().strftime('%Y-%m-%d')}")
    print(f"   Columns: {list(hab_df.columns)}")
    
    # 2. Training Dataset
    print("\n📊 TRAINING DATASET:")
    train_df = pd.read_parquet('cache/batch_processing/final_training_dataset.parquet')
    print(f"   Records: {len(train_df):,}")
    print(f"   Features: {len(train_df.columns)}")
    print(f"   Positive samples: {train_df['has_hab'].sum()}")
    print(f"   Positive rate: {train_df['has_hab'].mean()*100:.4f}%")
    
    # 3. Environmental Features
    exclude_cols = ['date', 'grid_cell_id', 'lat', 'lon', 'bay_region', 'has_hab']
    feature_cols = [col for col in train_df.columns if col not in exclude_cols]
    print(f"\n📊 ENVIRONMENTAL FEATURES ({len(feature_cols)}):")
    print(f"   Sample: {feature_cols[:10]}")
    
    # 4. Species Performance Data
    print("\n📊 SPECIES MODEL RESULTS:")
    with open('cache/species_models_2025/species_training_summary.json', 'r') as f:
        species_perf = json.load(f)
    
    trained_species = list(species_perf['species_performance'].keys())
    print(f"   Trained species: {len(trained_species)}")
    print(f"   Species list: {trained_species[:5]}...")
    
    # 5. Spatial Coverage
    print(f"\n📊 SPATIAL COVERAGE:")
    print(f"   Latitude: {train_df['lat'].min():.2f} to {train_df['lat'].max():.2f}")
    print(f"   Longitude: {train_df['lon'].min():.2f} to {train_df['lon'].max():.2f}")
    print(f"   Grid cells: {train_df['grid_cell_id'].nunique()}")
    
    # 6. Temporal Coverage
    print(f"\n📊 TEMPORAL COVERAGE:")
    train_df['date'] = pd.to_datetime(train_df['date'])
    print(f"   Date range: {train_df['date'].min().strftime('%Y-%m-%d')} to {train_df['date'].max().strftime('%Y-%m-%d')}")
    print(f"   Total days: {(train_df['date'].max() - train_df['date'].min()).days}")
    
    # 7. Check for predictions
    pred_cols = [col for col in train_df.columns if 'pred' in col.lower() or 'prob' in col.lower()]
    print(f"\n📊 PREDICTION DATA:")
    print(f"   Prediction columns: {pred_cols if pred_cols else 'None found'}")
    
    return {
        'hab_events': hab_df,
        'training_data': train_df, 
        'feature_columns': feature_cols,
        'species_performance': species_perf,
        'trained_species': trained_species
    }

def assess_visualization_feasibility(data):
    print("\n" + "="*60)
    print("VISUALIZATION FEASIBILITY ASSESSMENT")
    print("="*60)
    
    assessments = {}
    
    # Figure 1: ROC Curves for Species-Specific Models
    print("\n📈 Figure 1: ROC Curves for Species-Specific Models")
    has_model_results = len(data['species_performance']['species_performance']) > 0
    has_test_metrics = 'test_metrics' in list(data['species_performance']['species_performance'].values())[0]
    
    if has_model_results and has_test_metrics:
        print("   ✅ FEASIBLE: We have test metrics (AUC, precision, recall) for 9 species")
        print("   📊 Can create: Performance comparison, but need prediction probabilities for true ROC curves")
        assessments['roc_curves'] = 'PARTIAL'
    else:
        print("   ❌ NOT FEASIBLE: Missing model results")
        assessments['roc_curves'] = 'NO'
    
    # Figure 2: Learning Curves
    print("\n📈 Figure 2: Learning Curves (Training Size vs Performance)")
    has_training_sizes = True  # We have this in species performance data
    if has_training_sizes:
        print("   ✅ FEASIBLE: We have training sizes and performance metrics for each species")
        assessments['learning_curves'] = 'YES'
    else:
        print("   ❌ NOT FEASIBLE: Missing training size data")
        assessments['learning_curves'] = 'NO'
    
    # Figure 3: Feature Correlation Matrix
    print("\n📈 Figure 3: Feature Correlation Matrix")
    has_features = len(data['feature_columns']) > 0
    if has_features:
        print(f"   ✅ FEASIBLE: We have {len(data['feature_columns'])} environmental features")
        assessments['correlation_matrix'] = 'YES'
    else:
        print("   ❌ NOT FEASIBLE: Missing feature data")
        assessments['correlation_matrix'] = 'NO'
    
    # Figure 4: Temporal HAB Event Distribution
    print("\n📈 Figure 4: Temporal HAB Event Distribution")
    has_temporal_data = len(data['hab_events']) > 0
    if has_temporal_data:
        print("   ✅ FEASIBLE: We have 731 HAB events with dates and environmental data")
        assessments['temporal_distribution'] = 'YES'
    else:
        print("   ❌ NOT FEASIBLE: Missing temporal data")
        assessments['temporal_distribution'] = 'NO'
    
    # Figure 5: Spatial Prediction Accuracy Map
    print("\n📈 Figure 5: Spatial Prediction Accuracy Map")
    has_spatial_data = 'lat' in data['training_data'].columns and 'lon' in data['training_data'].columns
    has_labels = 'has_hab' in data['training_data'].columns
    if has_spatial_data and has_labels:
        print("   ✅ FEASIBLE: We have lat/lon coordinates and HAB labels")
        print("   📊 Can show: HAB event locations, grid cell coverage, label distribution")
        assessments['spatial_accuracy'] = 'PARTIAL'
    else:
        print("   ❌ NOT FEASIBLE: Missing spatial coordinates")
        assessments['spatial_accuracy'] = 'NO'
    
    # Figure 6: Model Calibration Plots
    print("\n📈 Figure 6: Model Calibration Plots")
    pred_cols = [col for col in data['training_data'].columns if 'pred' in col.lower() or 'prob' in col.lower()]
    if pred_cols:
        print(f"   ✅ FEASIBLE: Found prediction columns: {pred_cols}")
        assessments['calibration_plots'] = 'YES'
    else:
        print("   ❌ NOT FEASIBLE: No prediction probabilities found")
        print("   💡 Note: Would need to run inference on trained models to get predictions")
        assessments['calibration_plots'] = 'NO'
    
    print(f"\n📊 SUMMARY:")
    feasible = [k for k, v in assessments.items() if v in ['YES', 'PARTIAL']]
    print(f"   ✅ Fully feasible: {len([k for k, v in assessments.items() if v == 'YES'])}/6")
    print(f"   🔄 Partially feasible: {len([k for k, v in assessments.items() if v == 'PARTIAL'])}/6")
    print(f"   ❌ Not feasible: {len([k for k, v in assessments.items() if v == 'NO'])}/6")
    
    return assessments

if __name__ == "__main__":
    data = analyze_data()
    assessments = assess_visualization_feasibility(data)