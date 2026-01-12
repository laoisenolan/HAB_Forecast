#!/usr/bin/env python3
import pandas as pd
import pickle
import json
from pathlib import Path

def check_selector_training_features():
    """Check what features the SelectKBest selector was trained on"""
    print("🔍 CHECKING SELECTOR TRAINING FEATURES")
    print("=" * 50)
    
    # Load a model selector
    model_dir = Path("cache/species_models_ultimate/Chattonella")
    
    with open(model_dir / "selector.pkl", 'rb') as f:
        selector = pickle.load(f)
    
    # Check selector attributes
    print(f"Selector type: {type(selector)}")
    print(f"Selector k: {selector.k}")
    print(f"Has feature_names_in_: {hasattr(selector, 'feature_names_in_')}")
    
    if hasattr(selector, 'feature_names_in_'):
        original_features = selector.feature_names_in_
        print(f"Original training features ({len(original_features)}):")
        for i, feat in enumerate(original_features[:20], 1):
            print(f"  {i:2d}. {feat}")
        if len(original_features) > 20:
            print(f"  ... and {len(original_features) - 20} more")
        
        # Check current dataset features
        env_data = pd.read_parquet("cache/species_integrated/environmental_with_species.parquet")
        current_features = [col for col in env_data.columns 
                           if col not in ['date', 'grid_cell_id', 'lat', 'lon', 'bay_region', 'has_hab', 'hab_species', 'hab_occurrence']]
        
        print(f"\nCurrent dataset features ({len(current_features)}):")
        
        # Find mismatches
        missing_in_current = set(original_features) - set(current_features)
        new_in_current = set(current_features) - set(original_features)
        
        print(f"\n❌ Missing from current dataset ({len(missing_in_current)}):")
        for feat in sorted(missing_in_current):
            print(f"  - {feat}")
        
        print(f"\n➕ New in current dataset ({len(new_in_current)}):")
        for feat in sorted(new_in_current):
            print(f"  + {feat}")
        
        # Create aligned feature set
        aligned_features = [f for f in original_features if f in current_features]
        print(f"\n✅ Aligned features ({len(aligned_features)}):")
        print(f"   Can use {len(aligned_features)}/{len(original_features)} original features")
        
        return original_features, current_features, aligned_features
    else:
        print("❌ Selector doesn't have feature_names_in_ attribute")
        return None, None, None

if __name__ == "__main__":
    check_selector_training_features()