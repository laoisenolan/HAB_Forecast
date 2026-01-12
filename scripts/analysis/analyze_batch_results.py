#!/usr/bin/env python3
"""
Quick analysis of the batch processing dataset
"""

import pandas as pd
import numpy as np
from pathlib import Path

def analyze_dataset():
    data_path = "cache/batch_processing/final_training_dataset.parquet"
    
    print("🔍 Analyzing batch processing dataset...")
    print("=" * 60)
    
    # Load dataset
    try:
        print(f"📊 Loading: {data_path}")
        df = pd.read_parquet(data_path)
        print(f"✅ Loaded successfully!")
        
        # Basic info
        print(f"\n📊 DATASET OVERVIEW:")
        print(f"   Shape: {df.shape}")
        print(f"   Memory: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        # Column analysis
        print(f"\n📋 COLUMNS ({len(df.columns)}):")
        for i, col in enumerate(df.columns[:20]):  # Show first 20
            print(f"   {i+1:2d}. {col}")
        if len(df.columns) > 20:
            print(f"   ... and {len(df.columns) - 20} more columns")
        
        # Target analysis
        target_cols = ['has_hab', 'hab_occurrence', 'hab_species']
        print(f"\n🎯 TARGET ANALYSIS:")
        for col in target_cols:
            if col in df.columns:
                if col in ['has_hab', 'hab_occurrence']:
                    dist = df[col].value_counts()
                    print(f"   {col}: {dist.to_dict()}")
                elif col == 'hab_species':
                    species_count = df[col].value_counts()
                    print(f"   hab_species: {len(species_count)} unique species")
                    print(f"   Top 5 species:")
                    for species, count in species_count.head().items():
                        print(f"      - {species}: {count:,}")
        
        # Data types
        print(f"\n📈 DATA TYPES:")
        dtype_counts = df.dtypes.value_counts()
        for dtype, count in dtype_counts.items():
            print(f"   {dtype}: {count} columns")
        
        # Missing values
        missing = df.isnull().sum()
        missing = missing[missing > 0]
        if len(missing) > 0:
            print(f"\n⚠️ MISSING VALUES:")
            for col, count in missing.head(10).items():
                print(f"   {col}: {count:,} ({count/len(df)*100:.2f}%)")
        else:
            print(f"\n✅ NO MISSING VALUES")
        
        # Sample data
        print(f"\n📋 SAMPLE DATA (first 3 rows, key columns):")
        key_cols = ['date', 'latitude', 'longitude', 'has_hab', 'hab_species']
        sample_cols = [col for col in key_cols if col in df.columns]
        if sample_cols:
            print(df[sample_cols].head(3).to_string())
        
        return df
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

if __name__ == "__main__":
    df = analyze_dataset()
    if df is not None:
        print(f"\n🎉 Analysis complete! Dataset has {len(df):,} records.")