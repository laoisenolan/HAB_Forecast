#!/usr/bin/env python3
"""
Proper analysis of the batch processing dataset to understand what data we actually have
"""

import pandas as pd
import numpy as np

def analyze_batch_dataset():
    print("🔍 PROPER DATASET ANALYSIS")
    print("=" * 60)
    
    # Load the batch processing dataset
    df = pd.read_parquet('cache/batch_processing/final_training_dataset.parquet')
    print(f"Dataset shape: {df.shape}")
    
    # Check HAB-related columns
    print("\n🎯 HAB-RELATED COLUMNS:")
    hab_cols = [col for col in df.columns if 'hab' in col.lower()]
    for col in hab_cols:
        print(f"  {col}")
        if col in ['has_hab', 'hab_occurrence']:
            print(f"    Distribution: {df[col].value_counts().to_dict()}")
        elif col in ['hab_species', 'hab_severity']:
            unique_vals = df[col].dropna().unique()
            print(f"    Unique values: {len(unique_vals)}")
            if len(unique_vals) <= 10:
                print(f"    Values: {list(unique_vals)}")
            else:
                print(f"    Sample values: {list(unique_vals[:10])}")
    
    # Check location-related columns
    print("\n📍 LOCATION-RELATED COLUMNS:")
    location_keywords = ['location', 'region', 'bay', 'area', 'place', 'latitude', 'longitude']
    location_cols = []
    for col in df.columns:
        if any(keyword in col.lower() for keyword in location_keywords):
            location_cols.append(col)
    
    for col in location_cols[:10]:  # Show first 10
        print(f"  {col}")
        unique_vals = df[col].dropna().unique()
        print(f"    Unique values: {len(unique_vals)}")
        if len(unique_vals) <= 5:
            print(f"    Values: {list(unique_vals)}")
        else:
            print(f"    Sample values: {list(unique_vals[:5])}")
    
    # Check date column
    print("\n📅 DATE INFORMATION:")
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        print(f"  Date range: {df['date'].min()} to {df['date'].max()}")
        print(f"  Years covered: {sorted(df['date'].dt.year.unique())}")
        
        # HAB events by year
        if 'has_hab' in df.columns:
            hab_by_year = df.groupby(df['date'].dt.year)['has_hab'].sum()
            print(f"  HAB events by year: {hab_by_year.to_dict()}")
    
    # Check existing HAB labels distribution
    print("\n📊 EXISTING HAB LABELS ANALYSIS:")
    if 'has_hab' in df.columns:
        hab_positive = df[df['has_hab'] == 1]
        print(f"  Total HAB positive records: {len(hab_positive):,}")
        print(f"  HAB positive rate: {len(hab_positive)/len(df)*100:.4f}%")
        
        if 'hab_species' in df.columns and not hab_positive.empty:
            species_dist = hab_positive['hab_species'].value_counts()
            print(f"  Species in HAB positive records:")
            for species, count in species_dist.head(10).items():
                print(f"    {species}: {count}")
    
    # Check spatial coverage
    print("\n🗺️ SPATIAL COVERAGE:")
    if 'latitude' in df.columns and 'longitude' in df.columns:
        print(f"  Latitude range: {df['latitude'].min():.3f} to {df['latitude'].max():.3f}")
        print(f"  Longitude range: {df['longitude'].min():.3f} to {df['longitude'].max():.3f}")
        
        # Spatial distribution of HAB events
        if 'has_hab' in df.columns:
            hab_locs = df[df['has_hab'] == 1][['latitude', 'longitude']].drop_duplicates()
            print(f"  Unique HAB locations: {len(hab_locs)}")
    
    print("\n✅ Analysis complete!")
    return df

if __name__ == "__main__":
    df = analyze_dataset_properly()