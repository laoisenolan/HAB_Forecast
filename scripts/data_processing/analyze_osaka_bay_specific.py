#!/usr/bin/env python3
"""
RAPID OSAKA BAY ANALYSIS - Check existing data coverage
"""

import pandas as pd
import numpy as np

print("🚀 RAPID OSAKA BAY DATA ANALYSIS")
print("=" * 50)

# Load existing batch processing data
try:
    df = pd.read_csv('cache/batch_processing/batches/2017-01/spatial_temporal_training_dataset.csv')
    print(f"✅ Loaded batch processing data: {len(df):,} samples")
    
    # Basic data info
    print(f"\n📊 DATA OVERVIEW:")
    print(f"   Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"   Features: {len(df.columns)} columns")
    print(f"   Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
    
    # Spatial coverage
    print(f"\n🗺️ SPATIAL COVERAGE:")
    print(f"   Latitude:  {df['lat'].min():.3f} to {df['lat'].max():.3f}")
    print(f"   Longitude: {df['lon'].min():.3f} to {df['lon'].max():.3f}")
    
    # Osaka Bay bounds check
    osaka_bounds = (
        (df['lat'] >= 34.3) & (df['lat'] <= 34.8) &
        (df['lon'] >= 135.0) & (df['lon'] <= 135.6)
    )
    osaka_samples = df[osaka_bounds]
    
    print(f"\n🎯 OSAKA BAY COVERAGE:")
    print(f"   Samples in Osaka Bay area: {len(osaka_samples):,}")
    print(f"   Percentage of total: {len(osaka_samples)/len(df)*100:.1f}%")
    
    if len(osaka_samples) > 0:
        print(f"   Osaka lat range: {osaka_samples['lat'].min():.3f} to {osaka_samples['lat'].max():.3f}")
        print(f"   Osaka lon range: {osaka_samples['lon'].min():.3f} to {osaka_samples['lon'].max():.3f}")
    
    # HAB label analysis
    print(f"\n🌊 HAB LABELS:")
    print(f"   Total HAB positive: {df['has_hab'].sum():,}")
    print(f"   HAB rate: {df['has_hab'].mean()*100:.2f}%")
    
    if len(osaka_samples) > 0:
        osaka_hab = osaka_samples['has_hab'].sum()
        print(f"   Osaka HAB positive: {osaka_hab:,}")
        print(f"   Osaka HAB rate: {osaka_samples['has_hab'].mean()*100:.2f}%")
    
    # Bay region distribution
    if 'bay_region' in df.columns:
        print(f"\n🏞️ BAY REGIONS:")
        bay_counts = df['bay_region'].value_counts()
        for region, count in bay_counts.head().items():
            print(f"   {region}: {count:,} samples")
    
    # Feature summary
    core_features = [
        'sst', 'ocean_temperature_hycom', 'sea_surface_elevation',
        'ocean_current_u', 'ocean_current_v', 'wind_u', 'wind_v',
        'chlor_a', 'CHLA_AVE_mean', 'turbidity_index', 'ocean_salinity',
        'precipitation'
    ]
    
    available_features = [f for f in core_features if f in df.columns]
    print(f"\n⚡ AVAILABLE CORE FEATURES ({len(available_features)}/{len(core_features)}):")
    for feature in available_features:
        non_null = df[feature].notna().sum()
        print(f"   ✅ {feature}: {non_null:,}/{len(df):,} ({non_null/len(df)*100:.1f}%)")
    
    missing_features = [f for f in core_features if f not in df.columns]
    if missing_features:
        print(f"\n❌ MISSING FEATURES:")
        for feature in missing_features:
            print(f"   ❌ {feature}")
    
    # Time series check
    df['date'] = pd.to_datetime(df['date'])
    daily_counts = df.groupby('date').size()
    print(f"\n📅 TEMPORAL COVERAGE:")
    print(f"   Unique dates: {len(daily_counts)}")
    print(f"   Samples per day: {daily_counts.mean():.1f} ± {daily_counts.std():.1f}")
    print(f"   Date range: {daily_counts.index.min()} to {daily_counts.index.max()}")
    
    # Quick success probability assessment
    print(f"\n🎯 24-HOUR SUCCESS PROBABILITY:")
    
    success_factors = {
        'Data size': len(df) > 1000,
        'HAB labels': df['has_hab'].sum() > 50,
        'Osaka coverage': len(osaka_samples) > 100,
        'Core features': len(available_features) >= 8,
        'Temporal span': len(daily_counts) > 30
    }
    
    success_count = sum(success_factors.values())
    for factor, status in success_factors.items():
        print(f"   {'✅' if status else '❌'} {factor}")
    
    print(f"\n🏆 SUCCESS PROBABILITY: {success_count}/5 factors met")
    if success_count >= 4:
        print("   🚀 EXCELLENT - Ready for 24-hour sprint!")
    elif success_count >= 3:
        print("   ⚡ GOOD - Should work with minor adjustments")
    else:
        print("   ⚠️ CHALLENGING - May need more data")

except Exception as e:
    print(f"❌ Error loading data: {e}")
    print("\nChecking alternative data sources...")
    
    # Check smart sampling data
    try:
        smart_df = pd.read_csv('cache/smart_sampling/spatial_temporal_training_dataset.csv')
        print(f"✅ Found smart sampling data: {len(smart_df):,} samples")
        print("   Consider using this dataset instead!")
    except:
        print("❌ No smart sampling data found")

print("\n" + "=" * 50)
print("Ready for OSAKA BAY 24-hour sprint! 🏃‍♂️💨")