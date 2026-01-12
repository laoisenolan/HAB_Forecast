#!/usr/bin/env python3
"""
DURATION-AWARE HAB Analysis
Calculate total HAB DAYS not just events!
"""

import pandas as pd
import numpy as np

# Load HAB dataset
df = pd.read_csv('test_data/HAB_2017_2023.csv')
df['start_date'] = pd.to_datetime(df['start_date'])
df['duration_days'] = pd.to_numeric(df['duration_days'], errors='coerce')

print('🔥 DURATION-AWARE HAB ANALYSIS 🔥')
print('=' * 60)

# Filter out events with missing duration
df_with_duration = df[df['duration_days'].notna()].copy()
print(f'Events with duration data: {len(df_with_duration)}/{len(df)} ({len(df_with_duration)/len(df)*100:.1f}%)')

# Calculate total HAB days by location
def calculate_hab_days_by_location():
    locations = df_with_duration['location'].str.split(';').str[0].str.strip()
    location_stats = []
    
    for location in locations.unique():
        location_events = df_with_duration[locations == location].copy()
        
        stats = {
            'location': location,
            'event_count': len(location_events),
            'total_hab_days': location_events['duration_days'].sum(),
            'avg_duration': location_events['duration_days'].mean(),
            'max_duration': location_events['duration_days'].max(),
            'min_duration': location_events['duration_days'].min(),
            'median_duration': location_events['duration_days'].median()
        }
        location_stats.append(stats)
    
    return pd.DataFrame(location_stats).sort_values('total_hab_days', ascending=False)

location_analysis = calculate_hab_days_by_location()

print('\n📊 TOP 15 LOCATIONS BY TOTAL HAB DAYS:')
print('=' * 60)
for i, row in location_analysis.head(15).iterrows():
    print(f"{row['location']:<20} | Events: {row['event_count']:3d} | "
          f"Total Days: {row['total_hab_days']:6.0f} | "
          f"Avg: {row['avg_duration']:5.1f}d | "
          f"Max: {row['max_duration']:3.0f}d")

print('\n🎯 OSAKA BAY vs BUNGO CHANNEL DETAILED COMPARISON:')
print('=' * 60)

# Detailed analysis for top contenders
def analyze_location_details(location_name):
    df_location = df_with_duration[
        df_with_duration['location'].str.contains(location_name, na=False)
    ].copy()
    
    if len(df_location) == 0:
        return None
    
    total_days = df_location['duration_days'].sum()
    
    # Calculate temporal distribution
    df_location['year'] = df_location['start_date'].dt.year
    yearly_days = df_location.groupby('year')['duration_days'].sum()
    
    # Species breakdown
    species_days = df_location.groupby('plankton_species')['duration_days'].sum().sort_values(ascending=False)
    
    print(f'\n🔍 {location_name.upper()} ANALYSIS:')
    print(f'   📈 Total Events: {len(df_location)}')
    print(f'   📅 Total HAB Days: {total_days:.0f}')
    print(f'   📊 Average Duration: {df_location["duration_days"].mean():.1f} days')
    print(f'   ⏰ Days per Year: {total_days/7:.1f} (2017-2023)')
    print(f'   📋 HAB Days per Event: {total_days/len(df_location):.1f}')
    
    print(f'\n   📅 Yearly HAB Days Distribution:')
    for year, days in yearly_days.items():
        events_that_year = len(df_location[df_location['year'] == year])
        print(f'      {year}: {days:6.0f} days ({events_that_year:2d} events)')
    
    print(f'\n   🦠 Top Species by Total HAB Days:')
    for species, days in species_days.head(5).items():
        events = len(df_location[df_location['plankton_species'] == species])
        avg_duration = days / events
        print(f'      {species:<25}: {days:6.0f} days ({events:2d} events, {avg_duration:4.1f}d avg)')
    
    return {
        'total_events': len(df_location),
        'total_days': total_days,
        'avg_duration': df_location['duration_days'].mean(),
        'days_per_year': total_days/7,
        'yearly_distribution': yearly_days,
        'species_days': species_days
    }

osaka_stats = analyze_location_details('Osaka Bay')
bungo_stats = analyze_location_details('Bungo Channel')

print('\n🏆 WINNER COMPARISON:')
print('=' * 60)
if osaka_stats and bungo_stats:
    print(f'METRIC                    | OSAKA BAY    | BUNGO CHANNEL | WINNER')
    print(f'Total Events              | {osaka_stats["total_events"]:8d}     | {bungo_stats["total_events"]:8d}      | {"BUNGO" if bungo_stats["total_events"] > osaka_stats["total_events"] else "OSAKA"}')
    print(f'Total HAB Days            | {osaka_stats["total_days"]:8.0f}     | {bungo_stats["total_days"]:8.0f}      | {"BUNGO" if bungo_stats["total_days"] > osaka_stats["total_days"] else "OSAKA"}')
    print(f'HAB Days per Year         | {osaka_stats["days_per_year"]:8.1f}     | {bungo_stats["days_per_year"]:8.1f}      | {"BUNGO" if bungo_stats["days_per_year"] > osaka_stats["days_per_year"] else "OSAKA"}')
    print(f'Average Event Duration    | {osaka_stats["avg_duration"]:8.1f}     | {bungo_stats["avg_duration"]:8.1f}      | {"BUNGO" if bungo_stats["avg_duration"] > osaka_stats["avg_duration"] else "OSAKA"}')

print('\n💡 SAMPLING RATE CALCULATION:')
print('=' * 60)

def calculate_sampling_rates(stats, location_name):
    if not stats:
        return
    
    total_days_in_period = 7 * 365  # 2017-2023
    hab_days = stats['total_days']
    hab_rate = hab_days / total_days_in_period
    
    print(f'{location_name}:')
    print(f'   🎯 HAB Days: {hab_days:.0f} / {total_days_in_period} total days')
    print(f'   📊 HAB Rate: {hab_rate*100:.2f}% of all days')
    print(f'   🔢 Expected positive samples per 1000 grid-days: {hab_rate*1000:.1f}')
    
    # For temporal CV
    print(f'   📈 Leave-One-Year-Out CV:')
    for year, year_days in stats['yearly_distribution'].items():
        test_rate = year_days / 365
        print(f'      Test {year}: {year_days:.0f} days ({test_rate*100:.2f}% of year)')

calculate_sampling_rates(osaka_stats, 'OSAKA BAY')
calculate_sampling_rates(bungo_stats, 'BUNGO CHANNEL')

print('\n🎲 DATA COLLECTION IMPACT:')
print('=' * 60)

# Calculate grid-scale data requirements
def calculate_data_requirements(stats, location_name, grid_size=50):
    if not stats:
        return
    
    hab_days = stats['total_days']
    total_days = 7 * 365
    
    # Assume each HAB day affects multiple grid cells
    grid_cells_per_hab = 10  # Conservative estimate
    positive_samples = hab_days * grid_cells_per_hab
    total_samples = total_days * grid_size * grid_size
    positive_rate = positive_samples / total_samples
    
    print(f'{location_name} Data Requirements (50x50 grid):')
    print(f'   📦 Total samples (7 years): {total_samples:,}')
    print(f'   ✅ Positive samples: {positive_samples:,}')
    print(f'   📊 Positive rate: {positive_rate*100:.3f}%')
    print(f'   🎯 Samples per year: {total_samples/7:,.0f}')
    print(f'   💾 Storage (189 features): {total_samples * 189 / 1e6:.1f}M values')

calculate_data_requirements(osaka_stats, 'OSAKA BAY')
calculate_data_requirements(bungo_stats, 'BUNGO CHANNEL')

print('\n🚀 FINAL RECOMMENDATION:')
print('=' * 60)
if osaka_stats and bungo_stats:
    if bungo_stats['total_days'] > osaka_stats['total_days']:
        winner = 'BUNGO CHANNEL'
        winner_stats = bungo_stats
    else:
        winner = 'OSAKA BAY'
        winner_stats = osaka_stats
    
    print(f'🏆 WINNER: {winner}')
    print(f'   📈 {winner_stats["total_days"]:.0f} total HAB days vs competitor')
    print(f'   📊 {winner_stats["days_per_year"]:.1f} HAB days per year on average')
    print(f'   🎯 Much better sampling rate for ML training!')