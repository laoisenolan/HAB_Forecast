#!/usr/bin/env python3
"""
Analyze HAB events by location to inform bay-focused strategy
"""

import pandas as pd
import numpy as np

# Load HAB dataset
df = pd.read_csv('test_data/HAB_2017_2023.csv')

print('=== GEOGRAPHIC ANALYSIS ===')
print(f'Total HAB events: {len(df)}')

print('\n=== LOCATION DISTRIBUTION ===')
locations = df['location'].str.split(';').str[0].str.strip().value_counts()
print("Top 15 locations:")
print(locations.head(15))
print(f'\nTop 5 locations account for: {locations.head(5).sum()}/{len(df)} events ({locations.head(5).sum()/len(df)*100:.1f}%)')

print('\n=== OSAKA BAY ANALYSIS ===')
osaka_events = df[df['location'].str.contains('Osaka Bay', na=False)]
print(f'Osaka Bay events: {len(osaka_events)}')
if len(osaka_events) > 0:
    print('Osaka Bay HAB events (first 10):')
    for _, row in osaka_events.head(10).iterrows():
        print(f'  {row["start_date"]} - {row["plankton_species"]} - {row["cell_count"]}')

print('\n=== BUNGO CHANNEL ANALYSIS ===')
bungo_events = df[df['location'].str.contains('Bungo Channel', na=False)]
print(f'Bungo Channel events: {len(bungo_events)}')

print('\n=== TOSA BAY ANALYSIS ===')
tosa_events = df[df['location'].str.contains('Tosa Bay', na=False)]
print(f'Tosa Bay events: {len(tosa_events)}')

print('\n=== SUO-NADA ANALYSIS ===')
suo_events = df[df['location'].str.contains('Suo-nada', na=False)]
print(f'Suo-nada events: {len(suo_events)}')

print('\n=== TEMPORAL DISTRIBUTION ===')
df['start_date'] = pd.to_datetime(df['start_date'])
df['year'] = df['start_date'].dt.year
yearly = df['year'].value_counts().sort_index()
print('Events per year:')
for year, count in yearly.items():
    print(f'  {year}: {count} events')

print('\n=== SPECIES DISTRIBUTION IN TOP LOCATIONS ===')
print('\nOsaka Bay species:')
if len(osaka_events) > 0:
    osaka_species = osaka_events['plankton_species'].value_counts()
    print(osaka_species)

print('\nBungo Channel species:')
if len(bungo_events) > 0:
    bungo_species = bungo_events['plankton_species'].value_counts()
    print(bungo_species.head(5))

print('\n=== DURATION ANALYSIS ===')
df['duration_days'] = pd.to_numeric(df['duration_days'], errors='coerce')
print(f'Average HAB duration: {df["duration_days"].mean():.1f} days')
print(f'Median HAB duration: {df["duration_days"].median():.1f} days')
print(f'Max HAB duration: {df["duration_days"].max():.0f} days')

if len(osaka_events) > 0:
    osaka_events['duration_days'] = pd.to_numeric(osaka_events['duration_days'], errors='coerce')
    print(f'Osaka Bay average duration: {osaka_events["duration_days"].mean():.1f} days')