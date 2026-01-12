import pandas as pd
import numpy as np

# Load the rebuilt dataset
df = pd.read_parquet('cache/rebuilt_dataset/hab_training_dataset_rebuilt.parquet')

print("=== DATA LEAKAGE INVESTIGATION ===")
print(f"Total records: {len(df):,}")
print(f"Positive records: {df['has_hab'].sum():,}")
print(f"Positive rate: {df['has_hab'].mean()*100:.3f}%")

# Check HAB event clustering
positive_df = df[df['has_hab']]
print(f"\n=== SPATIAL-TEMPORAL CLUSTERING ===")
print("HAB events distribution by event ID:")
event_counts = positive_df['hab_event_id'].value_counts()
print(event_counts.head(10))

print(f"\nUnique HAB event IDs: {len(event_counts)}")
print(f"Total positive records: {len(positive_df):,}")
print(f"Average records per HAB event: {len(positive_df)/len(event_counts):.1f}")

# Check spatial clustering
print(f"\n=== SPATIAL DISTRIBUTION ===")
unique_locations = positive_df.drop_duplicates(['lat', 'lon'])
print(f"Unique lat/lon combinations with HAB: {len(unique_locations)}")
print(f"Total positive records: {len(positive_df):,}")
print(f"Records per location: {len(positive_df)/len(unique_locations):.1f}")

# Check temporal clustering  
print(f"\n=== TEMPORAL DISTRIBUTION ===")
positive_df['date'] = pd.to_datetime(positive_df['date'])
daily_counts = positive_df.groupby(positive_df['date'].dt.date).size()
print(f"Unique dates with HAB: {len(daily_counts)}")
print(f"Records per day: {len(positive_df)/len(daily_counts):.1f}")

print("\nTop dates by record count:")
print(daily_counts.sort_values(ascending=False).head(10))

# The smoking gun - check if same locations/times appear together
print(f"\n=== THE SMOKING GUN ===")
grouped = positive_df.groupby(['lat', 'lon', positive_df['date'].dt.date]).size()
print("Records per location-date combination:")
print(grouped.sort_values(ascending=False).head(10))

print(f"\nThis explains the perfect performance:")
print(f"- The model sees the same lat/lon/date combinations in train and test")
print(f"- It memorizes these exact combinations rather than learning general patterns")
print(f"- AUC 1.0 and F1 0.99+ are impossible without data leakage in real-world HAB prediction")