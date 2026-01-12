#!/usr/bin/env python3
"""
ULTIMATE HAB Species-Environment Data Merger
Merges the massive environmental dataset with HAB species from HAB_2017_2023.csv
"""

import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from pathlib import Path
import json

logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('species_merger_log.txt', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class SpeciesEnvironmentMerger:
    """Merges HAB species data with environmental data"""
    
    def __init__(self):
        self.coverage_bounds = {
            'lat_min': 33.575, 'lat_max': 34.425,
            'lon_min': 132.575, 'lon_max': 133.425
        }
        
        # Target bays in our coverage
        self.target_bays = [
            'Hiuchi-nada', 'Aki-nada', 'Iyo-nada', 
            'Hiroshima', 'Hiuchi', 'Aki', 'Iyo'
        ]
        
        logger.info("🧬 Species-Environment Merger Initialized")
        logger.info(f"Coverage area: {self.coverage_bounds}")
    
    def load_environmental_data(self):
        """Load the massive environmental dataset"""
        logger.info("📊 Loading massive environmental dataset...")
        
        env_df = pd.read_parquet('cache/batch_processing/final_training_dataset.parquet')
        logger.info(f"✅ Environmental data: {env_df.shape[0]:,} rows x {env_df.shape[1]} columns")
        logger.info(f"📅 Date range: {env_df['date'].min()} to {env_df['date'].max()}")
        
        # Check HAB labels
        hab_positive = env_df['has_hab'].sum()
        logger.info(f"🌊 HAB positive samples: {hab_positive:,} ({hab_positive/len(env_df)*100:.3f}%)")
        
        return env_df
    
    def load_hab_species_data(self):
        """Load HAB species data from CSV"""
        logger.info("🐟 Loading HAB species data...")
        
        hab_df = pd.read_csv('test_data/HAB_2017_2023.csv')
        logger.info(f"✅ HAB events: {len(hab_df):,} total events")
        
        # Filter for our coverage area
        our_hab_events = []
        for pattern in self.target_bays:
            pattern_events = hab_df[hab_df['location'].str.contains(pattern, na=False, case=False)]
            our_hab_events.extend(pattern_events.index.tolist())
        
        our_hab_events = list(set(our_hab_events))
        our_habs = hab_df.loc[our_hab_events].copy()
        
        logger.info(f"✅ HAB events in our area: {len(our_habs)}")
        
        # Parse dates
        our_habs['start_date'] = pd.to_datetime(our_habs['start_date'])
        our_habs['end_date'] = pd.to_datetime(our_habs['end_date'], errors='coerce')
        our_habs['duration_days'] = pd.to_numeric(our_habs['duration_days'], errors='coerce')
        
        # Fill missing end dates
        missing_end = our_habs['end_date'].isna()
        if missing_end.sum() > 0:
            logger.info(f"Filling {missing_end.sum()} missing end dates using duration")
            our_habs.loc[missing_end, 'end_date'] = (
                our_habs.loc[missing_end, 'start_date'] + 
                pd.to_timedelta(our_habs.loc[missing_end, 'duration_days'], unit='D')
            )
        
        # Show species distribution
        species_counts = our_habs['plankton_species'].value_counts()
        logger.info("🐟 Species in our area:")
        for species, count in species_counts.head(10).items():
            logger.info(f"  • {species}: {count} events")
        
        return our_habs
    
    def merge_species_with_environment(self, env_df, hab_df):
        """Merge species information with environmental data"""
        logger.info("🔗 Merging species data with environmental data...")
        
        # Start with environmental data
        merged_df = env_df.copy()
        merged_df['hab_species'] = 'none'
        merged_df['hab_event_id'] = -1
        merged_df['date'] = pd.to_datetime(merged_df['date'])
        
        species_matches = 0
        total_positive_labeled = 0
        
        # For each HAB event, find matching environmental samples
        for idx, hab_event in hab_df.iterrows():
            if pd.isna(hab_event['start_date']):
                continue
            
            species = hab_event['plankton_species']
            start_date = hab_event['start_date']
            end_date = hab_event['end_date'] if not pd.isna(hab_event['end_date']) else start_date
            
            # Create date range for this event
            try:
                event_dates = pd.date_range(start=start_date, end=end_date, freq='D')
            except:
                event_dates = [start_date]
            
            # Find environmental samples that match this HAB event
            event_matched = False
            for event_date in event_dates:
                # Match by date (all our environmental data is in our coverage area already)
                date_mask = (merged_df['date'].dt.date == event_date.date())
                
                if date_mask.sum() > 0:
                    # Only label samples that are already marked as HAB positive
                    hab_positive_mask = date_mask & (merged_df['has_hab'] == 1)
                    
                    if hab_positive_mask.sum() > 0:
                        merged_df.loc[hab_positive_mask, 'hab_species'] = species
                        merged_df.loc[hab_positive_mask, 'hab_event_id'] = idx
                        total_positive_labeled += hab_positive_mask.sum()
                        event_matched = True
            
            if event_matched:
                species_matches += 1
        
        logger.info(f"✅ Matched {species_matches} HAB events to environmental data")
        logger.info(f"✅ Labeled {total_positive_labeled:,} positive samples with species")
        
        # Show final species distribution
        species_in_env = merged_df[merged_df['has_hab'] == 1]['hab_species'].value_counts()
        logger.info("🐟 Final species distribution in environmental data:")
        for species, count in species_in_env.items():
            if species != 'none':
                logger.info(f"  • {species}: {count:,} samples")
        
        unlabeled_positive = (merged_df['has_hab'] == 1) & (merged_df['hab_species'] == 'none')
        logger.info(f"⚠️ Unlabeled positive samples: {unlabeled_positive.sum():,}")
        
        return merged_df
    
    def save_merged_dataset(self, merged_df):
        """Save the merged dataset"""
        logger.info("💾 Saving merged dataset...")
        
        output_dir = Path("cache/species_integrated")
        output_dir.mkdir(exist_ok=True)
        
        # Save full merged dataset
        output_path = output_dir / "environmental_with_species.parquet"
        merged_df.to_parquet(output_path, index=False)
        logger.info(f"✅ Merged dataset saved: {output_path}")
        
        # Save metadata
        metadata = {
            'created_at': datetime.now().isoformat(),
            'total_samples': len(merged_df),
            'hab_positive_samples': int(merged_df['has_hab'].sum()),
            'species_labeled_samples': int((merged_df['hab_species'] != 'none').sum()),
            'unique_species': merged_df[merged_df['hab_species'] != 'none']['hab_species'].nunique(),
            'species_distribution': merged_df['hab_species'].value_counts().to_dict(),
            'date_range': {
                'start': str(merged_df['date'].min()),
                'end': str(merged_df['date'].max())
            },
            'spatial_bounds': self.coverage_bounds
        }
        
        metadata_path = output_dir / "merger_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        logger.info(f"✅ Metadata saved: {metadata_path}")
        
        return output_path, metadata
    
    def run_merger(self):
        """Run the complete merger process"""
        logger.info("🚀 STARTING SPECIES-ENVIRONMENT MERGER")
        logger.info("=" * 60)
        
        start_time = datetime.now()
        
        try:
            # Load datasets
            env_df = self.load_environmental_data()
            hab_df = self.load_hab_species_data()
            
            # Merge datasets
            merged_df = self.merge_species_with_environment(env_df, hab_df)
            
            # Save results
            output_path, metadata = self.save_merged_dataset(merged_df)
            
            duration = datetime.now() - start_time
            
            logger.info("🏆 SPECIES-ENVIRONMENT MERGER COMPLETE!")
            logger.info(f"Duration: {duration.total_seconds():.0f} seconds")
            logger.info(f"Output: {output_path}")
            logger.info(f"Total samples: {metadata['total_samples']:,}")
            logger.info(f"HAB positive: {metadata['hab_positive_samples']:,}")
            logger.info(f"Species labeled: {metadata['species_labeled_samples']:,}")
            logger.info(f"Unique species: {metadata['unique_species']}")
            
            return output_path, metadata
            
        except Exception as e:
            logger.error(f"❌ Merger failed: {e}")
            import traceback
            traceback.print_exc()
            return None, None

def main():
    merger = SpeciesEnvironmentMerger()
    output_path, metadata = merger.run_merger()
    
    if output_path:
        print(f"\n🎉 SUCCESS! Merged dataset ready for species training!")
        print(f"📄 File: {output_path}")
        print(f"🐟 Species available: {metadata['unique_species']}")
    else:
        print("\n💥 FAILED!")

if __name__ == "__main__":
    main()