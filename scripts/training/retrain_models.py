#!/usr/bin/env python3
"""
Improved Model Retraining Script

This script retrains the HAB forecasting models with:
1. Proper feature exclusion (no data leakage)
2. Improved hyperparameters from config
3. Cross-validation for better evaluation
4. Better model performance metrics
"""

import sys
import logging
from pathlib import Path
from datetime import datetime

# Add src to path
current_dir = Path(__file__).parent
src_dir = current_dir / "src"
sys.path.insert(0, str(src_dir))

from species_trainer_standalone import StandaloneSpeciesTrainer

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'retraining_log_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def main():
    """Main retraining function"""
    
    logger.info("🚀 Starting improved model retraining")
    
    # Initialize trainer
    trainer = StandaloneSpeciesTrainer(
        cache_dir="cache/species_models_improved",
        random_state=42
    )
    
    # Training data path
    training_data_path = "data/real_collection/real_hab_training_dataset_20250806_151657.csv"
    
    # Target species (focus on the most important ones first)
    target_species = [
        "Karenia mikimotoi",
        "Heterosigma akashiwo",
        "Chattonella antiqua",
        "Cochlodinium polykrikoides"
    ]
    
    logger.info(f"📊 Training data: {training_data_path}")
    logger.info(f"🎯 Target species: {target_species}")
    
    try:
        # Train models
        results = trainer.train_all_species(
            data_path=training_data_path,
            target_species=target_species
        )
        
        # Generate report
        trainer.generate_report(results)
        
        logger.info("✅ Model retraining completed successfully!")
        
        # Print summary
        print("\n" + "="*60)
        print("📊 RETRAINING SUMMARY")
        print("="*60)
        
        for species, result in results.items():
            metrics = result['metrics']
            print(f"\n🧬 {species}:")
            print(f"   📈 F1 Score: {metrics['f1_score']:.3f}")
            print(f"   📊 AUC: {metrics['auc']:.3f}")
            print(f"   🔄 CV F1: {metrics['cv_f1_mean']:.3f} ± {metrics['cv_f1_std']:.3f}")
            print(f"   📋 Features: {len(result['selected_features'])}")
            print(f"   🎯 Positive Rate: {result['positive_rate']:.3f}")
        
        print(f"\n📁 Models saved to: {trainer.cache_dir}")
        
    except Exception as e:
        logger.error(f"❌ Retraining failed: {e}")
        raise

if __name__ == "__main__":
    main()
