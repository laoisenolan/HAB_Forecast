        # Step 1: Pre-filter environmental data for faster processing
        logger.info("📊 Step 1a: Pre-filtering environmental data for FAST processing...")
        import pandas as pd
        import os
        
        # Create filtered environmental data file
        env_df = pd.read_csv(config['environmental_data_file'])
        env_df['date'] = pd.to_datetime(env_df['date'])
        
        # Filter to date range
        start_dt = pd.to_datetime(config['start_date'])
        end_dt = pd.to_datetime(config['end_date'])
        env_df_filtered = env_df[(env_df['date'] >= start_dt) & (env_df['date'] <= end_dt)]
        
        logger.info(f"📊 Filtered environmental data: {len(env_df):,} → {len(env_df_filtered):,} records")
        
        # Save filtered data
        os.makedirs(config['output_dir'], exist_ok=True)
        filtered_env_path = f"{config['output_dir']}/environmental_data_filtered.csv"
        env_df_filtered.to_csv(filtered_env_path, index=False)
        
        # Step 1b: Create training dataset (FAST)
        logger.info("📊 Step 1b: Creating spatial-temporal training dataset (FAST)...")
        training_dataset_path = create_training_ready_dataset(
            hab_csv_path=config['hab_events_file'],
            environmental_csv_path=filtered_env_path,
            output_dir=config['output_dir'],
            grid_resolution=config['grid_resolution'],
            point_coordinates=config['center_coordinates']
        )r.info(f"   📁 Output: {config['output_dir']}")
    
    start_time = datetime.now()
    
    try:
        # Import the optimized spatial environmental mapper
        from feature_engineering.spatial_environmental_mapper import create_training_ready_dataset
        
        # Step 1: Pre-filter environmental data for faster processing
        logger.info("📊 Step 1a: Pre-filtering environmental data for FAST processing...")
        import pandas as pd
        import os
        
        # Create filtered environmental data file
        env_df = pd.read_csv(config['environmental_data_file'])
        env_df['date'] = pd.to_datetime(env_df['date'])
        
        # Filter to date range
        start_dt = pd.to_datetime(config['start_date'])
        end_dt = pd.to_datetime(config['end_date'])
        env_df_filtered = env_df[(env_df['date'] >= start_dt) & (env_df['date'] <= end_dt)]
        
        logger.info(f"📊 Filtered environmental data: {len(env_df):,} → {len(env_df_filtered):,} records")
        
        # Save filtered data
        os.makedirs(config['output_dir'], exist_ok=True)
        filtered_env_path = f"{config['output_dir']}/environmental_data_filtered.csv"
        env_df_filtered.to_csv(filtered_env_path, index=False)
        
        # Step 1b: Create training dataset (FAST)
        logger.info("📊 Step 1b: Creating spatial-temporal training dataset (FAST)...")
        training_dataset_path = create_training_ready_dataset(
            hab_csv_path=config['hab_events_file'],
            environmental_csv_path=filtered_env_path,
            output_dir=config['output_dir'],
            grid_resolution=config['grid_resolution'],
            point_coordinates=config['center_coordinates']
        )
        
        elapsed = datetime.now() - start_time
        logger.info(f"✅ Dataset created in {elapsed.total_seconds():.1f} seconds")
        logger.info(f"📁 Training dataset: {training_dataset_path}")
        
        # Step 2: Quick analysis
        logger.info("📊 Step 2: Analyzing training dataset...")
        
        import pandas as pd
        df = pd.read_parquet(training_dataset_path)
        
        logger.info(f"📊 Dataset shape: {df.shape[0]:,} records × {df.shape[1]} features")
        logger.info(f"🏷️ Label distribution:")
        logger.info(f"   Positive (HAB): {df['has_hab'].sum():,} ({df['has_hab'].mean()*100:.1f}%)")
        logger.info(f"   Negative (No HAB): {(~df['has_hab']).sum():,} ({(1-df['has_hab'].mean())*100:.1f}%)")
        
        # Grid analysis
        unique_cells = df['grid_cell_id'].nunique()
        unique_dates = df['date'].nunique()
        logger.info(f"🗺️ Coverage: {unique_cells:,} grid cells × {unique_dates:,} dates")
        
        # Class balance check
        pos_rate = df['has_hab'].mean()
        if pos_rate < 0.3:  # Less than 30% positive
            logger.info("✅ Class balance IMPROVED! (Much better than original 80%+ positive)")
        else:
            logger.warning("⚠️ Still some class imbalance - might need further optimization")
        
        # Step 3: Quick model training (optional)
        if config.get('train_quick_model', False):
            logger.info("📊 Step 3: Training quick XGBoost model...")
            from model.spatial_temporal_trainer import SpatialTemporalTrainer
            
            trainer = SpatialTemporalTrainer()
            models = trainer.train_global_models(
                training_dataset_path,
                output_dir=config['output_dir']
            )
            
            logger.info(f"✅ Trained {len(models)} models")
        
        # Success summary
        total_time = datetime.now() - start_time
        logger.info("=" * 60)
        logger.info("🎉 FAST Spatial-Temporal System - SUCCESS!")
        logger.info(f"⏱️ Total time: {total_time.total_seconds():.1f} seconds")
        logger.info(f"📁 Output directory: {config['output_dir']}")
        logger.info("✅ Ready for full-scale processing with optimized parameters")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Error in fast spatial-temporal system: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    if success:
        print("\n🚀 Run full system with: python run_spatial_temporal_hab_system.py")
        print("🔧 Or customize parameters based on fast results")
    else:
        print("\n❌ Fast system failed - check logs for details")
