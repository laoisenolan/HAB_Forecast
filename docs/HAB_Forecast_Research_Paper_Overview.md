## HAB Forecasting System: Complete Research Paper Overview and Technical Index (2017–2025)

### How to use this document
- This document is designed as a comprehensive, research-ready reference for the full HAB Forecast project. It blends: problem context, system architecture, data sources and availability windows, pipeline designs, implementation details, modeling strategies, operationalization, evaluation, interpretability, uncertainty, automation, and a consolidated index of the major classes, functions, and run scripts.
- It is organized for skim-first reading with deep links and indexes so you can jump to the specific topic you need.

### Table of Contents
- [Executive Summary](#executive-summary)
- [Scientific Problem Context](#scientific-problem-context)
- [Geographic Scope and Objectives](#geographic-scope-and-objectives)
- [Major 2025 Breakthroughs (Project Updates)](#major-2025-breakthroughs-project-updates)
- [System Architecture](#system-architecture)
- [Data Sources and Availability Windows](#data-sources-and-availability-windows)
- [Historical Data Collection: Period-Aware Workflow](#historical-data-collection-period-aware-workflow)
- [Data Processing and Quality](#data-processing-and-quality)
- [Feature Engineering: Comprehensive Pipeline](#feature-engineering-comprehensive-pipeline)
- [Modeling Strategies](#modeling-strategies)
- [Spatial-Temporal Class Imbalance Solution](#spatial-temporal-class-imbalance-solution)
- [Deployment Options and Runtimes](#deployment-options-and-runtimes)
- [Production Pipeline and Dashboard](#production-pipeline-and-dashboard)
- [Uncertainty, Interpretability, Early Warning](#uncertainty-interpretability-early-warning)
- [Automation: Monitoring and Retraining](#automation-monitoring-and-retraining)
- [Evaluation and Results](#evaluation-and-results)
- [Reproducibility and Environment Setup](#reproducibility-and-environment-setup)
- [Configuration Files](#configuration-files)
- [Datasets and Outputs](#datasets-and-outputs)
- [Trials, Errors, and Fixes](#trials-errors-and-fixes)
- [Research Protocols and Methods](#research-protocols-and-methods)
- [Security, Ethics, and Operational Concerns](#security-ethics-and-operational-concerns)
- [Comprehensive Code Index](#comprehensive-code-index)
- [Appendices](#appendices)

---

## Executive Summary

- The HAB Forecast project is a fully operational spatiotemporal forecasting platform for Harmful Algal Bloom (HAB) risk in the Seto Inland Sea region, integrating multi-source environmental data and predicting risk for 24+ HAB species.
- The system achieves end-to-end automation: data collection, processing, feature engineering, model training, risk prediction, monitoring, interpretability, uncertainty quantification, and production deployment with a live dashboard.
- A new spatial-temporal approach solved the historical class imbalance (from 18.4% positive to ~1.7% positive), enabling robust training at scale.
- In August 2025, we completed a period-aware 7-year historical data collection system (2017–2024) with 28+ variables, integrating Google Earth Engine (GEE), Copernicus Marine, and ERDDAP with intelligent date-window selection and robust fallbacks.
- Three deployment options (Ultra-Fast Demo, Smart Sampling, Batch Processing) match different needs from quick tests to production-scale datasets (~12M records).

Key numbers
- Species coverage: 24+ species (expanded)
- Variables: 25 core environmental variables (28+ implemented)
- Historical coverage: 2017–2024 end-to-end; extended access for 1950+ (ERA5), 1992+ (HYCOM), 2016+ (S3), etc.
- Spatial grid: 5–20 km cells depending on path; cell-specific risk predictions
- Records: up to ~12M for production batch mode

---

## Scientific Problem Context

- HABs threaten marine ecosystems, fisheries, aquaculture, and coastal economies. Predicting HAB risk demands multi-source environmental monitoring plus species-specific modeling.
- Challenges:
  - Severe class imbalance in naïve temporal labeling (HAB anywhere in region on date X).
  - Multi-source data with different availability windows, formats, resolutions, and latency.
  - Robustness against outages, missing days, or satellite coverage gaps (e.g., clouds) and variable aliasing across providers.
  - Spatial heterogeneity: conditions vary significantly within the region; risk must be localized.

---

## Geographic Scope and Objectives

- Primary focus: Seto Inland Sea and connected marine sub-basins (e.g., Osaka Bay, Harimanada, Suo-nada, Iyo-nada, etc.).
- Objectives:
  - Build a production-grade HAB risk forecasting system integrating 25+ variables.
  - Provide species-specific risk forecasting at cell-level.
  - Ensure operational reliability, explainability, and uncertainty-aware early warning.

---

## Major 2025 Breakthroughs (Project Updates)

Context: The 2025 updates record critical engineering achievements that unlocked reliable historical data, robust feature pipelines, and production-ready ML.

- Period-aware 7-year historical collection (2017–2024) using correct sources by time window:
  - 2017–2021: GEE-focused (ERA5, HYCOM, Sentinel-2/3, MODIS, JAXA)
  - 2022–2024: GEE + Copernicus Marine + ERDDAP
- Availability-aware logic preventing invalid queries to sources not available in early years (e.g., Copernicus 2022-06+, ERDDAP 2022-12+).
- Robust fallbacks and error handling across key variables:
  - Wind: ERA5 Daily → ERA5 Hourly → GLDAS → synthetic fallback
  - Precipitation: CHIRPS → ERA5 → synthetic fallback
  - Sentinel-2 turbidity fixed via cloud filtering improvements (20% → 50%).
- HYCOM suite expanded (temp, salinity, velocity U/V, sea surface elevation) through GEE.
- CSV processing and wide/long master matrices fixed and consolidated with weekly batching to avoid I/O bottlenecks.

Reference: `PROJECT_UPDATES_2025.md` (extensive changelog and rationale including v4.0–v3.0 milestones, coverage metrics, and feature discoveries).

---

## System Architecture

- Layers
  - Data Collection: Copernicus Marine, GEE, ERDDAP, historical satellite integrators with chunking and caching.
  - Data Processing: Integration, QC, spatial/temporal processing, regridding.
  - Feature Engineering: Species-, spatial-, temporal-, interaction-, turbidity-, and biogeochemical features.
  - Modeling: Species-specific XGBoost, deep learning variants, ensembles, spatial-temporal trainers.
  - Production: Real-time pipeline, performance monitoring, automated retraining, dashboard and API.

- Main orchestrators
  - `src/pipeline/main_orchestrator.py::HABForecastOrchestrator` — end-to-end runner across collection → processing → features → training → prediction.
  - `src/production/production_pipeline.py::ProductionPipeline` — real-time operations (streaming, scheduled updates, monitoring, auto-updates).

---

## Data Sources and Availability Windows

- Google Earth Engine (GEE):
  - ERA5 (wind, precipitation; 1950–present availability when using climatology/extended sources)
  - HYCOM (1992–present; ocean physics — temperature, salinity, currents, elevation)
  - Sentinel-2/3 (turbidity, ocean color)
  - MODIS (ocean color, SST)
  - JAXA GCOM-C (chlorophyll)

- Copernicus Marine (2022-06+ in this pipeline usage):
  - Physics (temperature, salinity, currents)
  - Biogeochemistry (nitrate, phosphate, silicate, dissolved iron, oxygen, primary production)

- ERDDAP (2022-12+ for atmospheric backup):
  - GFS/atmospheric fallbacks

- Availability logic is enforced in the period-aware collector to avoid requests to unavailable sources and to choose the optimal source in each period.

---

## Historical Data Collection: Period-Aware Workflow

Primary script: `scripts/collect_period_aware_7_years.py`.

High-level flow
- Define two temporal regimes with correct availability windows:
  - Pre-Copernicus: 2017-01-01 to 2021-12-31
  - Comprehensive: 2022-01-01 to 2024-12-31
- For each regime:
  - Create quarterly chunks (~90 days) to avoid GEE element limits.
  - Iterate day-by-day inside each chunk; parallelize per-day GEE pulls for variable families.
  - Convert to CSV via `CSVDataProcessor.convert_gee_data_to_csv` and buffer weekly writes to a master long CSV with deduplication keys.
  - Update the wide feature matrix periodically to keep a ready-to-use training matrix.

Key implementation details
- GEE calls are done per variable family per day with a `ThreadPoolExecutor` at up to 5 workers (safe for GEE limits).
- HYCOM variables are collected via dedicated helpers that call GEE image collections and reduce statistics over the region.
- Weekly flushing balances I/O and robustness; remaining partial weeks get flushed at chunk boundaries.
- For 2022+ chunks, add Copernicus and ERDDAP with proper date checks.

Artifacts
- Master long CSV (historical): `data/historical_csv/historical_2017_2024_master.csv`
- Master wide CSV (training matrix): `data/csv/historical/historical_2017_2024_master_wide.csv`
- Period-specific per-source CSVs (optional; for traceability if `keep_per_source_csv=True`).

Supporting modules
- `src/data_collection/historical_collector.py` — end-to-end historical collection with chunking and reports.
- `src/data_collection/historical_satellite_integrator.py` — multi-satellite historical integration (HYCOM, Sentinel-3, extended atmospherics).
- `src/data_collection/copernicus_downloader.py` — Copernicus Marine pulls.
- `src/data_collection/erddap_downloader.py` — ERDDAP pulls.
- `src/data_collection/gee_integrator.py` — GEE variable families (wind, precipitation, chlorophyll, turbidity, SST, etc.).
- `src/data_collection/csv_data_processor.py` — standardization, CSV conversion, long/wide builders, deduplication.

---

## Data Processing and Quality

Responsibilities
- Integration of per-source datasets into unified time-indexed frames
- Quality control and sanity checks
- Spatial processing (e.g., grid mapping)
- Temporal processing (gap filling, rolling stats prerequisites)

Key modules
- `src/data_processing/data_integrator.py` — merges environmental streams
- `src/data_processing/quality_control.py` and `quality_controller.py` — validation & cleaning
- `src/data_processing/spatial_processor.py` and `regridder.py` — spatial workflows
- `src/data_processing/temporal_processor.py` — temporal workflows

Outputs feed directly into the feature engineering pipeline.

---

## Feature Engineering: Comprehensive Pipeline

Primary orchestrator: `src/feature_engineering/comprehensive_feature_pipeline.py::ComprehensiveHABFeaturePipeline`

Integrated modules
- Species-specific features: `advanced_species_features.py`
- Spatial features: `advanced_spatial_features.py`
- Temporal features: `advanced_temporal_features.py`
- Interaction features: `advanced_interaction_features.py`
- Turbidity features: `advanced_turbidity_features.py`
- Biogeochemical features: `advanced_biogeochemical_features.py`

Feature families (illustrative)
- Temporal: multi-scale seasonality; long-term trends; lags (1, 7, 14, 30, 60, 90, 180d); rolling windows; anomalies; rate-of-change.
- Biogeochemical: N:P:Si ratios, Redfield comparisons, nutrient limitation indices, stress markers.
- Interaction: multiplicative interactions (Temp × Chl), environmental ratios, stress indices, thresholds.
- Turbidity: KD490, RRS-derived turbidity, Sentinel-2 turbidity harmonization, sediment/runoff indicators, resuspension proxies.
- Spatial: gradients/fronts/upwelling signatures, neighborhood statistics, coastline proximity, bathymetric interactions (as available).
- Species-specific: environmental thresholds and suitability features per species.

Pipeline behavior
- Adds only new columns when merging module outputs to avoid overwriting base variables.
- Offers caching and selection mechanisms; supports real-time computation for production.

How the spatial-temporal dataset is built (conceptual flow used by all three runners):
1) Input A: HAB events CSV (`test_data/HAB_2017_2023.csv`) with `start_date`, `end_date`, `location`, `plankton_species`, etc.
2) Input B: Environmental wide CSV (`historical_2017_2024_master_wide.csv`) with rows for `(date, lat, lon)` and columns for variables.
3) `create_training_ready_dataset(...)` maps B into grid cells (by `grid_resolution`) and aligns with A to derive labels per grid cell per date (`has_hab` and optional `hab_species`, `hab_occurrence`, `hab_severity`).
4) Optionally derive additional features (temporal lags/rollings, spatial context) and save parquet/csv + a dataset metadata JSON.

---

## Modeling Strategies

Model families
- Species-specific XGBoost classifiers/regressors for cell-level risk.
- Deep learning architectures (LSTM/GRU/TCN) for sequence modeling where applicable (`src/model/deep_learning_hab_trainer.py`, `src/model/deep_learning_models.py`).
- Ensemble methods: performance-weighted ensembles and stacking (`src/model/ensemble_hab_trainer.py`, `src/model/ensemble_trainer.py`).
- Spatial-temporal trainer for global/regional/cell-specific models (`src/model/spatial_temporal_trainer.py`).

Core trainer components
- `src/model/species_specific_trainer.py::SpeciesSpecificXGBoostTrainer` — per-species model flows.
- `src/model/model_trainer.py::XGBoostTrainer` and `src/model/xgboost_trainer.py::XGBoostTrainer` — lower-level utilities (splits, scaling, feature selection, persistence, metadata).
- `src/model/advanced_optimizer.py`, `src/model/optuna_hab_optimizer.py` — hyperparameter tuning.
- `src/model/advanced_cross_validation.py` — robust CV strategies; temporal splits.

Production-focused logic
- `src/model/predictor.py::HABPredictor` — inference orchestration.
- Automated retraining and performance monitoring are detailed in the automation section.

How species-specific training works (from `src/model/species_specific_trainer.py`):
1) Load the training dataset (CSV) and optionally filter to target species.
2) For each species present:
   - Build a per-species dataset: positives are rows labeled with that `hab_species`; negatives are sampled from all HAB-negative rows, maintaining a 3:1 negative ratio.
   - Prepare features: drop non-predictive columns; convert to arrays; impute NaNs to zeros; log class distribution.
   - Split into train/test; scale features with `StandardScaler`.
   - Select top-k features using `SelectKBest(f_classif)` with species-specific k; record selected feature names.
   - Hyperparameter optimization via Optuna (if available) using stratified CV on AUC; otherwise GridSearchCV fallback.
   - Train final `XGBClassifier` with early stopping; compute metrics (accuracy, precision, recall, F1, AUC, avg precision).
   - Store feature importances; save `model.pkl`, `scaler.pkl`, `selector.pkl`, and `metadata.json` under `cache/species_models/<species>/`.
3) Generate an aggregate CSV report sorted by F1 score across trained species.

---

## Spatial-Temporal Class Imbalance Solution

Problem
- Original labeling treated a date as positive if any HAB event occurred anywhere, leading to ~18.4% positives — heavily imbalanced and prone to false learning signals.

Solution
- Convert the task to spatial-temporal prediction: risk at specific cell (lat, lon) at specific date.
- This reduces positive incidence to ~1.7% in the ultra-fast demo configuration, thereby stabilizing training and improving calibration.

Proof-of-concept
- `run_ultra_fast_demo.py` demonstrates the improvement in 30 seconds using a coarse grid (0.2°) and 1-year slice (2023).
- Expected outputs and balance metrics are documented in `README.md` and `SETUP_FOR_NEW_LAPTOP.md`.

---

## Deployment Options and Runtimes

Reference: `DEPLOYMENT_OPTIONS.md`

- Ultra-Fast Demo (30 seconds)
  - Goal: verify end-to-end functionality and class balance solution.
  - Data: 2023 only; 20 km grid; target ~100K records.
  - Run: `python run_ultra_fast_demo.py`
  - How it works (control flow from `run_ultra_fast_demo.py`):
    1) Load wide environmental matrix `data/csv/historical/historical_2017_2024_master_wide.csv`, filter to 2023.
    2) Save filtered CSV to `cache/ultra_fast_demo/environmental_data_2023.csv`.
    3) Call `feature_engineering.spatial_environmental_mapper.create_training_ready_dataset(...)` to map environmental rows to spatial grid cells and align with HAB events `test_data/HAB_2017_2023.csv` at coarse resolution (`grid_resolution=0.2`).
    4) The mapper emits a training dataset (parquet or csv) with columns like `date`, `grid_cell_id`, `latitude`, `longitude`, engineered features, and `has_hab` labels.
    5) The script loads the dataset and computes class balance metrics, grid coverage, elapsed time, and writes `ultra_fast_demo_results.json` to `cache/ultra_fast_demo/`.

- Smart Sampling (15 minutes)
  - Goal: high-quality development models without full processing.
  - Strategy: include 100% HAB cells + random sample of non-HAB cells; keep full temporal range.
  - Records: ~2M (subset), accuracy ~95% of full dataset.
  - Run: `python run_smart_sampling.py`
  - How it works (control flow from `run_smart_sampling.py`):
    1) Build a sampling strategy: use all dates (full coverage), include all grid cells with HAB events, include 30% of non-HAB cells, ensure regional balance, prefer coastal cells.
    2) Load `test_data/HAB_2017_2023.csv` and the wide environmental CSV; mark dates near HAB events (via `has_hab_nearby`).
    3) Filter 2020–2023; if the estimated records exceed target, keep all HAB dates and sample a subset of non-HAB dates to hit the target.
    4) Save sampled environmental data to `cache/smart_sampling/environmental_data_sampled.csv`.
    5) Invoke `create_training_ready_dataset(...)` with `grid_resolution=0.08` to map to 8 km cells; get a training parquet/csv.
    6) Load the dataset, compute class balance and coverage metrics (unique grid cells, dates, HAB dates covered), speed KPIs and a simple quality score; save `smart_sampling_results.json`.

- Batch Processing (1–2 hours)
  - Goal: production-scale dataset (~12M records) at ~5 km resolution.
  - Strategy: 84 monthly chunks (2017–2024), combine at end.
  - Run: `python run_batch_processing.py`
  - How it works (control flow from `run_batch_processing.py`):
    1) Create monthly date ranges across the full interval.
    2) For each month, filter the wide environmental CSV to the month and save `environmental_data.csv` into `cache/batch_processing/batches/YYYY-MM/`.
    3) Invoke `create_training_ready_dataset(...)` with `grid_resolution=0.05` to emit a batch training dataset; record batch-level stats (records, positives, positive rate) and path.
    4) After all batches, combine successful outputs: concatenate dataframes and save `final_training_dataset.parquet`; compute global metrics and write a `batch_processing_summary.json`.

---

## Production Pipeline and Dashboard

Production pipeline: `src/production/production_pipeline.py::ProductionPipeline`
- Responsibilities: initialize/load model, stream/process new data, generate predictions, monitor performance, schedule updates, handle errors and recovery.
- Config highlights: update frequency, prediction horizon, model-update thresholds, retention windows, alert thresholds, monitoring intervals.
- Methods include training a production model, evaluating, saving, and tracking performance history; enqueued prediction workflows.

Concrete production loop (what actually happens):
- Start: `start_production_pipeline()` loads/initializes model (`_initialize_model`), starts a monitoring thread (`_monitoring_loop`) and schedules updates (`_schedule_updates`).
- Monitoring loop pulls new data, runs the feature pipeline, predicts risk, logs processing time/quality, and appends to performance history; thresholds trigger alerts and optional retraining.
- `_train_production_model()` collects historical data, prepares feature/target sets, splits temporally, scales, selects features, trains an ensemble, evaluates on test, saves model, and records metrics.

Dashboard: `src/web/dashboard_app.py` with runner `src/web/run_dashboard.py`
- Web-based visualization for real-time risk, historical trends, performance monitoring, system status, and alerts.
- Backend integrates with `ProductionPipeline` and `AdvancedMonitor`, serves endpoints and WebSocket updates.
- Quick start: `python src/web/run_dashboard.py` then open `http://localhost:5000`.

API server (optional): `src/web/api_server.py`.

---

## Uncertainty, Interpretability, Early Warning

- Uncertainty quantification: `src/model/uncertainty_quantification.py` produces confidence intervals, predictive uncertainty distributions. Visualizations cached under `cache/uncertainty_viz/`.
- Interpretability: `src/model/shap_analyzer.py` and `src/model/shap_hab_analyzer.py` compute SHAP values and feature importances; cached examples in `cache/interpretability/`.
- Early warning: `src/model/hab_early_warning_system.py` generates lead-time alerts and monitoring signals.

---

## Automation: Monitoring and Retraining

- Automated retraining pipeline: `src/model/automated_retraining.py::AutomatedRetrainingPipeline`.
  - Tracks performance drift; retrains when thresholds are crossed.
  - Integrates with production pipeline scheduling to refresh models safely.
- Production monitoring: `src/production/advanced_monitor.py` and `src/model/performance_monitoring.py`.
  - Maintains performance history, data quality metrics, processing times, and alerting.

---

## Evaluation and Results

- The improved data balance and species-specific modeling raised stability and predictive power across species.
- Detailed evaluation metrics are output by training pipelines into JSON reports and logs (see `models/hab_models_2025/` and `models/conservative_hab_models/`).
- Feature importance analyses repeatedly highlighted wind patterns, chlorophyll variability, and nutrient extremes as top predictors (see `PROJECT_UPDATES_2025.md`).

---

## Reproducibility and Environment Setup

Minimal steps
- Install: `pip install -r requirements.txt`
- Set credentials and environment variables (see `docs/SETUP_GUIDE.md`, `SETUP_INSTRUCTIONS.md`, and `config/env_template.txt`).
- Quick validation: `python run_ultra_fast_demo.py`.

Full training pipeline
- `python run_complete_7_year_training_pipeline.py` — orchestrates collection (optional) + enhanced training.
- Alternative: `scripts/complete_hab_training_pipeline.py`, `scripts/enhanced_model_training_pipeline.py`, or `scripts/conservative_hab_training_pipeline.py`.

---

## Configuration Files

- `config/data_sources.yaml` — dataset catalogs, spatial bounds, temporal settings, chunk configs.
- `config/features.yaml` — feature pipeline toggles and selections.
- `config/model_params.yaml` — model hyperparameters and training options.
- `config/processing.yaml` — pre-processing and QC parameters.
- `config/env_template.txt` — environment variables template (GEE, Copernicus).

---

## Datasets and Outputs

Inputs
- HAB occurrences: `test_data/HAB_2017_2023.csv` (curated timeseries with species, durations, locations, cell counts, fishery impact flags).
- Environmental data: produced via historical collection and stored in long and wide matrices.

Outputs
- `data/historical_csv/historical_2017_2024_master.csv` (long)
- `data/csv/historical/historical_2017_2024_master_wide.csv` (wide)
- Models: `models/hab_models_2025/*`, `cache/species_models_improved/*`, `models/conservative_hab_models/*`
- Interpretability and uncertainty visualizations: `cache/interpretability/*`, `cache/uncertainty_viz/*`
- Ultra-fast demo cache: `cache/ultra_fast_demo/*`

---

## Trials, Errors, and Fixes

This section consolidates key “lessons learned” underlying the robust design, largely documented and validated in `PROJECT_UPDATES_2025.md`.

Data source challenges and resolutions
- Copernicus Marine API migration: switched to new library and dataset IDs; fixed ResponseSubset handling.
- GEE authentication failures: introduced new project and credential handling; made GEE optional to avoid pipeline failure.
- Sentinel-2 turbidity gaps: cloud filtering relaxed to 50%; increased image counts dramatically.
- ERA5 daily gaps: introduced ERA5 hourly fallback; further fallback to GLDAS or synthetic if required.
- Precipitation reliability: CHIRPS used as first choice with ERA5 fallback.

System constraints and architectural responses
- GEE 5000+ element limit: quarterly chunking (~90 days) plus day-wise parallelization with max_workers tuned (5).
- I/O overhead from many small CSVs: batched weekly writes to master long; periodic refresh of wide feature matrix; optional per-source CSV saved only for traceability when enabled.
- Data leakage in early ML iterations: removed leaked outcome-dependent features; standardized on 13 clean environmental predictors in core model sets; clean feature gates in pipeline.

Modeling evolution
- Early single-model approaches replaced with species-specific and spatially explicit models, then ensembles.
- Deep learning introduced for sequence awareness; applied selectively where data density supports it.
- Spatial-temporal framing addressed the core class imbalance problem; improved calibration and generalization.

Operations
- Added production monitoring, explicit thresholds for acceptable degradation, and automated retraining triggers.
- Implemented dashboard and API for stakeholder access; background updates set to 5-minute intervals for live visibility.

---

## Research Protocols and Methods

Labeling
- Extract HAB events from `test_data/HAB_2017_2023.csv`; map to grid cells and dates.
- Generate cell-level labels per date using event localization rules; handle multi-species events and overlapping durations.

Feature set partitions
- Strict separation between predictors and any outcome-derived constructs; no leakage.
- Use temporal-only predictors in some ablation runs to quantify relative contribution of spatial and biogeochemical features.

Validation
- Temporal splits for train/val/test; cross-validation folds across contiguous blocks to prevent leakage.
- Spatial blocking (optional) to avoid overfitting to fixed locations.

Metrics
- Classification metrics: F1, AUC, precision/recall, calibration curves.
- Operational metrics: lead-time utility, false-alarm cost, missed-event penalties.

Interpretability & uncertainty
- SHAP for feature attribution; partial dependence for sanity checks.
- Confidence intervals and predictive distributions used to modulate early warning thresholds.

Reproducibility
- Fix random seeds for model training; track versions and metadata with model artifacts (`.json` sidecars).

---

## Security, Ethics, and Operational Concerns

- Data access credentials (GEE, Copernicus) are scoped least-privilege and kept outside source control.
- The system avoids releasing exact fishery-sensitive coordinates publicly; aggregations can be applied.
- Early warnings should be framed probabilistically with uncertainty ranges to prevent overconfidence.

---

## Comprehensive Code Index

This index maps the principal modules, classes, functions, and run scripts. It is intended as a quick locator, not a replacement for inline docstrings.

Pipelines and Orchestration
- `src/pipeline/main_orchestrator.py`
  - Class: `HABForecastOrchestrator`
    - Methods: `run_complete_pipeline(start_date, end_date, force_retrain, generate_predictions)` → orchestrates data collection → processing → features → training → predictions.
    - Internals: `_collect_data`, `_process_data`, `_engineer_features`, `_train_model`, `...` (see file for full list).

Production
- `src/production/production_pipeline.py`
  - Class: `ProductionPipeline`
    - Methods: `start_production_pipeline`, `stop_production_pipeline`, `_initialize_model`, `_monitoring_loop`, `_schedule_updates`, `_train_production_model`, `...`
- `src/production/advanced_monitor.py`
  - Class: `AdvancedMonitor` (system KPIs, performance history, alerts)

Data Collection
- `src/data_collection/data_orchestrator.py`
  - Class: `DataCollectionOrchestrator` (dataset selection, parallel download management, caching, GEE availability handling)
- `src/data_collection/historical_collector.py`
  - Class: `HistoricalDataCollector`
  - Dataclass: `HistoricalCollectionPeriod`
  - Methods: `run_historical_collection`, `collect_historical_copernicus_data`, `collect_historical_gee_data`, `collect_historical_erddap_data`, `merge_historical_datasets`, `save_historical_data`, `create_time_chunks`, `create_historical_collection_report`, `save_historical_report`.
- `src/data_collection/historical_satellite_integrator.py`
  - Class: `HistoricalSatelliteIntegrator` (HYCOM, Sentinel-3, extended atmospherics)
- `src/data_collection/copernicus_downloader.py` — functions for `download_weekly_data(..., data_type)`, dataset handling.
- `src/data_collection/erddap_downloader.py` — functions for GFS pulls.
- `src/data_collection/gee_integrator.py` — functions: `get_wind_data`, `get_precipitation_data`, `get_chlorophyll_data`, `get_jaxa_chlorophyll_data`, `get_sentinel2_turbidity_data`, `get_sst_data`, `get_modis_ocean_data`.
- `src/data_collection/csv_data_processor.py` — methods: `convert_gee_data_to_csv`, `save_csv`, `append_to_master_long`, `build_wide_feature_matrix`.

Data Processing
- `src/data_processing/data_integrator.py` — dataset merging and alignment.
- `src/data_processing/quality_control.py`, `src/data_processing/quality_controller.py` — QC workflow.
- `src/data_processing/spatial_processor.py`, `src/data_processing/regridder.py` — spatial grids and regridding.
- `src/data_processing/temporal_processor.py` — temporal handling (resampling, gap filling).

Feature Engineering
- `src/feature_engineering/comprehensive_feature_pipeline.py`
  - Class: `ComprehensiveHABFeaturePipeline`
  - Methods: `process_environmental_data(...)` plus module integrations.
- `src/feature_engineering/advanced_species_features.py` — `AdvancedSpeciesFeatureEngineer`
- `src/feature_engineering/advanced_spatial_features.py` — `AdvancedSpatialFeatureEngineer`
- `src/feature_engineering/advanced_temporal_features.py` — `AdvancedTemporalFeatureEngineer`
- `src/feature_engineering/advanced_interaction_features.py` — `AdvancedInteractionFeatureEngineer`
- `src/feature_engineering/advanced_turbidity_features.py` — `AdvancedTurbidityFeatureEngineer`
- `src/feature_engineering/advanced_biogeochemical_features.py` — `AdvancedBiogeochemicalFeatureEngineer`
- `src/feature_engineering/feature_selector.py` — feature selection utilities
- `src/feature_engineering/feature_scaler.py` — scaling utilities

Modeling
- `src/model/species_specific_trainer.py` — `SpeciesSpecificXGBoostTrainer`
- `src/model/model_trainer.py` — `XGBoostTrainer` (splits, scaling, selection, save/load, metadata)
- `src/model/xgboost_trainer.py` — standalone trainer utilities
- `src/model/ensemble_hab_trainer.py`, `src/model/ensemble_trainer.py` — ensemble models and trainers
- `src/model/deep_learning_hab_trainer.py`, `src/model/deep_learning_models.py` — deep learning training and inference
- `src/model/advanced_cross_validation.py` — temporal CV and related validators
- `src/model/advanced_optimizer.py`, `src/model/optuna_hab_optimizer.py` — hyperparameter search
- `src/model/predictor.py` — `HABPredictor` for inference orchestration
- `src/model/performance_monitoring.py` — evaluation and KPI tracking
- `src/model/transfer_learning_hab.py`, `src/model/multitask_learning_hab.py` — advanced ML strategies
- `src/model/shap_analyzer.py`, `src/model/shap_hab_analyzer.py` — interpretability
- `src/model/uncertainty_quantification.py` — uncertainty estimates and intervals
- `src/model/hab_early_warning_system.py` — early warning logic
- `src/model/automated_retraining.py` — `AutomatedRetrainingPipeline`

Visualization and Web
- `src/visualization/dashboard.py`, `src/visualization/map_visualizer.py`, `src/visualization/time_series_plotter.py`, `src/visualization/shap_visualizer.py`
- `src/web/dashboard_app.py`, `src/web/run_dashboard.py`, `src/web/api_server.py`, `src/web/templates/dashboard.html`

Run Scripts
- `run_ultra_fast_demo.py` — 30-second end-to-end validation with coarse grid.
- `run_smart_sampling.py` — 15-minute development-grade dataset and models.
- `run_batch_processing.py` — 1–2 hour production dataset generation (monthly chunking).
- `run_spatial_temporal_hab_system.py`, `run_spatial_temporal_hab_system_FAST.py` — alternative runners.
- `run_complete_7_year_training_pipeline.py` — high-level orchestration (collection + training).
- `scripts/complete_hab_training_pipeline.py`, `scripts/enhanced_model_training_pipeline.py`, `scripts/conservative_hab_training_pipeline.py` — training flows.
- `scripts/collect_3_year_historical_data.py`, `scripts/collect_5_year_historical_data.py`, `scripts/collect_7_year_historical_data.py`, `scripts/collect_comprehensive_historical_data.py`, `scripts/collect_intelligent_7_year_data.py`, `scripts/collect_period_aware_7_years.py`, `scripts/collect_realistic_2022_2024_data.py` — historical collection recipes.

Utilities
- `src/utils/config_loader.py` — `ConfigLoader`, `get_data_dir` and config resolvers
- `src/data_collection/data_cache.py` — caching layer
- `src/data_collection/quality_monitor.py` — data quality monitor
- `src/data_collection/parallel_downloader.py` — parallelization helpers
- `src/data_collection/dataset_selector.py`, `src/data_collection/enhanced_dataset_selector.py` — intelligent dataset selection

---

## Appendices

Appendix A. Key Species (subset)
- Karenia mikimotoi
- Heterosigma akashiwo
- Chattonella antiqua / Chattonella spp.
- Cochlodinium polykrikoides
- Noctiluca scintillans
- Prorocentrum minimum
- Skeletonema spp.
- Mesodinium rubrum
- Others as cataloged in `species_models/` and occurrences CSV

Appendix B. Variables (illustrative)
- Atmospheric: wind u/v, precipitation, air temp, pressure (as available)
- Ocean physics: SST, subsurface temperature, salinity, currents u/v, sea surface elevation
- Ocean color/biogeochem: chlorophyll-a, KD490 turbidity, RRS bands, nutrients (NO3, PO4, Si, Fe), O2, primary production

Appendix C. File Structure Highlights
- See repository snapshot; outputs are cached under `cache/` and `models/*`, datasets under `data/`.

Appendix D. Glossary
- HAB: Harmful Algal Bloom
- GEE: Google Earth Engine
- ERDDAP: Environmental Research Division Data Access Program
- KD490: Diffuse attenuation coefficient at 490nm (turbidity proxy)
- SHAP: Shapley Additive Explanations

Appendix E. Troubleshooting
- GEE authentication: ensure correct service account credentials and environment variables (see setup docs).
- Copernicus API: validate current dataset IDs; check credentials in `.env`.
- Memory/time: choose appropriate deployment path; verify chunk sizes; increase grid cell size for fast tests.
- Missing data: inspect logs for fallback activations and gaps; adjust chunk windows if necessary.

---

End of document.

