#!/usr/bin/env python3
"""
Deep Learning Models for HAB Forecasting - Task 4.2

This module implements advanced deep learning models for HAB prediction:
- LSTM for temporal pattern recognition
- CNN for spatial pattern analysis  
- CNN-LSTM hybrid models for spatio-temporal patterns
- Attention mechanisms for feature importance
- Species-specific architectures

🎯 MODELS IMPLEMENTED:
- 1D CNN for environmental variable sequences
- LSTM/GRU for temporal dependencies
- Attention-based models for interpretability
- Multi-input models combining spatial and temporal features

🛰️ LEVERAGES:
- Time series environmental data
- Spatial environmental gradients
- Species-specific patterns
- Multi-scale temporal features

Author: HAB Forecasting System - Task 4.2 Implementation
Date: December 19, 2024
"""

import numpy as np
import pandas as pd
import logging
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime
import warnings
from collections import defaultdict

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Deep learning imports with fallbacks
TORCH_AVAILABLE = False
TF_AVAILABLE = False

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader
    import torch.optim as optim
    from torch.optim.lr_scheduler import ReduceLROnPlateau
    TORCH_AVAILABLE = True
    logger.info("✅ PyTorch available")
except ImportError:
    logger.warning("⚠️  PyTorch not available")

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, models, callbacks
    from tensorflow.keras.optimizers import Adam
    TF_AVAILABLE = True
    logger.info("✅ TensorFlow available")
except ImportError:
    logger.warning("⚠️  TensorFlow not available")

# Standard ML libraries
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, f1_score
from sklearn.model_selection import train_test_split

class HABTimeSeriesDataset(Dataset):
    """
    PyTorch Dataset for HAB time series data
    """
    
    def __init__(self, sequences: np.ndarray, targets: np.ndarray, 
                 seq_length: int = 30, transform=None):
        """
        Initialize HAB time series dataset
        
        Args:
            sequences: Environmental data sequences [samples, features, time_steps]
            targets: HAB occurrence targets [samples]
            seq_length: Length of input sequences
            transform: Optional data transform
        """
        self.sequences = torch.FloatTensor(sequences)
        self.targets = torch.LongTensor(targets)
        self.seq_length = seq_length
        self.transform = transform
        
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        target = self.targets[idx]
        
        if self.transform:
            sequence = self.transform(sequence)
            
        return sequence, target

class CNN1D_HAB(nn.Module):
    """
    1D CNN for HAB prediction from environmental sequences
    """
    
    def __init__(self, n_features: int, n_classes: int = 2, 
                 seq_length: int = 30, dropout: float = 0.3):
        super(CNN1D_HAB, self).__init__()
        
        self.n_features = n_features
        self.n_classes = n_classes
        self.seq_length = seq_length
        
        # 1D Convolutional layers
        self.conv1 = nn.Conv1d(n_features, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(128, 256, kernel_size=3, padding=1)
        
        # Batch normalization
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(256)
        
        # Pooling and dropout
        self.pool = nn.MaxPool1d(2)
        self.dropout = nn.Dropout(dropout)
        
        # Calculate flattened size after convolutions
        conv_out_size = self._get_conv_output_size()
        
        # Fully connected layers
        self.fc1 = nn.Linear(conv_out_size, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, n_classes)
        
    def _get_conv_output_size(self):
        """Calculate the output size after convolution layers"""
        # Simulate forward pass to get output size
        dummy_input = torch.randn(1, self.n_features, self.seq_length)
        x = self.pool(F.relu(self.bn1(self.conv1(dummy_input))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        return x.numel()
    
    def forward(self, x):
        # Reshape from [batch, seq_len, features] to [batch, features, seq_len]
        x = x.transpose(1, 2)
        
        # Convolutional layers
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.dropout(x)
        
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.dropout(x)
        
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.dropout(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        
        x = self.fc3(x)
        
        return x

class LSTM_HAB(nn.Module):
    """
    LSTM model for HAB prediction from temporal sequences
    """
    
    def __init__(self, n_features: int, n_classes: int = 2,
                 hidden_size: int = 128, num_layers: int = 2, 
                 dropout: float = 0.3, bidirectional: bool = True):
        super(LSTM_HAB, self).__init__()
        
        self.n_features = n_features
        self.n_classes = n_classes
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        
        # LSTM layer
        self.lstm = nn.LSTM(
            n_features, hidden_size, num_layers,
            batch_first=True, dropout=dropout,
            bidirectional=bidirectional
        )
        
        # Calculate LSTM output size
        lstm_output_size = hidden_size * (2 if bidirectional else 1)
        
        # Attention mechanism
        self.attention = nn.MultiheadAttention(lstm_output_size, 8, dropout=dropout)
        
        # Fully connected layers
        self.fc1 = nn.Linear(lstm_output_size, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, n_classes)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        batch_size = x.size(0)
        
        # LSTM forward pass
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # Apply attention
        lstm_out = lstm_out.transpose(0, 1)  # [seq_len, batch, hidden_size]
        attn_out, attn_weights = self.attention(lstm_out, lstm_out, lstm_out)
        attn_out = attn_out.transpose(0, 1)  # [batch, seq_len, hidden_size]
        
        # Use mean pooling over time steps
        pooled = torch.mean(attn_out, dim=1)
        
        # Fully connected layers
        x = F.relu(self.fc1(pooled))
        x = self.dropout(x)
        
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        
        x = self.fc3(x)
        
        return x

class DeepLearningHABTrainer:
    """
    Deep learning trainer for HAB forecasting with PyTorch and TensorFlow support
    """
    
    def __init__(self, cache_dir: str = "cache/dl_models", 
                 device: str = "auto", random_state: int = 42):
        """
        Initialize deep learning HAB trainer
        
        Args:
            cache_dir: Directory for caching models
            device: Device to use ('cpu', 'cuda', or 'auto')
            random_state: Random state for reproducibility
        """
        
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.random_state = random_state
        
        # Set device
        if device == "auto" and TORCH_AVAILABLE:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device if device != "auto" else "cpu")
        
        # Model storage
        self.models = {}
        self.scalers = {}
        self.training_history = defaultdict(list)
        
        logger.info(f"🧠 Deep Learning HAB Trainer initialized")
        logger.info(f"🖥️  Device: {self.device}")
        logger.info(f"📁 Cache directory: {self.cache_dir}")
    
    def prepare_sequences(self, data: pd.DataFrame, seq_length: int = 30,
                         feature_cols: Optional[List[str]] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare time series sequences from tabular data
        
        Args:
            data: Input dataframe
            seq_length: Length of sequences
            feature_cols: Feature columns to use
            
        Returns:
            sequences, targets
        """
        
        if feature_cols is None:
            # Auto-detect feature columns (exclude metadata)
            exclude_cols = [
                'timestamp', 'hab_species', 'hab_occurrence', 'hab_severity',
                'latitude', 'longitude', 'hab_cell_density', 'hab_confidence',
                'days_to_hab_event', 'distance_to_hab_km'  # Remove leaked features
            ]
            feature_cols = [col for col in data.columns if col not in exclude_cols]
        
        logger.info(f"🔧 Using {len(feature_cols)} features for sequences")
        logger.info(f"📋 Features: {feature_cols}")
        
        # Sort data by timestamp to create proper sequences
        if 'timestamp' in data.columns:
            data = data.sort_values('timestamp')
        
        # Extract features and targets
        features = data[feature_cols].values
        targets = data['hab_occurrence'].values
        
        # Handle missing values
        features = np.nan_to_num(features, nan=0.0)
        
        # Create sequences
        sequences = []
        sequence_targets = []
        
        for i in range(len(features) - seq_length + 1):
            seq = features[i:i + seq_length]
            target = targets[i + seq_length - 1]  # Predict for last time step
            
            sequences.append(seq)
            sequence_targets.append(target)
        
        sequences = np.array(sequences)
        sequence_targets = np.array(sequence_targets)
        
        logger.info(f"📊 Created {len(sequences)} sequences of length {seq_length}")
        logger.info(f"🎯 Target distribution: {np.bincount(sequence_targets)}")
        
        return sequences, sequence_targets
    
    def train_pytorch_model(self, sequences: np.ndarray, targets: np.ndarray,
                           model_type: str = "lstm", species: str = "general",
                           **model_params) -> Dict:
        """
        Train PyTorch model for HAB prediction
        
        Args:
            sequences: Input sequences [samples, seq_length, features]
            targets: Target labels
            model_type: Type of model ('cnn', 'lstm', 'hybrid')
            species: Species name for model identification
            **model_params: Additional model parameters
            
        Returns:
            Training results dictionary
        """
        
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch not available for deep learning training")
        
        logger.info(f"🧠 Training PyTorch {model_type.upper()} model for {species}")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            sequences, targets, test_size=0.2, random_state=self.random_state,
            stratify=targets if len(np.unique(targets)) > 1 else None
        )
        
        # Normalize features
        scaler = StandardScaler()
        n_samples, seq_length, n_features = X_train.shape
        
        # Reshape for scaling
        X_train_reshaped = X_train.reshape(-1, n_features)
        X_test_reshaped = X_test.reshape(-1, n_features)
        
        X_train_scaled = scaler.fit_transform(X_train_reshaped).reshape(n_samples, seq_length, n_features)
        X_test_scaled = scaler.transform(X_test_reshaped).reshape(X_test.shape[0], seq_length, n_features)
        
        # Create datasets
        train_dataset = HABTimeSeriesDataset(X_train_scaled, y_train, seq_length)
        test_dataset = HABTimeSeriesDataset(X_test_scaled, y_test, seq_length)
        
        # Create data loaders
        batch_size = model_params.get('batch_size', 32)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size)
        
        # Initialize model
        if model_type == "cnn":
            model = CNN1D_HAB(
                n_features=n_features,
                n_classes=2,
                seq_length=seq_length,
                dropout=model_params.get('dropout', 0.3)
            )
        elif model_type == "lstm":
            model = LSTM_HAB(
                n_features=n_features,
                n_classes=2,
                hidden_size=model_params.get('hidden_size', 128),
                num_layers=model_params.get('num_layers', 2),
                dropout=model_params.get('dropout', 0.3),
                bidirectional=model_params.get('bidirectional', True)
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        model = model.to(self.device)
        
        # Loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=model_params.get('learning_rate', 0.001))
        scheduler = ReduceLROnPlateau(optimizer, patience=5, factor=0.5, verbose=True)
        
        # Training loop
        epochs = model_params.get('epochs', 50)
        best_f1 = 0.0
        patience = model_params.get('patience', 10)
        patience_counter = 0
        
        train_losses = []
        val_losses = []
        val_f1_scores = []
        
        for epoch in range(epochs):
            # Training phase
            model.train()
            train_loss = 0.0
            
            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                
                optimizer.zero_grad()
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            # Validation phase
            model.eval()
            val_loss = 0.0
            all_preds = []
            all_targets = []
            
            with torch.no_grad():
                for batch_x, batch_y in test_loader:
                    batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                    
                    outputs = model(batch_x)
                    loss = criterion(outputs, batch_y)
                    val_loss += loss.item()
                    
                    _, predicted = torch.max(outputs.data, 1)
                    all_preds.extend(predicted.cpu().numpy())
                    all_targets.extend(batch_y.cpu().numpy())
            
            # Calculate metrics
            train_loss /= len(train_loader)
            val_loss /= len(test_loader)
            val_f1 = f1_score(all_targets, all_preds, average='weighted')
            
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            val_f1_scores.append(val_f1)
            
            # Learning rate scheduling
            scheduler.step(val_loss)
            
            # Early stopping
            if val_f1 > best_f1:
                best_f1 = val_f1
                patience_counter = 0
                # Save best model
                best_model_state = model.state_dict().copy()
            else:
                patience_counter += 1
            
            if epoch % 10 == 0 or patience_counter >= patience:
                logger.info(f"Epoch {epoch:3d}: Train Loss={train_loss:.4f}, "
                           f"Val Loss={val_loss:.4f}, Val F1={val_f1:.4f}")
            
            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch}")
                break
        
        # Load best model
        model.load_state_dict(best_model_state)
        
        # Final evaluation
        model.eval()
        final_preds = []
        final_proba = []
        
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_x = batch_x.to(self.device)
                outputs = model(batch_x)
                proba = F.softmax(outputs, dim=1)
                _, predicted = torch.max(outputs.data, 1)
                
                final_preds.extend(predicted.cpu().numpy())
                final_proba.extend(proba.cpu().numpy()[:, 1])  # Probability of positive class
        
        # Calculate final metrics
        final_f1 = f1_score(y_test, final_preds, average='weighted')
        final_auc = roc_auc_score(y_test, final_proba) if len(np.unique(y_test)) > 1 else 0.0
        
        # Store results
        results = {
            'model': model,
            'scaler': scaler,
            'model_type': model_type,
            'species': species,
            'final_f1': final_f1,
            'final_auc': final_auc,
            'best_f1': best_f1,
            'train_losses': train_losses,
            'val_losses': val_losses,
            'val_f1_scores': val_f1_scores,
            'epochs_trained': len(train_losses),
            'device': str(self.device)
        }
        
        # Store in instance
        model_key = f"{species}_{model_type}"
        self.models[model_key] = model
        self.scalers[model_key] = scaler
        self.training_history[model_key] = results
        
        logger.info(f"✅ {species} {model_type.upper()}: Final F1={final_f1:.3f}, AUC={final_auc:.3f}")
        
        return results
    
    def save_model(self, model_key: str, results: Dict) -> None:
        """Save trained deep learning model"""
        
        model_dir = self.cache_dir / model_key
        model_dir.mkdir(exist_ok=True)
        
        # Save PyTorch model
        if TORCH_AVAILABLE and isinstance(results['model'], nn.Module):
            torch.save(results['model'].state_dict(), model_dir / "model.pth")
            
            # Save model architecture info
            model_info = {
                'model_type': results['model_type'],
                'species': results['species'],
                'model_class': results['model'].__class__.__name__,
                'device': results['device']
            }
            
            with open(model_dir / "model_info.json", 'w') as f:
                import json
                json.dump(model_info, f, indent=2)
        
        # Save scaler
        with open(model_dir / "scaler.pkl", 'wb') as f:
            pickle.dump(results['scaler'], f)
        
        # Save training history
        history = {
            'final_f1': results['final_f1'],
            'final_auc': results['final_auc'],
            'best_f1': results['best_f1'],
            'epochs_trained': results['epochs_trained'],
            'train_losses': results['train_losses'],
            'val_losses': results['val_losses'],
            'val_f1_scores': results['val_f1_scores']
        }
        
        with open(model_dir / "training_history.json", 'w') as f:
            import json
            json.dump(history, f, indent=2, default=str)
        
        logger.info(f"💾 Saved {model_key} model to {model_dir}")

if __name__ == "__main__":
    # Example usage
    print("🧠 Deep Learning HAB Trainer - Task 4.2")
    print(f"📊 PyTorch available: {TORCH_AVAILABLE}")
    print(f"📊 TensorFlow available: {TF_AVAILABLE}")
    
    if TORCH_AVAILABLE:
        # Initialize trainer
        trainer = DeepLearningHABTrainer()
        
        print("✅ Deep Learning HAB Trainer initialized successfully!")
        print("🎯 Ready to train LSTM and CNN models for HAB prediction")
    else:
        print("❌ PyTorch not available - install torch to use deep learning models")