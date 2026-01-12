"""
Deep Learning Models for HAB Forecasting

This module implements advanced deep learning models:
- LSTM/GRU for temporal pattern recognition
- CNN for spatial pattern analysis
- Transformer models for sequence modeling
- Hybrid models (CNN-LSTM)
"""

import sys
import os
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List, Union
import numpy as np
import pandas as pd
import xarray as xr
from datetime import datetime
import json
import pickle

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.utils.config_loader import ConfigLoader

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available. Deep learning models will not be available.")

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    logger.warning("TensorFlow not available. Some deep learning models will not be available.")

class HABDataset(Dataset):
    """Custom dataset for HAB forecasting"""
    
    def __init__(self, features: np.ndarray, targets: np.ndarray, sequence_length: int = 12):
        """
        Initialize dataset
        
        Args:
            features: Feature array (time, features)
            targets: Target array (time,)
            sequence_length: Length of temporal sequences
        """
        self.features = features
        self.targets = targets
        self.sequence_length = sequence_length
        
        # Create sequences
        self.sequences = []
        self.sequence_targets = []
        
        for i in range(len(features) - sequence_length):
            self.sequences.append(features[i:i+sequence_length])
            self.sequence_targets.append(targets[i+sequence_length])
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return torch.FloatTensor(self.sequences[idx]), torch.FloatTensor([self.sequence_targets[idx]])

class LSTMModel(nn.Module):
    """LSTM model for temporal pattern recognition"""
    
    def __init__(self, input_size: int, hidden_size: int = 64, num_layers: int = 2, dropout: float = 0.2):
        super(LSTMModel, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, 1)
        
    def forward(self, x):
        # x shape: (batch_size, sequence_length, input_size)
        lstm_out, _ = self.lstm(x)
        
        # Take the last output
        last_output = lstm_out[:, -1, :]
        
        # Apply dropout and final layer
        out = self.dropout(last_output)
        out = self.fc(out)
        
        return out

class CNNModel(nn.Module):
    """CNN model for spatial pattern analysis"""
    
    def __init__(self, input_channels: int, spatial_size: int, num_filters: int = 32, dropout: float = 0.2):
        super(CNNModel, self).__init__()
        
        self.conv1 = nn.Conv2d(input_channels, num_filters, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(num_filters, num_filters * 2, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(num_filters * 2, num_filters * 4, kernel_size=3, padding=1)
        
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(dropout)
        
        # Calculate flattened size
        flattened_size = (spatial_size // 8) ** 2 * (num_filters * 4)
        
        self.fc1 = nn.Linear(flattened_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)
        
        self.relu = nn.ReLU()
        
    def forward(self, x):
        # x shape: (batch_size, channels, height, width)
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        
        x = x.view(x.size(0), -1)  # Flatten
        
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.dropout(self.relu(self.fc2(x)))
        x = self.fc3(x)
        
        return x

class CNNLSTMModel(nn.Module):
    """Hybrid CNN-LSTM model for spatio-temporal patterns"""
    
    def __init__(self, input_channels: int, spatial_size: int, sequence_length: int, 
                 num_filters: int = 32, lstm_hidden: int = 64, dropout: float = 0.2):
        super(CNNLSTMModel, self).__init__()
        
        # CNN layers for spatial feature extraction
        self.conv1 = nn.Conv2d(input_channels, num_filters, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(num_filters, num_filters * 2, kernel_size=3, padding=1)
        
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(dropout)
        
        # Calculate CNN output size
        cnn_output_size = (spatial_size // 4) ** 2 * (num_filters * 2)
        
        # LSTM layers for temporal modeling
        self.lstm = nn.LSTM(
            input_size=cnn_output_size,
            hidden_size=lstm_hidden,
            num_layers=2,
            dropout=dropout,
            batch_first=True
        )
        
        self.fc1 = nn.Linear(lstm_hidden, 64)
        self.fc2 = nn.Linear(64, 1)
        
        self.relu = nn.ReLU()
        
    def forward(self, x):
        # x shape: (batch_size, sequence_length, channels, height, width)
        batch_size, seq_len, channels, height, width = x.size()
        
        # Reshape for CNN processing
        x = x.view(batch_size * seq_len, channels, height, width)
        
        # Apply CNN
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        
        # Flatten CNN output
        x = x.view(batch_size, seq_len, -1)
        
        # Apply LSTM
        lstm_out, _ = self.lstm(x)
        
        # Take last output
        last_output = lstm_out[:, -1, :]
        
        # Final layers
        out = self.dropout(self.relu(self.fc1(last_output)))
        out = self.fc2(out)
        
        return out

class DeepLearningTrainer:
    """Deep learning model trainer"""
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize the deep learning trainer"""
        self.config_loader = ConfigLoader()
        self.config = config or self.config_loader.get_model_params_config()
        
        # Deep learning configuration
        self.dl_config = {
            "lstm": {
                "hidden_size": 64,
                "num_layers": 2,
                "dropout": 0.2,
                "learning_rate": 0.001,
                "batch_size": 32,
                "epochs": 100,
                "sequence_length": 12
            },
            "cnn": {
                "num_filters": 32,
                "dropout": 0.2,
                "learning_rate": 0.001,
                "batch_size": 16,
                "epochs": 100
            },
            "cnn_lstm": {
                "num_filters": 32,
                "lstm_hidden": 64,
                "dropout": 0.2,
                "learning_rate": 0.001,
                "batch_size": 16,
                "epochs": 100,
                "sequence_length": 12
            }
        }
        
        self.models = {}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for deep learning models")
    
    def prepare_sequence_data(self, features_df: pd.DataFrame, target_series: pd.Series, 
                            sequence_length: int = 12) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare sequence data for deep learning models"""
        try:
            # Convert to numpy arrays
            features = features_df.values
            targets = target_series.values
            
            # Normalize features
            feature_mean = np.mean(features, axis=0)
            feature_std = np.std(features, axis=0)
            features_normalized = (features - feature_mean) / (feature_std + 1e-8)
            
            # Normalize targets
            target_mean = np.mean(targets)
            target_std = np.std(targets)
            targets_normalized = (targets - target_mean) / (target_std + 1e-8)
            
            logger.info(f"Prepared sequence data: {features_normalized.shape}, {targets_normalized.shape}")
            
            return features_normalized, targets_normalized, {
                "feature_mean": feature_mean,
                "feature_std": feature_std,
                "target_mean": target_mean,
                "target_std": target_std
            }
            
        except Exception as e:
            logger.error(f"Sequence data preparation failed: {e}")
            raise
    
    def train_lstm_model(self, features_df: pd.DataFrame, target_series: pd.Series) -> Dict[str, Any]:
        """Train LSTM model"""
        try:
            logger.info("Training LSTM model")
            
            # Prepare data
            features, targets, normalization = self.prepare_sequence_data(
                features_df, target_series, self.dl_config["lstm"]["sequence_length"]
            )
            
            # Create dataset
            dataset = HABDataset(features, targets, self.dl_config["lstm"]["sequence_length"])
            dataloader = DataLoader(dataset, batch_size=self.dl_config["lstm"]["batch_size"], shuffle=True)
            
            # Initialize model
            model = LSTMModel(
                input_size=features.shape[1],
                hidden_size=self.dl_config["lstm"]["hidden_size"],
                num_layers=self.dl_config["lstm"]["num_layers"],
                dropout=self.dl_config["lstm"]["dropout"]
            ).to(self.device)
            
            # Loss and optimizer
            criterion = nn.MSELoss()
            optimizer = optim.Adam(model.parameters(), lr=self.dl_config["lstm"]["learning_rate"])
            
            # Training loop
            model.train()
            train_losses = []
            
            for epoch in range(self.dl_config["lstm"]["epochs"]):
                epoch_loss = 0
                for batch_features, batch_targets in dataloader:
                    batch_features = batch_features.to(self.device)
                    batch_targets = batch_targets.to(self.device)
                    
                    optimizer.zero_grad()
                    outputs = model(batch_features)
                    loss = criterion(outputs, batch_targets)
                    loss.backward()
                    optimizer.step()
                    
                    epoch_loss += loss.item()
                
                avg_loss = epoch_loss / len(dataloader)
                train_losses.append(avg_loss)
                
                if epoch % 20 == 0:
                    logger.info(f"Epoch {epoch}, Loss: {avg_loss:.6f}")
            
            # Store model and metadata
            self.models["lstm"] = {
                "model": model,
                "normalization": normalization,
                "config": self.dl_config["lstm"],
                "train_losses": train_losses
            }
            
            logger.info("LSTM model training complete")
            return {
                "model": model,
                "normalization": normalization,
                "train_losses": train_losses,
                "final_loss": train_losses[-1]
            }
            
        except Exception as e:
            logger.error(f"LSTM training failed: {e}")
            raise
    
    def train_cnn_model(self, features_df: pd.DataFrame, target_series: pd.Series, 
                       spatial_shape: Tuple[int, int]) -> Dict[str, Any]:
        """Train CNN model for spatial patterns"""
        try:
            logger.info("Training CNN model")
            
            # Prepare data
            features, targets, normalization = self.prepare_sequence_data(features_df, target_series)
            
            # Reshape for CNN (batch, channels, height, width)
            # Assuming features are spatial data
            batch_size = len(features)
            channels = 1  # Single channel for now
            height, width = spatial_shape
            
            # Reshape features to spatial format
            features_spatial = features.reshape(batch_size, channels, height, width)
            
            # Create dataset
            dataset = list(zip(features_spatial, targets))
            dataloader = DataLoader(dataset, batch_size=self.dl_config["cnn"]["batch_size"], shuffle=True)
            
            # Initialize model
            model = CNNModel(
                input_channels=channels,
                spatial_size=height,
                num_filters=self.dl_config["cnn"]["num_filters"],
                dropout=self.dl_config["cnn"]["dropout"]
            ).to(self.device)
            
            # Loss and optimizer
            criterion = nn.MSELoss()
            optimizer = optim.Adam(model.parameters(), lr=self.dl_config["cnn"]["learning_rate"])
            
            # Training loop
            model.train()
            train_losses = []
            
            for epoch in range(self.dl_config["cnn"]["epochs"]):
                epoch_loss = 0
                for batch_features, batch_targets in dataloader:
                    batch_features = torch.FloatTensor(batch_features).to(self.device)
                    batch_targets = torch.FloatTensor(batch_targets).unsqueeze(1).to(self.device)
                    
                    optimizer.zero_grad()
                    outputs = model(batch_features)
                    loss = criterion(outputs, batch_targets)
                    loss.backward()
                    optimizer.step()
                    
                    epoch_loss += loss.item()
                
                avg_loss = epoch_loss / len(dataloader)
                train_losses.append(avg_loss)
                
                if epoch % 20 == 0:
                    logger.info(f"Epoch {epoch}, Loss: {avg_loss:.6f}")
            
            # Store model and metadata
            self.models["cnn"] = {
                "model": model,
                "normalization": normalization,
                "config": self.dl_config["cnn"],
                "train_losses": train_losses,
                "spatial_shape": spatial_shape
            }
            
            logger.info("CNN model training complete")
            return {
                "model": model,
                "normalization": normalization,
                "train_losses": train_losses,
                "final_loss": train_losses[-1]
            }
            
        except Exception as e:
            logger.error(f"CNN training failed: {e}")
            raise
    
    def predict(self, model_name: str, features: np.ndarray) -> np.ndarray:
        """Generate predictions using trained model"""
        try:
            if model_name not in self.models:
                raise ValueError(f"Model {model_name} not found")
            
            model_info = self.models[model_name]
            model = model_info["model"]
            normalization = model_info["normalization"]
            
            # Normalize features
            features_normalized = (features - normalization["feature_mean"]) / (normalization["feature_std"] + 1e-8)
            
            # Convert to tensor
            if model_name == "lstm":
                # For LSTM, create sequences
                sequence_length = model_info["config"]["sequence_length"]
                sequences = []
                for i in range(len(features_normalized) - sequence_length):
                    sequences.append(features_normalized[i:i+sequence_length])
                
                if not sequences:
                    return np.array([])
                
                features_tensor = torch.FloatTensor(sequences).to(self.device)
                
            elif model_name == "cnn":
                # For CNN, reshape to spatial format
                spatial_shape = model_info["spatial_shape"]
                features_tensor = torch.FloatTensor(features_normalized.reshape(-1, 1, *spatial_shape)).to(self.device)
            
            else:
                raise ValueError(f"Unknown model type: {model_name}")
            
            # Generate predictions
            model.eval()
            with torch.no_grad():
                predictions_normalized = model(features_tensor).cpu().numpy()
            
            # Denormalize predictions
            predictions = predictions_normalized * normalization["target_std"] + normalization["target_mean"]
            
            # Clip to valid range
            predictions = np.clip(predictions, 0, 100)
            
            return predictions.flatten()
            
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            raise
    
    def save_model(self, model_name: str, filepath: str):
        """Save trained model"""
        try:
            if model_name not in self.models:
                raise ValueError(f"Model {model_name} not found")
            
            model_data = {
                "model_state": self.models[model_name]["model"].state_dict(),
                "normalization": self.models[model_name]["normalization"],
                "config": self.models[model_name]["config"],
                "train_losses": self.models[model_name]["train_losses"],
                "model_name": model_name,
                "timestamp": datetime.now().isoformat()
            }
            
            torch.save(model_data, filepath)
            logger.info(f"Model {model_name} saved to {filepath}")
            
        except Exception as e:
            logger.error(f"Model saving failed: {e}")
            raise
    
    def load_model(self, model_name: str, filepath: str):
        """Load trained model"""
        try:
            model_data = torch.load(filepath, map_location=self.device)
            
            # Initialize model based on type
            if model_name == "lstm":
                model = LSTMModel(
                    input_size=len(model_data["normalization"]["feature_mean"]),
                    hidden_size=model_data["config"]["hidden_size"],
                    num_layers=model_data["config"]["num_layers"],
                    dropout=model_data["config"]["dropout"]
                )
            elif model_name == "cnn":
                spatial_shape = model_data.get("spatial_shape", (10, 10))
                model = CNNModel(
                    input_channels=1,
                    spatial_size=spatial_shape[0],
                    num_filters=model_data["config"]["num_filters"],
                    dropout=model_data["config"]["dropout"]
                )
            else:
                raise ValueError(f"Unknown model type: {model_name}")
            
            # Load state dict
            model.load_state_dict(model_data["model_state"])
            model.to(self.device)
            
            # Store in models dict
            self.models[model_name] = {
                "model": model,
                "normalization": model_data["normalization"],
                "config": model_data["config"],
                "train_losses": model_data["train_losses"]
            }
            
            logger.info(f"Model {model_name} loaded from {filepath}")
            
        except Exception as e:
            logger.error(f"Model loading failed: {e}")
            raise 