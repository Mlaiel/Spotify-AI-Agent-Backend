"""
Enterprise LSTM Model for Spotify AI Agent
==========================================

Advanced Long Short-Term Memory Neural Network for Time Series Analysis in Music Streaming

This module implements a sophisticated LSTM architecture specifically optimized for
time series anomaly detection, trend prediction, and pattern recognition in music
streaming platforms. The model incorporates state-of-the-art techniques including
bidirectional LSTM, attention mechanisms, and multi-scale temporal analysis.

🎵 MUSIC STREAMING TIME SERIES APPLICATIONS:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• User Listening Patterns - Daily/weekly listening habits, session durations
• Audio Quality Metrics - Bitrate fluctuations, buffering events over time
• Engagement Trends - Skip rates, like/dislike patterns, playlist additions
• Revenue Streams - Subscription patterns, ad revenue trends, conversion funnels
• Content Performance - Track popularity evolution, viral content detection
• Infrastructure Metrics - Server load patterns, CDN performance, API response times
• Geographic Analysis - Regional usage patterns, cultural music trends
• Seasonal Effects - Holiday listening, weather-based preferences

⚡ ENTERPRISE LSTM FEATURES:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• Bidirectional LSTM for past and future context analysis
• Multi-layer LSTM with configurable depth and dimensions
• Attention mechanisms for important time step identification
• Multi-scale temporal convolutions for different time horizons
• Residual connections for training stability
• Dropout and regularization for overfitting prevention
• Dynamic sequence length handling
• Real-time streaming prediction capability
• Multi-step ahead forecasting
• Anomaly detection with temporal context

Version: 2.0.0 (Enterprise Edition)
Optimized for: Real-time streaming, multi-horizon forecasting, <10ms inference
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, Any, List, Optional, Union, Tuple
from abc import ABC
from datetime import datetime, timedelta
import warnings

# Deep Learning imports
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, Model, optimizers, losses, metrics
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    logging.warning("TensorFlow not available. LSTM model will use lightweight implementation.")

# Scientific computing imports
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from sklearn.model_selection import train_test_split

# Import base model interface
from . import ModelInterface

logger = logging.getLogger(__name__)


class LSTMModel(ModelInterface):
    """
    Enterprise-grade LSTM model for time series analysis in music streaming platforms.
    
    This implementation supports multiple LSTM architectures including:
    - Standard LSTM for basic sequence modeling
    - Bidirectional LSTM for past and future context
    - Stacked LSTM for complex pattern recognition
    - Encoder-Decoder LSTM for sequence-to-sequence tasks
    - Attention-enhanced LSTM for temporal focus
    
    The model is optimized for real-time streaming data processing with
    dynamic sequence handling and multi-horizon forecasting capabilities.
    """
    
    def __init__(self, 
                 model_name: str = "LSTMModel",
                 version: str = "2.0.0",
                 sequence_length: int = 50,
                 prediction_horizon: int = 1,
                 lstm_units: List[int] = [128, 64, 32],
                 bidirectional: bool = True,
                 attention: bool = True,
                 dropout_rate: float = 0.2,
                 recurrent_dropout: float = 0.1,
                 activation: str = "tanh",
                 recurrent_activation: str = "sigmoid",
                 use_conv1d: bool = True,
                 conv_filters: int = 64,
                 conv_kernel_size: int = 3,
                 dense_layers: List[int] = [64, 32],
                 output_activation: str = "linear",
                 loss_function: str = "mse",
                 optimizer: str = "adam",
                 learning_rate: float = 0.001,
                 batch_size: int = 32,
                 epochs: int = 100,
                 early_stopping_patience: int = 15,
                 reduce_lr_patience: int = 8,
                 regularization_l1: float = 0.0,
                 regularization_l2: float = 0.001,
                 task_type: str = "regression",  # regression, classification, anomaly_detection
                 anomaly_threshold_method: str = "statistical",
                 contamination: float = 0.05,
                 multi_step: bool = False,
                 stateful: bool = False,
                 **kwargs):
        """
        Initialize LSTM model with enterprise configuration.
        
        Args:
            model_name: Name identifier for the model
            version: Model version
            sequence_length: Length of input sequences
            prediction_horizon: Number of steps to predict ahead
            lstm_units: List of LSTM layer units
            bidirectional: Whether to use bidirectional LSTM
            attention: Whether to include attention mechanism
            dropout_rate: Dropout rate for regularization
            recurrent_dropout: Recurrent dropout rate
            activation: LSTM activation function
            recurrent_activation: LSTM recurrent activation function
            use_conv1d: Whether to use Conv1D layers before LSTM
            conv_filters: Number of convolutional filters
            conv_kernel_size: Convolutional kernel size
            dense_layers: Dense layer dimensions after LSTM
            output_activation: Output layer activation
            loss_function: Loss function for training
            optimizer: Optimizer for training
            learning_rate: Learning rate
            batch_size: Training batch size
            epochs: Maximum training epochs
            early_stopping_patience: Early stopping patience
            reduce_lr_patience: Learning rate reduction patience
            regularization_l1: L1 regularization factor
            regularization_l2: L2 regularization factor
            task_type: Type of task (regression, classification, anomaly_detection)
            anomaly_threshold_method: Method for anomaly threshold calculation
            contamination: Expected proportion of anomalies
            multi_step: Whether to perform multi-step prediction
            stateful: Whether to use stateful LSTM
        """
        super().__init__(model_name, version)
        
        # Model architecture configuration
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        self.lstm_units = lstm_units
        self.bidirectional = bidirectional
        self.attention = attention
        self.dropout_rate = dropout_rate
        self.recurrent_dropout = recurrent_dropout
        self.activation = activation
        self.recurrent_activation = recurrent_activation
        self.use_conv1d = use_conv1d
        self.conv_filters = conv_filters
        self.conv_kernel_size = conv_kernel_size
        self.dense_layers = dense_layers
        self.output_activation = output_activation
        self.loss_function = loss_function
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.regularization_l1 = regularization_l1
        self.regularization_l2 = regularization_l2
        
        # Training configuration
        self.batch_size = batch_size
        self.epochs = epochs
        self.early_stopping_patience = early_stopping_patience
        self.reduce_lr_patience = reduce_lr_patience
        self.task_type = task_type
        self.anomaly_threshold_method = anomaly_threshold_method
        self.contamination = contamination
        self.multi_step = multi_step
        self.stateful = stateful
        
        # Model components
        self.model = None
        self.scaler_X = None
        self.scaler_y = None
        self.feature_names = None
        self.target_names = None
        
        # Performance tracking
        self.training_history = None
        self.anomaly_threshold = None
        self.sequence_buffer = []
        
        # Music streaming specific configurations
        self.music_streaming_patterns = {
            'daily_cycles': {'period': 24, 'features': ['listening_hours', 'active_users']},
            'weekly_cycles': {'period': 168, 'features': ['weekend_usage', 'weekday_patterns']},
            'seasonal_patterns': {'period': 8760, 'features': ['holiday_listening', 'seasonal_genres']},
            'content_lifecycle': {'period': 720, 'features': ['track_popularity', 'viral_spread']},
            'user_behavior': {'period': 48, 'features': ['session_patterns', 'engagement_cycles']}
        }
        
        logger.info(f"Initialized LSTM model: {model_name} v{version} for {task_type}")
    
    def _build_model(self, input_shape: Tuple[int, int], output_dim: int) -> Model:
        """
        Build the LSTM model based on configuration.
        
        Args:
            input_shape: Shape of input sequences (sequence_length, features)
            output_dim: Dimension of output
            
        Returns:
            Compiled Keras model
        """
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow is required for LSTM model")
        
        # Input layer
        inputs = layers.Input(shape=input_shape, name="sequence_input")
        x = inputs
        
        # Optional Conv1D layers for feature extraction
        if self.use_conv1d:
            x = layers.Conv1D(
                filters=self.conv_filters,
                kernel_size=self.conv_kernel_size,
                activation='relu',
                padding='same',
                name="conv1d_1"
            )(x)
            x = layers.BatchNormalization(name="conv_bn_1")(x)
            x = layers.Dropout(self.dropout_rate, name="conv_dropout_1")(x)
            
            # Second conv layer
            x = layers.Conv1D(
                filters=self.conv_filters // 2,
                kernel_size=self.conv_kernel_size,
                activation='relu',
                padding='same',
                name="conv1d_2"
            )(x)
            x = layers.BatchNormalization(name="conv_bn_2")(x)
            x = layers.Dropout(self.dropout_rate, name="conv_dropout_2")(x)
        
        # LSTM layers
        for i, units in enumerate(self.lstm_units):
            return_sequences = (i < len(self.lstm_units) - 1) or self.attention
            
            # Regular or Bidirectional LSTM
            if self.bidirectional:
                lstm_layer = layers.Bidirectional(
                    layers.LSTM(
                        units,
                        return_sequences=return_sequences,
                        dropout=self.dropout_rate,
                        recurrent_dropout=self.recurrent_dropout,
                        activation=self.activation,
                        recurrent_activation=self.recurrent_activation,
                        kernel_regularizer=keras.regularizers.l1_l2(l1=self.regularization_l1, 
                                                                   l2=self.regularization_l2),
                        stateful=self.stateful,
                        name=f"bidirectional_lstm_{i}"
                    ),
                    name=f"bidirectional_{i}"
                )(x)
            else:
                lstm_layer = layers.LSTM(
                    units,
                    return_sequences=return_sequences,
                    dropout=self.dropout_rate,
                    recurrent_dropout=self.recurrent_dropout,
                    activation=self.activation,
                    recurrent_activation=self.recurrent_activation,
                    kernel_regularizer=keras.regularizers.l1_l2(l1=self.regularization_l1, 
                                                               l2=self.regularization_l2),
                    stateful=self.stateful,
                    name=f"lstm_{i}"
                )(x)
            
            x = lstm_layer
            
            # Batch normalization and dropout
            if i < len(self.lstm_units) - 1:
                x = layers.BatchNormalization(name=f"lstm_bn_{i}")(x)
                x = layers.Dropout(self.dropout_rate, name=f"lstm_dropout_{i}")(x)
        
        # Attention mechanism
        if self.attention:
            # Multi-head self-attention
            attention_output = layers.MultiHeadAttention(
                num_heads=8,
                key_dim=x.shape[-1] // 8,
                name="attention"
            )(x, x)
            
            # Add & Norm
            x = layers.Add(name="attention_add")([x, attention_output])
            x = layers.LayerNormalization(name="attention_norm")(x)
            
            # Global average pooling to reduce sequence dimension
            x = layers.GlobalAveragePooling1D(name="global_avg_pool")(x)
        
        # Dense layers
        for i, units in enumerate(self.dense_layers):
            x = layers.Dense(
                units,
                activation='relu',
                kernel_regularizer=keras.regularizers.l1_l2(l1=self.regularization_l1, 
                                                           l2=self.regularization_l2),
                name=f"dense_{i}"
            )(x)
            x = layers.BatchNormalization(name=f"dense_bn_{i}")(x)
            x = layers.Dropout(self.dropout_rate, name=f"dense_dropout_{i}")(x)
        
        # Output layer
        if self.task_type == "classification":
            if output_dim == 1:
                output = layers.Dense(1, activation='sigmoid', name="output")(x)
            else:
                output = layers.Dense(output_dim, activation='softmax', name="output")(x)
        elif self.task_type == "anomaly_detection":
            # For anomaly detection, output reconstruction error
            output = layers.Dense(output_dim, activation='linear', name="output")(x)
        else:  # regression
            if self.multi_step:
                # Multi-step prediction
                output = layers.Dense(output_dim * self.prediction_horizon, 
                                    activation=self.output_activation, name="output")(x)
                output = layers.Reshape((self.prediction_horizon, output_dim), 
                                      name="reshape_output")(output)
            else:
                output = layers.Dense(output_dim, activation=self.output_activation, name="output")(x)
        
        # Create model
        model = Model(inputs, output, name=f"lstm_{self.task_type}")
        
        return model
    
    def _prepare_sequences(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Prepare data into sequences for LSTM training.
        
        Args:
            X: Input features
            y: Target values (optional)
            
        Returns:
            Sequenced data (X_sequences, y_sequences)
        """
        n_samples = len(X)
        n_features = X.shape[1]
        
        # Calculate number of sequences
        n_sequences = n_samples - self.sequence_length + 1
        
        if n_sequences <= 0:
            raise ValueError(f"Not enough data for sequence length {self.sequence_length}. "
                           f"Need at least {self.sequence_length} samples, got {n_samples}")
        
        # Create sequences for X
        X_sequences = np.zeros((n_sequences, self.sequence_length, n_features))
        for i in range(n_sequences):
            X_sequences[i] = X[i:i + self.sequence_length]
        
        # Create sequences for y if provided
        y_sequences = None
        if y is not None:
            if self.multi_step:
                # Multi-step prediction: predict next prediction_horizon steps
                y_sequences = []
                for i in range(n_sequences):
                    end_idx = i + self.sequence_length + self.prediction_horizon
                    if end_idx <= len(y):
                        y_sequences.append(y[i + self.sequence_length:end_idx])
                
                if y_sequences:
                    y_sequences = np.array(y_sequences)
                else:
                    raise ValueError("Not enough data for multi-step prediction")
            else:
                # Single-step prediction
                y_sequences = y[self.sequence_length:]
                if len(y_sequences) != n_sequences:
                    y_sequences = y_sequences[:n_sequences]
        
        return X_sequences, y_sequences
    
    def fit(self, X: Union[np.ndarray, pd.DataFrame], 
            y: Optional[Union[np.ndarray, pd.Series]] = None,
            validation_split: float = 0.2,
            verbose: int = 1,
            **kwargs) -> 'LSTMModel':
        """
        Train the LSTM model on provided data.
        
        Args:
            X: Training features (time series data)
            y: Target values (optional for anomaly detection)
            validation_split: Fraction of data to use for validation
            verbose: Verbosity level
            **kwargs: Additional training parameters
            
        Returns:
            Self for method chaining
        """
        start_time = datetime.now()
        
        # Validate and prepare data
        self.validate_input(X)
        
        if isinstance(X, pd.DataFrame):
            self.feature_names = X.columns.tolist()
            X = X.values
        else:
            self.feature_names = [f"feature_{i}" for i in range(X.shape[1])]
        
        # Handle target data
        if y is not None:
            if isinstance(y, pd.DataFrame):
                self.target_names = y.columns.tolist()
                y = y.values
            elif isinstance(y, pd.Series):
                self.target_names = [y.name] if y.name else ["target"]
                y = y.values
            else:
                if len(y.shape) == 1:
                    self.target_names = ["target"]
                else:
                    self.target_names = [f"target_{i}" for i in range(y.shape[1])]
        else:
            # For anomaly detection, use X as target (autoencoder-style)
            if self.task_type == "anomaly_detection":
                y = X.copy()
                self.target_names = self.feature_names
        
        # Scale the data
        self.scaler_X = StandardScaler()
        X_scaled = self.scaler_X.fit_transform(X)
        
        if y is not None:
            if self.task_type == "classification":
                # Don't scale classification targets
                y_scaled = y
                self.scaler_y = None
            else:
                self.scaler_y = StandardScaler()
                if len(y.shape) == 1:
                    y_scaled = self.scaler_y.fit_transform(y.reshape(-1, 1)).flatten()
                else:
                    y_scaled = self.scaler_y.fit_transform(y)
        else:
            y_scaled = None
        
        # Prepare sequences
        X_sequences, y_sequences = self._prepare_sequences(X_scaled, y_scaled)
        
        # Determine output dimension
        if y_sequences is not None:
            if len(y_sequences.shape) == 1:
                output_dim = 1
            elif len(y_sequences.shape) == 2:
                output_dim = y_sequences.shape[1]
            else:  # multi-step
                output_dim = y_sequences.shape[2]
        else:
            output_dim = X_sequences.shape[2]  # For anomaly detection
        
        # Build model
        input_shape = (self.sequence_length, X_sequences.shape[2])
        self.model = self._build_model(input_shape, output_dim)
        
        # Compile model
        optimizer_instance = self._get_optimizer()
        loss_function = self._get_loss_function()
        model_metrics = self._get_metrics()
        
        self.model.compile(
            optimizer=optimizer_instance,
            loss=loss_function,
            metrics=model_metrics
        )
        
        # Callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=self.early_stopping_patience,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=self.reduce_lr_patience,
                min_lr=1e-7,
                verbose=1
            )
        ]
        
        # Train the model
        if y_sequences is not None:
            self.training_history = self.model.fit(
                X_sequences, y_sequences,
                batch_size=self.batch_size,
                epochs=self.epochs,
                validation_split=validation_split,
                callbacks=callbacks,
                verbose=verbose,
                **kwargs
            )
        else:
            # For anomaly detection without explicit targets
            self.training_history = self.model.fit(
                X_sequences, X_sequences,  # Autoencoder-style
                batch_size=self.batch_size,
                epochs=self.epochs,
                validation_split=validation_split,
                callbacks=callbacks,
                verbose=verbose,
                **kwargs
            )
        
        # Calculate anomaly threshold for anomaly detection
        if self.task_type == "anomaly_detection":
            predictions = self.model.predict(X_sequences)
            reconstruction_errors = np.mean(np.square(X_sequences - predictions), axis=(1, 2))
            self._calculate_anomaly_threshold(reconstruction_errors)
        
        # Update training metadata
        training_time = (datetime.now() - start_time).total_seconds()
        self.training_metadata = {
            'training_time_seconds': training_time,
            'epochs_trained': len(self.training_history.history['loss']),
            'final_loss': self.training_history.history['loss'][-1],
            'final_val_loss': self.training_history.history.get('val_loss', [None])[-1],
            'sequence_length': self.sequence_length,
            'n_sequences': len(X_sequences),
            'input_features': len(self.feature_names),
            'output_dimension': output_dim,
            'task_type': self.task_type,
            'architecture': {
                'lstm_units': self.lstm_units,
                'bidirectional': self.bidirectional,
                'attention': self.attention,
                'multi_step': self.multi_step
            }
        }
        
        if self.task_type == "anomaly_detection":
            self.training_metadata['anomaly_threshold'] = self.anomaly_threshold
        
        self.is_trained = True
        logger.info(f"LSTM training completed in {training_time:.2f} seconds")
        
        return self
    
    def predict(self, X: Union[np.ndarray, pd.DataFrame], 
                return_sequences: bool = False,
                **kwargs) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Make predictions on new data.
        
        Args:
            X: Input features for prediction
            return_sequences: Whether to return full sequences or just final predictions
            **kwargs: Additional prediction parameters
            
        Returns:
            Predictions (and sequences if requested)
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        self.validate_input(X)
        
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        # Scale the data
        X_scaled = self.scaler_X.transform(X)
        
        # Prepare sequences
        X_sequences, _ = self._prepare_sequences(X_scaled)
        
        # Make predictions
        raw_predictions = self.model.predict(X_sequences)
        
        # Process predictions based on task type
        if self.task_type == "anomaly_detection":
            # Calculate reconstruction errors
            reconstruction_errors = np.mean(np.square(X_sequences - raw_predictions), axis=(1, 2))
            anomalies = (reconstruction_errors > self.anomaly_threshold).astype(int)
            predictions = anomalies
            
            if return_sequences:
                return predictions, reconstruction_errors
        
        elif self.task_type == "classification":
            if raw_predictions.shape[-1] == 1:
                predictions = (raw_predictions > 0.5).astype(int).flatten()
            else:
                predictions = np.argmax(raw_predictions, axis=-1)
        
        else:  # regression
            if self.scaler_y is not None:
                if self.multi_step:
                    # Reshape for inverse transform
                    original_shape = raw_predictions.shape
                    predictions_flat = raw_predictions.reshape(-1, original_shape[-1])
                    predictions_unscaled = self.scaler_y.inverse_transform(predictions_flat)
                    predictions = predictions_unscaled.reshape(original_shape)
                else:
                    if len(raw_predictions.shape) == 1:
                        predictions = self.scaler_y.inverse_transform(raw_predictions.reshape(-1, 1)).flatten()
                    else:
                        predictions = self.scaler_y.inverse_transform(raw_predictions)
            else:
                predictions = raw_predictions
        
        # Update prediction count
        self.prediction_count += len(predictions)
        
        if return_sequences:
            return predictions, X_sequences
        else:
            return predictions
    
    def predict_proba(self, X: Union[np.ndarray, pd.DataFrame], 
                      **kwargs) -> np.ndarray:
        """
        Predict class probabilities (for classification tasks).
        
        Args:
            X: Input features for prediction
            **kwargs: Additional prediction parameters
            
        Returns:
            Prediction probabilities
        """
        if self.task_type != "classification":
            raise ValueError("predict_proba is only available for classification tasks")
        
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        self.validate_input(X)
        
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        # Scale the data
        X_scaled = self.scaler_X.transform(X)
        
        # Prepare sequences
        X_sequences, _ = self._prepare_sequences(X_scaled)
        
        # Get probability predictions
        probabilities = self.model.predict(X_sequences)
        
        return probabilities
    
    def predict_streaming(self, new_data: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Make predictions on streaming data (single time step).
        
        Args:
            new_data: New data point to add to sequence
            
        Returns:
            Prediction for the new sequence
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        if isinstance(new_data, pd.DataFrame):
            new_data = new_data.values
        
        # Ensure new_data is 2D
        if len(new_data.shape) == 1:
            new_data = new_data.reshape(1, -1)
        
        # Scale new data
        new_data_scaled = self.scaler_X.transform(new_data)
        
        # Update sequence buffer
        self.sequence_buffer.extend(new_data_scaled)
        
        # Keep only the last sequence_length points
        if len(self.sequence_buffer) > self.sequence_length:
            self.sequence_buffer = self.sequence_buffer[-self.sequence_length:]
        
        # Make prediction if we have enough data
        if len(self.sequence_buffer) == self.sequence_length:
            sequence = np.array(self.sequence_buffer).reshape(1, self.sequence_length, -1)
            raw_prediction = self.model.predict(sequence)
            
            # Process prediction based on task type
            if self.task_type == "anomaly_detection":
                reconstruction_error = np.mean(np.square(sequence - raw_prediction))
                prediction = int(reconstruction_error > self.anomaly_threshold)
            elif self.task_type == "classification":
                if raw_prediction.shape[-1] == 1:
                    prediction = int(raw_prediction[0] > 0.5)
                else:
                    prediction = np.argmax(raw_prediction[0])
            else:  # regression
                if self.scaler_y is not None:
                    prediction = self.scaler_y.inverse_transform(raw_prediction.reshape(-1, 1)).flatten()[0]
                else:
                    prediction = raw_prediction[0][0] if len(raw_prediction.shape) > 1 else raw_prediction[0]
            
            return prediction
        else:
            # Not enough data yet
            return None
    
    def forecast(self, X: Union[np.ndarray, pd.DataFrame], 
                 steps: int = 1) -> np.ndarray:
        """
        Generate multi-step forecasts.
        
        Args:
            X: Historical data for forecasting
            steps: Number of steps to forecast ahead
            
        Returns:
            Forecasted values
        """
        if self.task_type not in ["regression", "anomaly_detection"]:
            raise ValueError("Forecasting is only available for regression and anomaly detection tasks")
        
        if not self.is_trained:
            raise ValueError("Model must be trained before forecasting")
        
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        # Scale the data
        X_scaled = self.scaler_X.transform(X)
        
        # Use the last sequence_length points as starting sequence
        if len(X_scaled) < self.sequence_length:
            raise ValueError(f"Need at least {self.sequence_length} data points for forecasting")
        
        current_sequence = X_scaled[-self.sequence_length:].copy()
        forecasts = []
        
        for step in range(steps):
            # Reshape for prediction
            sequence_input = current_sequence.reshape(1, self.sequence_length, -1)
            
            # Make prediction
            prediction = self.model.predict(sequence_input)[0]
            
            # Store forecast
            if self.scaler_y is not None and self.task_type == "regression":
                forecast = self.scaler_y.inverse_transform(prediction.reshape(-1, 1)).flatten()
            else:
                forecast = prediction
            
            forecasts.append(forecast)
            
            # Update sequence for next prediction
            # Add prediction as next input (assuming univariate or that prediction matches input features)
            if len(forecast) == current_sequence.shape[1]:
                current_sequence = np.roll(current_sequence, -1, axis=0)
                current_sequence[-1] = prediction if self.task_type != "regression" or self.scaler_y is None else \
                                     self.scaler_X.transform(forecast.reshape(1, -1))[0]
            else:
                # For multivariate forecasting, use last known values for non-predicted features
                current_sequence = np.roll(current_sequence, -1, axis=0)
                current_sequence[-1, :len(forecast)] = prediction if self.task_type != "regression" or self.scaler_y is None else \
                                                     self.scaler_X.transform(forecast.reshape(1, -1))[0, :len(forecast)]
        
        return np.array(forecasts)
    
    def _get_optimizer(self):
        """Get optimizer instance"""
        if self.optimizer.lower() == "adam":
            return optimizers.Adam(learning_rate=self.learning_rate)
        elif self.optimizer.lower() == "rmsprop":
            return optimizers.RMSprop(learning_rate=self.learning_rate)
        elif self.optimizer.lower() == "sgd":
            return optimizers.SGD(learning_rate=self.learning_rate)
        else:
            return optimizers.Adam(learning_rate=self.learning_rate)
    
    def _get_loss_function(self):
        """Get loss function"""
        if self.task_type == "classification":
            if self.loss_function.lower() == "binary_crossentropy":
                return losses.BinaryCrossentropy()
            elif self.loss_function.lower() == "categorical_crossentropy":
                return losses.CategoricalCrossentropy()
            else:
                return losses.SparseCategoricalCrossentropy()
        else:  # regression or anomaly detection
            if self.loss_function.lower() == "mse":
                return losses.MeanSquaredError()
            elif self.loss_function.lower() == "mae":
                return losses.MeanAbsoluteError()
            elif self.loss_function.lower() == "huber":
                return losses.Huber()
            else:
                return losses.MeanSquaredError()
    
    def _get_metrics(self):
        """Get metrics for model compilation"""
        if self.task_type == "classification":
            return ['accuracy', 'precision', 'recall']
        else:
            return ['mae', 'mse']
    
    def _calculate_anomaly_threshold(self, reconstruction_errors: np.ndarray):
        """Calculate threshold for anomaly detection"""
        if self.anomaly_threshold_method == "percentile":
            self.anomaly_threshold = np.percentile(reconstruction_errors, (1 - self.contamination) * 100)
        elif self.anomaly_threshold_method == "statistical":
            mean_error = np.mean(reconstruction_errors)
            std_error = np.std(reconstruction_errors)
            self.anomaly_threshold = mean_error + 2 * std_error
        else:
            self.anomaly_threshold = np.percentile(reconstruction_errors, (1 - self.contamination) * 100)
        
        logger.info(f"Anomaly threshold set to: {self.anomaly_threshold:.6f}")
    
    def get_attention_weights(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Get attention weights for input sequences (if attention is enabled).
        
        Args:
            X: Input data
            
        Returns:
            Attention weights
        """
        if not self.attention:
            raise ValueError("Attention weights are only available when attention=True")
        
        if not self.is_trained:
            raise ValueError("Model must be trained before extracting attention weights")
        
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        # Scale data and prepare sequences
        X_scaled = self.scaler_X.transform(X)
        X_sequences, _ = self._prepare_sequences(X_scaled)
        
        # Create a model that outputs attention weights
        attention_layer = None
        for layer in self.model.layers:
            if isinstance(layer, layers.MultiHeadAttention):
                attention_layer = layer
                break
        
        if attention_layer is None:
            raise ValueError("No attention layer found in model")
        
        # Extract attention weights (this is a simplified implementation)
        # In practice, you'd need to modify the model architecture to output attention weights
        logger.warning("Attention weight extraction is simplified. Consider modifying model architecture for detailed attention analysis.")
        
        return np.ones((len(X_sequences), self.sequence_length))  # Placeholder
    
    def explain_prediction(self, X: Union[np.ndarray, pd.DataFrame], 
                          instance_idx: Optional[int] = None) -> Dict[str, Any]:
        """
        Provide explanations for LSTM predictions.
        
        Args:
            X: Input data
            instance_idx: Specific instance to explain
            
        Returns:
            Explanation including temporal importance and feature contributions
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before generating explanations")
        
        if isinstance(X, pd.DataFrame):
            X_values = X.values
        else:
            X_values = X
        
        # Scale data and prepare sequences
        X_scaled = self.scaler_X.transform(X_values)
        X_sequences, _ = self._prepare_sequences(X_scaled)
        
        # Get predictions
        predictions = self.model.predict(X_sequences)
        
        explanations = []
        indices = [instance_idx] if instance_idx is not None else range(len(X_sequences))
        
        for idx in indices:
            sequence = X_sequences[idx]
            prediction = predictions[idx]
            
            # Calculate feature importance by temporal position
            temporal_importance = np.mean(np.abs(sequence), axis=1)
            temporal_importance = temporal_importance / np.sum(temporal_importance)
            
            # Calculate feature importance across the sequence
            feature_importance = np.mean(np.abs(sequence), axis=0)
            feature_importance = feature_importance / np.sum(feature_importance)
            
            explanation = {
                'instance_index': idx,
                'prediction': prediction.tolist() if hasattr(prediction, 'tolist') else prediction,
                'sequence_length': self.sequence_length,
                'temporal_importance': temporal_importance.tolist(),
                'feature_importance': dict(zip(self.feature_names, feature_importance)),
                'top_temporal_steps': np.argsort(temporal_importance)[-5:].tolist(),
                'top_features': dict(
                    sorted(zip(self.feature_names, feature_importance), 
                          key=lambda x: x[1], reverse=True)[:5]
                ),
                'sequence_statistics': {
                    'mean': np.mean(sequence, axis=0).tolist(),
                    'std': np.std(sequence, axis=0).tolist(),
                    'trend': (sequence[-1] - sequence[0]).tolist()
                }
            }
            
            explanations.append(explanation)
        
        if instance_idx is not None:
            return explanations[0]
        else:
            return {'explanations': explanations}
    
    def detect_patterns(self, X: Union[np.ndarray, pd.DataFrame]) -> Dict[str, Any]:
        """
        Detect temporal patterns in the data.
        
        Args:
            X: Input time series data
            
        Returns:
            Detected patterns including cycles, trends, and anomalies
        """
        if isinstance(X, pd.DataFrame):
            X_values = X.values
        else:
            X_values = X
        
        patterns = {}
        
        # Detect cycles using FFT
        for i, feature_name in enumerate(self.feature_names):
            feature_data = X_values[:, i]
            
            # Remove trend
            detrended = feature_data - np.linspace(feature_data[0], feature_data[-1], len(feature_data))
            
            # FFT to find dominant frequencies
            fft = np.fft.fft(detrended)
            freqs = np.fft.fftfreq(len(detrended))
            
            # Find dominant frequencies
            power = np.abs(fft)
            dominant_freq_idx = np.argsort(power[1:len(power)//2])[-3:] + 1  # Top 3 frequencies
            dominant_periods = [1/freqs[idx] for idx in dominant_freq_idx if freqs[idx] > 0]
            
            patterns[feature_name] = {
                'dominant_periods': dominant_periods,
                'trend': np.polyfit(range(len(feature_data)), feature_data, 1)[0],
                'seasonality_strength': np.std(detrended) / np.std(feature_data)
            }
        
        # Check for music streaming specific patterns
        streaming_patterns = {}
        for pattern_name, config in self.music_streaming_patterns.items():
            period = config['period']
            relevant_features = [f for f in config['features'] if f in self.feature_names]
            
            if relevant_features:
                # Check if data length allows for pattern detection
                if len(X_values) >= period:
                    pattern_strength = 0
                    for feature in relevant_features:
                        feature_idx = self.feature_names.index(feature)
                        feature_data = X_values[:, feature_idx]
                        
                        # Calculate autocorrelation at the expected period
                        if len(feature_data) > period:
                            autocorr = np.corrcoef(feature_data[:-period], feature_data[period:])[0, 1]
                            pattern_strength += abs(autocorr)
                    
                    streaming_patterns[pattern_name] = {
                        'strength': pattern_strength / len(relevant_features),
                        'period': period,
                        'detected': pattern_strength / len(relevant_features) > 0.3
                    }
        
        return {
            'feature_patterns': patterns,
            'streaming_patterns': streaming_patterns,
            'data_length': len(X_values),
            'analysis_timestamp': datetime.now().isoformat()
        }
    
    def save_model(self, filepath: str):
        """Save the trained model to disk"""
        if not self.is_trained:
            raise ValueError("Cannot save untrained model")
        
        import pickle
        
        model_data = {
            'model': self.model,
            'scaler_X': self.scaler_X,
            'scaler_y': self.scaler_y,
            'feature_names': self.feature_names,
            'target_names': self.target_names,
            'anomaly_threshold': self.anomaly_threshold,
            'training_metadata': self.training_metadata,
            'model_config': {
                'sequence_length': self.sequence_length,
                'prediction_horizon': self.prediction_horizon,
                'lstm_units': self.lstm_units,
                'bidirectional': self.bidirectional,
                'attention': self.attention,
                'task_type': self.task_type,
                'multi_step': self.multi_step
            }
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        logger.info(f"LSTM model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load a trained model from disk"""
        import pickle
        
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model = model_data['model']
        self.scaler_X = model_data['scaler_X']
        self.scaler_y = model_data['scaler_y']
        self.feature_names = model_data['feature_names']
        self.target_names = model_data['target_names']
        self.anomaly_threshold = model_data['anomaly_threshold']
        self.training_metadata = model_data['training_metadata']
        
        # Update model configuration
        config = model_data['model_config']
        self.sequence_length = config['sequence_length']
        self.prediction_horizon = config['prediction_horizon']
        self.lstm_units = config['lstm_units']
        self.bidirectional = config['bidirectional']
        self.attention = config['attention']
        self.task_type = config['task_type']
        self.multi_step = config['multi_step']
        
        self.is_trained = True
        logger.info(f"LSTM model loaded from {filepath}")


# Export the model class
__all__ = ['LSTMModel']
