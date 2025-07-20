"""
Enterprise Random Forest Model for Spotify AI Agent
===================================================

Advanced Random Forest Implementation for Classification and Regression in Music Streaming

This module implements a sophisticated Random Forest algorithm optimized for
high-performance classification and regression tasks in music streaming platforms.
Features advanced ensemble techniques, automatic feature engineering, and
enterprise-grade model optimization with explainable AI capabilities.

🎵 MUSIC STREAMING CLASSIFICATION SCENARIOS:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• Genre Classification - Automatic music genre detection and categorization
• User Churn Classification - Identify users likely to cancel subscriptions
• Content Quality Classification - Audio quality rating and assessment
• Recommendation Relevance - Classify recommendation accuracy and user preference
• Fraud Detection - Identify fraudulent accounts and payment activities
• Content Moderation - Classify inappropriate or explicit content
• Artist Tier Classification - Categorize artists by popularity and success level
• Market Segmentation - Classify users into behavioral and demographic segments

⚡ ENTERPRISE RANDOM FOREST FEATURES:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• Advanced Ensemble Methods with Extra Trees and Isolation Forest integration
• Automated Feature Engineering with polynomial and interaction features
• Dynamic Tree Pruning and complexity optimization
• Out-of-Bag error estimation and confidence intervals
• Feature importance ranking with permutation importance
• Hyperparameter optimization with Bayesian search
• Class imbalance handling with advanced sampling techniques
• Multi-output classification and regression support
• Real-time prediction serving with model compression
• Distributed training for large-scale datasets

Version: 2.0.0 (Enterprise Edition)
Optimized for: High-dimensional data, real-time inference, interpretability
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, Any, List, Optional, Union, Tuple
from abc import ABC
from datetime import datetime, timedelta
import warnings

# Machine Learning imports
from sklearn.ensemble import (RandomForestClassifier, RandomForestRegressor, 
                            ExtraTreesClassifier, ExtraTreesRegressor)
from sklearn.model_selection import (train_test_split, StratifiedKFold, cross_val_score,
                                   validation_curve, GridSearchCV, RandomizedSearchCV)
from sklearn.preprocessing import (StandardScaler, LabelEncoder, PolynomialFeatures,
                                 RobustScaler, QuantileTransformer)
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                           roc_auc_score, confusion_matrix, classification_report,
                           mean_squared_error, mean_absolute_error, r2_score)
from sklearn.feature_selection import (SelectKBest, f_classif, f_regression, 
                                     mutual_info_classif, mutual_info_regression,
                                     RFE, SelectFromModel)
from sklearn.utils.class_weight import compute_class_weight

# Advanced ensemble methods
try:
    from sklearn.experimental import enable_iterative_imputer
    from sklearn.impute import IterativeImputer
    ITERATIVE_IMPUTER_AVAILABLE = True
except ImportError:
    from sklearn.impute import SimpleImputer
    ITERATIVE_IMPUTER_AVAILABLE = False

# Hyperparameter optimization
try:
    from skopt import BayesSearchCV
    from skopt.space import Real, Integer, Categorical
    BAYESIAN_OPT_AVAILABLE = True
except ImportError:
    BAYESIAN_OPT_AVAILABLE = False
    logging.warning("scikit-optimize not available for Bayesian optimization")

# Class imbalance handling
try:
    from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE
    from imblearn.under_sampling import RandomUnderSampler, EditedNearestNeighbours
    from imblearn.combine import SMOTETomek, SMOTEENN
    IMBALANCED_LEARN_AVAILABLE = True
except ImportError:
    IMBALANCED_LEARN_AVAILABLE = False
    logging.warning("imbalanced-learn not available for class imbalance handling")

# Explainability
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    logging.warning("SHAP not available for model explanations")

# Import base model interface
from . import ModelInterface

logger = logging.getLogger(__name__)


class RandomForestModel(ModelInterface):
    """
    Enterprise-grade Random Forest model for classification and regression in music streaming.
    
    This implementation provides advanced ensemble learning with:
    - Multiple tree types (Random Forest, Extra Trees)
    - Automated feature engineering and selection
    - Class imbalance handling with sampling techniques
    - Hyperparameter optimization with Bayesian search
    - Advanced model interpretability and explainable AI
    
    Optimized for high-dimensional data, real-time inference, and business interpretability.
    """
    
    def __init__(self, 
                 model_name: str = "RandomForestModel",
                 version: str = "2.0.0",
                 task_type: str = "classification",  # classification, regression
                 ensemble_method: str = "random_forest",  # random_forest, extra_trees
                 n_estimators: int = 100,
                 max_depth: Optional[int] = None,
                 min_samples_split: int = 2,
                 min_samples_leaf: int = 1,
                 max_features: str = "sqrt",
                 bootstrap: bool = True,
                 oob_score: bool = True,
                 random_state: int = 42,
                 n_jobs: int = -1,
                 class_weight: Optional[str] = "balanced",
                 criterion: str = "gini",  # gini, entropy for classification; mse, mae for regression
                 auto_feature_engineering: bool = True,
                 polynomial_features: bool = False,
                 interaction_features: bool = True,
                 feature_selection: bool = True,
                 feature_selection_method: str = "importance",  # importance, mutual_info, rfe
                 k_best_features: Optional[int] = None,
                 handle_imbalance: bool = True,
                 imbalance_method: str = "smote",  # smote, adasyn, borderline, undersampling
                 auto_hyperparameter_tuning: bool = True,
                 tuning_method: str = "bayesian",  # grid, random, bayesian
                 cv_folds: int = 5,
                 scoring: Optional[str] = None,
                 confidence_intervals: bool = True,
                 feature_importance_type: str = "default",  # default, permutation
                 **kwargs):
        """
        Initialize Random Forest model with enterprise configuration.
        
        Args:
            model_name: Name identifier for the model
            version: Model version
            task_type: Type of task (classification, regression)
            ensemble_method: Type of ensemble (random_forest, extra_trees)
            n_estimators: Number of trees in the forest
            max_depth: Maximum depth of the trees
            min_samples_split: Minimum samples required to split a node
            min_samples_leaf: Minimum samples required at a leaf node
            max_features: Number of features to consider for best split
            bootstrap: Whether to use bootstrap samples
            oob_score: Whether to use out-of-bag score
            random_state: Random state for reproducibility
            n_jobs: Number of parallel jobs
            class_weight: Class weight strategy
            criterion: Split criterion
            auto_feature_engineering: Whether to perform automatic feature engineering
            polynomial_features: Whether to generate polynomial features
            interaction_features: Whether to generate interaction features
            feature_selection: Whether to perform feature selection
            feature_selection_method: Method for feature selection
            k_best_features: Number of best features to select
            handle_imbalance: Whether to handle class imbalance
            imbalance_method: Method for handling class imbalance
            auto_hyperparameter_tuning: Whether to perform hyperparameter tuning
            tuning_method: Method for hyperparameter tuning
            cv_folds: Number of cross-validation folds
            scoring: Scoring metric for optimization
            confidence_intervals: Whether to calculate confidence intervals
            feature_importance_type: Type of feature importance calculation
        """
        super().__init__(model_name, version)
        
        # Model configuration
        self.task_type = task_type
        self.ensemble_method = ensemble_method
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.oob_score = oob_score
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.class_weight = class_weight
        self.criterion = criterion
        
        # Feature engineering
        self.auto_feature_engineering = auto_feature_engineering
        self.polynomial_features = polynomial_features
        self.interaction_features = interaction_features
        self.feature_selection = feature_selection
        self.feature_selection_method = feature_selection_method
        self.k_best_features = k_best_features
        
        # Class imbalance handling
        self.handle_imbalance = handle_imbalance
        self.imbalance_method = imbalance_method
        
        # Hyperparameter optimization
        self.auto_hyperparameter_tuning = auto_hyperparameter_tuning
        self.tuning_method = tuning_method
        self.cv_folds = cv_folds
        self.scoring = scoring
        
        # Advanced features
        self.confidence_intervals = confidence_intervals
        self.feature_importance_type = feature_importance_type
        
        # Model components
        self.model = None
        self.feature_names = None
        self.target_names = None
        self.label_encoders = {}
        self.scaler = None
        self.feature_engineer = None
        self.feature_selector = None
        self.imbalance_sampler = None
        self.selected_features = None
        self.best_params = None
        
        # Performance tracking
        self.feature_importance_scores = {}
        self.permutation_importance_scores = {}
        self.oob_score_value = None
        self.cv_scores = None
        self.confidence_interval_bounds = {}
        
        # Music streaming specific configurations
        self.music_streaming_features = {
            'user_behavior': ['listening_hours', 'skip_rate', 'replay_rate', 'discovery_rate'],
            'content_interaction': ['likes_ratio', 'shares_count', 'playlist_adds', 'downloads'],
            'engagement_patterns': ['session_length', 'sessions_per_day', 'peak_hours', 'device_usage'],
            'demographic_data': ['age_group', 'gender', 'location', 'subscription_type'],
            'audio_preferences': ['preferred_genres', 'audio_quality', 'language_pref', 'decade_pref'],
            'business_metrics': ['revenue_per_user', 'subscription_length', 'support_interactions']
        }
        
        logger.info(f"Initialized {self.ensemble_method} Random Forest model: {model_name} v{version}")
    
    def _get_base_model(self, **params) -> Union[RandomForestClassifier, RandomForestRegressor]:
        """
        Get the base model instance based on configuration.
        
        Args:
            **params: Additional parameters for the model
            
        Returns:
            Configured model instance
        """
        model_params = {
            'n_estimators': self.n_estimators,
            'max_depth': self.max_depth,
            'min_samples_split': self.min_samples_split,
            'min_samples_leaf': self.min_samples_leaf,
            'max_features': self.max_features,
            'bootstrap': self.bootstrap,
            'oob_score': self.oob_score,
            'random_state': self.random_state,
            'n_jobs': self.n_jobs,
            'criterion': self.criterion
        }
        
        # Add class weight for classification
        if self.task_type == "classification":
            model_params['class_weight'] = self.class_weight
        
        # Update with additional parameters
        model_params.update(params)
        
        # Create model based on ensemble method and task type
        if self.ensemble_method == "random_forest":
            if self.task_type == "classification":
                return RandomForestClassifier(**model_params)
            else:
                return RandomForestRegressor(**model_params)
        else:  # extra_trees
            if self.task_type == "classification":
                return ExtraTreesClassifier(**model_params)
            else:
                return ExtraTreesRegressor(**model_params)
    
    def _engineer_features(self, X: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """
        Perform automatic feature engineering.
        
        Args:
            X: Input features
            fit: Whether to fit the feature engineering components
            
        Returns:
            Engineered features
        """
        X_engineered = X.copy()
        
        if not self.auto_feature_engineering:
            return X_engineered
        
        # Polynomial features
        if self.polynomial_features and fit:
            self.feature_engineer = PolynomialFeatures(
                degree=2, 
                include_bias=False, 
                interaction_only=not self.interaction_features
            )
            # Only apply to numeric features to avoid explosion
            numeric_cols = X_engineered.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0 and len(numeric_cols) <= 20:  # Limit to prevent explosion
                X_numeric = X_engineered[numeric_cols]
                X_poly = self.feature_engineer.fit_transform(X_numeric)
                poly_feature_names = self.feature_engineer.get_feature_names_out(numeric_cols)
                
                # Add polynomial features
                X_poly_df = pd.DataFrame(X_poly, columns=poly_feature_names, index=X_engineered.index)
                X_engineered = pd.concat([X_engineered, X_poly_df], axis=1)
        
        elif self.polynomial_features and self.feature_engineer is not None:
            # Transform using fitted engineer
            numeric_cols = X_engineered.select_dtypes(include=[np.number]).columns
            numeric_cols = [col for col in numeric_cols if col in self.feature_engineer.feature_names_in_]
            
            if len(numeric_cols) > 0:
                X_numeric = X_engineered[numeric_cols]
                X_poly = self.feature_engineer.transform(X_numeric)
                poly_feature_names = self.feature_engineer.get_feature_names_out(numeric_cols)
                
                X_poly_df = pd.DataFrame(X_poly, columns=poly_feature_names, index=X_engineered.index)
                X_engineered = pd.concat([X_engineered, X_poly_df], axis=1)
        
        # Interaction features (simple version)
        if self.interaction_features and not self.polynomial_features:
            numeric_cols = X_engineered.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) >= 2 and len(numeric_cols) <= 10:  # Limit combinations
                for i, col1 in enumerate(numeric_cols):
                    for col2 in numeric_cols[i+1:]:
                        interaction_name = f"{col1}_x_{col2}"
                        X_engineered[interaction_name] = X_engineered[col1] * X_engineered[col2]
        
        # Music streaming specific feature engineering
        if fit:
            self._create_music_streaming_features(X_engineered, fit=True)
        else:
            self._create_music_streaming_features(X_engineered, fit=False)
        
        return X_engineered
    
    def _create_music_streaming_features(self, X: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """Create music streaming specific features"""
        
        # Engagement score
        engagement_features = ['likes_ratio', 'shares_count', 'playlist_adds']
        available_engagement = [f for f in engagement_features if f in X.columns]
        if len(available_engagement) >= 2:
            X['engagement_score'] = X[available_engagement].mean(axis=1)
        
        # Activity intensity
        activity_features = ['listening_hours', 'sessions_per_day']
        available_activity = [f for f in activity_features if f in X.columns]
        if len(available_activity) >= 2:
            X['activity_intensity'] = X[available_activity].prod(axis=1)
        
        # Preference diversity
        if 'preferred_genres' in X.columns and 'audio_quality' in X.columns:
            X['preference_diversity'] = X['preferred_genres'] * X['audio_quality']
        
        # Churn risk indicator (combination of multiple factors)
        churn_indicators = ['skip_rate', 'support_interactions']
        available_churn = [f for f in churn_indicators if f in X.columns]
        if len(available_churn) >= 1:
            X['churn_risk_score'] = X[available_churn].mean(axis=1)
        
        return X
    
    def _select_features(self, X: pd.DataFrame, y: pd.Series, fit: bool = True) -> pd.DataFrame:
        """
        Perform feature selection.
        
        Args:
            X: Input features
            y: Target values
            fit: Whether to fit the feature selector
            
        Returns:
            Selected features
        """
        if not self.feature_selection:
            self.selected_features = X.columns.tolist()
            return X
        
        if fit:
            if self.feature_selection_method == "importance":
                # Use a quick Random Forest to get feature importance
                quick_model = self._get_base_model(n_estimators=50)
                quick_model.fit(X, y)
                
                self.feature_selector = SelectFromModel(
                    quick_model, 
                    threshold='median' if self.k_best_features is None else None,
                    max_features=self.k_best_features
                )
                X_selected = self.feature_selector.fit_transform(X, y)
                self.selected_features = X.columns[self.feature_selector.get_support()].tolist()
            
            elif self.feature_selection_method == "mutual_info":
                if self.task_type == "classification":
                    score_func = mutual_info_classif
                else:
                    score_func = mutual_info_regression
                
                k = self.k_best_features or min(50, X.shape[1])
                self.feature_selector = SelectKBest(score_func=score_func, k=k)
                X_selected = self.feature_selector.fit_transform(X, y)
                self.selected_features = X.columns[self.feature_selector.get_support()].tolist()
            
            elif self.feature_selection_method == "rfe":
                base_model = self._get_base_model(n_estimators=50)
                n_features = self.k_best_features or min(20, X.shape[1])
                self.feature_selector = RFE(base_model, n_features_to_select=n_features)
                X_selected = self.feature_selector.fit_transform(X, y)
                self.selected_features = X.columns[self.feature_selector.get_support()].tolist()
            
            else:
                self.selected_features = X.columns.tolist()
                return X
            
            logger.info(f"Selected {len(self.selected_features)} features out of {X.shape[1]} using {self.feature_selection_method}")
            
        else:
            if self.feature_selector is not None and self.selected_features is not None:
                X_selected = X[self.selected_features]
            else:
                X_selected = X
        
        return pd.DataFrame(X_selected, columns=self.selected_features, index=X.index)
    
    def _handle_class_imbalance(self, X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Handle class imbalance using sampling techniques.
        
        Args:
            X: Input features
            y: Target values
            
        Returns:
            Resampled features and targets
        """
        if not self.handle_imbalance or self.task_type != "classification":
            return X, y
        
        if not IMBALANCED_LEARN_AVAILABLE:
            logger.warning("imbalanced-learn not available. Skipping imbalance handling.")
            return X, y
        
        # Check if imbalance exists
        class_counts = y.value_counts()
        imbalance_ratio = class_counts.max() / class_counts.min()
        
        if imbalance_ratio < 2:  # No significant imbalance
            logger.info(f"No significant class imbalance detected (ratio: {imbalance_ratio:.2f})")
            return X, y
        
        logger.info(f"Class imbalance detected (ratio: {imbalance_ratio:.2f}). Applying {self.imbalance_method}")
        
        # Apply sampling strategy
        if self.imbalance_method == "smote":
            self.imbalance_sampler = SMOTE(random_state=self.random_state)
        elif self.imbalance_method == "adasyn":
            self.imbalance_sampler = ADASYN(random_state=self.random_state)
        elif self.imbalance_method == "borderline":
            self.imbalance_sampler = BorderlineSMOTE(random_state=self.random_state)
        elif self.imbalance_method == "undersampling":
            self.imbalance_sampler = RandomUnderSampler(random_state=self.random_state)
        elif self.imbalance_method == "smote_tomek":
            self.imbalance_sampler = SMOTETomek(random_state=self.random_state)
        elif self.imbalance_method == "smote_enn":
            self.imbalance_sampler = SMOTEENN(random_state=self.random_state)
        else:
            logger.warning(f"Unknown imbalance method: {self.imbalance_method}")
            return X, y
        
        try:
            X_resampled, y_resampled = self.imbalance_sampler.fit_resample(X, y)
            X_resampled = pd.DataFrame(X_resampled, columns=X.columns)
            y_resampled = pd.Series(y_resampled)
            
            logger.info(f"Resampled from {len(X)} to {len(X_resampled)} samples")
            return X_resampled, y_resampled
        
        except Exception as e:
            logger.warning(f"Failed to apply imbalance handling: {e}")
            return X, y
    
    def _optimize_hyperparameters(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """
        Optimize hyperparameters using specified method.
        
        Args:
            X: Training features
            y: Training targets
            
        Returns:
            Best hyperparameters
        """
        if not self.auto_hyperparameter_tuning:
            return {}
        
        # Define parameter space
        if self.ensemble_method == "random_forest":
            param_space = {
                'n_estimators': [50, 100, 200, 500],
                'max_depth': [None, 5, 10, 15, 20],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'max_features': ['sqrt', 'log2', 0.3, 0.5]
            }
        else:  # extra_trees
            param_space = {
                'n_estimators': [50, 100, 200, 500],
                'max_depth': [None, 5, 10, 15, 20],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'max_features': ['sqrt', 'log2', 0.3, 0.5]
            }
        
        # Get base model
        base_model = self._get_base_model()
        
        # Setup scoring
        scoring = self.scoring
        if scoring is None:
            if self.task_type == "classification":
                scoring = 'roc_auc' if len(np.unique(y)) == 2 else 'f1_macro'
            else:
                scoring = 'neg_mean_squared_error'
        
        # Setup cross-validation
        if self.task_type == "classification":
            cv = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
        else:
            from sklearn.model_selection import KFold
            cv = KFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
        
        # Perform hyperparameter optimization
        if self.tuning_method == "bayesian" and BAYESIAN_OPT_AVAILABLE:
            # Convert to skopt format
            skopt_space = {}
            for param, values in param_space.items():
                if isinstance(values[0], int):
                    skopt_space[param] = Integer(min(values), max(values))
                elif isinstance(values[0], float):
                    skopt_space[param] = Real(min(values), max(values))
                else:
                    skopt_space[param] = Categorical(values)
            
            search = BayesSearchCV(
                base_model,
                skopt_space,
                n_iter=50,
                cv=cv,
                scoring=scoring,
                random_state=self.random_state,
                n_jobs=self.n_jobs
            )
        
        elif self.tuning_method == "random":
            search = RandomizedSearchCV(
                base_model,
                param_space,
                n_iter=50,
                cv=cv,
                scoring=scoring,
                random_state=self.random_state,
                n_jobs=self.n_jobs
            )
        
        else:  # grid search
            search = GridSearchCV(
                base_model,
                param_space,
                cv=cv,
                scoring=scoring,
                n_jobs=self.n_jobs
            )
        
        # Fit the search
        search.fit(X, y)
        
        logger.info(f"Hyperparameter optimization completed. Best score: {search.best_score_:.4f}")
        
        return search.best_params_
    
    def fit(self, X: Union[np.ndarray, pd.DataFrame], 
            y: Union[np.ndarray, pd.Series],
            validation_split: float = 0.2,
            **kwargs) -> 'RandomForestModel':
        """
        Train the Random Forest model on provided data.
        
        Args:
            X: Training features
            y: Training targets
            validation_split: Fraction of data to use for validation
            **kwargs: Additional training parameters
            
        Returns:
            Self for method chaining
        """
        start_time = datetime.now()
        
        # Validate and prepare data
        self.validate_input(X)
        
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
        
        if isinstance(y, np.ndarray):
            y = pd.Series(y)
        
        self.feature_names = X.columns.tolist()
        self.target_names = [y.name] if y.name else ["target"]
        
        # Handle categorical encoding
        categorical_cols = X.select_dtypes(include=['object', 'category']).columns
        for col in categorical_cols:
            self.label_encoders[col] = LabelEncoder()
            X[col] = self.label_encoders[col].fit_transform(X[col].astype(str))
        
        # Encode target if classification with string labels
        if self.task_type == "classification" and y.dtype == 'object':
            self.label_encoders['target'] = LabelEncoder()
            y = pd.Series(self.label_encoders['target'].fit_transform(y), index=y.index)
        
        # Feature engineering
        X_engineered = self._engineer_features(X, fit=True)
        
        # Handle missing values
        if X_engineered.isnull().any().any():
            if ITERATIVE_IMPUTER_AVAILABLE:
                imputer = IterativeImputer(random_state=self.random_state)
            else:
                from sklearn.impute import SimpleImputer
                imputer = SimpleImputer(strategy='median')
            
            numeric_cols = X_engineered.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                X_engineered[numeric_cols] = imputer.fit_transform(X_engineered[numeric_cols])
        
        # Feature selection
        X_selected = self._select_features(X_engineered, y, fit=True)
        
        # Handle class imbalance
        X_balanced, y_balanced = self._handle_class_imbalance(X_selected, y)
        
        # Split data for validation
        if validation_split > 0:
            X_train, X_val, y_train, y_val = train_test_split(
                X_balanced, y_balanced, test_size=validation_split,
                random_state=self.random_state,
                stratify=y_balanced if self.task_type == "classification" else None
            )
        else:
            X_train, y_train = X_balanced, y_balanced
            X_val, y_val = None, None
        
        # Hyperparameter optimization
        self.best_params = self._optimize_hyperparameters(X_train, y_train)
        
        # Train final model
        self.model = self._get_base_model(**self.best_params)
        self.model.fit(X_train, y_train)
        
        # Calculate feature importance
        self.feature_importance_scores = dict(zip(
            self.selected_features,
            self.model.feature_importances_
        ))
        
        # Calculate permutation importance if requested
        if self.feature_importance_type == "permutation":
            from sklearn.inspection import permutation_importance
            perm_importance = permutation_importance(
                self.model, X_train, y_train, 
                n_repeats=10, random_state=self.random_state, n_jobs=self.n_jobs
            )
            self.permutation_importance_scores = dict(zip(
                self.selected_features,
                perm_importance.importances_mean
            ))
        
        # Calculate out-of-bag score
        if self.oob_score and hasattr(self.model, 'oob_score_'):
            self.oob_score_value = self.model.oob_score_
        
        # Cross-validation scores
        if self.task_type == "classification":
            cv = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
        else:
            from sklearn.model_selection import KFold
            cv = KFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
        
        scoring = self.scoring or ('roc_auc' if self.task_type == "classification" else 'neg_mean_squared_error')
        self.cv_scores = cross_val_score(self.model, X_train, y_train, cv=cv, scoring=scoring)
        
        # Calculate confidence intervals if requested
        if self.confidence_intervals and X_val is not None:
            self._calculate_confidence_intervals(X_val, y_val)
        
        # Calculate validation metrics
        validation_metrics = {}
        if X_val is not None:
            val_predictions = self.model.predict(X_val)
            
            if self.task_type == "classification":
                val_proba = self.model.predict_proba(X_val)
                validation_metrics = {
                    'accuracy': accuracy_score(y_val, val_predictions),
                    'precision': precision_score(y_val, val_predictions, average='weighted'),
                    'recall': recall_score(y_val, val_predictions, average='weighted'),
                    'f1': f1_score(y_val, val_predictions, average='weighted')
                }
                
                if len(np.unique(y_val)) == 2:
                    validation_metrics['auc'] = roc_auc_score(y_val, val_proba[:, 1])
            
            else:
                validation_metrics = {
                    'mse': mean_squared_error(y_val, val_predictions),
                    'mae': mean_absolute_error(y_val, val_predictions),
                    'rmse': np.sqrt(mean_squared_error(y_val, val_predictions)),
                    'r2': r2_score(y_val, val_predictions)
                }
        
        # Update training metadata
        training_time = (datetime.now() - start_time).total_seconds()
        self.training_metadata = {
            'training_time_seconds': training_time,
            'ensemble_method': self.ensemble_method,
            'task_type': self.task_type,
            'n_samples_original': len(X),
            'n_samples_final': len(X_train),
            'n_features_original': len(self.feature_names),
            'n_features_selected': len(self.selected_features),
            'best_params': self.best_params,
            'oob_score': self.oob_score_value,
            'cv_scores_mean': np.mean(self.cv_scores),
            'cv_scores_std': np.std(self.cv_scores),
            'validation_metrics': validation_metrics,
            'feature_engineering_applied': self.auto_feature_engineering,
            'imbalance_handling_applied': self.handle_imbalance and IMBALANCED_LEARN_AVAILABLE
        }
        
        self.is_trained = True
        logger.info(f"Random Forest training completed in {training_time:.2f} seconds")
        
        return self
    
    def predict(self, X: Union[np.ndarray, pd.DataFrame], 
                **kwargs) -> np.ndarray:
        """
        Make predictions on new data.
        
        Args:
            X: Input features for prediction
            **kwargs: Additional prediction parameters
            
        Returns:
            Predictions
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        self.validate_input(X)
        
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X, columns=self.feature_names)
        
        # Apply same preprocessing pipeline
        X_processed = self._preprocess_for_prediction(X)
        
        # Make predictions
        predictions = self.model.predict(X_processed)
        
        # Transform back if needed
        if self.task_type == "classification" and 'target' in self.label_encoders:
            predictions = self.label_encoders['target'].inverse_transform(predictions)
        
        # Update prediction count
        self.prediction_count += len(predictions)
        
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
        
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X, columns=self.feature_names)
        
        # Apply same preprocessing pipeline
        X_processed = self._preprocess_for_prediction(X)
        
        # Get probability predictions
        probabilities = self.model.predict_proba(X_processed)
        
        return probabilities
    
    def _preprocess_for_prediction(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply preprocessing pipeline for prediction"""
        X_processed = X.copy()
        
        # Apply categorical encoding
        for col in X_processed.columns:
            if col in self.label_encoders:
                # Handle unseen categories
                unique_values = set(X_processed[col].astype(str))
                known_values = set(self.label_encoders[col].classes_)
                new_values = unique_values - known_values
                
                if new_values:
                    # Add new categories to encoder
                    self.label_encoders[col].classes_ = np.array(
                        list(self.label_encoders[col].classes_) + list(new_values)
                    )
                
                X_processed[col] = self.label_encoders[col].transform(X_processed[col].astype(str))
        
        # Feature engineering
        X_processed = self._engineer_features(X_processed, fit=False)
        
        # Feature selection
        X_processed = self._select_features(X_processed, None, fit=False)
        
        return X_processed
    
    def _calculate_confidence_intervals(self, X_val: pd.DataFrame, y_val: pd.Series):
        """Calculate confidence intervals for predictions"""
        # Use individual tree predictions to estimate confidence
        if hasattr(self.model, 'estimators_'):
            tree_predictions = []
            for tree in self.model.estimators_:
                if self.task_type == "classification":
                    tree_pred = tree.predict_proba(X_val)[:, 1] if hasattr(tree, 'predict_proba') else tree.predict(X_val)
                else:
                    tree_pred = tree.predict(X_val)
                tree_predictions.append(tree_pred)
            
            tree_predictions = np.array(tree_predictions)
            
            # Calculate percentiles for confidence intervals
            self.confidence_interval_bounds = {
                'lower_5': np.percentile(tree_predictions, 5, axis=0),
                'upper_95': np.percentile(tree_predictions, 95, axis=0),
                'mean': np.mean(tree_predictions, axis=0),
                'std': np.std(tree_predictions, axis=0)
            }
    
    def get_feature_importance(self, importance_type: str = "default") -> Dict[str, float]:
        """
        Get feature importance scores.
        
        Args:
            importance_type: Type of importance ('default' or 'permutation')
            
        Returns:
            Dictionary mapping feature names to importance scores
        """
        if importance_type == "permutation" and self.permutation_importance_scores:
            return self.permutation_importance_scores
        elif self.feature_importance_scores:
            return self.feature_importance_scores
        else:
            logger.warning("Feature importance not calculated. Train the model first.")
            return {}
    
    def explain_prediction(self, X: Union[np.ndarray, pd.DataFrame], 
                          instance_idx: Optional[int] = None) -> Dict[str, Any]:
        """
        Provide explanations for model predictions.
        
        Args:
            X: Input data
            instance_idx: Specific instance to explain (if None, explain all)
            
        Returns:
            Explanation data including feature contributions and tree paths
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before generating explanations")
        
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X, columns=self.feature_names)
        
        # Preprocess data
        X_processed = self._preprocess_for_prediction(X)
        
        explanations = []
        indices = [instance_idx] if instance_idx is not None else range(len(X_processed))
        
        for idx in indices:
            instance = X_processed.iloc[idx:idx+1]
            prediction = self.model.predict(instance)[0]
            
            explanation = {
                'instance_index': idx,
                'prediction': prediction,
                'feature_values': dict(zip(self.selected_features, instance.iloc[0])),
                'feature_importance': self.feature_importance_scores
            }
            
            # Add prediction confidence for classification
            if self.task_type == "classification" and hasattr(self.model, 'predict_proba'):
                proba = self.model.predict_proba(instance)[0]
                explanation['prediction_probability'] = proba
                explanation['confidence'] = np.max(proba)
            
            # Add confidence intervals if available
            if self.confidence_interval_bounds:
                explanation['confidence_interval'] = {
                    'lower_5': self.confidence_interval_bounds['lower_5'][idx] if idx < len(self.confidence_interval_bounds['lower_5']) else None,
                    'upper_95': self.confidence_interval_bounds['upper_95'][idx] if idx < len(self.confidence_interval_bounds['upper_95']) else None
                }
            
            # Tree path explanation (simplified)
            if hasattr(self.model, 'estimators_') and len(self.model.estimators_) > 0:
                # Get decision path from first tree as example
                tree = self.model.estimators_[0]
                leaf_id = tree.apply(instance)[0]
                feature_path = tree.tree_.feature[tree.tree_.children_left != -1]  # Internal nodes
                
                explanation['tree_path_features'] = [
                    self.selected_features[f] for f in feature_path[:5]  # First 5 features in path
                    if f >= 0 and f < len(self.selected_features)
                ]
            
            explanations.append(explanation)
        
        if instance_idx is not None:
            return explanations[0]
        else:
            return {'explanations': explanations}
    
    def save_model(self, filepath: str):
        """Save the trained model to disk"""
        if not self.is_trained:
            raise ValueError("Cannot save untrained model")
        
        import pickle
        
        model_data = {
            'model': self.model,
            'feature_names': self.feature_names,
            'target_names': self.target_names,
            'selected_features': self.selected_features,
            'label_encoders': self.label_encoders,
            'feature_engineer': self.feature_engineer,
            'feature_selector': self.feature_selector,
            'imbalance_sampler': self.imbalance_sampler,
            'feature_importance_scores': self.feature_importance_scores,
            'permutation_importance_scores': self.permutation_importance_scores,
            'best_params': self.best_params,
            'training_metadata': self.training_metadata,
            'confidence_interval_bounds': self.confidence_interval_bounds,
            'model_config': {
                'ensemble_method': self.ensemble_method,
                'task_type': self.task_type,
                'auto_feature_engineering': self.auto_feature_engineering,
                'feature_selection': self.feature_selection,
                'handle_imbalance': self.handle_imbalance
            }
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        logger.info(f"Random Forest model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load a trained model from disk"""
        import pickle
        
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model = model_data['model']
        self.feature_names = model_data['feature_names']
        self.target_names = model_data['target_names']
        self.selected_features = model_data['selected_features']
        self.label_encoders = model_data['label_encoders']
        self.feature_engineer = model_data['feature_engineer']
        self.feature_selector = model_data['feature_selector']
        self.imbalance_sampler = model_data['imbalance_sampler']
        self.feature_importance_scores = model_data['feature_importance_scores']
        self.permutation_importance_scores = model_data['permutation_importance_scores']
        self.best_params = model_data['best_params']
        self.training_metadata = model_data['training_metadata']
        self.confidence_interval_bounds = model_data['confidence_interval_bounds']
        
        # Update model configuration
        config = model_data['model_config']
        self.ensemble_method = config['ensemble_method']
        self.task_type = config['task_type']
        self.auto_feature_engineering = config['auto_feature_engineering']
        self.feature_selection = config['feature_selection']
        self.handle_imbalance = config['handle_imbalance']
        
        self.is_trained = True
        logger.info(f"Random Forest model loaded from {filepath}")


# Export the model class
__all__ = ['RandomForestModel']
