"""
Agentic AI System for Autonomous Flood Prediction
=================================================

This system implements multiple autonomous agents that work together to:
- Monitor data quality and drift
- Automatically select and optimize models
- Engineer features autonomously
- Monitor performance and adapt
- Make deployment decisions
- Coordinate all activities through a central orchestrator
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Any, Optional, Tuple
from abc import ABC, abstractmethod
import json
import pickle
import os
from dataclasses import dataclass, asdict
import warnings
from threading import Thread, Lock
import time
import asyncio
from concurrent.futures import ThreadPoolExecutor

# ML Libraries
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.ensemble import RandomForestRegressor, VotingRegressor, StackingRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
import optuna
from scipy import stats
import joblib

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

@dataclass
class AgentConfig:
    """Configuration for individual agents"""
    name: str
    enabled: bool = True
    update_frequency: int = 3600  # seconds
    performance_threshold: float = 0.85
    max_retries: int = 3
    timeout: int = 300  # seconds

@dataclass
class ModelPerformance:
    """Model performance metrics"""
    model_name: str
    rmse: float
    r2: float
    mae: float
    timestamp: datetime
    cross_val_score: float
    feature_count: int

@dataclass
class DataQualityReport:
    """Data quality assessment"""
    timestamp: datetime
    missing_values: Dict[str, float]
    data_drift_score: float
    feature_correlations: Dict[str, float]
    outlier_percentage: float
    distribution_changes: Dict[str, float]

class BaseAgent(ABC):
    """Base class for all agents"""
    
    def __init__(self, config: AgentConfig):
        self.config = config
        self.logger = logging.getLogger(f"Agent.{config.name}")
        self.is_running = False
        self.last_execution = None
        self.performance_history = []
        self.lock = Lock()
        
    @abstractmethod
    async def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Main execution method for the agent"""
        pass
    
    def should_execute(self) -> bool:
        """Check if agent should execute based on frequency"""
        if not self.config.enabled:
            return False
        
        if self.last_execution is None:
            return True
            
        time_since_last = datetime.now() - self.last_execution
        return time_since_last.total_seconds() >= self.config.update_frequency
    
    async def run_safely(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute with error handling and retries"""
        for attempt in range(self.config.max_retries):
            try:
                result = await asyncio.wait_for(
                    self.execute(context), 
                    timeout=self.config.timeout
                )
                self.last_execution = datetime.now()
                return result
            except Exception as e:
                self.logger.error(f"Attempt {attempt + 1} failed: {str(e)}")
                if attempt == self.config.max_retries - 1:
                    raise
                await asyncio.sleep(2 ** attempt)  # Exponential backoff

class DataMonitoringAgent(BaseAgent):
    """Agent for monitoring data quality and detecting drift"""
    
    def __init__(self, config: AgentConfig):
        super().__init__(config)
        self.baseline_stats = None
        self.drift_threshold = 0.3
        
    async def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Monitor data quality and detect drift"""
        try:
            data = context.get('current_data')
            if data is None or data.empty:
                return {'status': 'no_data', 'action_required': False}
            
            quality_report = self._assess_data_quality(data)
            drift_detected = self._detect_drift(data)
            
            action_required = (
                quality_report.outlier_percentage > 0.15 or
                quality_report.data_drift_score > self.drift_threshold or
                any(missing > 0.1 for missing in quality_report.missing_values.values())
            )
            
            result = {
                'status': 'completed',
                'quality_report': asdict(quality_report),
                'drift_detected': drift_detected,
                'action_required': action_required,
                'recommendations': self._generate_recommendations(quality_report, drift_detected)
            }
            
            self.logger.info(f"Data monitoring completed. Drift detected: {drift_detected}, Action required: {action_required}")
            return result
            
        except Exception as e:
            self.logger.error(f"Data monitoring failed: {str(e)}")
            raise
    
    def _assess_data_quality(self, data: pd.DataFrame) -> DataQualityReport:
        """Assess data quality metrics"""
        missing_values = {col: data[col].isna().sum() / len(data) for col in data.columns}
        
        # Calculate outliers using IQR method
        outlier_count = 0
        for col in data.select_dtypes(include=[np.number]).columns:
            Q1 = data[col].quantile(0.25)
            Q3 = data[col].quantile(0.75)
            IQR = Q3 - Q1
            outliers = ((data[col] < (Q1 - 1.5 * IQR)) | (data[col] > (Q3 + 1.5 * IQR))).sum()
            outlier_count += outliers
        
        outlier_percentage = outlier_count / (len(data) * len(data.select_dtypes(include=[np.number]).columns))
        
        # Feature correlations
        numeric_data = data.select_dtypes(include=[np.number])
        feature_correlations = {}
        if len(numeric_data.columns) > 1:
            corr_matrix = numeric_data.corr()
            for i, col1 in enumerate(corr_matrix.columns):
                for j, col2 in enumerate(corr_matrix.columns):
                    if i < j:
                        feature_correlations[f"{col1}_vs_{col2}"] = corr_matrix.iloc[i, j]
        
        return DataQualityReport(
            timestamp=datetime.now(),
            missing_values=missing_values,
            data_drift_score=self._calculate_drift_score(data),
            feature_correlations=feature_correlations,
            outlier_percentage=outlier_percentage,
            distribution_changes=self._calculate_distribution_changes(data)
        )
    
    def _detect_drift(self, data: pd.DataFrame) -> bool:
        """Detect data drift using statistical tests"""
        if self.baseline_stats is None:
            self.baseline_stats = self._calculate_baseline_stats(data)
            return False
        
        current_stats = self._calculate_baseline_stats(data)
        drift_detected = False
        
        for feature in current_stats:
            if feature in self.baseline_stats:
                # Use Kolmogorov-Smirnov test for drift detection
                try:
                    baseline_values = self.baseline_stats[feature]
                    current_values = current_stats[feature]
                    
                    if len(baseline_values) > 0 and len(current_values) > 0:
                        ks_stat, p_value = stats.ks_2samp(baseline_values, current_values)
                        if p_value < 0.05:  # Significant difference
                            drift_detected = True
                            self.logger.warning(f"Drift detected in feature {feature}: KS stat={ks_stat:.4f}, p-value={p_value:.4f}")
                except Exception as e:
                    self.logger.warning(f"Could not perform drift test for {feature}: {str(e)}")
        
        return drift_detected
    
    def _calculate_baseline_stats(self, data: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Calculate baseline statistics for drift detection"""
        stats = {}
        for col in data.select_dtypes(include=[np.number]).columns:
            stats[col] = data[col].dropna().values
        return stats
    
    def _calculate_drift_score(self, data: pd.DataFrame) -> float:
        """Calculate overall drift score"""
        if self.baseline_stats is None:
            return 0.0
        
        drift_scores = []
        for col in data.select_dtypes(include=[np.number]).columns:
            if col in self.baseline_stats:
                try:
                    current_mean = data[col].mean()
                    current_std = data[col].std()
                    baseline_mean = np.mean(self.baseline_stats[col])
                    baseline_std = np.std(self.baseline_stats[col])
                    
                    if baseline_std > 0:
                        mean_drift = abs(current_mean - baseline_mean) / baseline_std
                        std_drift = abs(current_std - baseline_std) / baseline_std
                        drift_scores.append((mean_drift + std_drift) / 2)
                except Exception:
                    continue
        
        return np.mean(drift_scores) if drift_scores else 0.0
    
    def _calculate_distribution_changes(self, data: pd.DataFrame) -> Dict[str, float]:
        """Calculate distribution changes for numerical features"""
        changes = {}
        if self.baseline_stats is None:
            return changes
        
        for col in data.select_dtypes(include=[np.number]).columns:
            if col in self.baseline_stats:
                try:
                    baseline_values = self.baseline_stats[col]
                    current_values = data[col].dropna().values
                    
                    if len(baseline_values) > 0 and len(current_values) > 0:
                        ks_stat, _ = stats.ks_2samp(baseline_values, current_values)
                        changes[col] = ks_stat
                except Exception:
                    changes[col] = 0.0
        
        return changes
    
    def _generate_recommendations(self, quality_report: DataQualityReport, drift_detected: bool) -> List[str]:
        """Generate actionable recommendations"""
        recommendations = []
        
        if any(missing > 0.1 for missing in quality_report.missing_values.values()):
            recommendations.append("High missing values detected - consider data imputation or collection review")
        
        if quality_report.outlier_percentage > 0.15:
            recommendations.append("High outlier percentage - consider outlier removal or robust scaling")
        
        if drift_detected:
            recommendations.append("Data drift detected - retrain model with recent data")
        
        if quality_report.data_drift_score > self.drift_threshold:
            recommendations.append("Significant distribution changes - update feature engineering pipeline")
        
        return recommendations

class ModelSelectionAgent(BaseAgent):
    """Agent for intelligent model selection and optimization"""
    
    def __init__(self, config: AgentConfig):
        super().__init__(config)
        self.model_registry = self._initialize_model_registry()
        self.best_model = None
        self.best_performance = None
        
    def _initialize_model_registry(self) -> Dict[str, Any]:
        """Initialize available models"""
        return {
            'random_forest': {
                'class': RandomForestRegressor,
                'params': {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [10, 20, None],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4]
                }
            },
            'xgboost': {
                'class': XGBRegressor,
                'params': {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [3, 6, 10],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'subsample': [0.8, 0.9, 1.0]
                }
            },
            'lightgbm': {
                'class': LGBMRegressor,
                'params': {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [3, 6, 10],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'num_leaves': [31, 50, 100]
                }
            },
            'ridge': {
                'class': Ridge,
                'params': {
                    'alpha': [0.1, 1.0, 10.0, 100.0]
                }
            },
            'elastic_net': {
                'class': ElasticNet,
                'params': {
                    'alpha': [0.1, 1.0, 10.0],
                    'l1_ratio': [0.1, 0.5, 0.9]
                }
            }
        }
    
    async def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute model selection and optimization"""
        try:
            X_train = context.get('X_train')
            y_train = context.get('y_train')
            X_test = context.get('X_test', None)
            y_test = context.get('y_test', None)
            
            if X_train is None or y_train is None:
                return {'status': 'no_data', 'action_required': False}
            
            # Optimize models using Optuna
            best_models = await self._optimize_models(X_train, y_train)
            
            # Evaluate models
            evaluations = self._evaluate_models(best_models, X_train, y_train, X_test, y_test)
            
            # Select best model
            best_model_info = max(evaluations, key=lambda x: x['cross_val_score'])
            
            # Create ensemble if beneficial
            ensemble_model = self._create_ensemble(best_models, X_train, y_train)
            if ensemble_model:
                ensemble_eval = self._evaluate_single_model(ensemble_model, X_train, y_train, X_test, y_test, 'ensemble')
                if ensemble_eval['cross_val_score'] > best_model_info['cross_val_score']:
                    best_model_info = ensemble_eval
                    best_model_info['model'] = ensemble_model
            
            self.best_model = best_model_info['model']
            self.best_performance = ModelPerformance(
                model_name=best_model_info['name'],
                rmse=best_model_info['rmse'],
                r2=best_model_info['r2'],
                mae=best_model_info['mae'],
                timestamp=datetime.now(),
                cross_val_score=best_model_info['cross_val_score'],
                feature_count=X_train.shape[1]
            )
            
            result = {
                'status': 'completed',
                'best_model': best_model_info,
                'all_evaluations': evaluations,
                'performance': asdict(self.best_performance),
                'action_required': True,
                'model_path': self._save_model(self.best_model, best_model_info['name'])
            }
            
            self.logger.info(f"Model selection completed. Best model: {best_model_info['name']} with CV score: {best_model_info['cross_val_score']:.4f}")
            return result
            
        except Exception as e:
            self.logger.error(f"Model selection failed: {str(e)}")
            raise
    
    async def _optimize_models(self, X_train: pd.DataFrame, y_train: pd.Series) -> Dict[str, Any]:
        """Optimize models using Optuna"""
        best_models = {}
        
        for model_name, model_config in self.model_registry.items():
            try:
                study = optuna.create_study(direction='maximize', 
                                          study_name=f"{model_name}_optimization",
                                          sampler=optuna.samplers.TPESampler())
                
                def objective(trial):
                    # Suggest hyperparameters
                    params = {}
                    for param_name, param_values in model_config['params'].items():
                        if isinstance(param_values[0], int):
                            params[param_name] = trial.suggest_int(param_name, min(param_values), max(param_values))
                        elif isinstance(param_values[0], float):
                            params[param_name] = trial.suggest_float(param_name, min(param_values), max(param_values))
                        else:
                            params[param_name] = trial.suggest_categorical(param_name, param_values)
                    
                    # Create and evaluate model
                    model = model_config['class'](**params, random_state=42)
                    scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
                    return scores.mean()
                
                study.optimize(objective, n_trials=50, timeout=120)  # 2 minutes per model
                
                # Create best model
                best_model = model_config['class'](**study.best_params, random_state=42)
                best_model.fit(X_train, y_train)
                best_models[model_name] = {
                    'model': best_model,
                    'params': study.best_params,
                    'score': study.best_value
                }
                
                self.logger.info(f"Optimized {model_name}: score={study.best_value:.4f}")
                
            except Exception as e:
                self.logger.warning(f"Failed to optimize {model_name}: {str(e)}")
                # Fallback to default parameters
                default_model = model_config['class'](random_state=42)
                default_model.fit(X_train, y_train)
                best_models[model_name] = {
                    'model': default_model,
                    'params': {},
                    'score': 0.0
                }
        
        return best_models
    
    def _evaluate_models(self, models: Dict[str, Any], X_train: pd.DataFrame, y_train: pd.Series, 
                        X_test: Optional[pd.DataFrame], y_test: Optional[pd.Series]) -> List[Dict[str, Any]]:
        """Evaluate all models"""
        evaluations = []
        
        for name, model_info in models.items():
            eval_result = self._evaluate_single_model(
                model_info['model'], X_train, y_train, X_test, y_test, name
            )
            evaluations.append(eval_result)
        
        return evaluations
    
    def _evaluate_single_model(self, model: Any, X_train: pd.DataFrame, y_train: pd.Series,
                              X_test: Optional[pd.DataFrame], y_test: Optional[pd.Series], 
                              name: str) -> Dict[str, Any]:
        """Evaluate a single model"""
        # Cross-validation score
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
        cv_score = cv_scores.mean()
        
        # Training predictions
        y_train_pred = model.predict(X_train)
        train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
        train_r2 = r2_score(y_train, y_train_pred)
        train_mae = mean_absolute_error(y_train, y_train_pred)
        
        result = {
            'name': name,
            'model': model,
            'cross_val_score': cv_score,
            'cross_val_std': cv_scores.std(),
            'train_rmse': train_rmse,
            'train_r2': train_r2,
            'train_mae': train_mae,
            'rmse': train_rmse,
            'r2': train_r2,
            'mae': train_mae
        }
        
        # Test set evaluation if available
        if X_test is not None and y_test is not None:
            y_test_pred = model.predict(X_test)
            test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
            test_r2 = r2_score(y_test, y_test_pred)
            test_mae = mean_absolute_error(y_test, y_test_pred)
            
            result.update({
                'test_rmse': test_rmse,
                'test_r2': test_r2,
                'test_mae': test_mae,
                'rmse': test_rmse,
                'r2': test_r2,
                'mae': test_mae
            })
        
        return result
    
    def _create_ensemble(self, models: Dict[str, Any], X_train: pd.DataFrame, y_train: pd.Series) -> Optional[Any]:
        """Create ensemble model from best performers"""
        try:
            # Select top 3 models
            sorted_models = sorted(models.items(), key=lambda x: x[1]['score'], reverse=True)[:3]
            
            if len(sorted_models) < 2:
                return None
            
            # Create voting ensemble
            estimators = [(name, model_info['model']) for name, model_info in sorted_models]
            ensemble = VotingRegressor(estimators=estimators)
            ensemble.fit(X_train, y_train)
            
            return ensemble
            
        except Exception as e:
            self.logger.warning(f"Failed to create ensemble: {str(e)}")
            return None
    
    def _save_model(self, model: Any, model_name: str) -> str:
        """Save model to disk"""
        os.makedirs('models', exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = f'models/{model_name}_{timestamp}.pkl'
        
        try:
            joblib.dump(model, model_path)
            self.logger.info(f"Model saved to {model_path}")
            return model_path
        except Exception as e:
            self.logger.error(f"Failed to save model: {str(e)}")
            return ""

class FeatureEngineeringAgent(BaseAgent):
    """Agent for automated feature discovery and engineering"""
    
    def __init__(self, config: AgentConfig):
        super().__init__(config)
        self.feature_history = []
        self.successful_features = set()
        
    async def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute feature engineering"""
        try:
            X_train = context.get('X_train')
            y_train = context.get('y_train')
            
            if X_train is None or y_train is None:
                return {'status': 'no_data', 'action_required': False}
            
            original_features = X_train.shape[1]
            
            # Generate new features
            X_engineered = self._engineer_features(X_train.copy())
            
            # Select best features
            X_selected, selected_features = self._select_features(X_engineered, y_train)
            
            # Evaluate feature impact
            improvement = self._evaluate_feature_impact(X_train, X_selected, y_train)
            
            result = {
                'status': 'completed',
                'original_features': original_features,
                'engineered_features': X_engineered.shape[1],
                'selected_features': X_selected.shape[1],
                'feature_names': selected_features,
                'improvement': improvement,
                'action_required': improvement > 0.01,  # 1% improvement threshold
                'engineered_data': X_selected
            }
            
            if improvement > 0.01:
                self.successful_features.update(selected_features)
            
            self.logger.info(f"Feature engineering completed. Features: {original_features} → {X_selected.shape[1]}, Improvement: {improvement:.4f}")
            return result
            
        except Exception as e:
            self.logger.error(f"Feature engineering failed: {str(e)}")
            raise
    
    def _engineer_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Generate new features automatically"""
        X_new = X.copy()
        
        # Get numerical columns
        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        
        # Polynomial features for top correlated features
        if len(numeric_cols) >= 2:
            for i, col1 in enumerate(numeric_cols[:5]):  # Limit to top 5 to avoid explosion
                for col2 in numeric_cols[i+1:6]:
                    # Interaction features
                    X_new[f'{col1}_x_{col2}'] = X[col1] * X[col2]
                    
                    # Ratio features (avoid division by zero)
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        ratio = X[col1] / (X[col2] + 1e-8)
                        X_new[f'{col1}_div_{col2}'] = np.where(np.isfinite(ratio), ratio, 0)
        
        # Polynomial features (degree 2)
        for col in numeric_cols[:8]:  # Limit to prevent too many features
            X_new[f'{col}_squared'] = X[col] ** 2
            X_new[f'{col}_sqrt'] = np.sqrt(np.abs(X[col]))
            X_new[f'{col}_log'] = np.log1p(np.abs(X[col]))
        
        # Statistical features
        if len(numeric_cols) >= 3:
            row_stats = X[numeric_cols].agg(['mean', 'std', 'min', 'max'], axis=1)
            for stat in row_stats.columns:
                X_new[f'row_{stat}'] = row_stats[stat]
        
        # Binning features for continuous variables
        for col in numeric_cols[:5]:
            try:
                X_new[f'{col}_binned'] = pd.cut(X[col], bins=5, labels=False, duplicates='drop')
            except Exception:
                continue
        
        # Rolling statistics if we have enough data
        if len(X) > 10:
            for col in numeric_cols[:3]:
                # Simple rolling mean (assuming temporal ordering)
                X_new[f'{col}_rolling_mean'] = X[col].rolling(window=min(5, len(X)//2), min_periods=1).mean()
        
        return X_new
    
    def _select_features(self, X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, List[str]]:
        """Select best features using various methods"""
        from sklearn.feature_selection import SelectKBest, f_regression, RFE
        from sklearn.linear_model import LinearRegression
        
        # Remove features with high missing values
        missing_threshold = 0.5
        valid_features = [col for col in X.columns if X[col].isna().sum() / len(X) < missing_threshold]
        X_clean = X[valid_features].fillna(X[valid_features].median())
        
        # Remove constant features
        non_constant = [col for col in X_clean.columns if X_clean[col].nunique() > 1]
        X_clean = X_clean[non_constant]
        
        # Remove highly correlated features
        corr_matrix = X_clean.corr().abs()
        upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > 0.95)]
        X_clean = X_clean.drop(columns=to_drop)
        
        # Statistical feature selection
        max_features = min(50, X_clean.shape[1])  # Limit to prevent overfitting
        selector = SelectKBest(score_func=f_regression, k=max_features)
        
        try:
            X_selected = selector.fit_transform(X_clean, y)
            selected_features = X_clean.columns[selector.get_support()].tolist()
            X_result = pd.DataFrame(X_selected, columns=selected_features, index=X.index)
        except Exception as e:
            self.logger.warning(f"Feature selection failed: {str(e)}, using original features")
            selected_features = X_clean.columns.tolist()
            X_result = X_clean
        
        return X_result, selected_features
    
    def _evaluate_feature_impact(self, X_original: pd.DataFrame, X_engineered: pd.DataFrame, 
                                y: pd.Series) -> float:
        """Evaluate impact of feature engineering"""
        try:
            # Quick model evaluation
            from sklearn.ensemble import RandomForestRegressor
            
            model = RandomForestRegressor(n_estimators=50, random_state=42)
            
            # Original features score
            original_scores = cross_val_score(model, X_original, y, cv=3, scoring='r2')
            original_score = original_scores.mean()
            
            # Engineered features score
            engineered_scores = cross_val_score(model, X_engineered, y, cv=3, scoring='r2')
            engineered_score = engineered_scores.mean()
            
            improvement = engineered_score - original_score
            return improvement
            
        except Exception as e:
            self.logger.warning(f"Could not evaluate feature impact: {str(e)}")
            return 0.0

class PerformanceMonitoringAgent(BaseAgent):
    """Agent for monitoring model performance and triggering actions"""
    
    def __init__(self, config: AgentConfig):
        super().__init__(config)
        self.performance_history = []
        self.alert_thresholds = {
            'r2_decline': 0.05,
            'rmse_increase': 0.1,
            'consecutive_declines': 3
        }
        
    async def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Monitor performance and detect degradation"""
        try:
            current_performance = context.get('current_performance')
            if current_performance is None:
                return {'status': 'no_performance_data', 'action_required': False}
            
            # Add to history
            self.performance_history.append(current_performance)
            
            # Keep only recent history (last 50 records)
            if len(self.performance_history) > 50:
                self.performance_history = self.performance_history[-50:]
            
            # Analyze trends
            degradation_detected = self._detect_performance_degradation()
            alerts = self._generate_alerts()
            recommendations = self._generate_performance_recommendations()
            
            # Determine if action is required
            action_required = (
                degradation_detected or 
                len(alerts) > 0 or
                current_performance.r2 < self.config.performance_threshold
            )
            
            result = {
                'status': 'completed',
                'current_performance': asdict(current_performance),
                'degradation_detected': degradation_detected,
                'alerts': alerts,
                'recommendations': recommendations,
                'action_required': action_required,
                'trend_analysis': self._analyze_trends()
            }
            
            self.logger.info(f"Performance monitoring completed. Degradation: {degradation_detected}, Alerts: {len(alerts)}")
            return result
            
        except Exception as e:
            self.logger.error(f"Performance monitoring failed: {str(e)}")
            raise
    
    def _detect_performance_degradation(self) -> bool:
        """Detect if model performance is degrading"""
        if len(self.performance_history) < 2:
            return False
        
        recent_performances = self.performance_history[-5:]  # Last 5 records
        
        # Check for consistent decline in R²
        r2_scores = [p.r2 for p in recent_performances]
        if len(r2_scores) >= self.alert_thresholds['consecutive_declines']:
            consecutive_declines = 0
            for i in range(1, len(r2_scores)):
                if r2_scores[i] < r2_scores[i-1]:
                    consecutive_declines += 1
                else:
                    consecutive_declines = 0
            
            if consecutive_declines >= self.alert_thresholds['consecutive_declines'] - 1:
                return True
        
        # Check for significant performance drop
        if len(self.performance_history) >= 10:
            baseline_r2 = np.mean([p.r2 for p in self.performance_history[-10:-5]])
            current_r2 = np.mean([p.r2 for p in self.performance_history[-5:]])
            
            if baseline_r2 - current_r2 > self.alert_thresholds['r2_decline']:
                return True
        
        return False
    
    def _generate_alerts(self) -> List[Dict[str, Any]]:
        """Generate performance alerts"""
        alerts = []
        
        if not self.performance_history:
            return alerts
        
        current = self.performance_history[-1]
        
        # Low R² alert
        if current.r2 < self.config.performance_threshold:
            alerts.append({
                'type': 'low_performance',
                'severity': 'high',
                'message': f"Model R² ({current.r2:.4f}) below threshold ({self.config.performance_threshold})",
                'timestamp': current.timestamp
            })
        
        # High RMSE alert
        if len(self.performance_history) >= 2:
            previous = self.performance_history[-2]
            rmse_increase = (current.rmse - previous.rmse) / previous.rmse
            
            if rmse_increase > self.alert_thresholds['rmse_increase']:
                alerts.append({
                    'type': 'rmse_increase',
                    'severity': 'medium',
                    'message': f"RMSE increased by {rmse_increase:.2%}",
                    'timestamp': current.timestamp
                })
        
        return alerts
    
    def _generate_performance_recommendations(self) -> List[str]:
        """Generate recommendations based on performance analysis"""
        recommendations = []
        
        if not self.performance_history:
            return recommendations
        
        current = self.performance_history[-1]
        
        if current.r2 < 0.7:
            recommendations.append("Low R² score - consider feature engineering or model selection")
        
        if self._detect_performance_degradation():
            recommendations.append("Performance degradation detected - retrain model with recent data")
        
        if current.feature_count > 100:
            recommendations.append("High feature count - consider feature selection to reduce overfitting")
        
        return recommendations
    
    def _analyze_trends(self) -> Dict[str, Any]:
        """Analyze performance trends"""
        if len(self.performance_history) < 3:
            return {'trend': 'insufficient_data'}
        
        recent_r2 = [p.r2 for p in self.performance_history[-10:]]
        recent_rmse = [p.rmse for p in self.performance_history[-10:]]
        
        # Calculate trends
        from scipy.stats import linregress
        
        x = range(len(recent_r2))
        r2_slope, _, r2_r_value, _, _ = linregress(x, recent_r2)
        rmse_slope, _, rmse_r_value, _, _ = linregress(x, recent_rmse)
        
        trend_analysis = {
            'r2_trend': 'improving' if r2_slope > 0 else 'declining',
            'rmse_trend': 'improving' if rmse_slope < 0 else 'declining',
            'r2_slope': r2_slope,
            'rmse_slope': rmse_slope,
            'trend_strength': {
                'r2': abs(r2_r_value),
                'rmse': abs(rmse_r_value)
            }
        }
        
        return trend_analysis

class DecisionMakingAgent(BaseAgent):
    """Agent for autonomous decision making and deployment"""
    
    def __init__(self, config: AgentConfig):
        super().__init__(config)
        self.decision_history = []
        self.deployment_criteria = {
            'min_r2': 0.8,
            'max_rmse_increase': 0.1,
            'min_improvement': 0.02
        }
        
    async def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Make autonomous decisions about model deployment and actions"""
        try:
            # Gather all agent results
            monitoring_result = context.get('monitoring_result', {})
            model_result = context.get('model_result', {})
            feature_result = context.get('feature_result', {})
            performance_result = context.get('performance_result', {})
            
            # Make decisions
            decisions = await self._make_decisions(
                monitoring_result, model_result, feature_result, performance_result
            )
            
            # Execute approved decisions
            executed_actions = await self._execute_decisions(decisions, context)
            
            # Record decision
            decision_record = {
                'timestamp': datetime.now(),
                'decisions': decisions,
                'executed_actions': executed_actions,
                'context_summary': self._summarize_context(context)
            }
            self.decision_history.append(decision_record)
            
            result = {
                'status': 'completed',
                'decisions': decisions,
                'executed_actions': executed_actions,
                'action_required': len(executed_actions) > 0
            }
            
            self.logger.info(f"Decision making completed. Decisions: {len(decisions)}, Actions executed: {len(executed_actions)}")
            return result
            
        except Exception as e:
            self.logger.error(f"Decision making failed: {str(e)}")
            raise
    
    async def _make_decisions(self, monitoring_result: Dict, model_result: Dict, 
                            feature_result: Dict, performance_result: Dict) -> List[Dict[str, Any]]:
        """Make autonomous decisions based on agent results"""
        decisions = []
        
        # Decision 1: Model retraining
        if self._should_retrain_model(monitoring_result, performance_result):
            decisions.append({
                'type': 'retrain_model',
                'priority': 'high',
                'reason': 'Performance degradation or data drift detected',
                'auto_execute': True
            })
        
        # Decision 2: Feature engineering deployment
        if self._should_deploy_features(feature_result):
            decisions.append({
                'type': 'deploy_features',
                'priority': 'medium',
                'reason': 'Significant feature improvement detected',
                'auto_execute': True
            })
        
        # Decision 3: Model deployment
        if self._should_deploy_model(model_result):
            decisions.append({
                'type': 'deploy_model',
                'priority': 'high',
                'reason': 'New model meets deployment criteria',
                'auto_execute': True
            })
        
        # Decision 4: Alert escalation
        if self._should_escalate_alerts(performance_result):
            decisions.append({
                'type': 'escalate_alerts',
                'priority': 'critical',
                'reason': 'Critical performance issues require human intervention',
                'auto_execute': False
            })
        
        # Decision 5: Data collection
        if self._should_collect_more_data(monitoring_result):
            decisions.append({
                'type': 'collect_data',
                'priority': 'medium',
                'reason': 'Data quality issues or drift detected',
                'auto_execute': False
            })
        
        return decisions
    
    def _should_retrain_model(self, monitoring_result: Dict, performance_result: Dict) -> bool:
        """Decide if model should be retrained"""
        # Data drift detected
        if monitoring_result.get('drift_detected', False):
            return True
        
        # Performance degradation
        if performance_result.get('degradation_detected', False):
            return True
        
        # Low current performance
        current_perf = performance_result.get('current_performance', {})
        if current_perf.get('r2', 1.0) < self.deployment_criteria['min_r2']:
            return True
        
        return False
    
    def _should_deploy_features(self, feature_result: Dict) -> bool:
        """Decide if new features should be deployed"""
        improvement = feature_result.get('improvement', 0)
        return improvement > self.deployment_criteria['min_improvement']
    
    def _should_deploy_model(self, model_result: Dict) -> bool:
        """Decide if new model should be deployed"""
        if not model_result.get('best_model'):
            return False
        
        performance = model_result.get('performance', {})
        r2_score = performance.get('r2', 0)
        
        return r2_score >= self.deployment_criteria['min_r2']
    
    def _should_escalate_alerts(self, performance_result: Dict) -> bool:
        """Decide if alerts should be escalated to humans"""
        alerts = performance_result.get('alerts', [])
        critical_alerts = [a for a in alerts if a.get('severity') == 'high']
        
        return len(critical_alerts) > 0
    
    def _should_collect_more_data(self, monitoring_result: Dict) -> bool:
        """Decide if more data should be collected"""
        quality_report = monitoring_result.get('quality_report', {})
        
        # High missing values
        missing_values = quality_report.get('missing_values', {})
        if any(missing > 0.2 for missing in missing_values.values()):
            return True
        
        # High outlier percentage
        if quality_report.get('outlier_percentage', 0) > 0.2:
            return True
        
        return False
    
    async def _execute_decisions(self, decisions: List[Dict[str, Any]], context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Execute approved decisions"""
        executed_actions = []
        
        for decision in decisions:
            if not decision.get('auto_execute', False):
                continue
            
            try:
                action_result = await self._execute_single_decision(decision, context)
                executed_actions.append(action_result)
                
            except Exception as e:
                self.logger.error(f"Failed to execute decision {decision['type']}: {str(e)}")
                executed_actions.append({
                    'decision_type': decision['type'],
                    'status': 'failed',
                    'error': str(e)
                })
        
        return executed_actions
    
    async def _execute_single_decision(self, decision: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a single decision"""
        decision_type = decision['type']
        
        if decision_type == 'deploy_model':
            return await self._deploy_model(context)
        elif decision_type == 'deploy_features':
            return await self._deploy_features(context)
        elif decision_type == 'retrain_model':
            return await self._trigger_retraining(context)
        else:
            return {
                'decision_type': decision_type,
                'status': 'logged',
                'message': f"Decision {decision_type} logged for human review"
            }
    
    async def _deploy_model(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Deploy new model"""
        model_result = context.get('model_result', {})
        model_path = model_result.get('model_path', '')
        
        if model_path and os.path.exists(model_path):
            # In a real system, this would deploy to production
            deployment_path = 'models/production_model.pkl'
            os.makedirs(os.path.dirname(deployment_path), exist_ok=True)
            
            import shutil
            shutil.copy2(model_path, deployment_path)
            
            return {
                'decision_type': 'deploy_model',
                'status': 'success',
                'deployment_path': deployment_path,
                'timestamp': datetime.now()
            }
        else:
            return {
                'decision_type': 'deploy_model',
                'status': 'failed',
                'error': 'No valid model to deploy'
            }
    
    async def _deploy_features(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Deploy new feature engineering pipeline"""
        feature_result = context.get('feature_result', {})
        
        if feature_result.get('action_required', False):
            # Save feature configuration
            feature_config = {
                'selected_features': feature_result.get('feature_names', []),
                'improvement': feature_result.get('improvement', 0),
                'timestamp': datetime.now().isoformat()
            }
            
            os.makedirs('configs', exist_ok=True)
            config_path = 'configs/feature_config.json'
            
            with open(config_path, 'w') as f:
                json.dump(feature_config, f, indent=2, default=str)
            
            return {
                'decision_type': 'deploy_features',
                'status': 'success',
                'config_path': config_path,
                'improvement': feature_result.get('improvement', 0)
            }
        else:
            return {
                'decision_type': 'deploy_features',
                'status': 'skipped',
                'reason': 'No significant improvement'
            }
    
    async def _trigger_retraining(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Trigger model retraining"""
        # In a real system, this would trigger a retraining pipeline
        retrain_config = {
            'trigger_time': datetime.now().isoformat(),
            'reason': 'Autonomous decision based on performance/drift',
            'data_version': context.get('data_version', 'latest')
        }
        
        os.makedirs('configs', exist_ok=True)
        config_path = 'configs/retrain_trigger.json'
        
        with open(config_path, 'w') as f:
            json.dump(retrain_config, f, indent=2, default=str)
        
        return {
            'decision_type': 'retrain_model',
            'status': 'triggered',
            'config_path': config_path,
            'trigger_time': datetime.now()
        }
    
    def _summarize_context(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Summarize context for decision record"""
        return {
            'data_available': context.get('X_train') is not None,
            'monitoring_status': context.get('monitoring_result', {}).get('status'),
            'model_status': context.get('model_result', {}).get('status'),
            'feature_status': context.get('feature_result', {}).get('status'),
            'performance_status': context.get('performance_result', {}).get('status')
        }

class AgentOrchestrator:
    """Central orchestrator for managing all agents"""
    
    def __init__(self):
        self.agents = {}
        self.execution_history = []
        self.is_running = False
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # Initialize agents
        self._initialize_agents()
        
        # Setup logging
        self.logger = logging.getLogger("AgentOrchestrator")
        
    def _initialize_agents(self):
        """Initialize all agents with their configurations"""
        agent_configs = {
            'data_monitoring': AgentConfig(
                name='DataMonitoring',
                update_frequency=1800,  # 30 minutes
                performance_threshold=0.85
            ),
            'model_selection': AgentConfig(
                name='ModelSelection',
                update_frequency=7200,  # 2 hours
                performance_threshold=0.85,
                timeout=600  # 10 minutes for model training
            ),
            'feature_engineering': AgentConfig(
                name='FeatureEngineering',
                update_frequency=3600,  # 1 hour
                performance_threshold=0.85,
                timeout=300  # 5 minutes
            ),
            'performance_monitoring': AgentConfig(
                name='PerformanceMonitoring',
                update_frequency=900,  # 15 minutes
                performance_threshold=0.8
            ),
            'decision_making': AgentConfig(
                name='DecisionMaking',
                update_frequency=1800,  # 30 minutes
                performance_threshold=0.8
            )
        }
        
        self.agents = {
            'data_monitoring': DataMonitoringAgent(agent_configs['data_monitoring']),
            'model_selection': ModelSelectionAgent(agent_configs['model_selection']),
            'feature_engineering': FeatureEngineeringAgent(agent_configs['feature_engineering']),
            'performance_monitoring': PerformanceMonitoringAgent(agent_configs['performance_monitoring']),
            'decision_making': DecisionMakingAgent(agent_configs['decision_making'])
        }
    
    async def run_cycle(self, data_context: Dict[str, Any]) -> Dict[str, Any]:
        """Run a complete cycle of all agents"""
        try:
            cycle_start = datetime.now()
            results = {}
            
            self.logger.info("Starting agent orchestration cycle")
            
            # Step 1: Data Monitoring
            if self.agents['data_monitoring'].should_execute():
                self.logger.info("Executing data monitoring agent")
                results['monitoring_result'] = await self.agents['data_monitoring'].run_safely(data_context)
            
            # Step 2: Feature Engineering (if data is good)
            if (not results.get('monitoring_result', {}).get('action_required', True) and 
                self.agents['feature_engineering'].should_execute()):
                self.logger.info("Executing feature engineering agent")
                results['feature_result'] = await self.agents['feature_engineering'].run_safely(data_context)
                
                # Update context with engineered features
                if results['feature_result'].get('action_required', False):
                    data_context['X_train'] = results['feature_result'].get('engineered_data', data_context.get('X_train'))
            
            # Step 3: Model Selection (if needed)
            if self.agents['model_selection'].should_execute():
                self.logger.info("Executing model selection agent")
                results['model_result'] = await self.agents['model_selection'].run_safely(data_context)
            
            # Step 4: Performance Monitoring
            if self.agents['performance_monitoring'].should_execute():
                # Add current performance to context if available
                if 'model_result' in results:
                    model_perf = results['model_result'].get('performance')
                    if model_perf:
                        performance_obj = ModelPerformance(**model_perf)
                        data_context['current_performance'] = performance_obj
                
                self.logger.info("Executing performance monitoring agent")
                results['performance_result'] = await self.agents['performance_monitoring'].run_safely(data_context)
            
            # Step 5: Decision Making (always run if any other agent ran)
            if any(results.values()) and self.agents['decision_making'].should_execute():
                self.logger.info("Executing decision making agent")
                decision_context = {**data_context, **results}
                results['decision_result'] = await self.agents['decision_making'].run_safely(decision_context)
            
            # Record execution
            execution_record = {
                'timestamp': cycle_start,
                'duration': (datetime.now() - cycle_start).total_seconds(),
                'agents_executed': list(results.keys()),
                'success': True,
                'summary': self._generate_cycle_summary(results)
            }
            
            self.execution_history.append(execution_record)
            self.logger.info(f"Orchestration cycle completed in {execution_record['duration']:.2f} seconds")
            
            return {
                'execution_record': execution_record,
                'agent_results': results
            }
            
        except Exception as e:
            self.logger.error(f"Orchestration cycle failed: {str(e)}")
            execution_record = {
                'timestamp': cycle_start,
                'duration': (datetime.now() - cycle_start).total_seconds(),
                'agents_executed': list(results.keys()) if 'results' in locals() else [],
                'success': False,
                'error': str(e)
            }
            self.execution_history.append(execution_record)
            raise
    
    async def start_autonomous_mode(self, data_source: callable, check_interval: int = 300):
        """Start autonomous operation mode"""
        self.is_running = True
        self.logger.info("Starting autonomous mode")
        
        try:
            while self.is_running:
                try:
                    # Get current data from source
                    data_context = await self._get_data_context(data_source)
                    
                    # Run agent cycle
                    cycle_result = await self.run_cycle(data_context)
                    
                    # Log cycle results
                    self._log_cycle_results(cycle_result)
                    
                    # Wait before next cycle
                    await asyncio.sleep(check_interval)
                    
                except Exception as e:
                    self.logger.error(f"Error in autonomous cycle: {str(e)}")
                    await asyncio.sleep(check_interval)  # Continue running despite errors
                    
        except KeyboardInterrupt:
            self.logger.info("Autonomous mode stopped by user")
        finally:
            self.is_running = False
    
    def stop_autonomous_mode(self):
        """Stop autonomous operation"""
        self.is_running = False
        self.logger.info("Stopping autonomous mode")
    
    async def _get_data_context(self, data_source: callable) -> Dict[str, Any]:
        """Get current data context from data source"""
        try:
            if asyncio.iscoroutinefunction(data_source):
                return await data_source()
            else:
                return data_source()
        except Exception as e:
            self.logger.error(f"Failed to get data context: {str(e)}")
            return {}
    
    def _generate_cycle_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate summary of cycle execution"""
        summary = {
            'agents_run': len(results),
            'actions_required': 0,
            'decisions_made': 0,
            'alerts_generated': 0
        }
        
        for agent_name, result in results.items():
            if result.get('action_required', False):
                summary['actions_required'] += 1
            
            if agent_name == 'decision_result':
                summary['decisions_made'] = len(result.get('decisions', []))
            
            if agent_name == 'performance_result':
                summary['alerts_generated'] = len(result.get('alerts', []))
        
        return summary
    
    def _log_cycle_results(self, cycle_result: Dict[str, Any]):
        """Log cycle results for monitoring"""
        execution_record = cycle_result['execution_record']
        agent_results = cycle_result['agent_results']
        
        # Log summary
        summary = execution_record['summary']
        self.logger.info(f"Cycle Summary: {summary['agents_run']} agents run, "
                        f"{summary['actions_required']} actions required, "
                        f"{summary['decisions_made']} decisions made")
        
        # Log important alerts
        perf_result = agent_results.get('performance_result', {})
        alerts = perf_result.get('alerts', [])
        for alert in alerts:
            if alert.get('severity') in ['high', 'critical']:
                self.logger.warning(f"Alert: {alert['message']}")
    
    def get_agent_status(self) -> Dict[str, Any]:
        """Get status of all agents"""
        status = {}
        for name, agent in self.agents.items():
            status[name] = {
                'enabled': agent.config.enabled,
                'last_execution': agent.last_execution,
                'performance_history_length': len(agent.performance_history),
                'is_running': agent.is_running
            }
        return status
    
    def get_execution_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent execution history"""
        return self.execution_history[-limit:]

# Factory function to create and configure the orchestrator
def create_flood_prediction_orchestrator() -> AgentOrchestrator:
    """Create configured orchestrator for flood prediction"""
    orchestrator = AgentOrchestrator()
    
    # Configure agents for flood prediction specifics
    orchestrator.agents['data_monitoring'].drift_threshold = 0.25  # More sensitive for flood data
    orchestrator.agents['decision_making'].deployment_criteria['min_r2'] = 0.82  # Higher threshold for flood prediction
    
    return orchestrator

if __name__ == "__main__":
    # Example usage
    async def example_data_source():
        """Example data source function"""
        # In real implementation, this would load current data
        # For demo, we'll return dummy data
        return {
            'X_train': pd.DataFrame(np.random.randn(100, 10)),
            'y_train': pd.Series(np.random.randn(100)),
            'data_version': 'v1.0'
        }
    
    async def main():
        orchestrator = create_flood_prediction_orchestrator()
        
        # Run single cycle
        data_context = await example_data_source()
        result = await orchestrator.run_cycle(data_context)
        
        print("Orchestration completed successfully!")
        print(f"Agents executed: {result['execution_record']['agents_executed']}")
        print(f"Duration: {result['execution_record']['duration']:.2f} seconds")
    
    # Run example
    asyncio.run(main()) 