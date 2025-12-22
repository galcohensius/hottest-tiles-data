"""
Train APS prediction model using Lasso Regression.

Lasso Regression is good for feature selection and handling multicollinearity.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
import joblib
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import Lasso, LassoCV, ElasticNet, ElasticNetCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import xgboost as xgb
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

MODEL_DIR = REPO_ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)

from scripts.predictions.feature_engineering import load_data, get_feature_matrix_and_target


def train_lasso_model(X_train, y_train, alpha=None, cv=5):
    """
    Train Lasso regression model with optional alpha tuning.
    
    Args:
        X_train: Training features
        y_train: Training targets
        alpha: Regularization strength (if None, uses LassoCV to find optimal)
        cv: Number of CV folds for alpha selection
    
    Returns:
        Trained model and scaler
    """
    # Standardize features (important for Lasso)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    if alpha is None:
        # Use LassoCV to find optimal alpha
        print("   Finding optimal alpha using cross-validation...")
        alphas = np.logspace(-4, 1, 50)  # Range from 0.0001 to 10
        lasso_cv = LassoCV(alphas=alphas, cv=cv, random_state=42, max_iter=2000, n_jobs=-1)
        lasso_cv.fit(X_train_scaled, y_train)
        alpha = lasso_cv.alpha_
        print(f"   Optimal alpha: {alpha:.6f}")
    
    # Train final model with optimal alpha
    model = Lasso(alpha=alpha, max_iter=2000, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    return model, scaler, alpha


def train_elasticnet_model(X_train, y_train, alpha=None, l1_ratio=None, cv=5):
    """
    Train ElasticNet regression model (combines L1 and L2 regularization).
    
    Args:
        X_train: Training features
        y_train: Training targets
        alpha: Regularization strength (if None, uses ElasticNetCV to find optimal)
        l1_ratio: Mix of L1 vs L2 (0=Ridge, 1=Lasso, if None uses CV to find optimal)
        cv: Number of CV folds for parameter selection
    
    Returns:
        Trained model and scaler
    """
    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    if alpha is None or l1_ratio is None:
        # Use ElasticNetCV to find optimal parameters
        print("   Finding optimal alpha and l1_ratio using cross-validation...")
        alphas = np.logspace(-4, 1, 30)
        l1_ratios = [0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 0.99, 1.0]
        elastic_cv = ElasticNetCV(
            alphas=alphas, l1_ratio=l1_ratios, cv=cv, 
            random_state=42, max_iter=2000, n_jobs=-1
        )
        elastic_cv.fit(X_train_scaled, y_train)
        alpha = elastic_cv.alpha_
        l1_ratio = elastic_cv.l1_ratio_
        print(f"   Optimal alpha: {alpha:.6f}, l1_ratio: {l1_ratio:.4f}")
    
    # Train final model
    model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=2000, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    return model, scaler, alpha, l1_ratio


def evaluate_model(model, scaler, X, y, name="Model", use_scaler=True):
    """Evaluate model performance."""
    if use_scaler and scaler is not None:
        X_scaled = scaler.transform(X)
    else:
        X_scaled = X
    
    y_pred = model.predict(X_scaled)
    
    mae = mean_absolute_error(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    r2 = r2_score(y, y_pred)
    
    print(f"\n{name} Performance:")
    print(f"  MAE:  {mae:.4f}")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  R²:   {r2:.4f}")
    
    return {
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "predictions": y_pred,
    }


def train_random_forest(X_train, y_train, n_estimators=50, max_depth=5):
    """Train Random Forest regression model with regularization to prevent overfitting."""
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=10,  # Increased to reduce overfitting
        min_samples_leaf=5,    # Increased to reduce overfitting
        max_features='sqrt',   # Limit features per split
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    return model, None  # No scaler needed for tree-based models


def train_gradient_boosting(X_train, y_train, n_estimators=50, max_depth=3, learning_rate=0.05):
    """Train Gradient Boosting regression model with regularization to prevent overfitting."""
    model = GradientBoostingRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        min_samples_split=10,  # Increased to reduce overfitting
        min_samples_leaf=5,    # Increased to reduce overfitting
        max_features='sqrt',   # Limit features per split
        subsample=0.8,         # Use 80% of samples per tree
        random_state=42
    )
    model.fit(X_train, y_train)
    return model, None  # No scaler needed for tree-based models


def train_xgboost(X_train, y_train, n_estimators=50, max_depth=3, learning_rate=0.05):
    """Train XGBoost regression model with regularization to prevent overfitting."""
    model = xgb.XGBRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        min_child_weight=5,     # Similar to min_samples_leaf
        subsample=0.8,          # Use 80% of samples per tree
        colsample_bytree=0.8,   # Use 80% of features per tree
        reg_alpha=0.1,          # L1 regularization
        reg_lambda=1.0,         # L2 regularization
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    return model, None  # No scaler needed for tree-based models


def train_xgboost_tuned(X_train, y_train, cv=5):
    """Train XGBoost with aggressive hyperparameter tuning to prevent overfitting."""
    print("   Tuning hyperparameters with RandomizedSearchCV...")
    
    # More aggressive regularization parameters
    param_grid = {
        'n_estimators': [30, 50, 100],
        'max_depth': [2, 3, 4],  # Shallower trees
        'learning_rate': [0.01, 0.05, 0.1],  # Lower learning rates
        'min_child_weight': [5, 10, 15],  # Higher = more regularization
        'subsample': [0.6, 0.7, 0.8],  # Lower = more regularization
        'colsample_bytree': [0.6, 0.7, 0.8],  # Lower = more regularization
        'reg_alpha': [0.5, 1.0, 2.0],  # Higher L1 regularization
        'reg_lambda': [2.0, 5.0, 10.0],  # Higher L2 regularization
        'gamma': [0, 0.1, 0.2],  # Minimum loss reduction for split
    }
    
    # Base model
    base_model = xgb.XGBRegressor(random_state=42, n_jobs=-1)
    
    # Use RandomizedSearchCV for faster search (with small dataset, full grid would be fine too)
    search = RandomizedSearchCV(
        base_model,
        param_grid,
        n_iter=50,  # Try 50 random combinations
        cv=cv,
        scoring='r2',
        n_jobs=-1,
        random_state=42,
        verbose=0
    )
    
    search.fit(X_train, y_train)
    
    best_model = search.best_estimator_
    best_params = search.best_params_
    best_cv_score = search.best_score_
    
    print(f"   Best CV R²: {best_cv_score:.4f}")
    print(f"   Best params: max_depth={best_params['max_depth']}, "
          f"learning_rate={best_params['learning_rate']:.3f}, "
          f"reg_alpha={best_params['reg_alpha']:.1f}, "
          f"reg_lambda={best_params['reg_lambda']:.1f}")
    
    return best_model, None, best_params


def main():
    print("=" * 80)
    print("APS PREDICTION MODEL TRAINING (LASSO REGRESSION)")
    print("=" * 80)
    
    # Load data
    print("\n1. Loading data...")
    df = load_data()
    print(f"   Loaded {len(df)} levels")
    
    # Extract features and target
    print("\n2. Engineering features...")
    X, y, feature_names = get_feature_matrix_and_target(df, target_col="APS")
    print(f"   Features: {len(feature_names)}")
    print(f"   Samples: {len(y)}")
    
    # Split data
    print("\n3. Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"   Train: {len(X_train)} samples")
    print(f"   Test:  {len(X_test)} samples")
    
    # Try multiple models: Linear and Non-linear
    print("\n4. Training models...")
    
    # Standardize features for linear models
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    models_to_try = {}
    
    # Lasso (Linear)
    print("\n4a. Training Lasso Regression (Linear)...")
    lasso_model, _, lasso_alpha = train_lasso_model(X_train, y_train)
    lasso_cv_scores = cross_val_score(lasso_model, X_train_scaled, y_train, cv=5, scoring="r2")
    print(f"   CV R²: {lasso_cv_scores.mean():.4f} (+/- {lasso_cv_scores.std() * 2:.4f})")
    lasso_test_results = evaluate_model(lasso_model, scaler, X_test, y_test, "Lasso (Test)")
    lasso_train_results = evaluate_model(lasso_model, scaler, X_train, y_train, "Lasso (Train)")
    models_to_try['Lasso'] = {
        'model': lasso_model,
        'scaler': scaler,
        'test': lasso_test_results,
        'train': lasso_train_results,
        'cv': lasso_cv_scores,
        'params': {'alpha': lasso_alpha}
    }
    
    # ElasticNet (Linear)
    print("\n4b. Training ElasticNet Regression (Linear)...")
    elastic_model, _, elastic_alpha, elastic_l1_ratio = train_elasticnet_model(X_train, y_train)
    elastic_cv_scores = cross_val_score(elastic_model, X_train_scaled, y_train, cv=5, scoring="r2")
    print(f"   CV R²: {elastic_cv_scores.mean():.4f} (+/- {elastic_cv_scores.std() * 2:.4f})")
    elastic_test_results = evaluate_model(elastic_model, scaler, X_test, y_test, "ElasticNet (Test)")
    elastic_train_results = evaluate_model(elastic_model, scaler, X_train, y_train, "ElasticNet (Train)")
    models_to_try['ElasticNet'] = {
        'model': elastic_model,
        'scaler': scaler,
        'test': elastic_test_results,
        'train': elastic_train_results,
        'cv': elastic_cv_scores,
        'params': {'alpha': elastic_alpha, 'l1_ratio': elastic_l1_ratio}
    }
    
    # ElasticNet with Polynomial Features (Non-linear via interactions)
    print("\n4e. Training ElasticNet with Polynomial Features (Non-linear)...")
    poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
    X_train_poly = poly.fit_transform(X_train_scaled)
    X_test_poly = poly.transform(X_test_scaled)
    
    # Use higher alpha for polynomial features (more regularization needed)
    elastic_poly_alpha = elastic_alpha * 10  # More regularization for more features
    elastic_poly_model = ElasticNet(alpha=elastic_poly_alpha, l1_ratio=elastic_l1_ratio, max_iter=2000, random_state=42)
    elastic_poly_model.fit(X_train_poly, y_train)
    
    elastic_poly_cv_scores = cross_val_score(elastic_poly_model, X_train_poly, y_train, cv=5, scoring="r2")
    print(f"   CV R²: {elastic_poly_cv_scores.mean():.4f} (+/- {elastic_poly_cv_scores.std() * 2:.4f})")
    elastic_poly_test_results = evaluate_model(elastic_poly_model, None, X_test_poly, y_test, "ElasticNet-Poly (Test)", use_scaler=False)
    elastic_poly_train_results = evaluate_model(elastic_poly_model, None, X_train_poly, y_train, "ElasticNet-Poly (Train)", use_scaler=False)
    
    # Store poly transformer with model
    models_to_try['ElasticNet-Poly'] = {
        'model': elastic_poly_model,
        'scaler': poly,  # Store poly transformer
        'test': elastic_poly_test_results,
        'train': elastic_poly_train_results,
        'cv': elastic_poly_cv_scores,
        'params': {'alpha': elastic_poly_alpha, 'l1_ratio': elastic_l1_ratio, 'degree': 2}
    }
    
    # Random Forest (Non-linear) - with regularization
    print("\n4c. Training Random Forest (Non-linear, regularized)...")
    rf_model, _ = train_random_forest(X_train, y_train, n_estimators=50, max_depth=5)
    rf_cv_scores = cross_val_score(rf_model, X_train, y_train, cv=5, scoring="r2")
    print(f"   CV R²: {rf_cv_scores.mean():.4f} (+/- {rf_cv_scores.std() * 2:.4f})")
    rf_test_results = evaluate_model(rf_model, None, X_test, y_test, "Random Forest (Test)", use_scaler=False)
    rf_train_results = evaluate_model(rf_model, None, X_train, y_train, "Random Forest (Train)", use_scaler=False)
    models_to_try['RandomForest'] = {
        'model': rf_model,
        'scaler': None,
        'test': rf_test_results,
        'train': rf_train_results,
        'cv': rf_cv_scores,
        'params': {'n_estimators': 50, 'max_depth': 5}
    }
    
    # Gradient Boosting (Non-linear) - with regularization
    print("\n4d. Training Gradient Boosting (Non-linear, regularized)...")
    gb_model, _ = train_gradient_boosting(X_train, y_train, n_estimators=50, max_depth=3, learning_rate=0.05)
    gb_cv_scores = cross_val_score(gb_model, X_train, y_train, cv=5, scoring="r2")
    print(f"   CV R²: {gb_cv_scores.mean():.4f} (+/- {gb_cv_scores.std() * 2:.4f})")
    gb_test_results = evaluate_model(gb_model, None, X_test, y_test, "Gradient Boosting (Test)", use_scaler=False)
    gb_train_results = evaluate_model(gb_model, None, X_train, y_train, "Gradient Boosting (Train)", use_scaler=False)
    models_to_try['GradientBoosting'] = {
        'model': gb_model,
        'scaler': None,
        'test': gb_test_results,
        'train': gb_train_results,
        'cv': gb_cv_scores,
        'params': {'n_estimators': 50, 'max_depth': 3, 'learning_rate': 0.05}
    }
    
    # XGBoost (Non-linear) - with aggressive hyperparameter tuning
    print("\n4f. Training XGBoost (Non-linear, aggressively tuned)...")
    try:
        xgb_model, _, xgb_best_params = train_xgboost_tuned(X_train, y_train, cv=5)
        xgb_cv_scores = cross_val_score(xgb_model, X_train, y_train, cv=5, scoring="r2")
        print(f"   CV R²: {xgb_cv_scores.mean():.4f} (+/- {xgb_cv_scores.std() * 2:.4f})")
        xgb_test_results = evaluate_model(xgb_model, None, X_test, y_test, "XGBoost (Test)", use_scaler=False)
        xgb_train_results = evaluate_model(xgb_model, None, X_train, y_train, "XGBoost (Train)", use_scaler=False)
        models_to_try['XGBoost'] = {
            'model': xgb_model,
            'scaler': None,
            'test': xgb_test_results,
            'train': xgb_train_results,
            'cv': xgb_cv_scores,
            'params': xgb_best_params
        }
    except Exception as e:
        print(f"   ⚠️  XGBoost training failed: {e}")
        import traceback
        traceback.print_exc()

    
    # Choose best model by test R²
    best_model_name = max(models_to_try.keys(), key=lambda k: models_to_try[k]['test']['r2'])
    best_model_data = models_to_try[best_model_name]
    
    model = best_model_data['model']
    scaler = best_model_data['scaler']
    test_results = best_model_data['test']
    train_results = best_model_data['train']
    cv_scores = best_model_data['cv']
    model_params = best_model_data['params']
    
    print(f"\n   Best model: {best_model_name} (Test R²: {test_results['r2']:.4f})")
    
    # Save model and scaler/transformer
    print(f"\n5. Saving model...")
    model_path = MODEL_DIR / "aps_predictor_lasso.joblib"
    scaler_path = MODEL_DIR / "aps_predictor_scaler.joblib"
    
    joblib.dump(model, model_path)
    if scaler is not None:
        joblib.dump(scaler, scaler_path)
        print(f"   Model saved to: {model_path}")
        print(f"   Scaler/Transformer saved to: {scaler_path}")
    else:
        print(f"   Model saved to: {model_path}")
        print(f"   Scaler: None (tree-based model)")
    
    # Save feature names
    feature_names_path = MODEL_DIR / "aps_predictor_features.txt"
    with open(feature_names_path, "w") as f:
        for name in feature_names:
            f.write(f"{name}\n")
    print(f"   Feature names saved to: {feature_names_path}")
    
    # Feature importance/coefficients
    print(f"\n6. Feature Importance ({best_model_name}):")
    
    if hasattr(model, 'feature_importances_'):
        # Tree-based models
        importances = model.feature_importances_
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        print(f"\n   All Features by Importance:")
        for i, row in importance_df.iterrows():
            print(f"   {row['feature']:30s} {row['importance']:8.4f}")
    
    elif hasattr(model, 'coef_'):
        # Linear models
        if 'alpha' in model_params:
            print(f"   Alpha (regularization): {model_params['alpha']:.6f}")
        if 'l1_ratio' in model_params:
            print(f"   L1 ratio: {model_params['l1_ratio']:.4f}")
        print(f"   Non-zero coefficients: {np.sum(np.abs(model.coef_) > 1e-6)} / {len(model.coef_)}")
        
        coef_df = pd.DataFrame({
            'feature': feature_names,
            'coefficient': model.coef_,
            'abs_coefficient': np.abs(model.coef_)
        }).sort_values('abs_coefficient', ascending=False)
        
        # Separate non-zero and zero coefficients
        threshold = 1e-6
        non_zero = coef_df[coef_df['abs_coefficient'] > threshold]
        zero_coefs = coef_df[coef_df['abs_coefficient'] <= threshold]
        
        print(f"\n   All Features by Absolute Coefficient:")
        for idx, (i, row) in enumerate(non_zero.iterrows(), 1):
            sign = "+" if row['coefficient'] >= 0 else "-"
            print(f"   {idx:2d}. {sign} {row['feature']:30s} {row['coefficient']:8.4f}")
        
        if len(zero_coefs) > 0:
            print(f"\n   Zero-weight features ({len(zero_coefs)}):")
            zero_features = zero_coefs['feature'].tolist()
            # Print in a more readable format
            for i in range(0, len(zero_features), 5):
                chunk = zero_features[i:i+5]
                print(f"   {', '.join(chunk)}")
    
    # Summary
    print(f"\n" + "=" * 80)
    print("TRAINING SUMMARY")
    print("=" * 80)
    print(f"Best Model: {best_model_name}")
    if model_params:
        for param, value in model_params.items():
            print(f"{param}: {value}")
    print(f"Test R²:   {test_results['r2']:.4f}")
    print(f"Test MAE:  {test_results['mae']:.4f}")
    print(f"Test RMSE: {test_results['rmse']:.4f}")
    print(f"\nModel saved to: {model_path}")
    if scaler is not None:
        print(f"Scaler saved to: {scaler_path}")
    else:
        print(f"Scaler: None (tree-based model)")


if __name__ == "__main__":
    main()

