"""
Analyze features and their relationship with APS to improve model performance.
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.predictions.feature_engineering import load_data, engineer_features

def analyze_feature_correlations():
    """Analyze correlations between features and APS."""
    print("=" * 80)
    print("FEATURE ANALYSIS FOR APS PREDICTION")
    print("=" * 80)
    
    # Load data
    df = load_data()
    print(f"\nLoaded {len(df)} levels")
    
    # Engineer features
    features_df = engineer_features(df)
    features_df['APS'] = pd.to_numeric(df['APS'], errors='coerce')
    
    # Remove rows with NaN APS
    features_df = features_df.dropna(subset=['APS'])
    
    print(f"\n1. Feature Correlations with APS:")
    print("-" * 80)
    correlations = features_df.corr()['APS'].sort_values(ascending=False)
    correlations = correlations[correlations.index != 'APS']  # Remove self-correlation
    
    print(f"{'Feature':<35} {'Correlation':<15} {'Abs Corr':<15}")
    print("-" * 80)
    for feature, corr in correlations.items():
        print(f"{feature:<35} {corr:>10.4f}     {abs(corr):>10.4f}")
    
    # Check for missing data patterns
    print(f"\n2. Missing Data Analysis:")
    print("-" * 80)
    missing = features_df.isnull().sum()
    if missing.sum() > 0:
        print("Features with missing values:")
        for col, count in missing[missing > 0].items():
            print(f"  {col}: {count} ({count/len(features_df)*100:.1f}%)")
    else:
        print("No missing values in engineered features")
    
    # Analyze feature distributions
    print(f"\n3. Feature Statistics (for APS > 1.2 vs APS <= 1.2):")
    print("-" * 80)
    high_aps = features_df[features_df['APS'] > 1.2]
    low_aps = features_df[features_df['APS'] <= 1.2]
    
    print(f"High APS levels (APS > 1.2): {len(high_aps)}")
    print(f"Low APS levels (APS <= 1.2): {len(low_aps)}")
    
    # Compare means for top correlated features
    top_features = correlations.abs().nlargest(10).index
    print(f"\nMean values comparison for top features:")
    print(f"{'Feature':<35} {'Low APS':<15} {'High APS':<15} {'Difference':<15}")
    print("-" * 80)
    for feature in top_features:
        low_mean = low_aps[feature].mean()
        high_mean = high_aps[feature].mean()
        diff = high_mean - low_mean
        print(f"{feature:<35} {low_mean:>10.4f}     {high_mean:>10.4f}     {diff:>10.4f}")
    
    # Check raw data columns we might be missing
    print(f"\n4. Raw Data Columns Available:")
    print("-" * 80)
    raw_df = load_data()
    print(f"Total columns: {len(raw_df.columns)}")
    
    # Check if Difficulty might help
    if 'Difficulty' in raw_df.columns:
        print(f"\n5. APS by Difficulty Category:")
        print("-" * 80)
        difficulty_stats = raw_df.groupby('Difficulty')['APS'].agg(['mean', 'std', 'count'])
        print(difficulty_stats)
        
        # Check if we can encode difficulty
        print(f"\nDifficulty encoding could be useful:")
        difficulty_map = {'Normal': 1, 'Hard': 2, 'SuperHard': 3}
        raw_df['Difficulty_encoded'] = raw_df['Difficulty'].map(difficulty_map)
        corr = raw_df['Difficulty_encoded'].corr(raw_df['APS'])
        print(f"  Correlation with APS: {corr:.4f}")
    
    # Check for potential interaction features
    print(f"\n6. Potential Interaction Features:")
    print("-" * 80)
    
    # Grid size interactions
    if 'GridRows' in features_df.columns and 'GridCols' in features_df.columns:
        features_df['grid_size_interaction'] = features_df['GridRows'] * features_df['GridCols']
        corr = features_df['grid_size_interaction'].corr(features_df['APS'])
        print(f"  GridRows * GridCols: {corr:.4f}")
    
    # Moves per grid area
    if 'total_moves' in features_df.columns and 'grid_area' in features_df.columns:
        features_df['moves_density'] = features_df['total_moves'] / (features_df['grid_area'] + 1e-10)
        corr = features_df['moves_density'].corr(features_df['APS'])
        print(f"  total_moves / grid_area: {corr:.4f}")
    
    # Color complexity
    if 'num_colors' in features_df.columns and 'colors_entropy' in features_df.columns:
        features_df['color_complexity'] = features_df['num_colors'] * features_df['colors_entropy']
        corr = features_df['color_complexity'].corr(features_df['APS'])
        print(f"  num_colors * colors_entropy: {corr:.4f}")
    
    # Target complexity
    if 'num_targets' in features_df.columns and 'total_target_count' in features_df.columns:
        features_df['target_complexity'] = features_df['num_targets'] * features_df['total_target_count']
        corr = features_df['target_complexity'].corr(features_df['APS'])
        print(f"  num_targets * total_target_count: {corr:.4f}")
    
    return features_df, correlations

if __name__ == "__main__":
    features_df, correlations = analyze_feature_correlations()

