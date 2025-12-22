"""Check feature value distributions to see if most levels have the same value."""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.predictions.feature_engineering import load_data, engineer_features

df = load_data()
features = engineer_features(df)

print("=" * 80)
print("FEATURE VALUE DISTRIBUTION ANALYSIS")
print("=" * 80)

# Check color_complexity specifically
if 'color_complexity' in features.columns:
    print("\n1. color_complexity distribution:")
    print("-" * 80)
    cc = features['color_complexity']
    print(f"  Mean: {cc.mean():.4f}")
    print(f"  Std:  {cc.std():.4f}")
    print(f"  Min:  {cc.min():.4f}")
    print(f"  Max:  {cc.max():.4f}")
    print(f"  Unique values: {cc.nunique()} / {len(cc)}")
    
    # Check value counts
    print(f"\n  Value counts (top 10):")
    value_counts = cc.value_counts().head(10)
    for val, count in value_counts.items():
        pct = count / len(cc) * 100
        print(f"    {val:8.4f}: {count:3d} levels ({pct:5.1f}%)")
    
    # Check if most levels have the same value
    most_common = cc.value_counts().iloc[0]
    most_common_pct = most_common / len(cc) * 100
    print(f"\n  Most common value appears in {most_common} levels ({most_common_pct:.1f}%)")
    
    if most_common_pct > 50:
        print(f"  ⚠️  WARNING: Most levels have the same value! Low variance feature.")

# Check other key features
print("\n2. Other key features distribution:")
print("-" * 80)
key_features = ['num_colors', 'colors_entropy', 'target_complexity', 'moves_per_target']
for feat in key_features:
    if feat in features.columns:
        f = features[feat]
        unique_ratio = f.nunique() / len(f)
        most_common_pct = f.value_counts().iloc[0] / len(f) * 100
        
        print(f"\n  {feat}:")
        print(f"    Unique values: {f.nunique()} / {len(f)} ({unique_ratio*100:.1f}%)")
        print(f"    Most common: {most_common_pct:.1f}% of levels")
        print(f"    Mean: {f.mean():.4f}, Std: {f.std():.4f}")
        
        if most_common_pct > 50:
            print(f"    ⚠️  Low variance - most levels have same value")

# Check for features with very low variance
print("\n3. Features with low variance (most levels have same value):")
print("-" * 80)
low_variance_features = []
for col in features.columns:
    if features[col].dtype in [np.int64, np.float64]:
        unique_ratio = features[col].nunique() / len(features)
        most_common_pct = features[col].value_counts().iloc[0] / len(features) * 100
        
        if most_common_pct > 80:  # More than 80% have same value
            low_variance_features.append({
                'feature': col,
                'unique_ratio': unique_ratio,
                'most_common_pct': most_common_pct,
                'std': features[col].std()
            })

if low_variance_features:
    low_var_df = pd.DataFrame(low_variance_features).sort_values('most_common_pct', ascending=False)
    print(f"\n  Found {len(low_var_df)} features with >80% same value:")
    print(f"  {'Feature':<35} {'Unique %':<12} {'Most Common %':<15} {'Std':<10}")
    print("-" * 80)
    for _, row in low_var_df.iterrows():
        print(f"  {row['feature']:<35} {row['unique_ratio']*100:>8.1f}%     {row['most_common_pct']:>10.1f}%     {row['std']:>8.4f}")
else:
    print("  No features with extremely low variance found.")

