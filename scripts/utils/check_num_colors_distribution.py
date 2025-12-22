"""Check the distribution of num_colors to see all unique values."""

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
print("num_colors DISTRIBUTION")
print("=" * 80)

if 'num_colors' in features.columns:
    nc = features['num_colors']
    
    print(f"\nAll unique values and their counts:")
    print("-" * 80)
    value_counts = nc.value_counts().sort_index()
    for val, count in value_counts.items():
        pct = count / len(nc) * 100
        print(f"  num_colors = {val:2.0f}: {count:3d} levels ({pct:5.1f}%)")
    
    print(f"\nStatistics:")
    print(f"  Total levels: {len(nc)}")
    print(f"  Unique values: {nc.nunique()}")
    print(f"  Mean: {nc.mean():.4f}")
    print(f"  Std:  {nc.std():.4f}")
    print(f"  Min:  {nc.min():.0f}")
    print(f"  Max:  {nc.max():.0f}")
    
    # Check which levels have which num_colors
    print(f"\nSample levels for each num_colors value:")
    print("-" * 80)
    for val in sorted(nc.unique()):
        levels_with_val = df[nc == val]['Level'].head(5).tolist()
        print(f"  num_colors = {val:2.0f}: Levels {levels_with_val}")

