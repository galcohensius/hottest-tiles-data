"""Validate weighted_target_count by showing scores for each level."""

import pandas as pd
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.predictions.feature_engineering import load_data, engineer_features, parse_json_column

def parse_targets(target_json):
    """Parse LevelTarget JSON."""
    return parse_json_column(target_json)

def calculate_weighted_breakdown(target_json):
    """Calculate weighted target count with breakdown by type."""
    targets = parse_targets(target_json)
    if not targets or not isinstance(targets, list):
        return {
            'total_count': 0,
            'multicolor_count': 0,
            'color_count': 0,
            'blocker_count': 0,
            'weighted_total': 0
        }
    
    regular_colors = {
        "Red", "Blue", "Green", "Yellow", "Pink", "Purple", "Orange", 
        "Turquoise", "LightBlue", "LightGreen"
    }
    
    multicolor_count = 0
    color_count = 0
    blocker_count = 0
    total_count = 0
    
    for target in targets:
        if isinstance(target, dict):
            count = target.get("Count", 0)
            total_count += count
            color_name = target.get("ColorName", "")
            
            if "MultiColor" in color_name:
                multicolor_count += count
            elif color_name in regular_colors:
                color_count += count
            else:
                blocker_count += count
    
    # Apply learned weights
    weighted_total = (multicolor_count * 0.5) + (color_count * 1.2) + (blocker_count * 2.0)
    
    return {
        'total_count': total_count,
        'multicolor_count': multicolor_count,
        'color_count': color_count,
        'blocker_count': blocker_count,
        'weighted_total': weighted_total
    }

df = load_data()
features = engineer_features(df)

print("=" * 100)
print("WEIGHTED TARGET COUNT VALIDATION")
print("=" * 100)
print("\nWeights: MultiColor=0.5, Regular Colors=1.2, Blockers=2.0")
print("\n" + "=" * 100)

# Calculate breakdown for each level
results = []
for idx, row in df.iterrows():
    level = row.get('Level', idx + 1)
    aps = row.get('APS', 'N/A')
    total_moves = row.get('total_moves', 0)
    target_json = row.get('LevelTarget', '[]')
    
    breakdown = calculate_weighted_breakdown(target_json)
    calculated_weighted = breakdown['weighted_total']
    feature_weighted = features.iloc[idx]['weighted_target_count'] if 'weighted_target_count' in features.columns else 0
    
    results.append({
        'Level': level,
        'APS': aps,
        'total_moves': total_moves,
        'total_targets': breakdown['total_count'],
        'MultiColor': breakdown['multicolor_count'],
        'Colors': breakdown['color_count'],
        'Blockers': breakdown['blocker_count'],
        'weighted_count': calculated_weighted,
        'feature_weighted': feature_weighted,
        'match': abs(calculated_weighted - feature_weighted) < 0.01
    })

results_df = pd.DataFrame(results)

# Show all levels
print("\nAll Levels:")
print("-" * 100)
print(f"{'Level':<8} {'APS':<8} {'Moves':<8} {'Total':<8} {'MultiC':<8} {'Colors':<8} {'Block':<8} {'Weighted':<10} {'Match':<8}")
print("-" * 100)
for _, row in results_df.iterrows():
    match_str = "✓" if row['match'] else "✗"
    print(f"{row['Level']:<8} {row['APS']:<8.2f} {row['total_moves']:<8} {row['total_targets']:<8} "
          f"{row['MultiColor']:<8} {row['Colors']:<8} {row['Blockers']:<8} {row['weighted_count']:<10.2f} {match_str:<8}")

# Check for mismatches
mismatches = results_df[~results_df['match']]
if len(mismatches) > 0:
    print(f"\n⚠️  Found {len(mismatches)} mismatches between calculated and feature values:")
    print(mismatches[['Level', 'weighted_count', 'feature_weighted']].to_string(index=False))
else:
    print("\n✓ All weighted counts match!")

# Show some examples with target details
print("\n\n" + "=" * 100)
print("SAMPLE LEVELS WITH TARGET BREAKDOWN")
print("=" * 100)

# Show a few interesting examples
sample_levels = [0, 2, 4, 8, 20, 50, 100] if len(df) > 100 else [0, 2, 4, 8, 20]
for idx in sample_levels:
    if idx >= len(df):
        continue
    
    row = df.iloc[idx]
    level = row.get('Level', idx + 1)
    target_json = row.get('LevelTarget', '[]')
    targets = parse_targets(target_json)
    
    print(f"\nLevel {level}:")
    print(f"  APS: {row.get('APS', 'N/A')}")
    print(f"  Total Moves: {row.get('total_moves', 0)}")
    print(f"  Targets:")
    
    breakdown = calculate_weighted_breakdown(target_json)
    for target in targets:
        if isinstance(target, dict):
            color_name = target.get("ColorName", "")
            count = target.get("Count", 0)
            if "MultiColor" in color_name:
                weight = 0.5
            elif color_name in {"Red", "Blue", "Green", "Yellow", "Pink", "Purple", "Orange", "Turquoise", "LightBlue", "LightGreen"}:
                weight = 1.2
            else:
                weight = 2.0
            weighted = count * weight
            print(f"    - {color_name}: {count} × {weight} = {weighted:.1f}")
    
    print(f"  Total: {breakdown['total_count']} targets")
    print(f"  Weighted Total: {breakdown['weighted_total']:.2f}")

