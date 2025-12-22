"""
Feature engineering for APS prediction.

Extracts numerical features from level design data (excluding performance metrics).
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def parse_json_column(value: str) -> Any:
    """Parse JSON string column, return empty dict/list if invalid."""
    if pd.isna(value) or value == "":
        return {}
    try:
        return json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return {}


def extract_grid_features(grid_json: str) -> Dict[str, float]:
    """Extract features from Grid JSON."""
    grid = parse_json_column(grid_json)
    if not grid:
        return {
            "grid_total_tiles": 0,
            "grid_avg_stack_height": 0,
            "grid_max_stack_height": 0,
            "grid_filled_cells": 0,
            "grid_density": 0,
            "grid_multi_stack_cells": 0,
            "grid_has_bunny": 0,
            "grid_has_ice": 0,
            "grid_color_diversity": 0,
            "grid_stack_height_std": 0,
        }
    
    total_tiles = 0
    stack_heights = []
    filled_cells = len(grid)
    multi_stack_cells = 0
    has_bunny = 0
    has_ice = 0
    colors_in_grid = set()
    
    # Individual blocker counters
    blocker_counts = {
        "UniCorn": 0,
        "WoodBox": 0,
        "Bunny": 0,
        "Flower": 0,
        "Ice": 0,
        "Grass": 0,
        "Birdhouse": 0,
        "Dwarf": 0,
        "Fairy": 0,
        "Igloo": 0,
        "Bomb": 0,
        "Empty": 0,
        "Lock": 0,
        "ColorBlocker": 0,
        "ColorBlocker4": 0,
    }
    
    for pos, stacks in grid.items():
        if isinstance(stacks, list):
            if len(stacks) > 1:
                multi_stack_cells += 1
            
            for stack in stacks:
                tiles_count = stack.get("TilesCount", 0)
                total_tiles += tiles_count
                if tiles_count > 0:
                    stack_heights.append(tiles_count)
                
                # Check for special tiles (binary flags for backward compatibility)
                if stack.get("Bunny", False):
                    has_bunny = 1
                if stack.get("Ice", False):
                    has_ice = 1
                
                # Count each blocker type
                if stack.get("UniCorn", False):
                    blocker_counts["UniCorn"] += 1
                if stack.get("WoodBox", False):
                    blocker_counts["WoodBox"] += 1
                if stack.get("Bunny", False):
                    blocker_counts["Bunny"] += 1
                if stack.get("Flower", False):
                    blocker_counts["Flower"] += 1
                if stack.get("Ice", False):
                    blocker_counts["Ice"] += 1
                if stack.get("Grass", False):
                    blocker_counts["Grass"] += 1
                if stack.get("Birdhouse", False):
                    blocker_counts["Birdhouse"] += 1
                if stack.get("Dwarf", False):
                    blocker_counts["Dwarf"] += 1
                if stack.get("Fairy", False):
                    blocker_counts["Fairy"] += 1
                if stack.get("Igloo", 0) > 0:
                    blocker_counts["Igloo"] += 1
                if stack.get("Bomb", False):
                    blocker_counts["Bomb"] += 1
                if stack.get("Empty", False):
                    blocker_counts["Empty"] += 1
                if stack.get("Lock", False):
                    blocker_counts["Lock"] += 1
                if stack.get("ColorBlocker", False):
                    blocker_counts["ColorBlocker"] += 1
                if stack.get("ColorBlocker4", False):
                    blocker_counts["ColorBlocker4"] += 1
                
                # Track colors in grid
                color = stack.get("Color", "")
                if color:
                    colors_in_grid.add(color)
    
    # Calculate total blockers
    total_blockers = sum(blocker_counts.values())
    
    avg_stack = np.mean(stack_heights) if stack_heights else 0
    max_stack = max(stack_heights) if stack_heights else 0
    stack_std = np.std(stack_heights) if stack_heights else 0
    
    # Build return dictionary with all features
    result = {
        "grid_total_tiles": total_tiles,
        "grid_avg_stack_height": avg_stack,
        "grid_max_stack_height": max_stack,
        "grid_filled_cells": filled_cells,
        "grid_density": filled_cells / (10 * 12) if filled_cells > 0 else 0,
        "grid_multi_stack_cells": multi_stack_cells,
        "grid_has_bunny": has_bunny,
        "grid_has_ice": has_ice,
        "grid_color_diversity": len(colors_in_grid),
        "grid_stack_height_std": stack_std,
        "total_blockers": total_blockers,  # Sum of all blocker tiles
    }
    
    # Add individual blocker counts
    for blocker_type, count in blocker_counts.items():
        result[f"blocker_{blocker_type}"] = count
    
    return result


def extract_colors_features(colors_json: str) -> Dict[str, float]:
    """Extract features from ColorsProbability JSON."""
    colors = parse_json_column(colors_json)
    if not colors or not isinstance(colors, dict):
        return {
            "num_colors": 0,
            "colors_entropy": 0,
            "colors_std": 0,
        }
    
    color_probs = [v for v in colors.values() if isinstance(v, (int, float))]
    num_colors = len(color_probs)
    
    if num_colors == 0:
        return {
            "num_colors": 0,
            "colors_entropy": 0,
            "colors_std": 0,
        }
    
    # Normalize probabilities
    total = sum(color_probs)
    if total > 0:
        probs = np.array(color_probs) / total
        # Entropy: -sum(p * log(p))
        entropy = -np.sum(probs * np.log(probs + 1e-10))
    else:
        entropy = 0
    
    std = np.std(color_probs) if color_probs else 0
    
    return {
        "num_colors": num_colors,
        "colors_entropy": entropy,
        "colors_std": std,
    }


def extract_target_features(target_json: str) -> Dict[str, float]:
    """Extract features from LevelTarget JSON."""
    targets = parse_json_column(target_json)
    if not targets or not isinstance(targets, list):
        return {
            "num_targets": 0,
            "total_target_count": 0,
            "avg_target_count": 0,
            "target_diversity": 0,
            "has_bunny_target": 0,
            "has_ice_target": 0,
            "has_multicolor_target": 0,
            "target_complexity_score": 0,
            "LevelTarget_regular_color": 0,
            "LevelTarget_special": 0,
            "weighted_target_count": 0,
        }
    
    num_targets = len(targets)
    total_count = 0
    color_names = []
    has_bunny = 0
    has_ice = 0
    has_multicolor = 0
    target_counts = []
    
    # Regular color names (including MultiColor)
    regular_colors = {
        "Red", "Blue", "Green", "Yellow", "Pink", "Purple", "Orange", 
        "Turquoise", "LightBlue", "LightGreen", "MultiColor"
    }
    
    sum_regular_color_targets = 0
    sum_special_targets = 0
    weighted_target_count = 0  # Weighted by difficulty
    
    for target in targets:
        if isinstance(target, dict):
            count = target.get("Count", 0)
            total_count += count
            target_counts.append(count)
            color_name = target.get("ColorName", "")
            if color_name:
                color_names.append(color_name)
                
                # Weight targets by difficulty (learned optimal weights):
                # - MultiColor: easiest (weight = 0.5) - learned optimal
                # - Regular colors: medium (weight = 1.2) - learned optimal
                # - Blockers/special: hardest (weight = 2.0) - learned optimal
                # These weights were learned by testing 150 combinations for best APS correlation
                if "MultiColor" in color_name:
                    weighted_target_count += count * 0.5  # Easiest (learned)
                    has_multicolor = 1
                elif color_name in regular_colors:
                    weighted_target_count += count * 1.2  # Medium difficulty (learned)
                    sum_regular_color_targets += count
                else:
                    # Special target/blocker (Bunny, Ice, Bird, Flower, etc.) - hardest
                    weighted_target_count += count * 2.0  # Hardest (learned)
                    sum_special_targets += count
                
                if "Bunny" in color_name:
                    has_bunny = 1
                if "Ice" in color_name:
                    has_ice = 1
    
    avg_count = total_count / num_targets if num_targets > 0 else 0
    diversity = len(set(color_names)) if color_names else 0
    
    # Complexity score: more targets + special targets = harder
    complexity = num_targets + (has_bunny * 2) + (has_ice * 1.5) + (has_multicolor * 1.5)
    if target_counts:
        complexity += np.std(target_counts)  # Variation in target counts
    
    return {
        "num_targets": num_targets,
        "total_target_count": total_count,
        "avg_target_count": avg_count,
        "target_diversity": diversity,
        "has_bunny_target": has_bunny,
        "has_ice_target": has_ice,
        "has_multicolor_target": has_multicolor,
        "target_complexity_score": complexity,
        "LevelTarget_regular_color": sum_regular_color_targets,  # Sum of regular color targets
        "LevelTarget_special": sum_special_targets,  # Sum of special targets (Bird, Flower, etc.)
        "weighted_target_count": weighted_target_count,  # Weighted by difficulty (MultiColor=0.5, colors=1.0, blockers=2.0)
    }


def extract_tiles_prob_features(tiles_prob_json: str) -> Dict[str, float]:
    """
    Extract features from TilesProbability JSON array.
    
    Note: TilesProbability represents Stack Size Probabilities, not tile probabilities.
    The array contains probabilities for stack sizes 3, 4, 5, 6 (exactly 4 values).
    For example, [25,25,25,25] means 25% chance of stack size 3, 25% for size 4, 25% for size 5, 25% for size 6.
    """
    tiles_prob = parse_json_column(tiles_prob_json)
    if not tiles_prob or not isinstance(tiles_prob, list):
        return {
            "tiles_prob_count": 0,  # Number of stack size probability bins (for sizes 3, 4, 5, 6)
            "tiles_prob_mean": 0,   # Mean stack size probability
            "tiles_prob_std": 0,    # Std of stack size probabilities
        }
    
    probs = [p for p in tiles_prob if isinstance(p, (int, float))]
    
    return {
        "tiles_prob_count": len(probs),  # Number of stack size probability bins (for sizes 3, 4, 5, 6)
        "tiles_prob_mean": np.mean(probs) if probs else 0,  # Mean stack size probability
        "tiles_prob_std": np.std(probs) if probs else 0,   # Std of stack size probabilities
    }


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer features from level design columns.
    
    Returns a DataFrame with engineered features, excluding performance metrics.
    """
    features = pd.DataFrame()
    
    # Basic numeric features (level design only)
    # Option 1: Keep grid_area and GridCols, remove GridRows to avoid multicollinearity
    design_cols = [
        "GridCols",  # Keep GridCols for aspect ratio
        "total_moves",
        "MinFairy",
        "MinCoin",
        "DynamicBombPercents",
        "DynamicIcePercents",
        "MultiStack",
    ]
    
    # Get GridRows for calculations (but don't add as feature)
    grid_rows = None
    if "GridRows" in df.columns:
        grid_rows = pd.to_numeric(df["GridRows"], errors="coerce").fillna(0)
    
    # Add design columns
    for col in design_cols:
        if col in df.columns:
            features[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    
    # Derived features from GridRows and GridCols
    # Use grid_area (GridRows * GridCols) instead of GridRows to avoid multicollinearity
    if grid_rows is not None and "GridCols" in features.columns:
        # Grid area (total cells) - more informative than GridRows alone
        features["grid_area"] = grid_rows * features["GridCols"]
        # Aspect ratio (width/height)
        features["grid_aspect_ratio"] = features["GridCols"] / (grid_rows + 1e-10)
    elif "GridCols" in features.columns:
        # Fallback if only GridCols available
        features["grid_area"] = features["GridCols"] * 10  # Default assumption
        features["grid_aspect_ratio"] = 1.0
    else:
        features["grid_area"] = 0
        features["grid_aspect_ratio"] = 1.0
    
    if "total_moves" in features.columns:
        features["moves_per_cell"] = features["total_moves"] / (features.get("grid_area", 1) + 1e-10)
    
    # Extract features from JSON columns
    if "Grid" in df.columns:
        grid_features = df["Grid"].apply(extract_grid_features)
        grid_df = pd.DataFrame(grid_features.tolist())
        features = pd.concat([features, grid_df], axis=1)
    
    if "ColorsProbability" in df.columns:
        colors_features = df["ColorsProbability"].apply(extract_colors_features)
        colors_df = pd.DataFrame(colors_features.tolist())
        features = pd.concat([features, colors_df], axis=1)
    
    if "LevelTarget" in df.columns:
        target_features = df["LevelTarget"].apply(extract_target_features)
        target_df = pd.DataFrame(target_features.tolist())
        features = pd.concat([features, target_df], axis=1)
    
    if "TilesProbability" in df.columns:
        # Note: TilesProbability represents Stack Size Probabilities (for stack sizes 3, 4, 5, 6)
        tiles_features = df["TilesProbability"].apply(extract_tiles_prob_features)
        tiles_df = pd.DataFrame(tiles_features.tolist())
        features = pd.concat([features, tiles_df], axis=1)
    
    # Note: Difficulty is NOT included as it's a human label, not a level design feature
    
    # Add interaction features (high correlations)
    if "num_colors" in features.columns and "colors_entropy" in features.columns:
        features["color_complexity"] = features["num_colors"] * features["colors_entropy"]
    
    if "num_targets" in features.columns and "total_target_count" in features.columns:
        features["target_complexity"] = features["num_targets"] * features["total_target_count"]
    
    # Grid complexity interactions
    if "grid_multi_stack_cells" in features.columns and "grid_filled_cells" in features.columns:
        features["multi_stack_ratio"] = features["grid_multi_stack_cells"] / (features["grid_filled_cells"] + 1e-10)
    
    if "grid_has_bunny" in features.columns and "has_bunny_target" in features.columns:
        features["bunny_presence"] = features["grid_has_bunny"] + features["has_bunny_target"]
    
    if "grid_has_ice" in features.columns and "has_ice_target" in features.columns:
        features["ice_presence"] = features["grid_has_ice"] + features["has_ice_target"]
    
    # Overall difficulty indicators
    if "target_complexity_score" in features.columns:
        features["overall_complexity"] = features["target_complexity_score"]
    
    if "grid_color_diversity" in features.columns and "num_colors" in features.columns:
        features["color_usage_ratio"] = features["grid_color_diversity"] / (features["num_colors"] + 1e-10)
    
    if "grid_area" in features.columns and "GridCols" in features.columns:
        # Normalized grid size (using grid_area and GridCols)
        features["grid_size_normalized"] = (np.sqrt(features["grid_area"]) + features["GridCols"]) / 2
    
    # Add polynomial features for top correlated features
    # Note: Level and Level_sqrt removed as they're not level design features
    
    if "num_colors" in features.columns:
        features["num_colors_squared"] = features["num_colors"] ** 2
    
    if "total_moves" in features.columns:
        features["total_moves_squared"] = features["total_moves"] ** 2
        # Moves per grid cell (better than moves_per_cell)
        if "grid_area" in features.columns:
            features["moves_density"] = features["total_moves"] / (features["grid_area"] + 1e-10)
        
        # Tension between moves and targets (key difficulty indicator)
        # This captures: do you have enough moves to complete all targets?
        if "total_target_count" in features.columns:
            # Only calculate if targets > 0 (avoid division issues)
            valid_targets = features["total_target_count"] > 0
            # Moves per target - higher = easier (more moves per target)
            features["moves_per_target"] = np.where(
                valid_targets,
                features["total_moves"] / features["total_target_count"],
                0  # If no targets, set to 0
            )
            # Targets per move - higher = harder (more targets per move)
            # This is the key tension metric: how many targets must you complete per move?
            features["targets_per_move"] = features["total_target_count"] / (features["total_moves"] + 1e-10)
        
        # Weighted tension (accounts for target difficulty)
        if "weighted_target_count" in features.columns:
            # Only calculate if weighted targets > 0
            valid_weighted = features["weighted_target_count"] > 0
            # Moves per weighted target - accounts for difficulty
            features["moves_per_weighted_target"] = np.where(
                valid_weighted,
                features["total_moves"] / features["weighted_target_count"],
                0
            )
            # Weighted targets per move - better tension metric (accounts for difficulty)
            features["weighted_targets_per_move"] = features["weighted_target_count"] / (features["total_moves"] + 1e-10)
        
        if "num_targets" in features.columns:
            # Moves per target type (only if num_targets > 0)
            valid_target_types = features["num_targets"] > 0
            features["moves_per_target_type"] = np.where(
                valid_target_types,
                features["total_moves"] / features["num_targets"],
                0
            )
    
    # Grid complexity features
    if "grid_total_tiles" in features.columns and "grid_area" in features.columns:
        features["tile_density"] = features["grid_total_tiles"] / (features["grid_area"] + 1e-10)
    
    if "grid_avg_stack_height" in features.columns and "grid_max_stack_height" in features.columns:
        features["stack_height_variation"] = features["grid_max_stack_height"] - features["grid_avg_stack_height"]
    
    # Target difficulty features
    if "total_target_count" in features.columns and "grid_area" in features.columns:
        features["target_density"] = features["total_target_count"] / (features["grid_area"] + 1e-10)
    
    if "avg_target_count" in features.columns and "num_targets" in features.columns:
        features["target_balance"] = features["avg_target_count"] * features["num_targets"]
    
    return features


def load_data(csv_path: Path = None) -> pd.DataFrame:
    """Load unified CSV data."""
    if csv_path is None:
        csv_path = REPO_ROOT / "data" / "levels_unified.csv"
    return pd.read_csv(csv_path)


def get_feature_matrix_and_target(df: pd.DataFrame, target_col: str = "APS") -> tuple:
    """
    Extract feature matrix and target vector.
    
    Returns:
        X: Feature matrix (numpy array)
        y: Target vector (numpy array)
        feature_names: List of feature names
    """
    # Engineer features
    X_df = engineer_features(df)
    
    # Get target
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in data")
    
    y = pd.to_numeric(df[target_col], errors="coerce").values
    
    # Remove any rows with NaN targets
    valid_mask = ~np.isnan(y)
    X_df = X_df[valid_mask]
    y = y[valid_mask]
    
    # Fill any remaining NaN in features
    X_df = X_df.fillna(0)
    
    # Convert to numpy
    X = X_df.values
    feature_names = X_df.columns.tolist()
    
    return X, y, feature_names

