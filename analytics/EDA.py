import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set style for better-looking plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

# Resolve paths relative to repository root
project_root = Path(__file__).resolve().parent.parent
data_dir = project_root / "data"

# Load the data
data_path = data_dir / "hottestTiles_LevelsFunnel.csv"
df = pd.read_csv(data_path)

print("=" * 80)
print("EXPLORATORY DATA ANALYSIS - hottestTiles Levels Funnel")
print("=" * 80)

# 1. Basic Information
print("\n1. DATASET OVERVIEW")
print("-" * 80)
print(f"Shape: {df.shape[0]} rows, {df.shape[1]} columns")
print(f"\nColumn names:")
for i, col in enumerate(df.columns, 1):
    print(f"  {i:2d}. {col}")

# 2. Data Types and Missing Values
print("\n2. DATA TYPES AND MISSING VALUES")
print("-" * 80)
info_df = pd.DataFrame({
    'Data Type': df.dtypes,
    'Non-Null Count': df.count(),
    'Null Count': df.isnull().sum(),
    'Null Percentage': (df.isnull().sum() / len(df) * 100).round(2)
})
print(info_df.to_string())

# 3. Basic Statistics
print("\n3. DESCRIPTIVE STATISTICS")
print("-" * 80)
print(df.describe().to_string())

# 4. First and Last Few Rows
print("\n4. FIRST 5 ROWS")
print("-" * 80)
print(df.head().to_string())

print("\n5. LAST 5 ROWS")
print("-" * 80)
print(df.tail().to_string())

# 5. Key Metrics Analysis
print("\n6. KEY METRICS SUMMARY")
print("-" * 80)

# Clean numeric columns (remove % signs and convert)
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

# Player progression
if 'players' in df.columns:
    print(f"\nPlayer Progression:")
    print(f"  Starting players (Level 1): {df['players'].iloc[0]:,.0f}")
    print(f"  Ending players (Level {df['Level'].iloc[-1]}): {df['players'].iloc[-1]:,.0f}")
    print(f"  Total drop-off: {df['players'].iloc[0] - df['players'].iloc[-1]:,.0f} players")
    print(f"  Retention rate: {(df['players'].iloc[-1] / df['players'].iloc[0] * 100):.2f}%")

# Attempts analysis
if 'Attempts' in df.columns:
    print(f"\nAttempts Analysis:")
    print(f"  Total attempts: {df['Attempts'].sum():,.0f}")
    print(f"  Average attempts per level: {df['Attempts'].mean():.2f}")
    print(f"  Max attempts (Level {df.loc[df['Attempts'].idxmax(), 'Level']}): {df['Attempts'].max():,.0f}")

# Churn analysis
if 'Global Churn' in df.columns:
    print(f"\nChurn Analysis:")
    print(f"  Average Global Churn: {df['Global Churn'].str.rstrip('%').astype(float).mean():.2f}%")
    print(f"  Max Global Churn: {df['Global Churn'].str.rstrip('%').astype(float).max():.2f}%")

# 7. Visualizations
print("\n7. GENERATING VISUALIZATIONS...")
print("-" * 80)

# Create output directory for plots (now under analytics)
output_dir = Path(__file__).parent / "EDA_plots"
output_dir.mkdir(exist_ok=True)

# Plot 1: Player Progression Over Levels
if 'players' in df.columns:
    plt.figure(figsize=(14, 6))
    plt.plot(range(1, len(df) + 1), df['players'], marker='o', linewidth=2, markersize=4)
    plt.title('Player Count by Level', fontsize=16, fontweight='bold')
    plt.xlabel('Level', fontsize=12)
    plt.ylabel('Number of Players', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / '01_player_progression.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✓ Saved: 01_player_progression.png")

# Plot 2: Attempts per Level
if 'Attempts' in df.columns:
    plt.figure(figsize=(14, 6))
    plt.bar(range(1, len(df) + 1), df['Attempts'], alpha=0.7, color='steelblue')
    plt.title('Attempts per Level', fontsize=16, fontweight='bold')
    plt.xlabel('Level', fontsize=12)
    plt.ylabel('Number of Attempts', fontsize=12)
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(output_dir / '02_attempts_per_level.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✓ Saved: 02_attempts_per_level.png")

# Plot 3: Global Churn Rate
if 'Global Churn' in df.columns:
    churn_numeric = df['Global Churn'].str.rstrip('%').astype(float)
    plt.figure(figsize=(14, 6))
    plt.plot(range(1, len(df) + 1), churn_numeric, marker='o', linewidth=2, 
             markersize=4, color='coral')
    plt.title('Global Churn Rate by Level', fontsize=16, fontweight='bold')
    plt.xlabel('Level', fontsize=12)
    plt.ylabel('Churn Rate (%)', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / '03_global_churn.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✓ Saved: 03_global_churn.png")

# Plot 4: APS (Attempts per Success) Analysis
if 'APS' in df.columns:
    plt.figure(figsize=(14, 6))
    plt.plot(range(1, len(df) + 1), df['APS'], marker='o', linewidth=2, 
             markersize=4, color='green')
    plt.title('Attempts per Success (APS) by Level', fontsize=16, fontweight='bold')
    plt.xlabel('Level', fontsize=12)
    plt.ylabel('APS', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / '04_APS.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✓ Saved: 04_APS.png")
    
    # Plot 4b: APS and Pure APS - First 150 Levels
    if 'Pure APS' in df.columns:
        early_levels = df.head(150)
        plt.figure(figsize=(14, 6))
        levels_150 = range(1, len(early_levels) + 1)
        
        # Plot APS
        plt.plot(levels_150, early_levels['APS'], marker='o', linewidth=2, 
                markersize=4, label='APS', color='green', alpha=0.8)
        
        # Plot Pure APS
        plt.plot(levels_150, early_levels['Pure APS'], marker='s', linewidth=2, 
                markersize=4, label='Pure APS', color='blue', alpha=0.8)
        
        plt.title('APS vs Pure APS - First 150 Levels', fontsize=16, fontweight='bold')
        plt.xlabel('Level', fontsize=12)
        plt.ylabel('Attempts per Success', fontsize=12)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / '04b_APS_early_levels.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("  ✓ Saved: 04b_APS_early_levels.png")
        
        # Plot 4c: APS and Pure APS - First 80 Levels (without spike at level 99)
        very_early_levels = df.head(80)
        plt.figure(figsize=(14, 6))
        levels_80 = range(1, len(very_early_levels) + 1)
        
        # Plot APS
        plt.plot(levels_80, very_early_levels['APS'], marker='o', linewidth=2, 
                markersize=4, label='APS', color='green', alpha=0.8)
        
        # Plot Pure APS
        plt.plot(levels_80, very_early_levels['Pure APS'], marker='s', linewidth=2, 
                markersize=4, label='Pure APS', color='blue', alpha=0.8)
        
        plt.title('APS vs Pure APS - First 80 Levels', fontsize=16, fontweight='bold')
        plt.xlabel('Level', fontsize=12)
        plt.ylabel('Attempts per Success', fontsize=12)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / '04c_APS_first_80_levels.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("  ✓ Saved: 04c_APS_first_80_levels.png")
        
        # Plot 4d: APS Only - First 125 Levels
        early_levels_125 = df.head(125)
        plt.figure(figsize=(14, 6))
        levels_125 = range(1, len(early_levels_125) + 1)
        plt.plot(levels_125, early_levels_125['APS'], marker='o', linewidth=2, 
                markersize=4, color='green', alpha=0.8)
        
        plt.title('APS - First 125 Levels', fontsize=16, fontweight='bold')
        plt.xlabel('Level', fontsize=12)
        plt.ylabel('Attempts per Success (APS)', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / '04d_APS_only_first_125_levels.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("  ✓ Saved: 04d_APS_only_first_125_levels.png")

# Plot 5: Correlation Heatmap (select key numeric columns)
key_cols = ['players', 'Attempts', 'APS', 'Pure APS', 'Avg Win (s)', 
            'Avg Lose (s)', 'Avg Moves Left', 'Avg Combo', 'Avg SuperCombo']
available_cols = [col for col in key_cols if col in df.columns and df[col].dtype in [np.int64, np.float64]]
if len(available_cols) > 1:
    plt.figure(figsize=(12, 10))
    corr_matrix = df[available_cols].corr()
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8})
    plt.title('Correlation Heatmap - Key Metrics', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / '05_correlation_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✓ Saved: 05_correlation_heatmap.png")

# Plot 6: Boosters Usage
booster_cols = ['% Used Boosters', '% Used PG Boosters']
available_booster_cols = [col for col in booster_cols if col in df.columns]
if available_booster_cols:
    plt.figure(figsize=(14, 6))
    for col in available_booster_cols:
        booster_numeric = df[col].str.rstrip('%').astype(float)
        plt.plot(range(1, len(df) + 1), booster_numeric, marker='o', 
                linewidth=2, markersize=4, label=col)
    plt.title('Booster Usage by Level', fontsize=16, fontweight='bold')
    plt.xlabel('Level', fontsize=12)
    plt.ylabel('Usage Percentage (%)', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / '06_booster_usage.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✓ Saved: 06_booster_usage.png")

# Plot 7: Average Game Time
time_cols = ['Avg Win (s)', 'Avg Lose (s)']
available_time_cols = [col for col in time_cols if col in df.columns and df[col].notna().any()]
if available_time_cols:
    # Use only first 125 levels
    df_125 = df.head(125)
    plt.figure(figsize=(14, 6))
    for col in available_time_cols:
        plt.plot(range(1, len(df_125) + 1), df_125[col], marker='o', 
                linewidth=2, markersize=4, label=col, alpha=0.8)
    plt.title('Average Game Time by Level', fontsize=16, fontweight='bold')
    plt.xlabel('Level', fontsize=12)
    plt.ylabel('Time (seconds)', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / '07_avg_game_time.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✓ Saved: 07_avg_game_time.png")
    
    # Plot 7b: Smoothed Rolling Average Game Time
    window_size = 5
    # Use only first 125 levels
    df_125 = df.head(125)
    plt.figure(figsize=(14, 6))
    levels = range(1, len(df_125) + 1)
    
    for col in available_time_cols:
        # Original data (faded)
        plt.plot(levels, df_125[col], marker='o', linewidth=1, markersize=2, 
                label=f'{col} (original)', alpha=0.3)
        # Smoothed data (bold)
        smoothed = df_125[col].rolling(window=window_size, center=True, min_periods=5).mean()
        plt.plot(levels, smoothed, linewidth=3, label=f'{col} (smoothed)', alpha=0.9)
    
    plt.title('Average Game Time by Level (Smoothed Trends)', fontsize=16, fontweight='bold')
    plt.xlabel('Level', fontsize=12)
    plt.ylabel('Time (seconds)', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / '07b_avg_game_time_smoothed.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✓ Saved: 07b_avg_game_time_smoothed.png")
    
    # Plot 7c: Box Plot by Level Ranges
    # Use only first 125 levels
    df_125 = df.head(125)
    # Create level range categories
    def get_level_range(level_idx):
        level_num = level_idx + 1
        if level_num <= 40:
            return 'Early (1-40)'
        elif level_num <= 80:
            return 'Mid (41-80)'
        else:
            return 'Late (81-125)'
    
    df_125 = df_125.copy()
    df_125['Level_Range'] = [get_level_range(i) for i in range(len(df_125))]
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    for idx, col in enumerate(available_time_cols):
        ax = axes[idx]
        data_for_box = []
        labels = []
        for level_range in ['Early (1-40)', 'Mid (41-80)', 'Late (81-125)']:
            range_data = df_125[df_125['Level_Range'] == level_range][col].dropna()
            if len(range_data) > 0:
                data_for_box.append(range_data)
                labels.append(level_range)
        
        bp = ax.boxplot(data_for_box, tick_labels=labels, patch_artist=True)
        ax.set_title(f'{col} Distribution by Level Range', fontsize=14, fontweight='bold')
        ax.set_ylabel('Time (seconds)', fontsize=12)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Color the boxes
        colors = ['lightblue', 'lightgreen', 'lightcoral']
        for patch, color in zip(bp['boxes'], colors[:len(bp['boxes'])]):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
    
    plt.tight_layout()
    plt.savefig(output_dir / '07c_avg_game_time_distributions.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✓ Saved: 07c_avg_game_time_distributions.png")
    
    # Plot 7d: Win-Lose Time Difference
    if 'Avg Win (s)' in available_time_cols and 'Avg Lose (s)' in available_time_cols:
        # Use only first 125 levels
        df_125 = df.head(125)
        win_times = df_125['Avg Win (s)']
        lose_times = df_125['Avg Lose (s)']
        time_diff = lose_times - win_times
        levels = range(1, len(df_125) + 1)
        
        plt.figure(figsize=(14, 6))
        # Original difference
        plt.plot(levels, time_diff, marker='o', linewidth=1, markersize=2, 
                label='Difference (original)', alpha=0.4, color='purple')
        # Smoothed difference
        smoothed_diff = time_diff.rolling(window=window_size, center=True, min_periods=5).mean()
        plt.plot(levels, smoothed_diff, linewidth=3, label='Difference (smoothed)', 
                color='purple', alpha=0.9)
        # Zero line
        plt.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
        
        plt.title('Win-Lose Time Difference by Level', fontsize=16, fontweight='bold')
        plt.xlabel('Level', fontsize=12)
        plt.ylabel('Time Difference (seconds)', fontsize=12)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / '07d_win_lose_difference.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("  ✓ Saved: 07d_win_lose_difference.png")
    
    # Plot 7e: Percentile/Variability Bands
    # Use only first 125 levels
    df_125 = df.head(125)
    levels = range(1, len(df_125) + 1)
    plt.figure(figsize=(14, 6))
    for col in available_time_cols:
        # Calculate smoothed mean
        smoothed_mean = df_125[col].rolling(window=window_size, center=True, min_periods=5).mean()
        # Calculate rolling standard deviation as variability proxy
        smoothed_std = df_125[col].rolling(window=window_size, center=True, min_periods=5).std()
        
        # Plot bands
        upper_band = smoothed_mean + smoothed_std
        lower_band = smoothed_mean - smoothed_std
        
        # Fill between bands
        plt.fill_between(levels, lower_band, upper_band, alpha=0.2, 
                        label=f'{col} (±1 std dev)')
        # Plot smoothed mean
        plt.plot(levels, smoothed_mean, linewidth=2.5, label=f'{col} (smoothed mean)', alpha=0.9)
    
    plt.title('Average Game Time by Level (with Variability Bands)', fontsize=16, fontweight='bold')
    plt.xlabel('Level', fontsize=12)
    plt.ylabel('Time (seconds)', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / '07e_avg_game_time_percentiles.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✓ Saved: 07e_avg_game_time_percentiles.png")
    
    # Clean up temporary column
    if 'Level_Range' in df_125.columns:
        df_125.drop('Level_Range', axis=1, inplace=True, errors='ignore')

# Plot 8: Combo Statistics
combo_cols = ['Avg Combo', 'Avg SuperCombo', 'Avg MegaCombo', 'Avg BigCombo']
available_combo_cols = [col for col in combo_cols if col in df.columns and df[col].notna().any()]
if available_combo_cols:
    plt.figure(figsize=(14, 6))
    for col in available_combo_cols:
        plt.plot(range(1, len(df) + 1), df[col], marker='o', 
                linewidth=2, markersize=4, label=col, alpha=0.8)
    plt.title('Combo Statistics by Level', fontsize=16, fontweight='bold')
    plt.xlabel('Level', fontsize=12)
    plt.ylabel('Average Count', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / '08_combo_stats.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✓ Saved: 08_combo_stats.png")

# 8. Level Difficulty Analysis
print("\n8. LEVEL DIFFICULTY ANALYSIS")
print("-" * 80)
if 'APS' in df.columns:
    # Find hardest levels (highest APS)
    hardest_levels = df.nlargest(5, 'APS')[['Level', 'APS', 'players', 'Attempts']]
    print("\nTop 5 Hardest Levels (by APS):")
    print(hardest_levels.to_string(index=False))
    
    # Find easiest levels (lowest APS, excluding level 1)
    if len(df) > 1:
        easiest_levels = df[df['Level'] != 'Level 1'].nsmallest(5, 'APS')[['Level', 'APS', 'players', 'Attempts']]
        print("\nTop 5 Easiest Levels (by APS, excluding Level 1):")
        print(easiest_levels.to_string(index=False))

# 9. Power-ups Usage Analysis
print("\n9. POWER-UPS USAGE ANALYSIS")
print("-" * 80)
power_up_cols = ['Avg Switch', 'Avg Slingshots', 'Avg Hammers', 'Avg Rockets', 'Avg Colors']
available_power_cols = [col for col in power_up_cols if col in df.columns]
if available_power_cols:
    power_up_summary = df[available_power_cols].describe()
    print(power_up_summary.to_string())

# 10. Summary Statistics by Level Range
print("\n10. SUMMARY BY LEVEL RANGES")
print("-" * 80)
if 'players' in df.columns:
    total_levels = len(df)
    ranges = [
        (1, min(10, total_levels), "Early (1-10)"),
        (11, min(50, total_levels), "Mid (11-50)"),
        (51, total_levels, "Late (51+)")
    ]
    
    for start, end, label in ranges:
        if start <= total_levels:
            range_df = df.iloc[start-1:end]
            print(f"\n{label} Levels ({start}-{end}):")
            if 'players' in df.columns:
                print(f"  Players: {range_df['players'].mean():.0f} avg, {range_df['players'].min():.0f} min, {range_df['players'].max():.0f} max")
            if 'APS' in df.columns:
                print(f"  APS: {range_df['APS'].mean():.2f} avg, {range_df['APS'].min():.2f} min, {range_df['APS'].max():.2f} max")
            if 'Attempts' in df.columns:
                print(f"  Attempts: {range_df['Attempts'].mean():.0f} avg per level")

print("\n" + "=" * 80)
print("EDA COMPLETE!")
print(f"Visualizations saved to: {output_dir}")
print("=" * 80)

