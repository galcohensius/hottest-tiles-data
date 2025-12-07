import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set style for better-looking plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)

# ============================================================================
# 1. DATA QUALITY & PREPARATION
# ============================================================================

print("=" * 80)
print("GAME RETENTION EDA - hottestTiles Levels Funnel")
print("=" * 80)

# Load the data
data_path = Path(__file__).parent / "hottestTiles_LevelsFunnel.csv"
df = pd.read_csv(data_path)

print("\n1. DATA QUALITY & PREPARATION")
print("-" * 80)

# Basic info
print(f"Dataset shape: {df.shape[0]} rows, {df.shape[1]} columns")
print(f"Level range: {df['Level'].iloc[0]} to {df['Level'].iloc[-1]}")

# Handle missing values
print("\nMissing values analysis:")
missing_summary = pd.DataFrame({
    'Column': df.columns,
    'Missing Count': df.isnull().sum(),
    'Missing %': (df.isnull().sum() / len(df) * 100).round(2)
})
missing_summary = missing_summary[missing_summary['Missing Count'] > 0].sort_values('Missing %', ascending=False)
if len(missing_summary) > 0:
    print(missing_summary.to_string(index=False))
else:
    print("  No missing values found")

# Clean percentage columns
percentage_cols = ['Relative Churn', 'Global Churn', '% Revived', '% Used Boosters', '% Used PG Boosters']
for col in percentage_cols:
    if col in df.columns:
        df[col + '_numeric'] = df[col].str.rstrip('%').replace('', np.nan).astype(float)

# Extract level number for easier analysis
df['level_num'] = df['Level'].str.extract(r'(\d+)').astype(int)

# Create output directory
output_dir = Path(__file__).parent / "retention_plots"
output_dir.mkdir(exist_ok=True)

# ============================================================================
# 2. RETENTION FUNNEL ANALYSIS
# ============================================================================

print("\n2. RETENTION FUNNEL ANALYSIS")
print("-" * 80)

# Calculate retention rates
initial_players = df['players'].iloc[0]
df['retention_rate'] = (df['players'] / initial_players * 100).round(2)
df['cumulative_retention'] = df['retention_rate']

# Calculate churn rates
df['relative_churn_numeric'] = df['Relative Churn_numeric'].fillna(0)
df['global_churn_numeric'] = df['Global Churn_numeric'].fillna(0)

# Calculate actual churn (players lost between levels)
df['players_lost'] = df['players'].diff().fillna(0) * -1  # Negative diff means loss
df['churn_rate_calculated'] = (df['players_lost'] / df['players'].shift(1).fillna(initial_players) * 100).round(2)

# Identify critical drop-off points
df['drop_off_rank'] = df['players_lost'].rank(ascending=False, method='dense')

print(f"\nRetention Overview:")
print(f"  Starting players (Level 1): {initial_players:,.0f}")
print(f"  Ending players (Level {df['Level'].iloc[-1]}): {df['players'].iloc[-1]:,.0f}")
print(f"  Overall retention: {df['retention_rate'].iloc[-1]:.2f}%")
print(f"  Total players lost: {initial_players - df['players'].iloc[-1]:,.0f}")

# Find top drop-off levels
top_dropoffs = df.nlargest(10, 'players_lost')[['Level', 'players', 'players_lost', 'churn_rate_calculated', 'APS']]
print(f"\nTop 10 Critical Drop-off Points:")
print(top_dropoffs.to_string(index=False))

# Find sticky vs leaky levels
df['churn_category'] = pd.cut(df['churn_rate_calculated'], 
                               bins=[-np.inf, 5, 15, np.inf],
                               labels=['Sticky (Low Churn)', 'Moderate', 'Leaky (High Churn)'])

sticky_levels = df[df['churn_category'] == 'Sticky (Low Churn)'].nlargest(10, 'players')[['Level', 'players', 'churn_rate_calculated', 'APS']]
leaky_levels = df[df['churn_category'] == 'Leaky (High Churn)'].nlargest(10, 'players_lost')[['Level', 'players', 'churn_rate_calculated', 'APS', 'players_lost']]

print(f"\nTop 10 Sticky Levels (Low Churn, High Player Count):")
print(sticky_levels.to_string(index=False))

print(f"\nTop 10 Leaky Levels (High Churn):")
print(leaky_levels.to_string(index=False))

# Retention by level ranges (Day 1, Day 7, Day 30 equivalent)
level_milestones = {
    'Day 1 Equivalent': [1, 20],
    'Day 7 Equivalent': [21, 100],
    'Day 30 Equivalent': [101, 500]
}

print(f"\nRetention by Milestones:")
for milestone, (start, end) in level_milestones.items():
    milestone_df = df[(df['level_num'] >= start) & (df['level_num'] <= end)]
    if len(milestone_df) > 0:
        avg_retention = milestone_df['retention_rate'].mean()
        print(f"  {milestone} (Levels {start}-{end}): {avg_retention:.2f}% avg retention")

# ============================================================================
# 3. FACTORS AFFECTING RETENTION
# ============================================================================

print("\n3. FACTORS AFFECTING RETENTION")
print("-" * 80)

# Normalize difficulty scores
df['difficulty_score'] = ((df['APS'] - df['APS'].min()) / (df['APS'].max() - df['APS'].min()) * 100).fillna(0)
if 'Pure APS' in df.columns:
    df['pure_difficulty_score'] = ((df['Pure APS'] - df['Pure APS'].min()) / 
                                   (df['Pure APS'].max() - df['Pure APS'].min()) * 100).fillna(0)

# Create engagement score (composite of combos, time, power-ups)
engagement_components = []
if 'Avg Combo' in df.columns:
    engagement_components.append(df['Avg Combo'].fillna(0))
if 'Avg SuperCombo' in df.columns:
    engagement_components.append(df['Avg SuperCombo'].fillna(0))
if 'Avg Win (s)' in df.columns:
    # Normalize time (more time = more engagement, but cap at reasonable max)
    normalized_time = (df['Avg Win (s)'].fillna(0) / df['Avg Win (s)'].quantile(0.95) * 100).clip(0, 100)
    engagement_components.append(normalized_time)

if engagement_components:
    df['engagement_score'] = pd.concat(engagement_components, axis=1).mean(axis=1)
else:
    df['engagement_score'] = 0

# Create risk score (high churn + high difficulty)
df['risk_score'] = ((df['churn_rate_calculated'].fillna(0) / 100) * 0.6 + 
                    (df['difficulty_score'] / 100) * 0.4) * 100

# Correlation analysis
retention_factors = ['retention_rate', 'churn_rate_calculated', 'difficulty_score', 
                     'engagement_score', 'APS', 'Pure APS', 'Attempts', 'Quits',
                     '% Used Boosters_numeric', 'Avg Win (s)', 'Avg Combo']

available_factors = [col for col in retention_factors if col in df.columns]
correlation_matrix = df[available_factors].corr()

print("\nCorrelation with Retention Rate:")
retention_corr = correlation_matrix['retention_rate'].sort_values(ascending=False)
for factor, corr in retention_corr.items():
    if factor != 'retention_rate':
        print(f"  {factor}: {corr:.3f}")

print("\nCorrelation with Churn Rate:")
churn_corr = correlation_matrix['churn_rate_calculated'].sort_values(ascending=False)
for factor, corr in churn_corr.items():
    if factor != 'churn_rate_calculated':
        print(f"  {factor}: {corr:.3f}")

# Difficulty impact on retention
print("\nDifficulty Impact Analysis:")
try:
    difficulty_bins = pd.qcut(df['APS'].fillna(1), q=4, labels=['Very Easy', 'Easy', 'Hard', 'Very Hard'], duplicates='drop')
    difficulty_retention = df.groupby(difficulty_bins).agg({
        'retention_rate': 'mean',
        'churn_rate_calculated': 'mean',
        'players_lost': 'sum'
    }).round(2)
    print(difficulty_retention.to_string())
except ValueError:
    # Fallback to cut if qcut fails due to duplicates
    difficulty_bins = pd.cut(df['APS'].fillna(1), bins=4, labels=['Very Easy', 'Easy', 'Hard', 'Very Hard'], duplicates='drop')
    difficulty_retention = df.groupby(difficulty_bins).agg({
        'retention_rate': 'mean',
        'churn_rate_calculated': 'mean',
        'players_lost': 'sum'
    }).round(2)
    print(difficulty_retention.to_string())

# Engagement impact
print("\nEngagement Impact Analysis:")
try:
    engagement_bins = pd.qcut(df['engagement_score'].fillna(0), q=3, labels=['Low', 'Medium', 'High'], duplicates='drop')
    engagement_retention = df.groupby(engagement_bins).agg({
        'retention_rate': 'mean',
        'churn_rate_calculated': 'mean',
        'players': 'mean'
    }).round(2)
    print(engagement_retention.to_string())
except ValueError:
    # Fallback to cut if qcut fails due to duplicates
    engagement_bins = pd.cut(df['engagement_score'].fillna(0), bins=3, labels=['Low', 'Medium', 'High'], duplicates='drop')
    engagement_retention = df.groupby(engagement_bins).agg({
        'retention_rate': 'mean',
        'churn_rate_calculated': 'mean',
        'players': 'mean'
    }).round(2)
    print(engagement_retention.to_string())

# Monetization impact
if '% Used Boosters_numeric' in df.columns:
    print("\nMonetization Impact (Booster Usage):")
    try:
        booster_bins = pd.qcut(df['% Used Boosters_numeric'].fillna(0), q=3, labels=['Low Usage', 'Medium Usage', 'High Usage'], duplicates='drop')
        booster_retention = df.groupby(booster_bins).agg({
            'retention_rate': 'mean',
            'churn_rate_calculated': 'mean',
            'players': 'mean'
        }).round(2)
        print(booster_retention.to_string())
    except ValueError:
        # Fallback to cut if qcut fails due to duplicates
        booster_bins = pd.cut(df['% Used Boosters_numeric'].fillna(0), bins=3, labels=['Low Usage', 'Medium Usage', 'High Usage'], duplicates='drop')
        booster_retention = df.groupby(booster_bins).agg({
            'retention_rate': 'mean',
            'churn_rate_calculated': 'mean',
            'players': 'mean'
        }).round(2)
        print(booster_retention.to_string())

# Frustration indicators
print("\nFrustration Indicators Analysis:")
high_frustration = df[(df['Attempts'] > df['Attempts'].quantile(0.75)) | 
                      (df['Quits'] > df['Quits'].quantile(0.75)) |
                      (df['Avg Lose (s)'].fillna(0) > df['Avg Lose (s)'].quantile(0.75))]
print(f"  Levels with high frustration indicators: {len(high_frustration)}")
if len(high_frustration) > 0:
    print(f"  Average churn rate for high frustration levels: {high_frustration['churn_rate_calculated'].mean():.2f}%")
    print(f"  Average churn rate for other levels: {df[~df.index.isin(high_frustration.index)]['churn_rate_calculated'].mean():.2f}%")

# ============================================================================
# 4. KEY VISUALIZATIONS
# ============================================================================

print("\n4. GENERATING VISUALIZATIONS...")
print("-" * 80)

# Plot 1: Retention Funnel Chart
plt.figure(figsize=(16, 8))
plt.subplot(2, 2, 1)
plt.plot(df['level_num'], df['players'], linewidth=2, color='steelblue', alpha=0.8)
plt.fill_between(df['level_num'], df['players'], alpha=0.3, color='steelblue')
plt.title('Retention Funnel - Player Count by Level', fontsize=14, fontweight='bold')
plt.xlabel('Level Number', fontsize=11)
plt.ylabel('Number of Players', fontsize=11)
plt.grid(True, alpha=0.3)
plt.yscale('log')  # Log scale for better visualization

# Plot 2: Retention Rate Over Levels
plt.subplot(2, 2, 2)
plt.plot(df['level_num'], df['retention_rate'], linewidth=2, color='coral', alpha=0.8)
plt.title('Cumulative Retention Rate by Level', fontsize=14, fontweight='bold')
plt.xlabel('Level Number', fontsize=11)
plt.ylabel('Retention Rate (%)', fontsize=11)
plt.grid(True, alpha=0.3)
plt.axhline(y=50, color='r', linestyle='--', alpha=0.5, label='50% Retention')
plt.axhline(y=25, color='orange', linestyle='--', alpha=0.5, label='25% Retention')
plt.legend()

# Plot 3: Churn Rate Heatmap by Level Ranges
plt.subplot(2, 2, 3)
level_ranges = [(1, 20), (21, 50), (51, 100), (101, 200), (201, 500), (501, 1000), (1001, 2000)]
churn_by_range = []
range_labels = []
for start, end in level_ranges:
    range_df = df[(df['level_num'] >= start) & (df['level_num'] <= end)]
    if len(range_df) > 0:
        avg_churn = range_df['churn_rate_calculated'].mean()
        churn_by_range.append(avg_churn)
        range_labels.append(f"{start}-{end}")

if churn_by_range:
    colors = ['green' if x < 5 else 'orange' if x < 15 else 'red' for x in churn_by_range]
    plt.barh(range(len(churn_by_range)), churn_by_range, color=colors, alpha=0.7)
    plt.yticks(range(len(range_labels)), range_labels)
    plt.title('Average Churn Rate by Level Ranges', fontsize=14, fontweight='bold')
    plt.xlabel('Average Churn Rate (%)', fontsize=11)
    plt.ylabel('Level Range', fontsize=11)
    plt.grid(True, alpha=0.3, axis='x')

# Plot 4: Difficulty vs Retention Scatter
plt.subplot(2, 2, 4)
scatter = plt.scatter(df['APS'], df['retention_rate'], 
                      c=df['churn_rate_calculated'], 
                      s=df['players']/50, 
                      alpha=0.6, 
                      cmap='RdYlGn_r',
                      edgecolors='black', linewidth=0.5)
plt.colorbar(scatter, label='Churn Rate (%)')
plt.title('Difficulty (APS) vs Retention Rate', fontsize=14, fontweight='bold')
plt.xlabel('APS (Attempts per Success)', fontsize=11)
plt.ylabel('Retention Rate (%)', fontsize=11)
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(output_dir / '01_retention_overview.png', dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ Saved: 01_retention_overview.png")

# Plot 5: Time Series - Player Count and Churn
fig, ax1 = plt.subplots(figsize=(16, 6))
ax2 = ax1.twinx()

line1 = ax1.plot(df['level_num'], df['players'], 'b-', linewidth=2, label='Player Count', alpha=0.8)
line2 = ax2.plot(df['level_num'], df['churn_rate_calculated'], 'r-', linewidth=2, label='Churn Rate', alpha=0.8)

ax1.set_xlabel('Level Number', fontsize=12)
ax1.set_ylabel('Number of Players', fontsize=12, color='b')
ax1.tick_params(axis='y', labelcolor='b')
ax1.set_yscale('log')
ax1.grid(True, alpha=0.3)

ax2.set_ylabel('Churn Rate (%)', fontsize=12, color='r')
ax2.tick_params(axis='y', labelcolor='r')

lines = line1 + line2
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, loc='upper right')

plt.title('Player Count and Churn Rate Over Levels', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig(output_dir / '02_player_churn_timeseries.png', dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ Saved: 02_player_churn_timeseries.png")

# Plot 6: Box Plots - Metrics for High vs Low Churn Levels
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Define high/low churn
high_churn = df[df['churn_rate_calculated'] > df['churn_rate_calculated'].quantile(0.75)]
low_churn = df[df['churn_rate_calculated'] < df['churn_rate_calculated'].quantile(0.25)]

# APS comparison
axes[0, 0].boxplot([low_churn['APS'].dropna(), high_churn['APS'].dropna()], 
                   labels=['Low Churn', 'High Churn'])
axes[0, 0].set_title('APS Distribution: Low vs High Churn', fontsize=12, fontweight='bold')
axes[0, 0].set_ylabel('APS', fontsize=11)
axes[0, 0].grid(True, alpha=0.3, axis='y')

# Engagement comparison
axes[0, 1].boxplot([low_churn['engagement_score'].dropna(), high_churn['engagement_score'].dropna()], 
                   labels=['Low Churn', 'High Churn'])
axes[0, 1].set_title('Engagement Score: Low vs High Churn', fontsize=12, fontweight='bold')
axes[0, 1].set_ylabel('Engagement Score', fontsize=11)
axes[0, 1].grid(True, alpha=0.3, axis='y')

# Attempts comparison
axes[1, 0].boxplot([low_churn['Attempts'].dropna(), high_churn['Attempts'].dropna()], 
                   labels=['Low Churn', 'High Churn'])
axes[1, 0].set_title('Attempts: Low vs High Churn', fontsize=12, fontweight='bold')
axes[1, 0].set_ylabel('Attempts', fontsize=11)
axes[1, 0].grid(True, alpha=0.3, axis='y')

# Retention rate comparison
axes[1, 1].boxplot([low_churn['retention_rate'].dropna(), high_churn['retention_rate'].dropna()], 
                   labels=['Low Churn', 'High Churn'])
axes[1, 1].set_title('Retention Rate: Low vs High Churn', fontsize=12, fontweight='bold')
axes[1, 1].set_ylabel('Retention Rate (%)', fontsize=11)
axes[1, 1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(output_dir / '03_high_vs_low_churn_comparison.png', dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ Saved: 03_high_vs_low_churn_comparison.png")

# Plot 7: Correlation Matrix - Retention Drivers
plt.figure(figsize=(14, 12))
corr_cols = ['retention_rate', 'churn_rate_calculated', 'difficulty_score', 
             'engagement_score', 'APS', 'Attempts', 'Quits', '% Used Boosters_numeric']
available_corr_cols = [col for col in corr_cols if col in df.columns and df[col].notna().sum() > 10]

if len(available_corr_cols) > 1:
    corr_matrix = df[available_corr_cols].corr()
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', cmap='coolwarm', 
                center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8},
                xticklabels=[col.replace('_', ' ').title() for col in available_corr_cols],
                yticklabels=[col.replace('_', ' ').title() for col in available_corr_cols])
    plt.title('Correlation Matrix - Retention Drivers', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / '04_retention_drivers_correlation.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✓ Saved: 04_retention_drivers_correlation.png")

# Plot 8: Engagement vs Retention
plt.figure(figsize=(14, 6))
scatter = plt.scatter(df['engagement_score'], df['retention_rate'], 
                      c=df['churn_rate_calculated'], 
                      s=df['players']/30, 
                      alpha=0.6, 
                      cmap='RdYlGn_r',
                      edgecolors='black', linewidth=0.5)
plt.colorbar(scatter, label='Churn Rate (%)')
plt.title('Engagement Score vs Retention Rate', fontsize=16, fontweight='bold')
plt.xlabel('Engagement Score', fontsize=12)
plt.ylabel('Retention Rate (%)', fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(output_dir / '05_engagement_vs_retention.png', dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ Saved: 05_engagement_vs_retention.png")

# ============================================================================
# 5. ACTIONABLE INSIGHTS
# ============================================================================

print("\n5. ACTIONABLE INSIGHTS")
print("-" * 80)

# Top 10 Problematic Levels (high churn + high difficulty)
df['problem_score'] = (df['churn_rate_calculated'].fillna(0) * 0.6 + 
                       df['difficulty_score'] * 0.4)
problematic_levels = df.nlargest(10, 'problem_score')[['Level', 'players', 'churn_rate_calculated', 
                                                        'APS', 'Attempts', 'problem_score']]
print("\nTop 10 Problematic Levels (High Churn + High Difficulty):")
print(problematic_levels.to_string(index=False))

# Retention Opportunities (high player count but high churn)
opportunity_levels = df[(df['players'] > df['players'].quantile(0.5)) & 
                        (df['churn_rate_calculated'] > df['churn_rate_calculated'].quantile(0.5))]
opportunity_levels = opportunity_levels.nlargest(10, 'players')[['Level', 'players', 'churn_rate_calculated', 
                                                                 'APS', 'engagement_score']]
print("\nTop 10 Retention Opportunities (High Player Count, High Churn):")
print(opportunity_levels.to_string(index=False))

# Segment Analysis
print("\nSegment Analysis:")
segments = {
    'Early Game (1-20)': (1, 20),
    'Mid Game (21-100)': (21, 100),
    'Late Game (100+)': (101, df['level_num'].max())
}

segment_summary = []
for segment_name, (start, end) in segments.items():
    segment_df = df[(df['level_num'] >= start) & (df['level_num'] <= end)]
    if len(segment_df) > 0:
        summary = {
            'Segment': segment_name,
            'Avg Players': segment_df['players'].mean(),
            'Avg Retention': segment_df['retention_rate'].mean(),
            'Avg Churn': segment_df['churn_rate_calculated'].mean(),
            'Avg APS': segment_df['APS'].mean(),
            'Avg Engagement': segment_df['engagement_score'].mean()
        }
        segment_summary.append(summary)

segment_df_summary = pd.DataFrame(segment_summary)
print(segment_df_summary.to_string(index=False))

# Recommendations
print("\nRecommendations:")
print("  1. Focus on balancing levels with high problem scores (high churn + high difficulty)")
print("  2. Investigate retention opportunities - levels with many players but high churn")
print("  3. Consider difficulty adjustments for levels with APS > 2.0")
print("  4. Enhance engagement features for levels with low engagement scores")
print("  5. Monitor booster usage patterns - high usage may indicate frustration")

# Executive Summary
print("\n" + "=" * 80)
print("EXECUTIVE SUMMARY")
print("=" * 80)
print(f"• Overall retention rate: {df['retention_rate'].iloc[-1]:.2f}%")
print(f"• Total players lost: {initial_players - df['players'].iloc[-1]:,.0f}")
print(f"• Average churn rate: {df['churn_rate_calculated'].mean():.2f}%")
print(f"• Most problematic level: {problematic_levels.iloc[0]['Level']} (Problem Score: {problematic_levels.iloc[0]['problem_score']:.2f})")
print(f"• Best retention opportunity: {opportunity_levels.iloc[0]['Level']} ({opportunity_levels.iloc[0]['players']:.0f} players, {opportunity_levels.iloc[0]['churn_rate_calculated']:.2f}% churn)")

print("\n" + "=" * 80)
print("RETENTION EDA COMPLETE!")
print(f"Visualizations saved to: {output_dir}")
print("=" * 80)

