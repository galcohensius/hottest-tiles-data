# Game Analytics & Level Data Tools

This repo contains two main parts:
1) Game analytics on the level funnel and retention.
2) Utilities to export/merge level JSON files with CSV, plus a round‑trip test.

## Running the analytics
- EDA: `python analytics/EDA.py` → prints dataset overview/stats and saves plots under `analytics/EDA_plots/` (progression, attempts, churn, APS, combos, time, boosters, correlations).
- Retention: `python analytics/retention_EDA.py` → computes retention/churn metrics, highlights drop-offs, correlates difficulty/engagement/monetization, and saves visuals under `analytics/retention_plots/`.
- Data source: `data/hottestTiles_LevelsFunnel.csv`.

## Level data utilities (JSON ↔ CSV)
- `scripts/export_levels_to_csv.py`: Reads `data/levels/Level_*.json` (123 levels) and writes a flattened table to `data/levels_export.csv`.
- `scripts/inverse_csv_to_json.py`: Rebuilds JSON levels into `data/levels_from_csv/` from `levels_export.csv`; also normalizes for comparison.
- `scripts/merge_levels_csv.py`: Merges `levels_export.csv` with funnel metrics from `hottestTiles_LevelsFunnel.csv` into `data/levels_unified.csv` (drops Grid, renames LevelNumber→Level, LevelDifficulty→Difficulty, appends funnel columns).
- `scripts/clean_levels.py`: Helpers to clean/normalize level files (see script for details).

## Tests
- `tests/test_roundtrip_levels.py`: Runs export → inverse → compare to ensure regenerated levels match originals.
  - Run with `python tests/test_roundtrip_levels.py`.

## Data layout
- `data/levels/`: Original level JSONs (123).
- `data/levels_from_csv/`: Regenerated JSONs from CSV.
- `data/levels_export.csv`: Flattened level data produced by export script.
- `data/levels_unified.csv` / `.xlsx`: Merged level + funnel data.
- `data/hottestTiles_LevelsFunnel.csv`: Funnel/retention metrics input.
- `backup/`: Copies of original/unused levels.

## Requirements
- Python 3.x
- Install: `pip install -r requirements.txt`
