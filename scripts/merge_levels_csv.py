import csv
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
LEVELS_EXPORT_CSV = REPO_ROOT / "data" / "levels_export.csv"
FUNNEL_CSV = REPO_ROOT / "data" / "hottestTiles_LevelsFunnel.csv"
OUTPUT_CSV = REPO_ROOT / "data" / "levels_unified.csv"


def extract_level_number(level_str):
    """Extract level number from 'Level X' format."""
    if level_str.startswith("Level "):
        try:
            return int(level_str.split()[1])
        except (IndexError, ValueError):
            return None
    return None


def load_levels_export():
    """Load levels_export.csv and return dict keyed by LevelNumber."""
    levels = {}
    with LEVELS_EXPORT_CSV.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            level_num = int(row["LevelNumber"])
            levels[level_num] = row
    return levels


def load_funnel_data():
    """Load hottestTiles_LevelsFunnel.csv and return dict keyed by level number."""
    funnel_data = {}
    with FUNNEL_CSV.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            level_num = extract_level_number(row["Level"])
            if level_num is not None:
                funnel_data[level_num] = row
    return funnel_data


def main():
    levels_export = load_levels_export()
    funnel_data = load_funnel_data()
    
    # Get column names
    # All columns from levels_export except Grid (last column)
    with LEVELS_EXPORT_CSV.open("r", encoding="utf-8") as f:
        export_reader = csv.DictReader(f)
        export_columns = export_reader.fieldnames
    
    # Remove Grid from export columns (it's the last one)
    # Also rename LevelNumber to Level and LevelDifficulty to Difficulty
    export_columns_without_grid = []
    for col in export_columns:
        if col == "Grid":
            continue
        elif col == "LevelNumber":
            export_columns_without_grid.append("Level")
        elif col == "LevelDifficulty":
            export_columns_without_grid.append("Difficulty")
        else:
            export_columns_without_grid.append(col)
    
    # All columns from funnel (except Level, since we're renaming LevelNumber to Level)
    with FUNNEL_CSV.open("r", encoding="utf-8") as f:
        funnel_reader = csv.DictReader(f)
        funnel_columns = [col for col in funnel_reader.fieldnames if col != "Level"]
    
    # Final column order: export (without Grid, LevelNumber renamed to Level), funnel (without Level), Grid
    fieldnames = export_columns_without_grid + list(funnel_columns) + ["Grid"]
    
    # Write merged CSV
    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        # Process each level from levels_export
        for level_num in sorted(levels_export.keys()):
            level_row = levels_export[level_num]
            funnel_row = funnel_data.get(level_num, {})
            
            # Build merged row
            merged_row = {}
            
            # Add export columns (except Grid)
            # Map LevelNumber to Level and LevelDifficulty to Difficulty
            for col in export_columns_without_grid:
                if col == "Level":
                    merged_row[col] = level_row.get("LevelNumber", "")
                elif col == "Difficulty":
                    merged_row[col] = level_row.get("LevelDifficulty", "")
                else:
                    merged_row[col] = level_row.get(col, "")
            
            # Add funnel columns
            for col in funnel_columns:
                merged_row[col] = funnel_row.get(col, "")
            
            # Add Grid (last)
            merged_row["Grid"] = level_row.get("Grid", "")
            
            writer.writerow(merged_row)
    
    print(f"Wrote {len(levels_export)} levels to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()

