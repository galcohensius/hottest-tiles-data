import csv
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
LEVELS_DIR = REPO_ROOT / "data" / "levels"
OUTPUT_DIR = REPO_ROOT / "data" / "levels_from_csv"
INPUT_CSV = REPO_ROOT / "data" / "levels_export.csv"

GRID_ROWS = 10
GRID_COLS = 12


def load_rows():
    with INPUT_CSV.open() as f:
        reader = csv.DictReader(f)
        return list(reader)


def parse_colors_prob(colors_json):
    data = json.loads(colors_json) if colors_json else {}
    return [{"ColorName": name, "Chance": chance} for name, chance in data.items()]


def parse_grid(grid_json):
    grid_map = json.loads(grid_json) if grid_json else {}
    cells = []
    for r in range(1, GRID_ROWS + 1):
        for c in range(1, GRID_COLS + 1):
            key = f"{r},{c}"
            stacks = grid_map.get(key, [])
            # Only include cells with non-empty stacks
            if stacks:
                cells.append({"GridPosition": [r, c], "Stacks": stacks})
    return cells


def row_to_level(row):
    level_number = int(row["LevelNumber"])
    board = {
        "GridRows": int(row["GridRows"]) if row.get("GridRows") and row["GridRows"] != "" else 0,
        "GridCols": int(row["GridCols"]) if row.get("GridCols") and row["GridCols"] != "" else 0,
        "Grid": parse_grid(row["Grid"]),
        "TilesProbability": json.loads(row["TilesProbability"]) if row["TilesProbability"] else [],
        "MultiStack": [int(row["MultiStack"])] if row.get("MultiStack") and row["MultiStack"] != "" else [],
        "ColorsProbability": parse_colors_prob(row["ColorsProbability"]),
        "LevelTarget": json.loads(row["LevelTarget"]) if row["LevelTarget"] else [],
        "TotalMoves": int(row["total_moves"]) if row["total_moves"] else None,
        "MinFairy": int(row["MinFairy"]) if row["MinFairy"] else 0,
        "MinCoin": int(row["MinCoin"]) if row["MinCoin"] else 0,
    }

    # Optional dynamic fields
    if row.get("DynamicBombPercents"):
        board["DynamicBombPercents"] = int(row["DynamicBombPercents"])
    if row.get("DynamicIcePercents"):
        board["DynamicIcePercents"] = int(row["DynamicIcePercents"])

    return {
        "Level": {
            "LevelNumber": level_number,
            "LevelDifficulty": row.get("LevelDifficulty"),
            "Board": board,
        }
    }


def write_levels(levels):
    OUTPUT_DIR.mkdir(exist_ok=True)
    for level in levels:
        level_num = level["Level"]["LevelNumber"]
        out_path = OUTPUT_DIR / f"Level_{level_num}.json"
        out_path.write_text(json.dumps(level, indent=2))


def normalize_level(data):
    """Canonicalize ordering for comparison."""
    board = data["Level"]["Board"]

    # Build a full 10x12 grid map, defaulting missing cells to empty stacks
    grid_map = {}
    for cell in board.get("Grid", []) or []:
        pos = cell.get("GridPosition", [])
        if len(pos) == 2:
            r, c = pos
            grid_map[f"{r},{c}"] = cell.get("Stacks", []) or []
    for r in range(1, GRID_ROWS + 1):
        for c in range(1, GRID_COLS + 1):
            grid_map.setdefault(f"{r},{c}", [])
    grid_sorted = [
        {"GridPosition": [r, c], "Stacks": grid_map[f"{r},{c}"]}
        for r in range(1, GRID_ROWS + 1)
        for c in range(1, GRID_COLS + 1)
    ]

    # Sort colors probability by ColorName
    colors = sorted(board.get("ColorsProbability", []), key=lambda x: x.get("ColorName", ""))

    # Sort level targets by ColorName (stable for comparison)
    targets = sorted(board.get("LevelTarget", []), key=lambda x: x.get("ColorName", ""))

    normalized = {
        "Level": {
            "LevelNumber": data["Level"]["LevelNumber"],
            "LevelDifficulty": data["Level"].get("LevelDifficulty"),
            "Board": {
                "GridRows": board.get("GridRows", 0),
                "GridCols": board.get("GridCols", 0),
                "Grid": grid_sorted,
                "TilesProbability": board.get("TilesProbability", []),
                "MultiStack": board.get("MultiStack", []),
                "ColorsProbability": colors,
                "LevelTarget": targets,
                "TotalMoves": board.get("TotalMoves"),
                "MinFairy": board.get("MinFairy", 0),
                "MinCoin": board.get("MinCoin", 0),
            },
        }
    }

    # Preserve optional dynamics if present
    if "DynamicBombPercents" in board:
        normalized["Level"]["Board"]["DynamicBombPercents"] = board["DynamicBombPercents"]
    if "DynamicIcePercents" in board:
        normalized["Level"]["Board"]["DynamicIcePercents"] = board["DynamicIcePercents"]

    return normalized


def load_original_levels():
    originals = {}
    for path in sorted(LEVELS_DIR.glob("Level_*.json"), key=lambda p: int(p.stem.split("_")[1])):
        data = json.loads(path.read_text())
        level_number = data.get("Level", {}).get("LevelNumber")
        if level_number is not None:
            originals[level_number] = data
    return originals


def verify_roundtrip(regenerated):
    originals = load_original_levels()
    mismatches = []
    for level in regenerated:
        num = level["Level"]["LevelNumber"]
        orig = originals.get(num)
        if not orig:
            mismatches.append((num, "missing_original"))
            continue

        norm_orig = normalize_level(orig)
        norm_regen = normalize_level(level)
        if norm_orig != norm_regen:
            mismatches.append((num, "diff"))

    return mismatches


def main():
    rows = load_rows()
    levels = [row_to_level(row) for row in rows]
    write_levels(levels)
    print(f"Wrote {len(levels)} levels to {OUTPUT_DIR}")

    mismatches = verify_roundtrip(levels)
    if mismatches:
        details = ", ".join(f"{num}:{reason}" for num, reason in mismatches[:5])
        print(f"Roundtrip mismatches ({len(mismatches)}): {details}")
    else:
        print("Roundtrip verification passed for all levels.")


if __name__ == "__main__":
    main()

