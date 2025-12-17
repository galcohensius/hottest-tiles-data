import csv
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
LEVELS_DIR = REPO_ROOT / "data" / "levels"
OUTPUT_CSV = REPO_ROOT / "data" / "levels_export.csv"

# Fixed grid dimensions
GRID_ROWS = 10
GRID_COLS = 12


def load_levels():
    levels = []
    for path in sorted(LEVELS_DIR.glob("Level_*.json"), key=lambda p: int(p.stem.split("_")[1])):
        data = json.loads(path.read_text())
        level = data.get("Level", {})
        board = level.get("Board", {})

        # Core fields
        level_number = level.get("LevelNumber")
        level_difficulty = level.get("LevelDifficulty")
        total_moves = board.get("TotalMoves")
        min_fairy = board.get("MinFairy", 0)
        min_coin = board.get("MinCoin", 0)
        dynamic_bomb_percents = board.get("DynamicBombPercents")
        dynamic_ice_percents = board.get("DynamicIcePercents")

        # Probabilities and arrays
        tiles_prob = board.get("TilesProbability", [])
        multi_stack_list = board.get("MultiStack", [])
        # MultiStack is an array but we want just the first value as int
        multi_stack = multi_stack_list[0] if multi_stack_list and len(multi_stack_list) > 0 else None

        # Colors probability as map color: prob
        colors_prob_list = board.get("ColorsProbability", []) or []
        colors_prob = {
            entry.get("ColorName"): entry.get("Chance")
            for entry in colors_prob_list
            if entry.get("ColorName") is not None
        }

        # Level targets keep as array
        level_target = board.get("LevelTarget", []) or []

        # Grid as map "r,c" -> stacks (list)
        grid_map = {}
        grid = board.get("Grid", []) or []
        for cell in grid:
            pos = cell.get("GridPosition", [])
            if len(pos) == 2:
                r, c = pos
                key = f"{r},{c}"
                grid_map[key] = cell.get("Stacks", []) or []

        # Normalize to fixed 10x12: fill missing as empty list
        for r in range(1, GRID_ROWS + 1):
            for c in range(1, GRID_COLS + 1):
                key = f"{r},{c}"
                grid_map.setdefault(key, [])

        levels.append(
            {
                "LevelNumber": level_number,
                "LevelDifficulty": level_difficulty,
                "total_moves": total_moves,
                "MinFairy": min_fairy,
                "MinCoin": min_coin,
                "DynamicBombPercents": dynamic_bomb_percents,
                "DynamicIcePercents": dynamic_ice_percents,
                "TilesProbability": tiles_prob,
                "MultiStack": multi_stack,
                "ColorsProbability": colors_prob,
                "LevelTarget": level_target,
                "Grid": grid_map,
            }
        )

    return levels


def write_csv(levels):
    fieldnames = [
        "LevelNumber",
        "LevelDifficulty",
        "total_moves",
        "MinFairy",
        "MinCoin",
        "DynamicBombPercents",
        "DynamicIcePercents",
        "TilesProbability",
        "MultiStack",
        "ColorsProbability",
        "LevelTarget",
        "Grid",
    ]

    with OUTPUT_CSV.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for level in levels:
            row = {
                "LevelNumber": level["LevelNumber"],
                "LevelDifficulty": level["LevelDifficulty"],
                "total_moves": level["total_moves"],
                "MinFairy": level["MinFairy"],
                "MinCoin": level["MinCoin"],
                "DynamicBombPercents": level["DynamicBombPercents"],
                "DynamicIcePercents": level["DynamicIcePercents"],
                "TilesProbability": json.dumps(level["TilesProbability"], separators=(",", ":")),
                "MultiStack": level["MultiStack"],
                "ColorsProbability": json.dumps(level["ColorsProbability"], separators=(",", ":")),
                "LevelTarget": json.dumps(level["LevelTarget"], separators=(",", ":")),
                "Grid": json.dumps(level["Grid"], separators=(",", ":")),
            }
            writer.writerow(row)


def main():
    levels = load_levels()
    write_csv(levels)
    print(f"Wrote {len(levels)} levels to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()

