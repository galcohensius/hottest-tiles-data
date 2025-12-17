import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
LEVELS_DIR = REPO_ROOT / "data" / "levels"

# Default values that do not carry information when false/zero (stack-level)
DEFAULT_FLAGS = {
    "Fairy": False,
    "Ice": False,
    "Grass": False,
    "Flower": False,
    "Bunny": False,
    "Dwarf": False,
    "Birdhouse": False,
    "Empty": False,
}
DEFAULT_ZERO_FIELDS_STACK = {"Igloo": 0}
# Board-level zero-value fields to drop (none - keep MinFairy and MinCoin even if 0)
DEFAULT_ZERO_FIELDS_BOARD = {}


def compact_stack(stack: dict) -> dict:
    """Return a stack dict without redundant default-valued fields."""
    cleaned = {}
    for key, value in stack.items():
        if key in DEFAULT_FLAGS and value is False:
            continue
        if key in DEFAULT_ZERO_FIELDS_STACK and value == DEFAULT_ZERO_FIELDS_STACK[key]:
            continue
        cleaned[key] = value
    return cleaned


def process_level(path: Path) -> bool:
    """Compact one level file in place. Returns True if changed."""
    before = path.read_text()
    data = json.loads(before)

    level = data.get("Level", {})
    board = level.get("Board", {})

    grid = board.get("Grid", [])
    cleaned_grid = []
    for cell in grid:
        stacks = cell.get("Stacks", [])
        cell["Stacks"] = [compact_stack(stack) for stack in stacks]
        # Only keep cells with non-empty stacks
        if cell["Stacks"]:
            cleaned_grid.append(cell)
    board["Grid"] = cleaned_grid

    # Drop zero-value board-level defaults
    for key in list(DEFAULT_ZERO_FIELDS_BOARD.keys()):
        if board.get(key) == DEFAULT_ZERO_FIELDS_BOARD[key]:
            board.pop(key, None)

    after = json.dumps(data, indent=2)
    if after != before:
        path.write_text(after + "\n")
        return True
    return False


def main():
    changed = 0
    for path in sorted(LEVELS_DIR.glob("Level_*.json")):
        if process_level(path):
            changed += 1
            print(f"cleaned {path.name}")
    print(f"Done. Updated {changed} files.")


if __name__ == "__main__":
    main()

