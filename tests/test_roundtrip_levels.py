"""
Round-trip test: JSON levels -> CSV -> JSON, then compare for equality.

Usage:
  python tests/test_roundtrip_levels.py
"""

import json
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import export_levels_to_csv as exporter  # noqa: E402
from scripts import inverse_csv_to_json as inverter  # noqa: E402

CSV_PATH = exporter.OUTPUT_CSV
OUTPUT_DIR = inverter.OUTPUT_DIR
LEVELS_DIR = exporter.LEVELS_DIR


def load_levels_from_dir(directory: Path):
    levels = {}
    for path in sorted(directory.glob("Level_*.json"), key=lambda p: int(p.stem.split("_")[1])):
        data = json.loads(path.read_text())
        number = data.get("Level", {}).get("LevelNumber")
        if number is not None:
            # If duplicate LevelNumber, keep the first one (original behavior)
            if number not in levels:
                levels[number] = data
    return levels


def clean_outputs():
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    if CSV_PATH.exists():
        CSV_PATH.unlink()


def main():
    clean_outputs()

    # Step 1: export JSON levels to CSV
    exporter.main()

    # Step 2: regenerate JSON from CSV
    inverter.main()

    # Step 3: compare originals vs regenerated
    originals = load_levels_from_dir(LEVELS_DIR)
    regenerated = load_levels_from_dir(OUTPUT_DIR)

    mismatches = []
    missing = []

    for num, orig in originals.items():
        regen = regenerated.get(num)
        if regen is None:
            missing.append(num)
            continue
        if inverter.normalize_level(orig) != inverter.normalize_level(regen):
            mismatches.append(num)

    extra = sorted(set(regenerated.keys()) - set(originals.keys()))

    if missing or extra or mismatches:
        print(f"Missing regenerated levels: {missing}" if missing else "Missing regenerated levels: none")
        print(f"Extra regenerated levels: {extra}" if extra else "Extra regenerated levels: none")
        print(f"Mismatched content: {mismatches}" if mismatches else "Mismatched content: none")
        raise SystemExit(1)

    print(f"Roundtrip succeeded for {len(originals)} levels.")


if __name__ == "__main__":
    main()

