"""Load and clean DEAM, then write the coverage plot and report.

Usage (from the repo root):
    python scripts/prepare_data.py                    # uses data/DEAM
    python scripts/prepare_data.py --data-dir path/to/DEAM

Outputs:
    data/songs_clean.csv              cleaned song table (not committed)
    results/coverage_plot.png         where songs are + gap heatmap
    results/coverage_grid.csv         gap for all 81 integer slider settings
    results/data_report.txt           cleaning counts + coverage summary
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from isoharmonix import config  # noqa: E402
from isoharmonix.coverage import coverage_table, plot_coverage, summarise  # noqa: E402
from isoharmonix.data import load_clean_songs  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", default=config.DEFAULT_DATA_DIR)
    p.add_argument("--out-csv", default=config.DEFAULT_CLEAN_CSV)
    p.add_argument("--results-dir", default="results")
    args = p.parse_args()

    songs, report = load_clean_songs(args.data_dir)
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    songs.to_csv(args.out_csv, index=False)

    results = Path(args.results_dir)
    results.mkdir(parents=True, exist_ok=True)
    grid = coverage_table(songs, step=1.0)
    grid.to_csv(results / "coverage_grid.csv", index=False)
    plot_coverage(songs, results / "coverage_plot.png")

    text = (
        "== Cleaning ==\n" + report.as_text()
        + "\n\n== Coverage (integer slider grid) ==\n" + summarise(grid) + "\n"
    )
    (results / "data_report.txt").write_text(text)
    print(text)
    print(f"Saved {args.out_csv} and outputs in {results}/")


if __name__ == "__main__":
    main()
