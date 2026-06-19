from __future__ import annotations

import sys

from scrabble_plotter.main import main as run_scrabble_plotter


def main() -> int:
    argv = sys.argv[1:] or ["gui"]
    return run_scrabble_plotter(argv)


if __name__ == "__main__":
    raise SystemExit(main())
