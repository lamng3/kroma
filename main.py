"""Backward-compatible entry point for `python -m main`."""

from kroma.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
