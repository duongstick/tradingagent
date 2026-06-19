"""Convenience entry point: `python main.py` runs the full pipeline.

For the richer CLI use `vnquant --help` (installed entry point) or `python -m cli.main`.
"""

from cli.main import app

if __name__ == "__main__":
    app()
