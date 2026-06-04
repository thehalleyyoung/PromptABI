"""Module entrypoint for ``python -m promptabi``."""

from .cli import main


if __name__ == "__main__":
    raise SystemExit(main())
