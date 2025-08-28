from __future__ import annotations

from typing import Optional


def main(argv: Optional[list[str]] = None) -> None:
    from .cli.benchmark import main as cli_main

    cli_main(argv)


if __name__ == "__main__":
    main()
