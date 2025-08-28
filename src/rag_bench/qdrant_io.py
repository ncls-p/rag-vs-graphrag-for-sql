from __future__ import annotations

from typing import Optional

from .io.qdrant import QdrantIO as _QdrantIO


def _combined_text(rec):
    # Backward compatibility: unused here, retained for imports
    from .utils.text import combined_text

    return combined_text(rec)


class QdrantIO(_QdrantIO):
    # Backward-compat wrapper if someone imports from rag_bench.qdrant_io
    pass


def main(argv: Optional[list[str]] = None) -> None:
    from .cli.qdrant import main as cli_main

    cli_main(argv)


if __name__ == "__main__":
    main()
