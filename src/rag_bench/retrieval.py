from __future__ import annotations

from typing import Optional

# Re-exports for backward compatibility
from .retrievals import QdrantRetriever, Neo4jRetriever, Hit  # noqa: F401


def main(argv: Optional[list[str]] = None) -> None:
    from .cli.retrieval import main as cli_main

    cli_main(argv)


if __name__ == "__main__":
    main()
