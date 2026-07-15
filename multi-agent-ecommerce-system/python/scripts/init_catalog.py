"""Initialize the local SQLite catalog and inventory data."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import get_settings
from services.catalog_repository import CatalogRepository


def main() -> None:
    settings = get_settings()
    repository = CatalogRepository(settings.database_url)
    inserted = repository.initialize()
    product_count = len(repository.list_products())
    print(f"Catalog initialized: inserted={inserted} products={product_count}")


if __name__ == "__main__":
    main()