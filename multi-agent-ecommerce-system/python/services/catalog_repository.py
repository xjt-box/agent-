"""SQLite-backed product catalog and inventory repository."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from sqlalchemy import ForeignKey, Integer, String, create_engine, select
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, relationship, sessionmaker
from sqlalchemy.types import JSON, Float

from models.schemas import Product


class Base(DeclarativeBase):
    """Base class for catalog persistence models."""


class ProductRecord(Base):
    """Persisted product attributes used during recommendation."""

    __tablename__ = "products"

    product_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    name: Mapped[str] = mapped_column(String(255))
    category: Mapped[str] = mapped_column(String(128), index=True)
    price: Mapped[float] = mapped_column(Float)
    description: Mapped[str] = mapped_column(String(1000), default="")
    brand: Mapped[str] = mapped_column(String(128), default="")
    seller_id: Mapped[str] = mapped_column(String(64), default="")
    tags: Mapped[list[str]] = mapped_column(JSON, default=list)
    image_url: Mapped[str] = mapped_column(String(1000), default="")
    inventory: Mapped["InventoryRecord"] = relationship(back_populates="product", uselist=False)


class InventoryRecord(Base):
    """Current on-hand inventory for one product."""

    __tablename__ = "inventory"

    product_id: Mapped[str] = mapped_column(ForeignKey("products.product_id"), primary_key=True)
    stock: Mapped[int] = mapped_column(Integer, default=0)
    product: Mapped[ProductRecord] = relationship(back_populates="inventory")


@dataclass(frozen=True)
class SeedProduct:
    """Product data used to initialize a local development catalog."""

    product_id: str
    name: str
    category: str
    price: float
    brand: str
    seller_id: str
    stock: int
    tags: list[str]


DEFAULT_CATALOG = [
    SeedProduct("P001", "Nova Phone Pro", "phones", 7999, "Nova", "S01", 500, ["flagship", "new"]),
    SeedProduct("P002", "Orion Phone", "phones", 5999, "Orion", "S02", 300, ["flagship"]),
    SeedProduct("P003", "Cloud Buds", "headphones", 1899, "Cloud", "S01", 1000, ["wireless", "noise-cancelling"]),
    SeedProduct("P004", "Studio Headset", "headphones", 2499, "Studio", "S03", 200, ["over-ear", "noise-cancelling"]),
    SeedProduct("P005", "Slate Tablet", "tablets", 4799, "Slate", "S01", 400, ["study", "office"]),
    SeedProduct("P006", "Pulse Charger", "accessories", 399, "Pulse", "S05", 80, ["fast-charge", "portable"]),
]


class CatalogRepository:
    """Provides persisted product and inventory access for recommendation agents."""

    def __init__(self, database_url: str):
        connect_args = {"check_same_thread": False} if database_url.startswith("sqlite") else {}
        self.engine = create_engine(database_url, connect_args=connect_args)
        self.session_factory = sessionmaker(bind=self.engine, expire_on_commit=False)

    def close(self) -> None:
        """Release database connections held by the repository engine."""
        self.engine.dispose()

    def initialize(self, seed_products: Iterable[SeedProduct] = DEFAULT_CATALOG) -> int:
        """Create tables and seed the catalog once. Returns inserted product count."""
        seed_items = tuple(seed_products)
        Base.metadata.create_all(self.engine)
        with self.session_factory() as session:
            if session.scalar(select(ProductRecord.product_id).limit(1)) is not None:
                return 0

            for seed in seed_items:
                session.add(
                    ProductRecord(
                        product_id=seed.product_id,
                        name=seed.name,
                        category=seed.category,
                        price=seed.price,
                        brand=seed.brand,
                        seller_id=seed.seller_id,
                        tags=seed.tags,
                    )
                )
                session.add(InventoryRecord(product_id=seed.product_id, stock=seed.stock))
            session.commit()
        return len(seed_items)

    def list_products(self, limit: int | None = None) -> list[Product]:
        """Return catalog products with their current stock values."""
        statement = select(ProductRecord, InventoryRecord.stock).join(InventoryRecord).order_by(ProductRecord.product_id)
        if limit is not None:
            statement = statement.limit(limit)

        with self.session_factory() as session:
            rows = session.execute(statement).all()
        return [
            Product(
                product_id=record.product_id,
                name=record.name,
                category=record.category,
                price=record.price,
                description=record.description,
                brand=record.brand,
                seller_id=record.seller_id,
                stock=stock,
                tags=record.tags or [],
                image_url=record.image_url,
            )
            for record, stock in rows
        ]

    def get_stock(self, product_ids: Iterable[str]) -> dict[str, int]:
        """Return current stock values for the requested product IDs."""
        requested_ids = list(product_ids)
        if not requested_ids:
            return {}

        statement = select(InventoryRecord.product_id, InventoryRecord.stock).where(
            InventoryRecord.product_id.in_(requested_ids)
        )
        with self.session_factory() as session:
            return dict(session.execute(statement).all())

    def set_stock(self, product_id: str, stock: int) -> None:
        """Update on-hand stock for one persisted product."""
        if stock < 0:
            raise ValueError("Stock cannot be negative.")

        with self.session_factory() as session:
            inventory = session.get(InventoryRecord, product_id)
            if inventory is None:
                raise KeyError(f"Unknown product ID: {product_id}")
            inventory.stock = stock
            session.commit()