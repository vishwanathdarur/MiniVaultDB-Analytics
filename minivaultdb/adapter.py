"""
minivaultdb/adapter.py

Thin wrapper around your existing MiniVaultDB engine so the analytics
pipeline can call a consistent API regardless of your internal implementation.

IMPORTANT: Replace the MiniVaultDB class below with an import of your
actual engine. For example:
    from minivaultdb.db import MiniVaultDB          # adjust path as needed
    from minivaultdb.storage import StorageEngine   # or whatever your class is

The wrapper expects your engine to expose:
    db.put(key: str, value: str)   -> None
    db.get(key: str)               -> str | None
    db.scan()                      -> list[tuple[str, str]]   (all key-value pairs)
    db.delete(key: str)            -> None  (optional)

If your API differs slightly (e.g. get_all() instead of scan()), just
update the methods in MiniVaultDBAdapter below.
"""

import json
from typing import Any

from minivaultdb.db import MiniVaultDB


# ──────────────────────────────────────────────────────────────────────────────
# Adapter — used by the entire analytics pipeline
# ──────────────────────────────────────────────────────────────────────────────
class MiniVaultDBAdapter:
    """
    Wraps MiniVaultDB with JSON serialization helpers so the pipeline can
    store and retrieve rich Python objects (dicts, lists) transparently.
    """

    def __init__(self, db_path: str = "vault_data"):
        self.db = MiniVaultDB(db_path)
        self.db_path = db_path

    # ------------------------------------------------------------------
    # Core write / read
    # ------------------------------------------------------------------

    def put_record(self, key: str, record: dict[str, Any]) -> None:
        """Serialize a dict to JSON and write it to the DB."""
        self.db.put(key, json.dumps(record))

    def get_record(self, key: str) -> dict[str, Any] | None:
        """Fetch a JSON record by key and return it as a dict."""
        raw = self.db.get(key)
        if raw is None:
            return None
        return json.loads(raw)

    def delete_record(self, key: str) -> None:
        self.db.delete(key)

    # ------------------------------------------------------------------
    # Bulk operations
    # ------------------------------------------------------------------

    def put_many(self, records: list[tuple[str, dict[str, Any]]]) -> None:
        """Write a list of (key, record) tuples in one call."""
        for key, record in records:
            self.put_record(key, record)

    def scan_all(self) -> list[tuple[str, dict[str, Any]]]:
        """Return all stored records as (key, dict) pairs."""
        return [(k, json.loads(v)) for k, v in self.db.scan()]

    def count(self) -> int:
        return len(self.db.scan())

    def __repr__(self) -> str:
        return f"MiniVaultDBAdapter(path={self.db_path!r}, records={self.count()})"
