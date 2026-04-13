"""
minivaultdb/db.py

A minimal MiniVaultDB-compatible engine for use by the analytics pipeline.
This implementation is a simple JSON-backed key-value store that exposes the
same interface expected by minivaultdb.adapter.MiniVaultDBAdapter.
"""

from __future__ import annotations

import json
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

try:
    from ._native import MiniVaultDB
except ImportError:
    class MiniVaultDB:  # type: ignore
        def __init__(self, path: str = "vault_data"):
            self.path = Path(path)
            if self.path.is_dir() or self.path.suffix == "":
                self.path.mkdir(parents=True, exist_ok=True)
                self.file_path = self.path / "minivaultdb.json"
            else:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                self.file_path = self.path

            self._store: dict[str, str] = {}
            self._load()
            print(f"[MiniVaultDB STUB] Using JSON store at {self.file_path}")

        def _load(self) -> None:
            if not self.file_path.exists():
                self._store = {}
                return

            try:
                with self.file_path.open("r", encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, dict):
                        self._store = {str(k): str(v) for k, v in data.items()}
                    else:
                        self._store = {}
            except (json.JSONDecodeError, OSError):
                self._store = {}

        def _persist(self) -> None:
            self.file_path.parent.mkdir(parents=True, exist_ok=True)
            with NamedTemporaryFile("w", delete=False, encoding="utf-8", dir=str(self.file_path.parent)) as tmp:
                json.dump(self._store, tmp, ensure_ascii=False, indent=2)
                tmp.flush()
                tmp_name = Path(tmp.name)
            tmp_name.replace(self.file_path)

        def put(self, key: str, value: str) -> None:
            self._store[key] = value
            self._persist()

        def get(self, key: str) -> str | None:
            return self._store.get(key)

        def scan(self) -> list[tuple[str, str]]:
            return list(self._store.items())

        def delete(self, key: str) -> None:
            self._store.pop(key, None)
            self._persist()
