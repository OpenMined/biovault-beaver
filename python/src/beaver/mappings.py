"""Mapping helpers for private URL -> local path resolution (BioVault)."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import yaml


class MappingStore:
    """
    Store for private_url -> local_path mappings.

    Default location: BIOVAULT_HOME/mapping.yaml
    """

    def __init__(self, path: Path):
        self.path = path
        self._data: dict[str, str] = {}
        self._loaded = False

    @classmethod
    def from_env(cls, allow_missing: bool = True) -> Optional[MappingStore]:
        home = os.environ.get("BIOVAULT_HOME")
        if home:
            return cls(Path(home) / "mapping.yaml")
        # Fallback to mapping.yaml in current working directory if env not set
        cwd_path = Path.cwd() / "mapping.yaml"
        if cwd_path.exists():
            return cls(cwd_path)
        if allow_missing:
            return None
        raise ValueError("BIOVAULT_HOME is not set; cannot locate mapping.yaml")

    def load(self) -> None:
        if self._loaded:
            return
        self._loaded = True
        if not self.path.exists():
            return
        try:
            raw = self.path.read_text()
            data = yaml.safe_load(raw) or {}
            if isinstance(data, dict):
                # Support either flat mapping or nested under "mappings"
                mapping_dict = data.get("mappings") if "mappings" in data else data
                if isinstance(mapping_dict, dict):
                    self._data = {
                        str(k): str(v)
                        for k, v in mapping_dict.items()
                        if isinstance(k, str) and isinstance(v, (str, Path))
                    }
        except Exception:
            # Ignore load errors; act as empty
            self._data = {}

    def save(self) -> None:
        if not self.path.parent.exists():
            self.path.parent.mkdir(parents=True, exist_ok=True)
        yaml.safe_dump(self._data, self.path.open("w"))

    def all(self) -> dict[str, str]:
        self.load()
        return dict(self._data)

    def get(self, private_url: str) -> Optional[Path]:
        self.load()
        val = self._data.get(private_url)
        if val is None:
            return None
        p = Path(val)
        if not p.is_absolute():
            p = (self.path.parent / p).resolve()
        return p

    def set(self, private_url: str, local_path: str | Path, persist: bool = True) -> None:
        self.load()
        self._data[str(private_url)] = str(local_path)
        if persist:
            self.save()

    def __repr__(self) -> str:
        return f"MappingStore(path={self.path}, entries={len(self._data or {})})"

    def _repr_html_(self):
        self.load()
        rows = []
        for url, path in self._data.items():
            rows.append(f"<tr><td><code>{url}</code></td><td><code>{path}</code></td></tr>")
        body = "".join(rows) if rows else "<tr><td colspan='2'><i>No mappings</i></td></tr>"
        return f"""
        <div style='font-family: monospace; border-left: 3px solid #9C27B0; padding-left: 10px;'>
            <b>Private Mappings</b> <span style='color:#888;font-size:11px;'>{self.path}</span>
            <table style='margin-top:6px; border-collapse: collapse;'>
                <thead><tr><th style='padding:4px 8px; text-align:left;'>private_url</th><th style='padding:4px 8px; text-align:left;'>local_path</th></tr></thead>
                <tbody>{body}</tbody>
            </table>
        </div>
        """
