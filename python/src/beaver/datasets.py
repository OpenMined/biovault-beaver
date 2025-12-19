"""Dataset discovery and Twin construction from Biovault dataset manifests."""

from __future__ import annotations

import builtins
import os
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from uuid import uuid4

import yaml

from .mappings import MappingStore
from .syft_url import parse_syft_url, path_to_syft_url, sanitize_syft_path
from .twin import Twin


@dataclass
class _DatasetResource:
    name: str
    path: Path
    schema: Optional[str] = None
    version: Optional[str] = None


class DatasetRegistry:
    """
    Discover datasets published to SyftBox datasites and materialize them as Twins.

    Access pattern:
        bv.datasets["client1@sandbox.local"]["single_cell"]
    """

    def __init__(self, base_dir: Path, backend=None, mapping_store: Optional[MappingStore] = None):
        self._base_dir = Path(base_dir)
        self._backend = backend
        self._mapping_store = mapping_store
        self._owner_cache: dict[str, _DatasetOwnerView] = {}

    def owners(self) -> Iterable[str]:
        """List owners that have a datasets.yaml index."""
        datasites = self._base_dir / "datasites"
        if not datasites.exists():
            return []
        owners = []
        for owner_dir in datasites.iterdir():
            if not owner_dir.is_dir():
                continue
            index = owner_dir / "public" / "biovault" / "datasets.yaml"
            if index.exists():
                owners.append(owner_dir.name)
        return owners

    def list(self) -> dict[str, Iterable[str]]:
        """List datasets per owner."""
        result: dict[str, Iterable[str]] = {}
        for owner in self.owners():
            result[owner] = builtins.list(self[owner].names())
        return result

    # Display helpers
    def __repr__(self) -> str:
        owners = ", ".join(self.owners())
        return f"DatasetRegistry(owners=[{owners}])"

    def _repr_html_(self):
        owners = builtins.list(self.owners())
        if not owners:
            return "<div><b>Datasets</b><br><i>No datasets found.</i></div>"
        rows = []
        mapping = self.list()
        for owner in owners:
            ds_names = ", ".join(mapping.get(owner, [])) or "<i>none</i>"
            rows.append(f"<tr><td><b>{owner}</b></td><td>{ds_names}</td></tr>")
        return """
        <div style='font-family: monospace; border-left: 3px solid #2196F3; padding-left: 10px;'>
            <b>Datasets</b>
            <table style='margin-top: 8px; border-collapse: collapse;'>
                <thead><tr><th style='padding:4px 8px; text-align:left;'>Owner</th><th style='padding:4px 8px; text-align:left;'>Datasets</th></tr></thead>
                <tbody>
                    {rows}
                </tbody>
            </table>
        </div>
        """.format(rows="".join(rows))

    def __getitem__(self, owner: str):
        """Get datasets published by a specific owner."""
        if owner not in self._owner_cache:
            self._owner_cache[owner] = _DatasetOwnerView(
                owner=owner,
                base_dir=self._base_dir,
                backend=self._backend,
                mapping_store=self._mapping_store,
            )
        return self._owner_cache[owner]

    def load_from_url(self, public_url: str) -> Dataset:
        """Load a dataset manifest directly from its syft:// public URL."""
        owner, path = _syft_url_to_owner_and_path(public_url, self._base_dir)
        return Dataset(
            owner=owner,
            manifest_path=path,
            base_dir=self._base_dir,
            mapping_store=self._mapping_store,
        )

    def find_by_twin_id(self, owner: Optional[str], twin_id: str) -> Optional[Twin]:
        """
        Locate and materialize a Twin by its twin_id (asset UUID) and owner.

        Args:
            owner: Owner email if known; if None, search all owners.
            twin_id: Asset UUID used as Twin ID
        """
        owners_to_search = [owner] if owner else builtins.list(self.owners())
        for o in owners_to_search:
            view = self[o]
            twin = view.get_asset_by_id(twin_id)
            if twin:
                return twin
        return None


class _DatasetOwnerView:
    """Helper to access datasets for a single owner."""

    def __init__(
        self,
        owner: str,
        base_dir: Path,
        backend=None,
        mapping_store: Optional[MappingStore] = None,
    ):
        self.owner = owner
        self._base_dir = Path(base_dir)
        self._backend = backend
        self._index_path = (
            self._base_dir / "datasites" / owner / "public" / "biovault" / "datasets.yaml"
        )
        self._index_cache: Optional[dict[str, _DatasetResource]] = None
        self._dataset_cache: dict[str, Dataset] = {}
        self._mapping_store = mapping_store

    def names(self) -> Iterable[str]:
        """List dataset names for this owner."""
        return builtins.list(self._load_index().keys())

    def __getitem__(self, dataset_name: str):
        if dataset_name not in self._dataset_cache:
            resources = self._load_index()
            if dataset_name not in resources:
                raise KeyError(f"Dataset '{dataset_name}' not found for {self.owner}")
            res = resources[dataset_name]
            dataset_path = res.path
            self._dataset_cache[dataset_name] = Dataset(
                owner=self.owner,
                manifest_path=dataset_path,
                base_dir=self._base_dir,
                mapping_store=self._mapping_store,
            )
        return self._dataset_cache[dataset_name]

    def get_asset_by_id(self, asset_id: str) -> Optional[Twin]:
        """Try each dataset to find an asset by its UUID."""
        for dataset in self._dataset_cache.values():
            twin = dataset.get_asset_by_id(asset_id)
            if twin:
                return twin
        # If not cached, scan manifest paths from index
        for name in self.names():
            if name in self._dataset_cache:
                continue
            ds = self[name]
            twin = ds.get_asset_by_id(asset_id)
            if twin:
                return twin
        return None

    def _load_index(self) -> dict[str, _DatasetResource]:
        if self._index_cache is not None:
            return self._index_cache

        resources: dict[str, _DatasetResource] = {}
        if not self._index_path.exists():
            self._index_cache = resources
            return resources

        try:
            raw = self._index_path.read_text()
            data = yaml.safe_load(raw) or {}
        except Exception:
            self._index_cache = resources
            return resources

        base = self._index_path.parent
        for entry in data.get("resources", []):
            name = entry.get("name")
            path_str = entry.get("path")
            if not name or not path_str:
                continue
            path = Path(path_str)
            if not path.is_absolute():
                path = (base / path).resolve()
            resources[name] = _DatasetResource(
                name=name,
                path=path,
                schema=entry.get("schema"),
                version=entry.get("version"),
            )

        self._index_cache = resources
        return resources

    def __repr__(self) -> str:
        names = ", ".join(self.names())
        return f"Datasets(owner={self.owner}, datasets=[{names}])"

    def _repr_html_(self):
        resources = self._load_index()
        rows = []
        for name, res in resources.items():
            meta = []
            if res.schema:
                meta.append(res.schema)
            if res.version:
                meta.append(f"v{res.version}")
            meta_str = " • ".join(meta) if meta else ""
            rows.append(
                f"<tr><td><b>{name}</b></td><td>{meta_str}</td><td><code>{res.path}</code></td></tr>"
            )
        if not rows:
            rows_html = "<tr><td colspan='3'><i>No datasets</i></td></tr>"
        else:
            rows_html = "".join(rows)
        return f"""
        <div style='font-family: monospace; border-left: 3px solid #4CAF50; padding-left: 10px;'>
            <b>Datasets for {self.owner}</b>
            <table style='margin-top: 8px; border-collapse: collapse;'>
                <thead><tr><th style='padding:4px 8px; text-align:left;'>Name</th><th style='padding:4px 8px; text-align:left;'>Details</th><th style='padding:4px 8px; text-align:left;'>Manifest</th></tr></thead>
                <tbody>
                    {rows_html}
                </tbody>
            </table>
        </div>
        """


class Dataset:
    """Represents a single dataset manifest and its assets."""

    def __init__(
        self,
        owner: str,
        manifest_path: Path,
        base_dir: Path,
        mapping_store: Optional[MappingStore] = None,
    ):
        self.owner = owner
        self.manifest_path = Path(manifest_path)
        self._base_dir = Path(base_dir)
        self._manifest = self._load_manifest(self.manifest_path)
        self._asset_cache: dict[str, Twin] = {}
        self._asset_ids: dict[str, str] = {}
        self._mapping_store = mapping_store or MappingStore.from_env(allow_missing=True)
        self._private_mappings = self._load_private_mappings()
        for key, asset in (self._manifest.get("assets") or {}).items():
            asset_id = (asset or {}).get("id")
            if asset_id:
                self._asset_ids[asset_id] = key

    @property
    def name(self) -> str:
        return self._manifest.get("name", "(unnamed)")

    def assets(self) -> Iterable[str]:
        return builtins.list(self._manifest.get("assets", {}).keys())

    def __getitem__(self, asset_key: str) -> Twin:
        return self._get_asset(asset_key)

    def __getattr__(self, name: str) -> Twin:
        assets = self._manifest.get("assets", {})
        if name in assets:
            return self._get_asset(name)
        raise AttributeError(name)

    def get_asset_by_id(self, asset_id: str) -> Optional[Twin]:
        """Return Twin by asset UUID if present in this dataset."""
        asset_key = self._asset_ids.get(asset_id)
        if not asset_key:
            return None
        return self._get_asset(asset_key)

    def _load_manifest(self, path: Path) -> dict:
        if not path.exists():
            raise FileNotFoundError(f"Dataset manifest not found at {path}")
        raw = path.read_text()
        data = yaml.safe_load(raw) or {}
        return data

    def _load_private_mappings(self) -> dict[str, Path]:
        """
        Load private URL -> local path mappings from BIOVAULT_HOME/mapping.yaml.
        Format:
            syft://.../private/...: /absolute/or/relative/path
        """
        mappings: dict[str, Path] = {}
        if not self._mapping_store:
            return mappings
        for k, v in self._mapping_store.all().items():
            p = Path(v)
            if not p.is_absolute():
                p = (self._mapping_store.path.parent / p).resolve()
            mappings[k] = p
        return mappings

    def _resolve_template(self, value: str, asset: dict) -> str:
        """Resolve simple template placeholders in URLs."""
        manifest_public = self._manifest.get("public_url", "")
        manifest_private = self._manifest.get("private_url", "")
        resolved_url = asset.get("_resolved_url") or asset.get("url", "")
        resolved = value
        resolved = resolved.replace("{root.public_url}", manifest_public or "")
        resolved = resolved.replace("{root.private_url}", manifest_private or "")
        resolved = resolved.replace("{url}", resolved_url)
        return resolved

    def _to_local_path(self, url_or_path: str) -> Path:
        # Strip fragment if present
        path_part = url_or_path.split("#", 1)[0]
        if path_part.startswith("syft://"):
            owner, path = _syft_url_to_owner_and_path(path_part, self._base_dir)
            return path
        candidate = Path(path_part)
        if candidate.is_absolute():
            return candidate
        return (self.manifest_path.parent / candidate).resolve()

    def _load_file(self, path: Path):
        """
        Build a TrustedLoader-style descriptor for the file so resolution
        uses the same approval flow as Beaver serialization.
        """
        suffix = path.suffix.lower()

        def loader_src(body: str) -> dict:
            path_str = path_to_syft_url(path) or str(path)
            return {
                "_trusted_loader": True,
                "name": f"dataset_loader{suffix}",
                "path": path_str,
                "deserializer_src": body,
            }

        if suffix in {".h5ad", ".h5"}:
            src = "def load(path):\n    import anndata as ad\n    return ad.read_h5ad(path)\n"
            return loader_src(src)

        if suffix in {".csv", ".tsv"}:
            sep = "\\t" if suffix == ".tsv" else ","
            src = (
                "def load(path):\n"
                "    import pandas as pd\n"
                f"    return pd.read_csv(path, sep='{sep}')\n"
            )
            return loader_src(src)

        if suffix == ".json":
            src = "def load(path):\n    import json\n    with open(path) as f:\n        return json.load(f)\n"
            return loader_src(src)

        # Fallback: raw bytes loader
        src = "def load(path):\n    with open(path, 'rb') as f:\n        return f.read()\n"
        return loader_src(src)

    def _resolve_mock_path(self, asset: dict) -> Optional[Path]:
        mock_val = asset.get("mock")
        if isinstance(mock_val, dict):
            # YAML mapping like {mock: "..."}
            mock_val = mock_val.get("mock")
        if not mock_val:
            return None
        resolved = self._resolve_template(str(mock_val), asset)
        try:
            return self._to_local_path(resolved)
        except Exception:
            return None

    def _resolve_private_path(self, asset: dict) -> Optional[Path]:
        # First, try the canonical mapping key: {private_url}#assets.{asset_key}
        # This is the standard format for mapping private dataset assets
        asset_key = asset.get("_asset_key")  # Set by _resolve_asset
        canonical_key = None
        if asset_key:
            # Get private_url: prefer explicit, else derive from public_url
            private_url = self._manifest.get("private_url")
            if not private_url:
                public_url = self._manifest.get("public_url", "")
                if public_url:
                    private_url = public_url.replace("/public/", "/private/")

            if private_url:
                canonical_key = f"{private_url}#assets.{asset_key}"
                if canonical_key in self._private_mappings:
                    return self._private_mappings[canonical_key]

        # Fallback: try the asset's explicit private field
        private_val = asset.get("private")
        if isinstance(private_val, dict):
            private_val = private_val.get("private")
        if not private_val:
            # No private field and no mapping found - show what we tried
            if canonical_key:
                print(f"⚠️  No private mapping found for: {canonical_key}")
                print(f"   ({len(self._private_mappings)} entries in mapping.yaml)")
            return None
        resolved = self._resolve_template(str(private_val), asset)
        # Mapping override: BIOVAULT_HOME/mapping.yaml (private URL -> local path)
        if resolved in self._private_mappings:
            return self._private_mappings[resolved]
        try:
            return self._to_local_path(resolved)
        except Exception:
            # Show what we tried when resolution fails
            print("⚠️  Could not resolve private path:")
            if canonical_key:
                print(f"   Tried mapping key: {canonical_key}")
            print(f"   Tried explicit field: {resolved}")
            return None

    def _format_size(self, size_bytes: int) -> str:
        units = ["B", "KB", "MB", "GB"]
        size = float(size_bytes)
        unit = "B"
        for u in units:
            unit = u
            if size < 1024 or u == units[-1]:
                break
            size /= 1024.0
        return f"{size:.1f}{unit}"

    def _mock_info(self, asset: dict) -> str:
        path = self._resolve_mock_path(asset)
        if path is None:
            return "—"
        if not path.exists():
            return f"missing ({path.name})"
        try:
            size_str = self._format_size(path.stat().st_size)
        except Exception:
            size_str = "?"
        ext = path.suffix.lstrip(".") or "file"
        return f"{ext} • {size_str}"

    def _resolve_asset(self, asset_key: str) -> Twin:
        assets = self._manifest.get("assets", {})
        if asset_key not in assets:
            raise KeyError(f"Asset '{asset_key}' not found in dataset '{self.name}'")

        asset = assets[asset_key] or {}
        asset_id = asset.get("id") or uuid4().hex
        asset["_asset_id"] = asset_id
        asset["_asset_key"] = asset_key  # For mapping key construction

        # Pre-resolve {url} placeholder for downstream templating
        if "url" in asset:
            asset["_resolved_url"] = self._resolve_template(str(asset["url"]), asset)

        public_value = None
        private_value = None

        mock_path = self._resolve_mock_path(asset)
        if mock_path and mock_path.exists():
            public_value = self._load_file(mock_path)
            if public_value is None:
                public_value = mock_path  # fallback to path
        elif mock_path:
            print(f"⚠️  Mock asset not found at {mock_path}")

        private_path = self._resolve_private_path(asset)
        if private_path and private_path.exists():
            private_value = self._load_file(private_path)
            if private_value is None:
                private_value = private_path

        if public_value is None and private_value is None:
            raise FileNotFoundError(
                f"No asset data found for '{asset_key}' in dataset '{self.name}'"
            )

        twin_owner = self._manifest.get("author") or self.owner
        twin = Twin(
            public=public_value,
            private=private_value,
            owner=twin_owner,
            name=asset_key,
            twin_id=str(asset_id),
            syft_url=self._manifest.get("public_url"),
            dataset_asset=asset_key,
        )
        return twin

    def _get_asset(self, asset_key: str) -> Twin:
        if asset_key not in self._asset_cache:
            self._asset_cache[asset_key] = self._resolve_asset(asset_key)
        return self._asset_cache[asset_key]

    def __repr__(self) -> str:
        assets = ", ".join(self.assets())
        return f"Dataset(name={self.name}, assets=[{assets}])"

    def _repr_html_(self):
        meta = []
        if self._manifest.get("schema"):
            meta.append(self._manifest.get("schema"))
        if self._manifest.get("version"):
            meta.append(f"v{self._manifest.get('version')}")
        meta_str = " • ".join(meta)
        desc = self._manifest.get("description") or ""
        public_url = self._manifest.get("public_url", "")

        asset_rows = []
        for key, asset in (self._manifest.get("assets") or {}).items():
            kind = asset.get("kind") or asset.get("type") or "unknown"
            has_mock = "✓" if asset.get("mock") else "✗"
            has_priv = "✓" if asset.get("private") else "✗"
            mock_info = self._mock_info(asset)
            asset_rows.append(
                f"<tr>"
                f"<td><b>{key}</b></td>"
                f"<td>{kind}</td>"
                f"<td>{has_mock}</td>"
                f"<td>{mock_info}</td>"
                f"<td>{has_priv}</td>"
                f"</tr>"
            )
        if not asset_rows:
            asset_rows_html = "<tr><td colspan='5'><i>No assets</i></td></tr>"
        else:
            asset_rows_html = "".join(asset_rows)

        return f"""
        <div style='font-family: monospace; border-left: 3px solid #FF9800; padding-left: 10px;'>
            <b>Dataset:</b> {self.name}<br>
            <div style='color: #666; margin-top:2px;'>{meta_str}</div>
            <div style='margin-top:6px;'>{desc}</div>
            <div style='margin-top:6px; font-size: 12px; color: #777;'>
                public_url: <code>{public_url}</code>
            </div>
            <table style='margin-top:10px; border-collapse: collapse;'>
                <thead>
                    <tr>
                        <th style='padding:4px 8px; text-align:left;'>Asset</th>
                        <th style='padding:4px 8px; text-align:left;'>Type</th>
                        <th style='padding:4px 8px; text-align:left;'>Mock</th>
                        <th style='padding:4px 8px; text-align:left;'>Mock Info</th>
                        <th style='padding:4px 8px; text-align:left;'>Private</th>
                    </tr>
                </thead>
                <tbody>
                    {asset_rows_html}
                </tbody>
            </table>
        </div>
        """


# ---------------------------------------------------------------------------
# Convenience accessors (module-level) for interactive environments
# ---------------------------------------------------------------------------

_default_registry: Optional[DatasetRegistry] = None


def _get_default_registry() -> DatasetRegistry:
    """Lazy-create a registry using environment variables."""
    global _default_registry
    if _default_registry is not None:
        return _default_registry

    data_dir = os.environ.get("SYFTBOX_DATA_DIR") or os.environ.get("BIOVAULT_HOME")
    if not data_dir:
        raise ValueError(
            "Set SYFTBOX_DATA_DIR or BIOVAULT_HOME to discover datasets from datasites/"
        )
    _default_registry = DatasetRegistry(base_dir=Path(data_dir))
    return _default_registry


def list() -> dict[str, Iterable[str]]:
    """
    List datasets per owner using the default registry.

    Example:
        beaver.datasets.list()
    """
    return _get_default_registry().list()


def owners() -> Iterable[str]:
    """List owners that have datasets published."""
    return _get_default_registry().owners()


def get(owner: str, name: str) -> Dataset:
    """
    Fetch a dataset by owner/name using the default registry.

    Example:
        ds = beaver.datasets.get("client1@sandbox.local", "single_cell")
        twin = ds.sc_rnaseq
    """
    return _get_default_registry()[owner][name]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _syft_url_to_owner_and_path(url: str, base_dir: Path) -> tuple[str, Path]:
    try:
        owner, path = parse_syft_url(url)
        path = sanitize_syft_path(path)
    except Exception as exc:
        raise ValueError(f"Invalid syft URL: {url}") from exc
    return owner, base_dir / "datasites" / owner / path
