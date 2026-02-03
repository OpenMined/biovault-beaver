"""Sample datasets for beaver - download and use example data easily."""

from __future__ import annotations

import json
import os
import sys
import tempfile
import types
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import yaml

try:
    import requests

    _HAS_REQUESTS = True
except ImportError:
    _HAS_REQUESTS = False

_ENV_CACHE_DIR = os.environ.get("BEAVER_SAMPLE_DATA_DIR")
DEFAULT_CACHE_DIR = (
    Path(_ENV_CACHE_DIR).expanduser()
    if _ENV_CACHE_DIR
    else Path.home() / ".biovault" / "cache" / "beaver" / "sample-data"
)

BASE_URL = "https://raw.githubusercontent.com/OpenMined/biovault-data/main"


@dataclass
class DatasetPart:
    """A single part of a dataset (e.g., mock or real)."""

    name: str
    kind: str  # "mock" or "real"
    yaml_url: str
    description: str
    size_bytes: int
    compressed_size_bytes: int
    file_format: str
    _manifest: Optional[dict] = None

    def __repr__(self) -> str:
        size_mb = self.size_bytes / (1024 * 1024)
        compressed_mb = self.compressed_size_bytes / (1024 * 1024)
        return (
            f"DatasetPart({self.kind})\n"
            f"  Description: {self.description}\n"
            f"  Format: {self.file_format}\n"
            f"  Size: {size_mb:.1f} MB (compressed: {compressed_mb:.1f} MB)\n"
            f"  URL: {self.yaml_url}\n"
            f"  \n"
            f"  .download(path=None) -> Path  # Download to path or cache\n"
            f"  .info()                       # Show detailed manifest info"
        )

    def _fetch_manifest(self) -> dict:
        """Fetch and cache the YAML manifest."""
        if self._manifest is not None:
            return self._manifest

        cached_path = DEFAULT_CACHE_DIR / self.name / self._guess_original_filename()
        cached_manifest = self._load_cached_manifest(cached_path)
        url = _convert_github_url(self.yaml_url)
        try:
            self._manifest = _fetch_with_retry(url, is_yaml=True)
            return self._manifest
        except Exception:
            if cached_manifest is not None:
                self._manifest = cached_manifest
                return self._manifest
            raise

    def _guess_original_filename(self) -> str:
        url_path = urllib.parse.urlparse(self.yaml_url).path
        name = Path(urllib.parse.unquote(url_path)).name
        if name.endswith(".yaml"):
            return name[:-5]
        if name.endswith(".yml"):
            return name[:-4]
        return name

    @staticmethod
    def _manifest_cache_path(output_path: Path) -> Path:
        return output_path.with_name(f"{output_path.name}.manifest.json")

    def _load_cached_manifest(self, output_path: Path) -> Optional[dict]:
        manifest_path = self._manifest_cache_path(output_path)
        if not manifest_path.exists():
            return None
        try:
            return json.loads(manifest_path.read_text())
        except Exception:
            return None

    def _write_cached_manifest(self, output_path: Path, manifest: dict) -> None:
        manifest_path = self._manifest_cache_path(output_path)
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))

    def info(self) -> None:
        """Print detailed manifest information."""
        manifest = self._fetch_manifest()
        print(f"📦 {self.name} ({self.kind})")
        print(f"   File: {manifest['file']['original']}")
        print(f"   Size: {manifest['file']['original_size_bytes']:,} bytes")
        print(f"   Compressed: {manifest['file']['compressed_size_bytes']:,} bytes")
        print(f"   Shards: {len(manifest['file']['shards'])}")
        print(f"   Compression: {manifest['compression']}")
        print(f"   Generated: {manifest.get('generated_at', 'unknown')}")

    def download(self, path: Optional[Path] = None, verbose: bool = True) -> Path:
        """
        Download this dataset part.

        Args:
            path: Destination path. If None, uses default cache directory.
            verbose: Print progress information.

        Returns:
            Path to the downloaded file.
        """
        cache_dir: Optional[Path] = None
        guess_filename = self._guess_original_filename()
        if path is None:
            cache_dir = DEFAULT_CACHE_DIR / self.name
            cache_dir.mkdir(parents=True, exist_ok=True)
            output_guess = cache_dir / guess_filename
        elif path.is_dir():
            output_guess = path / guess_filename
        else:
            output_guess = path

        output_guess.parent.mkdir(parents=True, exist_ok=True)

        try:
            manifest = self._fetch_manifest()
        except Exception:
            if output_guess.exists():
                cached_manifest = self._load_cached_manifest(output_guess)
                if cached_manifest is not None:
                    self._manifest = cached_manifest
                    if verbose:
                        print(f"⚠️  Failed to reach manifest; using cached data: {output_guess}")
                    try:
                        actual_hash = _compute_blake3(output_guess)
                        if actual_hash == cached_manifest["file"]["original_b3sum"]:
                            return output_guess
                        if verbose:
                            print("  Cached checksum mismatch, cannot use offline copy.")
                    except Exception:
                        if verbose:
                            print("  Could not verify cached checksum, cannot use offline copy.")
                else:
                    try:
                        if output_guess.stat().st_size == self.size_bytes:
                            if verbose:
                                print(
                                    "⚠️  Failed to reach manifest; using cached data: "
                                    f"{output_guess}"
                                )
                            return output_guess
                    except OSError:
                        pass
            raise

        original_filename = manifest["file"]["original"]
        if path is None:
            cache_dir = cache_dir or (DEFAULT_CACHE_DIR / self.name)
            cache_dir.mkdir(parents=True, exist_ok=True)
            output_path = cache_dir / original_filename
        elif path.is_dir():
            output_path = path / original_filename
        else:
            output_path = path

        output_path.parent.mkdir(parents=True, exist_ok=True)

        if output_path.exists():
            if verbose:
                print(f"✓ Already downloaded: {output_path}")
            try:
                actual_hash = _compute_blake3(output_path)
                if actual_hash == manifest["file"]["original_b3sum"]:
                    return output_path
                if verbose:
                    print("  Checksum mismatch, re-downloading...")
            except Exception:
                if verbose:
                    print("  Could not verify checksum, re-downloading...")
        elif output_guess.exists() and output_guess != output_path:
            if verbose:
                print(f"✓ Already downloaded: {output_guess}")
            try:
                actual_hash = _compute_blake3(output_guess)
                if actual_hash == manifest["file"]["original_b3sum"]:
                    try:
                        self._write_cached_manifest(output_guess, manifest)
                    except Exception:
                        if verbose:
                            print("  ⚠️  Failed to write manifest cache (continuing)")
                    return output_guess
                if verbose:
                    print("  Checksum mismatch, re-downloading...")
            except Exception:
                if verbose:
                    print("  Could not verify checksum, re-downloading...")

        # Download
        output_path = _download_from_manifest(manifest, self.yaml_url, output_path, verbose)
        try:
            self._write_cached_manifest(output_path, manifest)
        except Exception:
            if verbose:
                print("  ⚠️  Failed to write manifest cache (continuing)")
        return output_path


@dataclass
class SampleDataset:
    """A sample dataset with mock and real parts."""

    name: str
    description: str
    category: str
    mock: Optional[DatasetPart] = None
    real: Optional[DatasetPart] = None

    def __repr__(self) -> str:
        lines = [
            f"SampleDataset: {self.name}",
            f"  Category: {self.category}",
            f"  Description: {self.description}",
            "",
        ]

        if self.mock:
            mock_mb = self.mock.size_bytes / (1024 * 1024)
            lines.append(f"  .mock  - {mock_mb:.1f} MB - {self.mock.description}")

        if self.real:
            real_mb = self.real.size_bytes / (1024 * 1024)
            lines.append(f"  .real  - {real_mb:.1f} MB - {self.real.description}")

        lines.extend(
            [
                "",
                "Methods:",
                "  .download()      # Download both mock and real to cache",
                "  .mock.download() # Download only mock data",
                "  .real.download() # Download only real data",
                "  .twin()          # Get a Twin object with both parts loaded",
            ]
        )

        return "\n".join(lines)

    def download(self, path: Optional[Path] = None, verbose: bool = True) -> dict[str, Path]:
        """
        Download both mock and real parts.

        Args:
            path: Destination directory. If None, uses default cache.
            verbose: Print progress information.

        Returns:
            Dict with 'mock' and 'real' keys pointing to downloaded paths.
        """
        result = {}

        if self.mock:
            if verbose:
                print("\n📥 Downloading mock data...")
            result["mock"] = self.mock.download(path, verbose)

        if self.real:
            if verbose:
                print("\n📥 Downloading real data...")
            result["real"] = self.real.download(path, verbose)

        return result

    def twin(
        self,
        download: bool = True,
        path: Optional[Path] = None,
        owner: str = "sample_data",
    ) -> Any:
        """
        Get a Twin object with mock (public) and real (private) data loaded.

        Args:
            download: If True, automatically download data if not cached.
            path: Optional path to download to (uses cache if None).
            owner: Owner string for the Twin.

        Returns:
            Twin object with .public (mock) and .private (real) data.
        """
        from ..twin import Twin

        mock_path = None
        real_path = None

        if self.mock:
            if download:
                mock_path = self.mock.download(path, verbose=True)
            else:
                # Check cache
                cache_path = (
                    DEFAULT_CACHE_DIR / self.name / self.mock._fetch_manifest()["file"]["original"]
                )
                if cache_path.exists():
                    mock_path = cache_path
                else:
                    raise FileNotFoundError(
                        "Mock data not found. Run .twin(download=True) first or .mock.download()"
                    )

        if self.real:
            if download:
                real_path = self.real.download(path, verbose=True)
            else:
                cache_path = (
                    DEFAULT_CACHE_DIR / self.name / self.real._fetch_manifest()["file"]["original"]
                )
                if cache_path.exists():
                    real_path = cache_path
                else:
                    raise FileNotFoundError(
                        "Real data not found. Run .twin(download=True) first or .real.download()"
                    )

        # Load the data based on file format
        mock_data = _load_file(mock_path) if mock_path else None
        real_data = _load_file(real_path) if real_path else None

        return Twin(
            name=self.name,
            public=mock_data,
            private=real_data,
            owner=owner,
        )


def _load_file(path: Path) -> Any:
    """Load a file based on its extension."""
    suffix = path.suffix.lower()

    if suffix == ".h5ad":
        try:
            import anndata as ad

            return ad.read_h5ad(path)
        except ImportError as e:
            raise ImportError(
                "anndata is required to load .h5ad files. Install with: pip install anndata"
            ) from e
    elif suffix == ".csv":
        try:
            import pandas as pd

            return pd.read_csv(path)
        except ImportError as e:
            raise ImportError(
                "pandas is required to load .csv files. Install with: pip install pandas"
            ) from e
    elif suffix == ".parquet":
        try:
            import pandas as pd

            return pd.read_parquet(path)
        except ImportError as e:
            raise ImportError(
                "pandas and pyarrow are required to load .parquet files. "
                "Install with: pip install pandas pyarrow"
            ) from e
    else:
        # Return path for unknown formats
        return path


def _convert_github_url(url: str) -> str:
    """Convert GitHub blob URL to raw content URL."""
    if "github.com" in url and "/blob/" in url:
        return url.replace("github.com", "raw.githubusercontent.com").replace("/blob/", "/")
    return url


def _compute_blake3(file_path: Path) -> str:
    """Compute BLAKE3 hash of a file."""
    try:
        import blake3

        hasher = blake3.blake3()
        with open(file_path, "rb") as f:
            while chunk := f.read(8192):
                hasher.update(chunk)
        return hasher.hexdigest()
    except ImportError:
        import subprocess

        result = subprocess.run(["b3sum", str(file_path)], capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"b3sum failed: {result.stderr}") from None
        return result.stdout.split()[0]


def _fetch_with_retry(
    url: str, is_yaml: bool = False, max_retries: int = 3, base_timeout: int = 60
) -> Any:
    """Fetch URL content with exponential backoff retry."""
    import time

    last_error = None
    for attempt in range(max_retries):
        timeout = base_timeout * (attempt + 1)  # Increase timeout each attempt
        try:
            if _HAS_REQUESTS:
                response = requests.get(url, timeout=timeout)
                response.raise_for_status()
                if is_yaml:
                    return yaml.safe_load(response.text)
                return response.content
            else:
                with urllib.request.urlopen(url, timeout=timeout) as response:
                    content = response.read()
                    if is_yaml:
                        return yaml.safe_load(content.decode("utf-8"))
                    return content
        except Exception as e:
            last_error = e
            if attempt < max_retries - 1:
                wait_time = 2 ** (attempt + 1)  # 2, 4, 8 seconds
                print(f"  ⚠️  Attempt {attempt + 1} failed: {e}")
                print(f"  Retrying in {wait_time}s...")
                time.sleep(wait_time)
    raise last_error  # type: ignore[misc]


def _download_file(url: str, dest_path: Path, desc: str = "Downloading") -> None:
    """Download file with progress reporting (uses tqdm if available)."""
    try:
        from tqdm import tqdm

        _has_tqdm = True
    except ImportError:
        _has_tqdm = False

    if _HAS_REQUESTS:
        response = requests.get(url, stream=True, timeout=180)
        response.raise_for_status()
        total_size = int(response.headers.get("content-length", 0))

        if _has_tqdm:
            with (
                tqdm(
                    total=total_size,
                    unit="B",
                    unit_scale=True,
                    unit_divisor=1024,
                    desc=desc,
                ) as pbar,
                open(dest_path, "wb") as f,
            ):
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
        else:
            print(f"{desc}: {url}")
            downloaded = 0
            with open(dest_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total_size > 0:
                            pct = (downloaded / total_size) * 100
                            print(
                                f"\r  Progress: {downloaded:,} / {total_size:,} bytes ({pct:.1f}%)",
                                end="",
                            )
            if total_size > 0:
                print()
    else:
        # Fallback to urllib
        with urllib.request.urlopen(url, timeout=180) as response:
            total_size = int(response.headers.get("content-length", 0))

            if _has_tqdm:
                with (
                    tqdm(
                        total=total_size,
                        unit="B",
                        unit_scale=True,
                        unit_divisor=1024,
                        desc=desc,
                    ) as pbar,
                    open(dest_path, "wb") as f,
                ):
                    while True:
                        chunk = response.read(8192)
                        if not chunk:
                            break
                        f.write(chunk)
                        pbar.update(len(chunk))
            else:
                print(f"{desc}: {url}")
                downloaded = 0
                with open(dest_path, "wb") as f:
                    while True:
                        chunk = response.read(8192)
                        if not chunk:
                            break
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total_size > 0:
                            pct = (downloaded / total_size) * 100
                            print(
                                f"\r  Progress: {downloaded:,} / {total_size:,} bytes ({pct:.1f}%)",
                                end="",
                            )
                if total_size > 0:
                    print()


def _download_from_manifest(
    manifest: dict, yaml_url: str, output_path: Path, verbose: bool = True
) -> Path:
    """Download and assemble a file from its manifest."""
    try:
        import zstandard as zstd
    except ImportError as e:
        raise ImportError(
            "zstandard is required for downloading sample data. Install with: pip install zstandard"
        ) from e

    base_url = yaml_url.rsplit("/", 1)[0] + "/"
    base_url = _convert_github_url(base_url)

    if verbose:
        print(f"📦 File: {manifest['file']['original']}")
        print(f"   Size: {manifest['file']['original_size_bytes']:,} bytes")

    shards = manifest["file"].get("shards")

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        compressed_path = tmpdir_path / manifest["file"]["compressed"]

        if shards:
            # Sharded download
            if verbose:
                print(f"   Shards: {len(shards)}")
                print("\nStep 1: Downloading shards")
            shard_paths = []

            for i, shard in enumerate(shards):
                shard_name = shard["name"]
                shard_url = base_url + shard_name
                shard_path = tmpdir_path / shard_name

                if verbose:
                    _download_file(shard_url, shard_path, f"  Shard {i + 1}/{len(shards)}")
                else:
                    if _HAS_REQUESTS:
                        response = requests.get(shard_url, timeout=180)
                        response.raise_for_status()
                        shard_path.write_bytes(response.content)
                    else:
                        with urllib.request.urlopen(shard_url, timeout=180) as response:
                            shard_path.write_bytes(response.read())

                # Verify shard checksum
                actual_hash = _compute_blake3(shard_path)
                if actual_hash != shard["b3sum"]:
                    raise ValueError(f"Checksum mismatch for shard {shard_name}")
                if verbose:
                    print("  ✓ Checksum verified")

                shard_paths.append(shard_path)

            # Combine shards
            if verbose:
                print("\nStep 2: Combining shards")

            with open(compressed_path, "wb") as outfile:
                for shard_path in shard_paths:
                    with open(shard_path, "rb") as infile:
                        while chunk := infile.read(8192):
                            outfile.write(chunk)

            if verbose:
                print(f"  ✓ Combined into {compressed_path.name}")
        else:
            # Single file download
            if verbose:
                print("\nStep 1: Downloading compressed file")

            compressed_filename = urllib.parse.quote(manifest["file"]["compressed"])
            compressed_url = base_url + compressed_filename
            _download_file(compressed_url, compressed_path, "  Downloading")

        # Verify compressed file
        if verbose:
            print("\nStep 2: Verifying checksum" if not shards else "")
        actual_hash = _compute_blake3(compressed_path)
        if actual_hash != manifest["file"]["compressed_b3sum"]:
            raise ValueError("Checksum mismatch for compressed file")
        if verbose:
            print("  ✓ Checksum verified")

        # Decompress
        if verbose:
            print("\nStep 3: Decompressing" if shards else "\nStep 2: Decompressing")

        dctx = zstd.ZstdDecompressor()
        with open(compressed_path, "rb") as ifh, open(output_path, "wb") as ofh:
            dctx.copy_stream(ifh, ofh)

        if verbose:
            print(f"  ✓ Decompressed to {output_path}")

        # Verify final file
        actual_hash = _compute_blake3(output_path)
        if actual_hash != manifest["file"]["original_b3sum"]:
            raise ValueError("Checksum mismatch for final file")
        if verbose:
            print("  ✓ Checksum verified")

    if verbose:
        print(f"\n✅ Success! File written to:\n   {output_path}")

    return output_path


# =============================================================================
# Dataset Registry
# =============================================================================

single_cell = SampleDataset(
    name="single_cell",
    description="Single-cell RNA sequencing dataset (5% downsampled)",
    category="bioinformatics",
    mock=DatasetPart(
        name="single_cell",
        kind="mock",
        yaml_url=f"{BASE_URL}/single_cell/sample1/sc_RNAseq_adata_downsampled_to5percent.mock.5k.h5ad.yaml",
        description="Synthetic mock scRNA-seq data for development",
        size_bytes=131563411,
        compressed_size_bytes=34244379,
        file_format="h5ad",
    ),
    real=DatasetPart(
        name="single_cell",
        kind="real",
        yaml_url=f"{BASE_URL}/single_cell/sample1/sc_RNAseq_adata_downsampled_to5percent.private.h5ad.yaml",
        description="Real scRNA-seq data (private/sensitive)",
        size_bytes=461184953,
        compressed_size_bytes=157467028,
        file_format="h5ad",
    ),
)

ecg_arrhythmia = SampleDataset(
    name="ecg_arrhythmia",
    description="MIT-BIH Arrhythmia ECG dataset (Kaggle)",
    category="cardiology",
    real=DatasetPart(
        name="ecg_arrhythmia",
        kind="real",
        yaml_url=(
            f"{BASE_URL}/kaggle/ecg-arrhythmia-classification-dataset/"
            "MIT-BIH%20Arrhythmia%20Database.csv.yaml"
        ),
        description="MIT-BIH Arrhythmia Database (CSV)",
        size_bytes=45482621,
        compressed_size_bytes=15576743,
        file_format="csv",
    ),
)

_HELP_TEXT = """beaver.sample_data - Download and use example datasets

List available datasets:
    sample_data.list_datasets()

Get a dataset:
    ds = sample_data.single_cell
    ds = sample_data.ecg_arrhythmia
    ds = sample_data.get('single_cell')
    ds = sample_data.get('ecg_arrhythmia')

Download data:
    ds.mock.download()       # Download mock data only
    ds.real.download()       # Download real data only
    ds.download()            # Download both

Create a Twin:
    twin = ds.twin()         # Downloads & returns Twin(public=mock, private=real)

Inspect a dataset part:
    ds.mock.info()           # Show manifest details

Cache location: ~/.biovault/cache/beaver/sample-data/
"""


def list_datasets() -> list[str]:
    """List all available sample datasets."""
    return ["single_cell", "ecg_arrhythmia"]


def help() -> None:
    """Print usage help for sample_data module."""
    print(_HELP_TEXT)


def get(name: str) -> SampleDataset:
    """Get a sample dataset by name."""
    datasets = {
        "single_cell": single_cell,
        "ecg_arrhythmia": ecg_arrhythmia,
    }
    if name not in datasets:
        available = ", ".join(datasets.keys())
        raise ValueError(f"Unknown dataset '{name}'. Available: {available}")
    return datasets[name]


__all__ = [
    "single_cell",
    "ecg_arrhythmia",
    "list_datasets",
    "get",
    "help",
    "SampleDataset",
    "DatasetPart",
    "DEFAULT_CACHE_DIR",
]


class _SampleDataModule(types.ModuleType):
    def __repr__(self) -> str:
        return _HELP_TEXT


_module = _SampleDataModule(__name__)
_module.__dict__.update(globals())
sys.modules[__name__] = _module
