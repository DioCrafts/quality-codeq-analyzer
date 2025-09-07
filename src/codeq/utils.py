"""
Utility functions and classes for CodeQ

Common utilities for file handling, path operations, caching, and other
helper functions used throughout the codebase.
"""

import os
import hashlib
import json
import pickle
from typing import Dict, List, Any, Optional, Union, Set
from pathlib import Path
import time
from functools import wraps
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass


class FileUtils:
    """Utilities for file operations."""

    @staticmethod
    def read_file_content(file_path: Union[str, Path], encoding: str = 'utf-8') -> Optional[str]:
        """Read file content with fallback encoding."""
        file_path = Path(file_path)

        if not file_path.exists():
            return None

        try:
            with open(file_path, 'r', encoding=encoding) as f:
                return f.read()
        except UnicodeDecodeError:
            # Try with different encoding
            try:
                with open(file_path, 'r', encoding='latin-1') as f:
                    return f.read()
            except UnicodeDecodeError:
                return None
        except Exception:
            return None

    @staticmethod
    def get_file_hash(file_path: Union[str, Path]) -> Optional[str]:
        """Calculate MD5 hash of file content."""
        content = FileUtils.read_file_content(file_path)
        if content is None:
            return None

        return hashlib.md5(content.encode('utf-8')).hexdigest()

    @staticmethod
    def is_text_file(file_path: Union[str, Path]) -> bool:
        """Check if file is a text file."""
        file_path = Path(file_path)

        # Check extension
        text_extensions = {
            '.py', '.js', '.ts', '.tsx', '.jsx', '.java', '.cpp', '.c', '.h',
            '.hpp', '.cs', '.php', '.rb', '.go', '.rs', '.swift', '.kt',
            '.scala', '.clj', '.hs', '.ml', '.fs', '.elm', '.dart', '.lua',
            '.r', '.m', '.pl', '.tcl', '.sh', '.bash', '.zsh', '.fish',
            '.ps1', '.bat', '.cmd', '.sql', '.xml', '.json', '.yaml', '.yml',
            '.toml', '.ini', '.cfg', '.conf', '.md', '.rst', '.txt'
        }

        if file_path.suffix.lower() in text_extensions:
            return True

        # Check file content (first 1024 bytes)
        try:
            with open(file_path, 'rb') as f:
                data = f.read(1024)
                # Check if it's valid UTF-8
                data.decode('utf-8')
                return True
        except (UnicodeDecodeError, OSError):
            return False

    @staticmethod
    def find_files_by_extension(
        directory: Union[str, Path],
        extensions: List[str],
        exclude_dirs: Optional[List[str]] = None
    ) -> List[Path]:
        """Find all files with given extensions in directory."""
        directory = Path(directory)
        exclude_dirs = exclude_dirs or ['.git', '__pycache__', 'node_modules', '.venv']

        files = []
        for ext in extensions:
            pattern = f"**/*{ext}"
            for file_path in directory.glob(pattern):
                if file_path.is_file():
                    # Check if path contains excluded directories
                    if not any(excl in str(file_path) for excl in exclude_dirs):
                        files.append(file_path)

        return files

    @staticmethod
    def get_file_size_mb(file_path: Union[str, Path]) -> float:
        """Get file size in MB."""
        file_path = Path(file_path)
        if not file_path.exists():
            return 0.0

        size_bytes = file_path.stat().st_size
        return size_bytes / (1024 * 1024)


class PathUtils:
    """Utilities for path operations."""

    @staticmethod
    def normalize_path(path: Union[str, Path]) -> str:
        """Normalize path to use forward slashes and resolve relative components."""
        return str(Path(path).resolve())

    @staticmethod
    def get_relative_path(path: Union[str, Path], base: Union[str, Path]) -> str:
        """Get relative path from base directory."""
        try:
            return str(Path(path).relative_to(Path(base)))
        except ValueError:
            return str(path)

    @staticmethod
    def is_subdirectory(path: Union[str, Path], parent: Union[str, Path]) -> bool:
        """Check if path is subdirectory of parent."""
        try:
            Path(path).relative_to(Path(parent))
            return True
        except ValueError:
            return False

    @staticmethod
    def get_common_prefix(paths: List[Union[str, Path]]) -> str:
        """Get common prefix path for a list of paths."""
        if not paths:
            return ""

        path_objects = [Path(p) for p in paths]
        common_parts = []

        # Get the shortest path as reference
        min_length = min(len(p.parts) for p in path_objects)
        reference_path = path_objects[0]

        for i in range(min_length):
            part = reference_path.parts[i]
            if all(p.parts[i] == part for p in path_objects):
                common_parts.append(part)
            else:
                break

        if common_parts:
            return str(Path(*common_parts))
        return ""


@dataclass
class CacheEntry:
    """Cache entry with metadata."""
    key: str
    data: Any
    timestamp: float
    ttl: Optional[float] = None  # Time to live in seconds

    def is_expired(self) -> bool:
        """Check if cache entry is expired."""
        if self.ttl is None:
            return False
        return time.time() - self.timestamp > self.ttl


class Cache:
    """Simple file-based cache with TTL support."""

    def __init__(self, cache_dir: Union[str, Path], default_ttl: Optional[float] = None):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.default_ttl = default_ttl

    def _get_cache_path(self, key: str) -> Path:
        """Get cache file path for key."""
        # Use hash of key to create filename
        key_hash = hashlib.md5(key.encode()).hexdigest()
        return self.cache_dir / f"{key_hash}.cache"

    def get(self, key: str) -> Optional[Any]:
        """Get cached data."""
        cache_path = self._get_cache_path(key)

        if not cache_path.exists():
            return None

        try:
            with open(cache_path, 'rb') as f:
                entry = pickle.load(f)

            if entry.is_expired():
                cache_path.unlink()  # Remove expired entry
                return None

            return entry.data
        except Exception:
            return None

    def set(self, key: str, data: Any, ttl: Optional[float] = None) -> None:
        """Set cached data."""
        ttl = ttl or self.default_ttl
        entry = CacheEntry(key, data, time.time(), ttl)

        cache_path = self._get_cache_path(key)
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(entry, f)
        except Exception:
            pass  # Silently fail if caching fails

    def clear(self) -> None:
        """Clear all cache entries."""
        for cache_file in self.cache_dir.glob("*.cache"):
            try:
                cache_file.unlink()
            except Exception:
                pass

    def cleanup_expired(self) -> int:
        """Remove expired cache entries. Returns number of removed entries."""
        removed = 0
        for cache_file in self.cache_dir.glob("*.cache"):
            try:
                with open(cache_file, 'rb') as f:
                    entry = pickle.load(f)

                if entry.is_expired():
                    cache_file.unlink()
                    removed += 1
            except Exception:
                # Remove corrupted cache files
                cache_file.unlink()
                removed += 1

        return removed


class ProgressTracker:
    """Track progress of long-running operations."""

    def __init__(self, total: int, description: str = ""):
        self.total = total
        self.current = 0
        self.description = description
        self.start_time = time.time()
        self.lock = threading.Lock()

    def update(self, increment: int = 1) -> None:
        """Update progress."""
        with self.lock:
            self.current += increment

    def get_progress(self) -> tuple[int, float, float]:
        """Get current progress, percentage, and ETA."""
        with self.lock:
            if self.total == 0:
                return 0, 0.0, 0.0

            percentage = (self.current / self.total) * 100

            elapsed = time.time() - self.start_time
            if self.current > 0:
                eta = elapsed * (self.total - self.current) / self.current
            else:
                eta = 0.0

            return self.current, percentage, eta

    def print_progress(self) -> None:
        """Print current progress."""
        current, percentage, eta = self.get_progress()
        eta_str = f"{eta:.1f}s" if eta < 60 else f"{eta/60:.1f}m"

        print(f"\r{self.description}: {current}/{self.total} ({percentage:.1f}%) ETA: {eta_str}", end="", flush=True)

        if current >= self.total:
            total_time = time.time() - self.start_time
            print(f"\nCompleted in {total_time:.2f}s")


def parallel_process(items: List[Any], process_func, max_workers: int = 4, description: str = ""):
    """
    Process items in parallel with progress tracking.

    Args:
        items: List of items to process
        process_func: Function to process each item
        max_workers: Maximum number of worker threads
        description: Progress description

    Returns:
        List of results in original order
    """
    if not items:
        return []

    progress = ProgressTracker(len(items), description)
    results = [None] * len(items)  # Pre-allocate to maintain order

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_index = {
            executor.submit(process_func, item): i
            for i, item in enumerate(items)
        }

        # Process completed tasks
        for future in as_completed(future_to_index):
            index = future_to_index[future]
            try:
                result = future.result()
                results[index] = result
            except Exception as e:
                print(f"Error processing item {index}: {e}")
                results[index] = None

            progress.update()
            if description:
                progress.print_progress()

    return results


def timing_decorator(func):
    """Decorator to measure function execution time."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()

        print(f"{func.__name__} took {end_time - start_time:.4f} seconds")
        return result

    return wrapper


class ConfigLoader:
    """Load configuration from various formats."""

    @staticmethod
    def load_yaml(file_path: Union[str, Path]) -> Optional[Dict[str, Any]]:
        """Load YAML configuration file."""
        try:
            import yaml
            with open(file_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except ImportError:
            print("PyYAML not installed. Install with: pip install PyYAML")
            return None
        except Exception as e:
            print(f"Error loading YAML config: {e}")
            return None

    @staticmethod
    def load_json(file_path: Union[str, Path]) -> Optional[Dict[str, Any]]:
        """Load JSON configuration file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading JSON config: {e}")
            return None

    @staticmethod
    def load_toml(file_path: Union[str, Path]) -> Optional[Dict[str, Any]]:
        """Load TOML configuration file."""
        try:
            import tomllib  # Python 3.11+
            with open(file_path, 'rb') as f:
                return tomllib.load(f)
        except ImportError:
            try:
                import tomli  # Fallback for older Python
                with open(file_path, 'rb') as f:
                    return tomli.load(f)
            except ImportError:
                print("TOML support not available. Install with: pip install tomli")
                return None
        except Exception as e:
            print(f"Error loading TOML config: {e}")
            return None


class StatsCollector:
    """Collect and calculate statistics."""

    @staticmethod
    def calculate_percentiles(data: List[float], percentiles: List[float]) -> Dict[float, float]:
        """Calculate percentiles for a dataset."""
        if not data:
            return {p: 0.0 for p in percentiles}

        sorted_data = sorted(data)
        n = len(sorted_data)

        results = {}
        for p in percentiles:
            if p <= 0:
                results[p] = sorted_data[0]
            elif p >= 100:
                results[p] = sorted_data[-1]
            else:
                # Linear interpolation
                k = (n - 1) * (p / 100)
                f = int(k)
                c = k - f

                if f + 1 < n:
                    results[p] = sorted_data[f] + c * (sorted_data[f + 1] - sorted_data[f])
                else:
                    results[p] = sorted_data[f]

        return results

    @staticmethod
    def calculate_histogram(data: List[float], bins: int = 10) -> Dict[str, Any]:
        """Calculate histogram for a dataset."""
        if not data:
            return {"bins": [], "counts": [], "edges": []}

        min_val = min(data)
        max_val = max(data)

        if min_val == max_val:
            return {
                "bins": [min_val],
                "counts": [len(data)],
                "edges": [min_val, max_val]
            }

        bin_width = (max_val - min_val) / bins
        edges = [min_val + i * bin_width for i in range(bins + 1)]

        counts = [0] * bins
        for value in data:
            bin_index = min(int((value - min_val) / bin_width), bins - 1)
            counts[bin_index] += 1

        bin_centers = [(edges[i] + edges[i + 1]) / 2 for i in range(bins)]

        return {
            "bins": bin_centers,
            "counts": counts,
            "edges": edges
        }
