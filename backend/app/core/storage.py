"""Storage abstraction for datasets, batch outputs and model artefacts.

Keys are POSIX-style relative paths (``org/<org_id>/datasets/<id>/v1.csv``).
Two backends:

* ``LocalStorage`` — files under ``BOTSHIELD_STORAGE_LOCAL_ROOT`` (development,
  single-node deployments with a persistent volume).
* ``S3Storage`` — any S3-compatible object store (AWS S3, MinIO, …) via boto3.

ML code needs real file paths (joblib / numpy), so ``local_dir(prefix)`` gives a
directory containing the objects under a prefix: for local storage that is the
directory itself; for S3 the objects are downloaded into a per-process cache.
"""

from __future__ import annotations

import io
import shutil
import threading
from abc import ABC, abstractmethod
from functools import lru_cache
from pathlib import Path, PurePosixPath
from typing import BinaryIO, Iterator

from app.core.config import get_settings


class StorageError(RuntimeError):
    pass


def _validate_key(key: str) -> str:
    key = key.replace("\\", "/").strip("/")
    if not key or ".." in PurePosixPath(key).parts or key.startswith("/"):
        raise StorageError("Invalid storage key")
    return key


class Storage(ABC):
    @abstractmethod
    def put_bytes(self, key: str, data: bytes, content_type: str = "application/octet-stream") -> None: ...

    @abstractmethod
    def put_file(self, key: str, path: Path, content_type: str = "application/octet-stream") -> None: ...

    @abstractmethod
    def get_bytes(self, key: str) -> bytes: ...

    @abstractmethod
    def open(self, key: str) -> BinaryIO: ...

    @abstractmethod
    def exists(self, key: str) -> bool: ...

    @abstractmethod
    def delete(self, key: str) -> None: ...

    @abstractmethod
    def delete_prefix(self, prefix: str) -> int: ...

    @abstractmethod
    def list(self, prefix: str) -> list[str]: ...

    @abstractmethod
    def size(self, key: str) -> int: ...

    @abstractmethod
    def local_dir(self, prefix: str) -> Path:
        """A local directory holding every object under ``prefix`` (read-only view)."""

    @abstractmethod
    def upload_dir(self, local_dir: Path, prefix: str) -> list[str]:
        """Store every file of ``local_dir`` under ``prefix``; return the keys written."""

    @abstractmethod
    def describe(self) -> dict[str, str]: ...


class LocalStorage(Storage):
    def __init__(self, root: Path) -> None:
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        key = _validate_key(key)
        target = (self.root / key).resolve()
        if self.root != target and self.root not in target.parents:
            raise StorageError("Invalid storage key")
        return target

    def put_bytes(self, key: str, data: bytes, content_type: str = "application/octet-stream") -> None:
        p = self._path(key)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(p.suffix + ".tmp")
        tmp.write_bytes(data)
        tmp.replace(p)

    def put_file(self, key: str, path: Path, content_type: str = "application/octet-stream") -> None:
        p = self._path(key)
        p.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, p)

    def get_bytes(self, key: str) -> bytes:
        p = self._path(key)
        if not p.exists():
            raise StorageError(f"Object not found: {key}")
        return p.read_bytes()

    def open(self, key: str) -> BinaryIO:
        p = self._path(key)
        if not p.exists():
            raise StorageError(f"Object not found: {key}")
        return p.open("rb")

    def exists(self, key: str) -> bool:
        return self._path(key).exists()

    def delete(self, key: str) -> None:
        p = self._path(key)
        if p.exists():
            p.unlink()

    def delete_prefix(self, prefix: str) -> int:
        p = self._path(prefix)
        if not p.exists():
            return 0
        count = sum(1 for _ in p.rglob("*") if _.is_file()) if p.is_dir() else 1
        if p.is_dir():
            shutil.rmtree(p, ignore_errors=True)
        else:
            p.unlink()
        return count

    def list(self, prefix: str) -> list[str]:
        p = self._path(prefix)
        if not p.exists():
            return []
        if p.is_file():
            return [_validate_key(prefix)]
        return sorted(str(f.relative_to(self.root)).replace("\\", "/") for f in p.rglob("*") if f.is_file())

    def size(self, key: str) -> int:
        return self._path(key).stat().st_size

    def local_dir(self, prefix: str) -> Path:
        return self._path(prefix)

    def upload_dir(self, local_dir: Path, prefix: str) -> list[str]:
        keys: list[str] = []
        for f in sorted(Path(local_dir).rglob("*")):
            if f.is_file():
                key = f"{_validate_key(prefix)}/{f.relative_to(local_dir).as_posix()}"
                self.put_file(key, f)
                keys.append(key)
        return keys

    def describe(self) -> dict[str, str]:
        return {"backend": "local", "root": str(self.root)}


class S3Storage(Storage):
    def __init__(self, bucket: str, prefix: str = "", endpoint_url: str | None = None, region: str | None = None, access_key: str | None = None, secret_key: str | None = None, cache_dir: Path | None = None) -> None:
        import boto3  # imported lazily: only needed when the backend is selected

        self.bucket = bucket
        self.prefix = prefix.strip("/")
        self.cache_dir = Path(cache_dir or get_settings().storage_cache_dir).resolve()
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._client = boto3.client(
            "s3",
            endpoint_url=endpoint_url,
            region_name=region,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
        )

    def _k(self, key: str) -> str:
        key = _validate_key(key)
        return f"{self.prefix}/{key}" if self.prefix else key

    def put_bytes(self, key: str, data: bytes, content_type: str = "application/octet-stream") -> None:
        self._client.put_object(Bucket=self.bucket, Key=self._k(key), Body=data, ContentType=content_type)

    def put_file(self, key: str, path: Path, content_type: str = "application/octet-stream") -> None:
        self._client.upload_file(str(path), self.bucket, self._k(key), ExtraArgs={"ContentType": content_type})

    def get_bytes(self, key: str) -> bytes:
        try:
            return self._client.get_object(Bucket=self.bucket, Key=self._k(key))["Body"].read()
        except self._client.exceptions.NoSuchKey as exc:
            raise StorageError(f"Object not found: {key}") from exc

    def open(self, key: str) -> BinaryIO:
        return io.BytesIO(self.get_bytes(key))

    def exists(self, key: str) -> bool:
        try:
            self._client.head_object(Bucket=self.bucket, Key=self._k(key))
            return True
        except Exception:  # noqa: BLE001 - botocore ClientError 404
            return False

    def delete(self, key: str) -> None:
        self._client.delete_object(Bucket=self.bucket, Key=self._k(key))

    def delete_prefix(self, prefix: str) -> int:
        keys = self.list(prefix)
        for i in range(0, len(keys), 1000):
            chunk = keys[i : i + 1000]
            self._client.delete_objects(Bucket=self.bucket, Delete={"Objects": [{"Key": self._k(k)} for k in chunk]})
        return len(keys)

    def list(self, prefix: str) -> list[str]:
        full = self._k(prefix)
        out: list[str] = []
        paginator = self._client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=self.bucket, Prefix=full):
            for obj in page.get("Contents", []) or []:
                k = obj["Key"]
                out.append(k[len(self.prefix) + 1 :] if self.prefix and k.startswith(self.prefix + "/") else k)
        return sorted(out)

    def size(self, key: str) -> int:
        return int(self._client.head_object(Bucket=self.bucket, Key=self._k(key))["ContentLength"])

    def local_dir(self, prefix: str) -> Path:
        prefix = _validate_key(prefix)
        target = self.cache_dir / prefix
        with self._lock:
            keys = self.list(prefix)
            for k in keys:
                rel = k[len(prefix) + 1 :] if k.startswith(prefix + "/") else PurePosixPath(k).name
                dest = target / rel
                if dest.exists():
                    continue
                dest.parent.mkdir(parents=True, exist_ok=True)
                self._client.download_file(self.bucket, self._k(k), str(dest))
        return target

    def upload_dir(self, local_dir: Path, prefix: str) -> list[str]:
        keys: list[str] = []
        for f in sorted(Path(local_dir).rglob("*")):
            if f.is_file():
                key = f"{_validate_key(prefix)}/{f.relative_to(local_dir).as_posix()}"
                self.put_file(key, f)
                keys.append(key)
        return keys

    def describe(self) -> dict[str, str]:
        return {"backend": "s3", "bucket": self.bucket, "prefix": self.prefix}


@lru_cache(maxsize=1)
def get_storage() -> Storage:
    s = get_settings()
    if s.storage_backend == "s3":
        return S3Storage(
            bucket=s.s3_bucket or "",
            prefix=s.s3_prefix,
            endpoint_url=s.s3_endpoint_url,
            region=s.s3_region,
            access_key=s.s3_access_key_id,
            secret_key=s.s3_secret_access_key,
            cache_dir=s.storage_cache_dir,
        )
    return LocalStorage(s.storage_local_root)


def iter_chunks(data: bytes, size: int = 1 << 20) -> Iterator[bytes]:
    for i in range(0, len(data), size):
        yield data[i : i + size]
