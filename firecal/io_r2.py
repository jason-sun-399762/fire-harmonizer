"""Cloudflare R2 access (S3-compatible). Keys come from the environment, never from the repo:
AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY (optional R2_ENDPOINT, R2_BUCKET override the config)."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pandas as pd


def client(cfg: dict):
    import boto3

    if not (os.environ.get("AWS_ACCESS_KEY_ID") and os.environ.get("AWS_SECRET_ACCESS_KEY")):
        raise RuntimeError("Set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY (the R2 token) in the environment.")
    from botocore.config import Config

    return boto3.client(
        "s3",
        region_name="auto",
        endpoint_url=os.environ.get("R2_ENDPOINT", cfg["r2"]["endpoint"]),
        config=Config(retries={"max_attempts": 10, "mode": "adaptive"}, max_pool_connections=64),
    )


def bucket(cfg: dict) -> str:
    return os.environ.get("R2_BUCKET", cfg["r2"]["bucket"])


def list_keys(cfg: dict, prefix: str, limit: int | None = None) -> list[dict]:
    s3 = client(cfg)
    out, token = [], None
    while True:
        kw = {"Bucket": bucket(cfg), "Prefix": prefix, "MaxKeys": 1000}
        if token:
            kw["ContinuationToken"] = token
        resp = s3.list_objects_v2(**kw)
        out += [{"key": o["Key"], "size": o["Size"]} for o in resp.get("Contents", [])]
        if limit and len(out) >= limit:
            return out[:limit]
        if not resp.get("IsTruncated"):
            return out
        token = resp["NextContinuationToken"]


def download(cfg: dict, key: str, dest: str | Path, retries: int = 3) -> Path:
    """Download and verify the local size equals the object's size; retry on a short read."""
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    s3 = client(cfg)
    expected = s3.head_object(Bucket=bucket(cfg), Key=key)["ContentLength"]
    for attempt in range(1, retries + 1):
        s3.download_file(bucket(cfg), key, str(dest))
        got = dest.stat().st_size
        if got == expected:
            return dest
        if attempt == retries:
            raise OSError(f"{key}: downloaded {got} of {expected} bytes after {retries} attempts")
    return dest


def upload(cfg: dict, path: str | Path, key: str) -> None:
    client(cfg).upload_file(str(path), bucket(cfg), key)


def read_parquet(cfg: dict, key: str, **kw) -> pd.DataFrame:
    with tempfile.TemporaryDirectory() as td:
        p = download(cfg, key, Path(td) / Path(key).name)
        return pd.read_parquet(p, **kw)
