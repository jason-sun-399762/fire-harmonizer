"""Write Parquet with provenance (config hash, git commit, inputs) in the file's own metadata."""

from __future__ import annotations

import datetime as dt
import json
import subprocess
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from . import __version__
from .config import config_hash

META_KEY = b"firecal"


def git_commit(repo_dir: str | Path | None = None) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo_dir or Path(__file__).parent, stderr=subprocess.DEVNULL, text=True
        ).strip()
    except Exception:
        return None


def write_parquet(df: pd.DataFrame, path: str | Path, cfg: dict, dataset: str, inputs: list[str] | None = None,
                  extra: dict | None = None) -> dict:
    meta = {
        "dataset": dataset,
        "firecal_version": __version__,
        "config_hash": config_hash(cfg),
        "grid": cfg["grid"],
        "git_commit": git_commit(),
        "written_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "n_inputs": len(inputs or []),
        "inputs_sample": (inputs or [])[:20],
        **(extra or {}),
    }
    table = pa.Table.from_pandas(df, preserve_index=False)
    existing = table.schema.metadata or {}
    table = table.replace_schema_metadata({**existing, META_KEY: json.dumps(meta, default=str).encode()})
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path, compression="zstd")
    return meta


def read_meta(path: str | Path) -> dict:
    md = pq.read_schema(path).metadata or {}
    return json.loads(md[META_KEY]) if META_KEY in md else {}


def output_key(cfg: dict, grid_name: str, dataset: str, part: str) -> str:
    """R2 key: features/<version>/<grid>/<dataset>/<part>.parquet (never overwrite another version)."""
    return f"{cfg['r2']['out_prefix']}/{cfg['version']}/{grid_name}/{dataset}/{part}.parquet"
