from __future__ import annotations

import io
import json
import tarfile
import urllib.request
from pathlib import Path, PurePosixPath

from .papers import Paper, get_paper


def _download_bytes(url: str) -> bytes:
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "hefty-tm/0.1 (+https://arxiv.org/)"},
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        return response.read()


def _should_extract(name: str, include_full_source: bool) -> bool:
    parts = [part for part in PurePosixPath(name).parts if part not in ("", ".")]
    if not parts:
        return False
    clean_name = PurePosixPath(*parts)
    top_name = clean_name.name
    if clean_name.parts[0] == "anc":
        return True
    if top_name in {"00README.json", "README.txt"}:
        return True
    if include_full_source and str(clean_name).endswith(".tex"):
        return True
    if include_full_source and str(clean_name).endswith(".pdf"):
        return True
    return False


def safe_extract_member_path(output_dir: Path, member_name: str) -> Path:
    parts = [part for part in PurePosixPath(member_name).parts if part not in ("", ".")]
    if not parts or any(part == ".." for part in parts):
        raise ValueError(f"Unsafe archive member path: {member_name}")

    target = (output_dir / Path(*parts)).resolve()
    output_root = output_dir.resolve()
    if not target.is_relative_to(output_root):
        raise ValueError(f"Archive member escapes output directory: {member_name}")
    return target


def fetch_paper(
    paper: Paper,
    output_root: Path,
    *,
    include_full_source: bool = False,
) -> dict[str, object]:
    output_dir = output_root / paper.arxiv_id
    output_dir.mkdir(parents=True, exist_ok=True)

    payload = _download_bytes(paper.source_url)
    extracted: list[str] = []

    with tarfile.open(fileobj=io.BytesIO(payload), mode="r:gz") as archive:
        for member in archive.getmembers():
            if not member.isfile():
                continue
            if not _should_extract(member.name, include_full_source):
                continue

            target = safe_extract_member_path(output_dir, member.name)
            target.parent.mkdir(parents=True, exist_ok=True)

            source = archive.extractfile(member)
            if source is None:
                continue
            target.write_bytes(source.read())
            extracted.append(str(target.relative_to(output_root)))

    manifest = {
        "key": paper.key,
        "arxiv_id": paper.arxiv_id,
        "posted": paper.posted,
        "title": paper.title,
        "role": paper.role,
        "source_url": paper.source_url,
        "has_ancillary_tables": paper.has_ancillary_tables,
        "include_full_source": include_full_source,
        "files": sorted(extracted),
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def fetch_many(
    keys: list[str] | tuple[str, ...],
    output_root: Path,
    *,
    include_full_source: bool = False,
) -> list[dict[str, object]]:
    results = []
    for key in keys:
        results.append(
            fetch_paper(
                get_paper(key),
                output_root,
                include_full_source=include_full_source,
            )
        )
    return results
