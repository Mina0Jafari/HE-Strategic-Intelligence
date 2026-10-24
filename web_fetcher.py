"""Refresh a small, approved list of public HE evidence sources.

Run: python web_fetcher.py
Then: python document_retriever.py --rebuild

This is intentionally not a general-purpose web crawler. Review each approved
URL and its publication's terms before adding it to external_sources.csv.
"""
from __future__ import annotations

import csv
import hashlib
import re
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

import requests

ROOT = Path(__file__).resolve().parent
CONFIG = ROOT / "external_sources.csv"
INVENTORY = ROOT / "external_source_inventory.csv"
DOWNLOAD_DIR = ROOT / "data" / "external"
ALLOWED_DOMAINS = ("officeforstudents.org.uk", "hesa.ac.uk", "uel.ac.uk")
MIME_EXTENSIONS = {
    "application/pdf": ".pdf",
    "text/csv": ".csv",
    "application/csv": ".csv",
    "application/vnd.ms-excel": ".xls",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": ".xlsx",
    "text/html": ".html",
    "application/xhtml+xml": ".html",
    "text/plain": ".txt",
}
SUPPORTED = {".pdf", ".csv", ".xls", ".xlsx", ".html", ".txt"}
MAX_BYTES = 20 * 1024 * 1024


def is_approved(url: str) -> bool:
    parsed = urlparse(url)
    hostname = (parsed.hostname or "").lower()
    return (parsed.scheme == "https" and not parsed.username and
            (hostname in ALLOWED_DOMAINS or
             any(hostname.endswith("." + d) for d in ALLOWED_DOMAINS)))


def _extension(response: requests.Response) -> str:
    mimetype = response.headers.get("Content-Type", "").split(";", 1)[0].strip().lower()
    if mimetype in MIME_EXTENSIONS:
        return MIME_EXTENSIONS[mimetype]
    suffix = Path(urlparse(response.url).path).suffix.lower()
    if suffix in SUPPORTED:
        return suffix
    raise ValueError(f"Unsupported file type: {mimetype or suffix}")


def _download(url: str, redirects_remaining: int = 5) -> tuple[bytes, str, str]:
    # Check redirect destinations as well as the initial URL.
    if not is_approved(url):
        raise ValueError("URL domain is not approved")
    with requests.get(url, timeout=(10, 45), stream=True,
                      allow_redirects=False,
                      headers={"User-Agent": "HE-Strategic-AI-Research/1.0"}) as response:
        if 300 <= response.status_code < 400:
            if redirects_remaining <= 0:
                raise ValueError("Too many redirects")
            location = response.headers.get("Location", "")
            from urllib.parse import urljoin
            destination = urljoin(url, location)
            if not is_approved(destination):
                raise ValueError("Redirect to non-approved domain")
            return _download(destination, redirects_remaining - 1)
        response.raise_for_status()
        extension = _extension(response)
        content = bytearray()
        for chunk in response.iter_content(chunk_size=65536):
            content.extend(chunk)
            if len(content) > MAX_BYTES:
                raise ValueError(f"File exceeds {MAX_BYTES // (1024*1024)} MB limit")
        if not content:
            raise ValueError("Empty response")
        return bytes(content), extension, response.url


def refresh_sources() -> list[dict]:
    if not CONFIG.exists():
        raise FileNotFoundError(f"Missing {CONFIG.name}; use the supplied example")
    DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
    with CONFIG.open(newline="", encoding="utf-8-sig") as file:
        entries = list(csv.DictReader(file))

    inventory = []
    for entry in entries:
        source_id = entry.get("source_id", "").strip()
        url = entry.get("url", "").strip()
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,60}", source_id):
            raise ValueError(f"Invalid source_id: {source_id!r}")
        result = {"source_id": source_id, "url": url,
                  "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
                  "status": "", "local_file": "", "sha256": "", "error": ""}
        try:
            contents, extension, final_url = _download(url)
            path = DOWNLOAD_DIR / f"{source_id}{extension}"
            sha256 = hashlib.sha256(contents).hexdigest()
            old_hash = hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None
            # Clean up a previous format of the same source.
            for old in DOWNLOAD_DIR.glob(f"{source_id}.*"):
                if old != path and old.suffix.lower() in SUPPORTED:
                    old.unlink()
            if old_hash != sha256:
                path.write_bytes(contents)
            result.update(status="unchanged" if old_hash == sha256 else "downloaded",
                          local_file=str(path.relative_to(ROOT)), sha256=sha256, url=final_url)
        except (requests.RequestException, ValueError, OSError) as exc:
            result.update(status="error", error=str(exc)[:300])
        inventory.append(result)
        print(f"{source_id}: {result['status']} {result['error']}")

    with INVENTORY.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(inventory[0]) if inventory else
                                ["source_id", "url", "retrieved_at_utc", "status", "local_file", "sha256", "error"])
        writer.writeheader()
        writer.writerows(inventory)
    return inventory


if __name__ == "__main__":
    refresh_sources()
