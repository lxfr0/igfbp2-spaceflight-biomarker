"""
Download raw data/metadata files for a NASA OSDR study accession.

Usage:
    python src/download_osdr.py 977 data/raw/OSD-977
    python src/download_osdr.py 342 data/raw/OSD-342

Requires only `requests` (already used elsewhere in this project).
"""
import sys
import zipfile
from pathlib import Path
from urllib.parse import quote

import requests

FILES_API = "https://osdr.nasa.gov/osdr/data/osd/files/{osd_id}"
DOWNLOAD_BASE = "https://osdr.nasa.gov"


def list_files(osd_id: int) -> list[dict]:
    """Return the file-listing JSON for an OSD-### study from the OSDR files API."""
    resp = requests.get(FILES_API.format(osd_id=osd_id), timeout=60)
    resp.raise_for_status()
    payload = resp.json()
    # the API nests results under studies -> "OSD-###" -> study_files
    study_key = f"OSD-{osd_id}"
    return payload["studies"][study_key]["study_files"]


def download_all(osd_id: int, dest_dir: str, extract_zips: bool = True) -> None:
    """Download every listed file for a study into dest_dir, extracting any .zip archives."""
    dest = Path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)

    for f in list_files(osd_id):
        filename = f["file_name"]
        remote_url = f["remote_url"]
        if not remote_url.startswith("http"):
            remote_url = DOWNLOAD_BASE + remote_url
        out_path = dest / filename

        print(f"Downloading {filename} ({f['file_size']:,} bytes)...")
        r = requests.get(remote_url, timeout=300)
        r.raise_for_status()
        out_path.write_bytes(r.content)

        if extract_zips and filename.lower().endswith(".zip"):
            extract_to = dest / (filename[: -len(".zip")] + "_extracted")
            with zipfile.ZipFile(out_path) as zf:
                zf.extractall(extract_to)
            print(f"  extracted -> {extract_to}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(1)
    download_all(int(sys.argv[1]), sys.argv[2])
