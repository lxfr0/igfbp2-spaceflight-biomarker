"""
Top-level entry point: runs both dataset pipelines end-to-end.

    python run_pipeline.py

Requires data already downloaded into data/raw/OSD-977 and data/raw/OSD-342
(see src/download_osdr.py or README.md).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import osd342_pipeline  # noqa: E402
import osd977_pipeline  # noqa: E402

if __name__ == "__main__":
    print("=" * 60)
    print("OSD-977: rat femur microCT structural analysis")
    print("=" * 60)
    osd977_pipeline.main()

    print()
    print("=" * 60)
    print("OSD-342: mouse plasma SomaScan bone-marker screen")
    print("=" * 60)
    osd342_pipeline.main()
