# Postflight Plasma IGFBP2 and Spaceflight-Associated Cancellous Bone Loss

Reproducible secondary analysis of two public NASA Open Science Data Repository
(OSDR) studies supporting the manuscript *Postflight Plasma IGFBP2 as a
Candidate Indicator of Spaceflight-Associated Cancellous Bone Loss*.

- **OSD-977** — rat femur microCT (cancellous BV/TV, cortical thickness),
  STS-57 (male) and STS-62 (ovariectomized female), Baseline / Ground
  Control / Space Flight.
- **OSD-342** — mouse plasma SomaScan proteomics (~5,283 targets), 33-day
  ISS mission, Space Flight vs. Ground Control (myostatin/activin
  intervention arms excluded from the comparison below).

## Versioned manuscript snapshot

The analysis state supporting the submission-ready manuscript is tagged
[`v1.0.0`](https://github.com/lxfr0/igfbp2-spaceflight-biomarker/tree/v1.0.0)
(23 October 2026). The tag fixes the code version; source data remain available
from NASA OSDR under OSD-977 and OSD-342 and are not redistributed here.

## Setup

```
pip install -r requirements.txt
```

## Get the data

Data is not committed to this repo. Download it first:

```
python src/download_osdr.py 977 data/raw/OSD-977
python src/download_osdr.py 342 data/raw/OSD-342
```

This hits the public OSDR files API (`osdr.nasa.gov/osdr/data/osd/files/<id>`)
and pulls down every listed file for the study, extracting any zip archives.

## Run the analysis

```
python run_pipeline.py
python src/export_figures_gsr.py
```

Runs both pipelines and writes everything to `results/`:

- `results/tables/osd977_group_stats.csv` — per-group mean/sd + one-way ANOVA
- `results/tables/osd977_pairwise_effectsizes.csv` — Welch t-tests, Hedges' g,
  Welch-Satterthwaite-aligned 95% intervals, and BH-FDR correction
- `results/tables/osd342_bone_marker_stats.csv` — 39-marker bone-remodeling
  panel screen, Flight vs. Ground Control, with Hedges' g, 95% CI, and
  Benjamini-Hochberg FDR correction
- `results/figures/osd977_bvtv_by_mission.png` — cancellous BV/TV by group/mission
- `results/figures/osd342_<top_gene>_boxplot.png` — top FDR hit, per-mouse values

Each pipeline can also be run/imported standalone:
`python src/osd977_pipeline.py`, `python src/osd342_pipeline.py`.

## Headline result

Cancellous bone volume fraction (BV/TV) drops significantly after
spaceflight in both missions, but only reaches significance vs. **baseline**
(not just ground control) in the STS-62 ovariectomized females — consistent
with the source paper's "net bone loss" conclusion for that cohort. Cortical
thickness is unaffected in either mission, also matching the paper.

Screening OSD-342's plasma proteome against a 39-gene bone-remodeling marker
panel (`src/bone_markers.py`), only **IGFBP2** survives FDR correction
(q≈0.006, ~36% lower in flight vs. ground control). IGFBP2 has independent
mechanistic literature support: it drives both osteoblast and osteoclast
differentiation, and restoring it pharmacologically prevents disuse-induced
bone loss in rats. This motivates a prospective hypothesis: does suppression
of circulating IGFBP2 track cancellous bone loss when plasma and bone outcomes
are measured in the same animals?

## Honest limitations

- OSD-977 (rat) and OSD-342 (mouse) are **different species, different
  missions, no matched animals** — this is a cross-study hypothesis-
  generating comparison, not a validated biomarker.
- OSD-342's untreated flight/ground arms are n=8 per group; the bone-marker
  screen tests 39 markers, so the single FDR survivor should be treated as a
  lead to validate, not a proven finding.
- No public animal-matched plasma IGFBP2 and bone micro-CT records were
  identified for OSD-977. Recovery of any unarchived matched records would
  require confirmation from the original investigators.

## Project layout

```
data/
  raw/OSD-977/       downloaded ISA metadata + microCT xlsx/csv
  raw/OSD-342/       downloaded ISA metadata + SomaScan xlsx + GPL annotation
  processed/         merged_OSD-977.csv (sample metadata + measurements)
src/
  download_osdr.py   generic OSDR file-list + download helper
  bone_markers.py     shared candidate bone-remodeling gene panel
  effect_size.py       shared Welch's t-test + Hedges' g + 95% CI helper
  osd977_pipeline.py  ISA parsing, merge, group stats + effect sizes, boxplot
  osd342_pipeline.py  SomaScan parsing, marker screen + effect sizes, FDR, boxplot
results/
  tables/            CSV outputs (stats)
  figures/           PNG outputs (plots)
run_pipeline.py       runs both pipelines end-to-end
requirements.txt
```
