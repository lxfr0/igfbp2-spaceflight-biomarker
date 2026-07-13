# NASA OSDR Bone-Loss Biomarker Project

Reanalysis of two public NASA Open Science Data Repository (OSDR) studies to
generate a testable hypothesis about circulating biomarkers of
spaceflight-induced bone loss, for the BPS proposal.

- **OSD-977** — rat femur microCT (cancellous BV/TV, cortical thickness),
  STS-57 (male) and STS-62 (ovariectomized female), Baseline / Ground
  Control / Space Flight.
- **OSD-342** — mouse plasma SomaScan proteomics (~5,283 targets), 33-day
  ISS mission, Space Flight vs. Ground Control (myostatin/activin
  intervention arms excluded from the comparison below).

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
```

Runs both pipelines and writes everything to `results/`:

- `results/tables/osd977_group_stats.csv` — per-group mean/sd + one-way ANOVA
- `results/tables/osd977_pairwise_effectsizes.csv` — Welch t-tests, Flight vs.
  Ground/Baseline, with Cohen's d, 95% CI, and BH-FDR correction across all
  12 comparisons (2 missions x 2 metrics x 3 pairwise contrasts)
- `results/tables/osd342_bone_marker_stats.csv` — 39-marker bone-remodeling
  panel screen, Flight vs. Ground Control, with Cohen's d, 95% CI, and
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
bone loss in rats. This is the basis for the BPS proposal hypothesis: does
spaceflight suppression of circulating IGFBP2 track the severity/sex-
dependence of the cancellous bone loss seen in OSD-977?

## Honest limitations

- OSD-977 (rat) and OSD-342 (mouse) are **different species, different
  missions, no matched animals** — this is a cross-study hypothesis-
  generating comparison, not a validated biomarker.
- OSD-342's untreated flight/ground arms are n=8 per group; the bone-marker
  screen tests 39 markers, so the single FDR survivor should be treated as a
  lead to validate, not a proven finding.
- No rat-tissue omics exist from the original OSD-977 animals (samples were
  formalin-fixed for histology/microCT, which precludes RNA/protein
  recovery) — there is no way to get literally matched omics for OSD-977.

## Project layout

```
data/
  raw/OSD-977/       downloaded ISA metadata + microCT xlsx/csv
  raw/OSD-342/       downloaded ISA metadata + SomaScan xlsx + GPL annotation
  processed/         merged_OSD-977.csv (sample metadata + measurements)
src/
  download_osdr.py   generic OSDR file-list + download helper
  bone_markers.py     shared candidate bone-remodeling gene panel
  effect_size.py       shared Welch's t-test + Cohen's d + 95% CI helper
  osd977_pipeline.py  ISA parsing, merge, group stats + effect sizes, boxplot
  osd342_pipeline.py  SomaScan parsing, marker screen + effect sizes, FDR, boxplot
results/
  tables/            CSV outputs (stats)
  figures/           PNG outputs (plots)
run_pipeline.py       runs both pipelines end-to-end
requirements.txt
```
