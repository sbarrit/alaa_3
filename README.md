# Comparing the Effect of Ranibizumab vs. Aflibercept on Visual Acuity Outcomes for Neovascular AMD

Code for: CPH200B Project — Advaith Veturi, Sami Barrit, Marcus Moen

## Overview

Causal inference analysis of anti-VEGF treatments for neovascular age-related macular degeneration, using the Moorfields Eye Hospital longitudinal cohort (Fu et al., *JAMA Ophthalmology*, 2021).

This repository implements a **fuzzy regression discontinuity design** exploiting the October 2013 introduction of aflibercept at MEH as a natural experiment, with the treatment era as an instrumental variable for drug assignment.

## Requirements

- Python 3.9
- Dependencies: `pip install -r requirements.txt`

## Data

The analysis uses the MEH AMD Survival Outcomes Database, publicly available on Dryad:

> Fu DJ et al. (2020). Insights from survival analyses during 12 years of anti-VEGF therapy for neovascular age-related macular degeneration. Dryad. https://doi.org/10.5061/dryad.nvx0k6drg

Download `MEH_AMD_survivaloutcomes_database.csv` and place it at `input/MEH_AMD_survivaloutcomes_database.csv`.

## Reproduce

```bash
# 1. Preprocess raw data
python3 code/scripts/run_preprocess.py

# 2. Run fuzzy RDD analysis
python3 code/scripts/run_rdd.py
```

Results are written to `output/tables/` (summary CSVs) and `output/results/` (full datasets, not tracked in git).

The technical report is at [`output/RDD.md`](output/RDD.md).

## Repository Structure

```
alaa_3/
├── code/
│   ├── scripts/
│   │   ├── run_preprocess.py   # Data cleaning and curation
│   │   └── run_rdd.py          # Fuzzy RDD / IV analysis
│   └── plot/                   # (plotting scripts)
├── output/
│   ├── RDD.md                  # Technical report
│   └── tables/                 # Summary CSVs
├── requirements.txt
└── README.md
```

## License

MIT
