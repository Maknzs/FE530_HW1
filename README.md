# FE 530 – Homework 1 Starter

This repo gives you a clean, reproducible setup to complete HW1.

## Quick start

```bash
# (optional) create env with conda
conda env create -f environment.yml
conda activate fe530-hw1

# or use venv + pip
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# pull data (SPY 2012–2022 monthly, SHY 2021–2022 monthly)
python src/data_download.py

# run the analyses (each script writes tables/figures to data/ and figures/)
python src/q1_binomial.py
python src/q2_pension.py
python src/q3_portfolio.py
python src/q4_forwards.py

# render the Quarto report
quarto render hw1.qmd
```

Folders:
- `src/` analysis scripts and utilities
- `data/` CSVs saved by scripts
- `figures/` charts saved by scripts
- `hw1.qmd` main report to submit as PDF

> Tip: use `Makefile` targets: `make data`, `make analysis`, `make report`.
