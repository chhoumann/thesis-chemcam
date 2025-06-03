# ChemCam Oxide Prediction Thesis

This repository contains all material for our master's thesis on machine learning models for NASA's **ChemCam** instrument. The project replicates the _Multivariate Oxide Composition_ (MOC) baseline used by the ChemCam team and explores improvements through modern regression techniques and ensemble learning.

## Domain Overview

ChemCam is a laser‑induced breakdown spectroscopy (LIBS) instrument onboard the _Curiosity_ rover. When the instrument fires its laser at Martian rocks, the resulting plasma emits light that is captured as spectra. Predicting the concentration of major oxides from these spectra forms a challenging multivariate regression task due to the high dimensionality and complex physical processes involved.

Our work focuses on

- Reproducing the published MOC pipeline combining Partial Least Squares Sub‑Models (PLS‑SM) and Independent Component Analysis (ICA).
- Investigating twelve additional machine learning models and data preprocessing techniques.
- Developing a cross‑validation strategy, outlier removal method, and an automated Optuna‑based optimisation framework.
- Contributing these methods back to the open source **PyHAT** toolset for hyperspectral analysis.

The results and discussion are documented in the `report_thesis` directory.

## Repository Structure

```
.
├── baseline/          # Python package with the MOC replication and experiments
├── p9-presentation/   # Quarto slides for the 9th semester project presentation
├── report_pre_thesis/ # Pre‑thesis technical report (LaTeX)
└── report_thesis/     # Final thesis (LaTeX)
```

### `baseline`

The `baseline` package contains the code used to reproduce the MOC model and run additional experiments. Important modules include:

- `PLS_SM/` – training and evaluating the PLS sub‑models. The `full_flow.py` script orchestrates preprocessing, cross‑validation and outlier removal.
- `ica/` – generation of ICA scores and regression models (`main.py`).
- `lib/` – utilities for data handling, normalization, plotting, and reproduction of the published configuration. Configuration is handled via environment variables defined in `lib/config.py`.
- `experiments/` – Jupyter notebooks exploring alternative models such as gradient boosting, neural networks and stacking ensembles.

A minimal setup to run the baseline is:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r baseline/requirements.txt
# create a .env file with the variables required by baseline/lib/config.py
python -m PLS_SM.full_flow train
```

The dataset used by the ChemCam team is not included. Paths to the spectral data and composition tables must be provided via the environment variables `DATA_PATH` and `COMPOSITION_DATA_PATH`.

### `p9-presentation`

Contains our Quarto based presentation. After installing [Quarto](https://quarto.org/), you can preview the slides with:

```bash
quarto preview
```

### `report_pre_thesis` and `report_thesis`

Both reports are written in LaTeX and can be built using [Tectonic](https://tectonic-typesetting.github.io/). Each folder contains a `Tectonic.toml` file specifying the build configuration. Running `tectonic` in the respective directory will produce a PDF in `build/`.

## Contribution

The codebase demonstrates how modern machine learning approaches can improve oxide prediction from ChemCam LIBS spectra. We provide a catalogue of model configurations, an automated optimisation pipeline, and integrate a new outlier detection method into PyHAT. The thesis text and presentation summarise the findings and outline directions for future work.

We hope this repository serves as a helpful starting point for further research on quantitative LIBS analysis on Mars and other planetary bodies.
