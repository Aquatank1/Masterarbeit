# Predicting Wheat yield on county-level using Deep Learning

Code of the master thesis of Frederick Wagner, Department of Computer Science, Chair of Computer Science VIII,
University of Würzburg (submitted 30 September 2026).

The thesis predicts the wheat yield of US counties from Harmonized Landsat and Sentinel-2 (HLS) satellite
observations and the yield history of each county. Satellite data are extracted over the wheat pixels of the
USDA Cropland Data Layer on Google Earth Engine, aggregated to county time series of 20-day windows and fed to
a small Transformer. The headline model reaches an RMSE of 0.51 t/ha, an MAE of 0.39 t/ha and an R² of 0.87
on the 282 counties of the 2025 test season (the historical-yield baseline: 0.77 t/ha RMSE).

The thesis is the reference for methods, experiments and interpretation. This repository shows how the data
were produced and how every number was computed.

## What is included and what is not

| included | not included |
|---|---|
| the Earth Engine export scripts and the preprocessing pipeline | the raw satellite exports (`Data_*` folders, up to about 20 GB per season) |
| the yield labels (`Ertragsdaten_USA.csv`) | the SoilGrids rasters and the soil cell files |
| the processed modelling tables (`Claude/data_cache/`, about 40 MB) | the US state boundaries (US Census, download link below) |
| the notebooks of the headline model and of every experiment, with outputs | the Earth Engine account, Cloud Storage bucket and service account key |
| the cache of every trained model's predictions and all result tables | earlier development stages of the code |
| a script that retrains the 30 headline models and checks them | the weights of the trained models (about 110 MB) |
| all figures of the thesis | |
| the county boundaries (GADM 3.6) | |

With the included tables and cache the models and the experiment notebook run without any raw data. The raw
data can only be rebuilt with your own Earth Engine account (see [DATA.md](DATA.md)).

## Repository layout

```
.
├── README.md                     this file
├── DATA.md                       structure of every data folder and file, raw and processed
├── AGENT_RULES.md                the rules the AI coding agents worked under
├── requirements.txt
├── Ertragsdaten_USA.csv          county, state and national wheat yields 2007-2025 (USDA NASS)
├── GEE-On_server_HLSL.py         Earth Engine export, Landsat part of HLS   -> Data_<year>/
├── GEE-On_server_HLSS.py         Earth Engine export, Sentinel-2 part of HLS -> Data_<year>/
├── preprocessing.ipynb           Data_<year> -> Data_Fused_<year> -> Data_Windowed_<year>_{10d,20d}
├── soil_download.py              SoilGrids rasters per county and property -> soil_data_others/ (collected in Data_Asoil_data/)
├── Building_soil_years.ipynb     soil values of every 1 km wheat cell -> Data_Astatic_wheat_soil_cells/
├── fixingsoil.py                 soil values at a single point, read directly from the SoilGrids rasters online, to patch gaps
├── analyze_missing_soil_cells.ipynb  finds the cells whose soil values are missing
├── integrating_sat_soil_20d.ipynb    the per-cell dataset (county year x window x cell) of the earlier cell-level experiments
├── Data_Ashape/                  US county boundaries (GADM 3.6); the US state boundaries go here too
└── Claude/
    ├── data_cache/               the modelling tables built by 00 (parquet)
    └── final_results_claude/
        ├── README.md             details of the modelling code, run order and caveats
        ├── 00_preprocessing.ipynb     windows + yields -> modelling tables and the 2025 test set
        ├── 01_headline_register.ipynb the headline model (30 models) and its in-season forecast
        ├── 03_track_10d.ipynb         only its first three code cells: the 10d modelling table
        ├── 08_experiments.ipynb       every experiment of chapters 7 and 8 of the thesis
        ├── build_soil_selection.py    the 18 soil features of the soil experiments
        ├── headline_weights.py        retrains the 30 headline models, checks and evaluates them
        ├── history_input_probe.py     side question: the model with a manipulated history input
        ├── oracle_ensemble.py         side question: the best model subset chosen on the test set
        ├── redraw_satellite_ranking_gci.py  redraws one figure with the index name used in the thesis
        ├── cache/                     predictions of every trained model and all result tables
        ├── figures/                   every figure of 08 (the thesis figures)
        └── figs/                      the figures of 01
```

The folder `Claude/` holds the modelling code. It was written with the coding agent Claude Code under the
direction of the author, as stated in the declaration of the thesis. Its name is kept because the notebooks
find their data through it.

## Pipeline

1. **Acquisition (Earth Engine).** `GEE-On_server_HLS{L,S}.py` run a scout pass per county and season (which
   1 km cells contain more than 11 wheat pixels of the CDL) and then export, for every usable scene, the
   statistics of the clear wheat pixels of every cell: one CSV per county, scene and tile.
2. **Fusing and windowing.** `preprocessing.ipynb` merges the scene files of a county and season into one
   file, harmonizes the Sentinel-2 band names to the Landsat names, and aggregates the season (7 April to
   24 August) into seven 20-day or fourteen 10-day windows weighted by wheat pixel count. An empty window
   between two observed ones is filled from its neighbours; an edge gap or two adjacent gaps drop the county year.
3. **Soil.** `soil_download.py` downloads the SoilGrids properties per county; `Building_soil_years.ipynb`
   samples them on the wheat cells of every season. Soil is not part of the headline model.
4. **Modelling tables.** `00_preprocessing.ipynb` joins the windows with the yields and the leak-free
   five-year yield history and writes `Claude/data_cache/`. `build_soil_selection.py` builds the 18 soil features.
5. **Models.** `01_headline_register.ipynb` trains the headline ensemble; `08_experiments.ipynb` runs every
   experiment of the thesis: window length, ensemble size, network size, prediction target, random forest,
   MLP and combinations, leave-one-year-out, input ablations and feature selection.

## Quick start

```
pip install -r requirements.txt
python Claude/final_results_claude/headline_weights.py
```

The weights of the trained models are not part of the repository (about 110 MB for the 30 headline models
alone). `headline_weights.py` retrains the 30 seeded headline models with the code of `08` (about seven
minutes on a GPU), saves their weights in `cache/e07_weights_width256/`, checks every model against its
cached prediction and prints

```
10d test set (282 counties): RMSE 0.510 t/ha, MAE 0.395 t/ha, R2 0.867
20d test set (426 counties): RMSE 0.556 t/ha, MAE 0.437 t/ha, R2 0.831
```

A second run only loads the saved weights. These 30 models are the seeded ensemble of the headline
configuration that serves as the reference in the experiment tables of the thesis and as the 2025 fold of
the leave-one-year-out evaluation; in the original run they reproduced their cached predictions exactly. The
headline ensemble of `01` is unseeded and cannot be recreated exactly; it reaches 0.508 / 0.395 / 0.868.

`08_experiments.ipynb` executed top to bottom with the included cache reproduces every number and every
figure of chapter 8 except the maps in about a minute, and `01_headline_register.ipynb` its results in
seconds (both tested on a fresh copy of this repository: nothing is retrained and every result table is
reproduced exactly). Drawing the maps is commented out, because it needs the US state
boundaries of the US Census Bureau, which are not part of the repository; the numbers behind the maps are
still computed. The maps are in `Claude/final_results_claude/figures/` (the files with `map` in their name).
To redraw them, download https://www2.census.gov/geo/tiger/GENZ2023/shp/cb_2023_us_state_20m.zip into
`Data_Ashape/` and follow the note at the top of the map geometry cell. A file or folder
deleted from `cache/` is retrained; from an empty cache the notebook trains about 3,300 networks and needs
the raw data for some experiments (see `Claude/final_results_claude/README.md`).

## Main results (t/ha)

| | RMSE | MAE | R² |
|---|---|---|---|
| Headline, 2025, 282 counties (10d test set) | 0.51 | 0.39 | 0.87 |
| Headline, 2025, 426 counties (20d test set) | 0.56 | 0.44 | 0.83 |
| Historical-yield baseline, 282 counties | 0.77 | 0.62 | 0.70 |
| Leave-one-year-out, mean over nine seasons, 10d requirements | 0.56 | 0.44 | 0.77 |
| Random forest, 2025, 282 counties | 0.61 | 0.48 | 0.81 |
| MLP, 2025, 282 counties | 0.59 | 0.46 | 0.82 |

## Reproducibility

- Python 3.13, PyTorch 2.11 with CUDA 12.8; the other versions are in `requirements.txt`.
- On the Blackwell GPU used here the flash and memory-efficient attention kernels are disabled; the notebooks
  set this at the top.
- `01` is unseeded. `08` seeds every model, but GPU training is not bit-for-bit deterministic, so a retrain
  reproduces the results within the training variability (independent 30-model ensembles differ by at most
  about 0.005 t/ha RMSE), not to the last digit.
- The Earth Engine scripts need your own Earth Engine project (replace the placeholder `project-master` in
  `ee.Initialize` and `ASSET_PATH`), your own Cloud Storage bucket for the scout files (replace
  `example-wheat2020`), the path to your own service account key (line 7; never commit the key) and the
  county boundaries uploaded as the asset `projects/<your project>/assets/USshape` with the county key in the
  property `id_comb`. The county list at the end of each script is the last batch that was exported.
- Comments and printed messages in the notebooks were written while the code evolved. Where a comment and the
  code disagree, the code is right; where the code and the thesis disagree in wording, the thesis is right.
  The thesis calls the index GCVI of the code GCI.

## Data sources

- Yields: USDA National Agricultural Statistics Service county estimates, provided by GreenSpin.
- Satellite: NASA Harmonized Landsat and Sentinel-2 (HLSL30 and HLSS30 v2.0), accessed through Google Earth Engine.
- Wheat mask: USDA NASS Cropland Data Layer.
- Soil: SoilGrids 2.0, ISRIC.
- Boundaries: GADM 3.6 (counties) and US Census Bureau cartographic boundary files (states).
