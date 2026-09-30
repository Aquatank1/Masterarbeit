# final_results_claude

The code behind the results of the thesis: the modelling tables, the headline model and every experiment.
The thesis is the reference for methods and interpretation; this folder is how the numbers were produced.
The structure of every data file is described in [DATA.md](../../DATA.md).

## Read this first

- **Comments, docstrings, markdown cells and printed messages are not necessarily up to date.**
  They were written while the code evolved and were not proofread like the thesis. Where a comment
  and the code disagree, the code is right; where a comment and the thesis disagree, the thesis is right.
- **The numbers that count** are the executed outputs of `01_headline_register.ipynb` and
  `08_experiments.ipynb` and the CSV files they write into `cache/`.
- `01` reports dt/ha (headline RMSE 5.08 dt/ha); `08` reports **t/ha** (0.51 t/ha), rounded to two decimals.
- The index called GCVI in the code is called GCI in the thesis.

## Files

| file | what it does |
|---|---|
| `../../preprocessing.ipynb` (repository root) | per-scene exports `Data_<year>` → `Data_Fused_<year>` → county windows `Data_Windowed_<year>_{20d,10d}` |
| `00_preprocessing.ipynb` | county windows + yield labels → the modelling tables in `Claude/data_cache/` and the 2025 test set |
| `01_headline_register.ipynb` | the headline model (30 models) and its in-season forecast |
| `03_track_10d.ipynb` | only its first three code cells are needed: `Claude/data_cache/features_10d_leakfree.parquet`. The rest is an earlier stage of the 10d experiment and is superseded by `08` |
| `08_experiments.ipynb` | every other experiment of the thesis in the order of chapters 7 and 8, an appendix with experiments the thesis does not use, the figures and a summary table |
| `build_soil_selection.py` | the soil features of the soil variants: the 18 properties that are the first of their correlation group |
| `headline_weights.py` | retrains the 30 headline models (or loads them once saved), checks them against the cache and evaluates the ensemble on 2025 |
| `history_input_probe.py` | side question: what the network does when its history input is replaced |
| `oracle_ensemble.py` | side question: the best subset of models chosen on the test set (an upper bound, not a model) |
| `redraw_satellite_ranking_gci.py` | redraws the satellite ranking figure with the index name GCI used in the thesis |

## Run order

1. *(only when the raw windows change, slow, needs the raw data)* `preprocessing.ipynb` in the repository
   root. The season starts on 7 April; the window standard deviation is the exact pooled county standard deviation.
2. *(needs the raw data)* `00_preprocessing.ipynb` → `Claude/data_cache/features_20d_leakfree.parquet`,
   `counts_20d.parquet`, `counts_10d.parquet`, `soil_means_20d.parquet` and `cache/cohort_10d_2025.txt`.
3. *(needs the raw data)* the first three code cells of `03_track_10d.ipynb` →
   `Claude/data_cache/features_10d_leakfree.parquet`.
4. *(needs the raw soil data)* `python Claude/final_results_claude/build_soil_selection.py` →
   `Claude/data_cache/soil_means_20d_selected.parquet`, the 18 soil properties the soil variants of `08` use.
   The final selection is hardcoded in the builder. About 100 minutes from nothing.
5. `01_headline_register.ipynb` → `cache/headline_seeds/` (30 models), `cache/inseason_D/` (7 horizons × 30
   models) and `figs/`.
6. `08_experiments.ipynb`, top to bottom.

The outputs of steps 2 to 4 are part of the repository (the output of step 1, the windowed raw data, is not),
so steps 5 and 6 run from the included tables and cache.

Every trained model is cached as its own file (`cache/<experiment>/model_<ii>.npz`), so an interrupted run
continues where it stopped. Deleting a file or folder retrains exactly that part. With the included cache `08`
executes in about a minute. From an empty cache it trains about 3,300 networks and reads about 70 GB of
per-cell exports (appendix E15, E16) and about 36,000 soil rasters (E18); the last full run took about 4.5 hours
with several processes sharing one RTX 5060 Ti.

**Figures.** Every figure of `08` is written to `figures/` as PNG at 300 dpi, with `index.csv` listing them.
The notebook sections follow chapter 7, so the prefix is the section of the thesis:
`2-3_larger_and_deeper.png` belongs to section 7.2.3 (and 8.2.3). `A-` marks the appendix of the notebook,
`S_` its summary. The maps in `figures/` come from the original run: drawing them needs the US Census state
boundaries (`Data_Ashape/cb_2023_us_state_20m.zip`, not included), so those lines are commented out with the
prefix `#CENSUS ` (the note at the top of the map geometry cell explains how to restore them). Without the raw
SoilGrids rasters, E18 takes its candidate properties from the cached county means.

## The setup in brief

- **Data.** 4,120 county-years of wheat in 21 states, seasons 2016–2023 and 2025, seven 20-day windows from
  7 April to 24 August.
- **Split.** Training on every season except 2023 and 2025, validation (early stopping and every choice) on
  2023, test on 2025.
- **Test sets.** 282 counties = the 2025 counties with a complete 10-day season (the headline set, "10d test
  set"); 426 counties = all of 2025 ("20d test set").
- **Headline model.** 27 values per window (8 bands × mean and standard deviation, 6 spectral indices,
  5 deltas), a one-layer Transformer of width 256, cross-attention to 18 learned register tokens,
  attention pooling biased by satellite coverage, fused with the 5-year yield history. The target is
  the deviation from the history. 30 models averaged.
- **Model weights.** The weights are not part of the repository (about 110 MB). `headline_weights.py`
  retrains the 30 seeded models of the headline configuration (E7 in `08`, the reference of sections 2–4 and
  the 2025 fold of leave-one-year-out) into `cache/e07_weights_width256/`. Their ensemble gives RMSE 0.510,
  MAE 0.395, R² 0.867 on the 282 counties. The headline ensemble of `01` is unseeded (0.508 / 0.395 / 0.868).
- **Leave one year out.** Every season held out in turn; its validation season is the next later season
  in the list, for the last season the previous one. Everything fitted from data is refitted per fold.

## Noise rule

Independent 30-model ensembles of the same configuration differ by at most about 0.005 t/ha RMSE. A
difference in RMSE on the 282 counties is called real only if it is at least 0.01 t/ha, unrounded (printed
with three decimals); smaller differences are reported as "within noise". The rule covers the randomness of
training, not the choice of test counties or seasons.

## Reproducibility

- Python 3.13, torch 2.11 (CUDA 12.8), scikit-learn 1.8, pandas 3.0, numpy 2.4, geopandas 1.1,
  matplotlib 3.10, netCDF4 1.7.
- On the Blackwell GPU the flash and memory-efficient attention kernels are disabled and the math kernel
  is used; the notebooks set this at the top.
- `01` is unseeded. `08` seeds every model, but GPU training is not bit-for-bit deterministic, so a retrain
  reproduces the results within the observed training variability, not to the last digit. In the original
  run the 30 retrained headline models reproduced their cached predictions exactly.

## The soil features

The soil variants use the **18 properties that are the first of their correlation group**: going down the
random-forest ranking of section 5.2 of `08`, a property is kept unless it correlates at |r| ≥ 0.85 with a
property already kept. The final selection is frozen, in ranking order, as `SELECTED_SOIL_PROPERTIES` in
`build_soil_selection.py`. The builder reads the wheat cells of `Data_Astatic_wheat_soil_cells` (not
included) with the extraction function of the root `Building_soil_years.ipynb`, unchanged.

The 641 cell files of the season 2020 come from an older version of the extraction. Since soil is static,
`build_soil_selection.py` recomputes every season with the current function, which removes that
inconsistency from `soil_means_20d_selected.parquet`. `soil_means_20d.parquet` (built by `00`) still contains
the older 2020 values and is not read by `08`.

## Caveats a reader of the code should know

- **Leave one year out** is not as strict as the 2025 holdout: the 5-year history of a training row from a
  later season contains the held-out season's label (never the held-out prediction itself), and 2025 is a
  training season of the other folds.
- **2023 is used twice** where something is chosen after training: blend weights, the MLP configuration and
  the changes combined in E25 are chosen on the same season that already chose each network's checkpoint,
  so their validation RMSE is optimistic.
- **The 2025 test season was inspected repeatedly during development** (see the limitations of the thesis).
- **Oracle blends** in `08` section 3 choose their weight on 2025. They are upper bounds, not models.
- **Appendix A.6 of `08`** (transfer to unseen counties and states) is described but not run.
- **Leave one year out for the random forest and the MLP** keeps the settings chosen on 2023 in every fold,
  as the Transformer keeps its configuration.
