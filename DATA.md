# Data structure

All paths are relative to the repository root. The raw folders (`Data_*`) are not part of the repository;
their structure is described so the code can be read and rerun. Yields are stored in **dt/ha** everywhere in
the data and converted to t/ha only when results are reported.

## Keys

- **County key** `"{state}_{county}"`, for example `17_2`. The state number runs from 1 to 51 in alphabetical
  order of the states including the District of Columbia (1 Alabama, 9 District of Columbia, 14 Illinois,
  17 Kansas, ...). The county number counts the counties within their state. In the tables this key is
  `county_id`, in the yield file `id_combo`, in the Earth Engine county asset `id_comb`.
- **Season** is the year of the harvest. The seasons used are 2016 to 2023 and 2025; 2024 has no county
  estimates because the USDA discontinued them for that year.

## Yield labels: `Ertragsdaten_USA.csv` (included)

One row per region: the United States, the states and 3,117 counties. USDA NASS estimates for all wheat.

| column | meaning |
|---|---|
| `Name` | name of the country, state or county |
| `Source` | always `NASS USDA` |
| `lev0` | country code (22) |
| `lev1` | state number (-1 for the national row) |
| `lev2` | county number within the state (-1 for national and state rows) |
| `lev3`, `ID`, `x` | not used |
| `Prod {year}` | production in t, 2007-2025 |
| `Area {year}` | harvested area in ha |
| `Yield {year}` | yield in dt/ha |
| `id_combo` | county key `{lev1}_{lev2}` |

A county year without a reported yield is empty. For 2024 only 93 rows have a value, 57 of them counties.

## Raw satellite data (not included)

### `Data_<year>/`: Earth Engine export

Written by `GEE-On_server_HLSL.py` (Landsat part of HLS) and `GEE-On_server_HLSS.py` (Sentinel-2 part), both
into the same folder. Only scenes between 7 April and 2 September with less than 85 % reported cloud cover
are used. The scripts first run a scout pass per county and season that keeps the 1 km cells (EPSG:5070 grid)
with more than 11 CDL wheat pixels and records which cells have clear wheat pixels on which scene (exported
to Cloud Storage as `Scouts/Scout_County_*.csv`). The harvester pass then exports one file per county, scene
and tile:

```
Harvester_{year}_{state}_{county}_{tile}_{acquisition timestamp}.csv
e.g. Harvester_2025_4_38_T15SXA_20250414T163550.csv
```

One row per 1 km cell with at least one clear wheat pixel on that scene. Wheat pixels are CDL classes 22, 23
and 24 (durum, spring and winter wheat) of the same season. Pixels that the HLS Fmask flags as cirrus, cloud,
adjacent to cloud or shadow, cloud shadow or high aerosol are removed; snow and water are not masked.

| columns | meaning |
|---|---|
| `image_id` | Earth Engine scene identifier |
| `longitude`, `latitude` | centre of the cell |
| `{band}_{statistic}` | 8 bands × 14 statistics over the clear wheat pixels of the cell |

Bands: `B2` blue, `B3` green, `B4` red, `B5` NIR, `B6` SWIR1, `B7` SWIR2 (Landsat names; the Sentinel-2 files
carry `B8A`, `B11`, `B12` instead of `B5`, `B6`, `B7`), plus `NDVI` and `GCVI` computed per pixel on Earth
Engine. Statistics: `mean`, `stdDev`, `count` (number of clear wheat pixels), `skew`, `kurtosis`, `p10` to `p90`.
A cell with a single pixel has `stdDev` 0, all percentiles equal to the mean and empty `skew` and `kurtosis`.

### `Data_Fused_<year>/`: one file per county and season

Written by `preprocessing.ipynb` (`fuse_harvester_files`). `Harvester_{year}_{state}_{county}.csv` holds all
scene files of the county, sorted by time, with the Sentinel-2 band names renamed to the Landsat names and an
added column `date` (day.month as a number, for example `14.4` for 14 April). One row per cell and scene.

### `Data_Windowed_<year>_{20d,10d}/`: county time series

Written by `preprocessing.ipynb`. `Harvester_{year}_{state}_{county}.csv` has one row per window: 7 windows of
20 days or 14 windows of 10 days from 7 April to 24 August.

| columns | meaning |
|---|---|
| `date_start`, `date_end` | first and last day of the window (day.month) |
| `from_data` | 1 if observed, 0 if an isolated empty window was filled with the mean of its two neighbours |
| `{band}_{statistic}` | 8 bands × 13 statistics (`count` is used for the weighting and dropped) |

Every statistic is a mean over the cell observations of the window, weighted by their number of clear wheat
pixels. For `mean` this equals the mean over all wheat pixels of the county; `stdDev` is the exact pooled
county standard deviation; skewness, kurtosis and percentiles are weighted means of the cell statistics.
A county year with an empty first or last window or two adjacent empty windows has no file.

## Raw soil and mask data (not included)

- `Data_Asoil_data/{state}_{county}_{property}.nc`: SoilGrids 2.0 rasters (250 m, NetCDF) of one county and
  one property, for example `14_100_bdod_0-5cm_mean.nc`, downloaded by `soil_download.py` (which writes them
  to `soil_data_others/`; they were collected in `Data_Asoil_data/`). Depth-range means,
  `awc_0_30` and the depth gradients are derived from the native layers.
- `Data_AUSDA-Cropland/{year}_30m_cdls/{year}_30m_cdls.tif`: the national Cropland Data Layer rasters used to
  find the wheat cells of each season for the soil sampling.
- `Data_Astatic_wheat_soil_cells/county_{state}_{county}_{year}_static_wheat_soil_cells.csv`: written by
  `Building_soil_years.ipynb`. One row per 1 km wheat cell of the county and season with `x_5070`, `y_5070`,
  `longitude`, `latitude`, `static_wheat_pixels`, `static_wheat_pixels_incounty`, `cdl_valid_pixels`,
  `incounty_ratio` and, per soil property, `{property}_mean_1km` and `{property}_soil_pixel_count_1km`.
- `Data_Ashape/gadm36_USA_2.shp`: US county boundaries, GADM 3.6 level 2 (included), with the county key in
  `id_comb`. `Data_Ashape/cb_2023_us_state_20m.zip`: US state boundaries of the US Census Bureau, used for the
  map background (not included; drawing the maps in `08` is commented out without it, see the README).

## Modelling tables: `Claude/data_cache/` (included)

Built by `Claude/final_results_claude/00_preprocessing.ipynb`, except `features_10d_leakfree.parquet` (first
three code cells of `03_track_10d.ipynb`) and `soil_means_20d_selected.parquet` (`build_soil_selection.py`).

| file | rows | content |
|---|---|---|
| `features_20d_leakfree.parquet` | 4,120 | `county_id`, `year`, `yield` (dt/ha), `yield_5yr_avg` (mean yield of the five previous seasons, dt/ha, empty without any previous season) and `w00_` to `w06_` × the 104 window statistics above |
| `features_10d_leakfree.parquet` | 2,331 | the same for the 14 windows of 10 days (`w00_` to `w13_`) |
| `counts_20d.parquet`, `counts_10d.parquet` | 5,387 | per county year: `n_cells`, `wheat_pix` (CDL wheat pixels), `wheat_pix_incty` (of them inside the county), `soil_valid_frac`, and `satc_wNN`, the clear wheat pixel observations of every window; the coverage input of the model is `satc_wNN / wheat_pix`, capped at 1.5 |
| `soil_means_20d_selected.parquet` | 5,387 | county means of the 18 soil features used in the thesis |
| `soil_means_20d.parquet` | 5,387 | an earlier selection of 18 soil features, built by `00`; not used by `08` |

"leakfree" means that the history of a season uses only earlier seasons. The model inputs (the 27 features of
a window, the indices and deltas) are derived from these tables inside the notebooks.

## Results: `Claude/final_results_claude/cache/` (included)

- `<experiment>/model_{ii}.npz`: the predictions of one trained model (`validation`: 2023, `test`: 2025, in
  dt/ha, in the row order of the season in `features_20d_leakfree.parquet`, for the 10d experiments in
  `features_10d_leakfree.parquet`). A leave-one-year-out experiment
  has one folder per held-out season (`fold_{year}`).
- `<experiment>/results*.csv`: the tables behind chapter 8, in t/ha.
- `cohort_10d_2025.txt`: the county keys of the 282 counties of the 10d test set.
- `headline_seeds/`, `inseason_D/`: the predictions of the 30 headline models of `01` and of its in-season models.
- `e07_weights_width256/model_{ii}.pt` (not included, written by `headline_weights.py`): the weights of the 30
  headline models (PyTorch; keys `state_dict`, `target_mean`, `target_std`, `seed`, `test`).
- `e99_summary/summary_all_experiments.csv`: every experiment in one table, unrounded.
