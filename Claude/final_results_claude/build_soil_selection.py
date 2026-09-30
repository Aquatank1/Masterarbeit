"""Soil features for the soil variants of 08: the first property of every correlation group.

WHY. The 47 SoilGrids candidates were ranked by a random forest on the county mean yield of the training
seasons (section 5.2 of 08_experiments.ipynb) and grouped by correlation: going down the ranking, a property
was kept as the "first of its group" unless it correlated at |r| >= SOIL_GROUP_THRESHOLD with a property
already kept. At 0.85 that gave 18 groups. SELECTED_SOIL_PROPERTIES freezes that final selection in ranking
order, verified against the existing model inputs on 2026-09-25. No ranking or candidate cache from 08 is
needed to build the table. Section 5.2 of 08 can reproduce the ranking and check the selection afterwards.

TWO EXTRACTION VERSIONS IN THE STORED CELLS (found 2026-09-22). The 641 files of the season 2020 were
written by an older version of the extraction: they carry no soil pixel-count columns and their values differ
slightly from what the current function produces (county means differ by about 0.1 in SoilGrids units, and a
few more cells come out empty). All other 4,746 files reproduce to 1e-14. Soil is static, so a county should
carry the same soil in every season; all 15 properties are therefore recomputed here with the current
function for every county-year. For the ten properties that already existed this changes only the 2020 rows.

HOW. The 18 properties were attached to the wheat cells by Building_soil_years.ipynb (project root): a 1 km
grid per county, the CDL of each season decides which cells are wheat cells (more than 11 wheat pixels in
the 1 km box), and every active cell receives the mean of the SoilGrids pixels inside it. The result is one
file per county-year in Data_Astatic_wheat_soil_cells/, which lists the active cells by their centre.
The CDL step does not depend on the soil property, so it is NOT repeated here: the active cells are rebuilt
from the stored centres (1 km boxes, EPSG:5070) and the five new properties are attached with the
unchanged function add_soil_means_fast, executed from Building_soil_years.ipynb itself.

CHECKS (both must pass before anything is written for use):
  1. Two of the existing 18 properties recomputed this way equal the values in Data_Astatic_wheat_soil_cells
     on a sample of county-years (same cells, same method).
  2. The county means of the ten existing properties, averaged like notebook 00 (plain mean over the cells),
     equal Claude/data_cache/soil_means_20d.parquet.

OUTPUT (nothing in Data_* is written, notebook 00 is not changed):
  Claude/data_cache/wheat_soil_cells_selected/county_<id>_<year>_soil_cells.csv   per cell, every property
  Claude/data_cache/soil_means_20d_selected.parquet                               per county-year

RUN ORDER: 00_preprocessing.ipynb -> this script -> 08_experiments.ipynb.
    python Claude/final_results_claude/build_soil_selection.py
"""
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import rioxarray  # noqa: F401  registers the .rio accessor the builder function relies on

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
CELL_FOLDER = os.path.join(ROOT, "Data_Astatic_wheat_soil_cells")          # read only
SOIL_FOLDER = os.path.join(ROOT, "Data_Asoil_data")                         # read only
BUILDER_NOTEBOOK = os.path.join(ROOT, "Building_soil_years.ipynb")          # read only
DATA = os.path.join(ROOT, "Claude", "data_cache")
EXTRA_FOLDER = os.path.join(DATA, "wheat_soil_cells_selected")
OUTPUT_TABLE = os.path.join(DATA, "soil_means_20d_selected.parquet")
EXISTING_TABLE = os.path.join(DATA, "soil_means_20d.parquet")
FILE_PATTERN = re.compile(r"county_(\d+)_(\d+)_(\d{4})_static_wheat_soil_cells\.csv")
CHECK_PROPERTIES = ["ocd_0-5cm_mean", "awc_0_30"]
N_CHECK_FILES = 60
SOIL_GROUP_THRESHOLD = 0.85          # provenance of the frozen selection; not recomputed here
EXPECTED_PROPERTIES = 18
SELECTED_SOIL_PROPERTIES = (
    "ocd_0-5cm_mean",
    "phh2o_30-60cm_mean",
    "sand_gradient_30_100_minus_0_30",
    "soc_0-5cm_mean",
    "clay_gradient_30_100_minus_0_30",
    "bdod_0-5cm_mean",
    "wv0010_0-5cm_mean",
    "nitrogen_0-5cm_mean",
    "awc_0_30",
    "phh2o_gradient_30_100_minus_0_30",
    "ocd_30-60cm_mean",
    "nitrogen_60-100cm_mean",
    "ocd_15-30cm_mean",
    "clay_60-100cm_mean",
    "wv1500_15-30cm_mean",
    "nitrogen_gradient_30_100_minus_0_30",
    "soc_15-30cm_mean",
    "sand_0-5cm_mean",
)

# 1. the function that attached the 18 properties, taken unchanged from Building_soil_years.ipynb
builder = json.load(open(BUILDER_NOTEBOOK, encoding="utf-8"))
namespace = {}
exec(compile("".join(builder["cells"][1]["source"]), "<Building_soil_years cell 1>", "exec"), namespace)
add_soil_means_fast = namespace["add_soil_means_fast"]
gpd, xr, box = namespace["gpd"], namespace["xr"], namespace["box"]
GRID_CRS = namespace["GRID_CRS"]

# 2. the final feature selection is fixed, so an empty experiment cache is sufficient
selected = list(SELECTED_SOIL_PROPERTIES)
assert len(selected) == len(set(selected)) == EXPECTED_PROPERTIES, "expected 18 distinct frozen soil properties"
existing_columns = [c for c in pd.read_csv(os.path.join(CELL_FOLDER, sorted(os.listdir(CELL_FOLDER))[0]), nrows=1).columns
                    if c.endswith("_mean_1km") and "pixel_count" not in c]
existing_properties = [c[:-len("_mean_1km")] for c in existing_columns]
new_properties = [p for p in selected if p not in existing_properties]
kept_existing = [p for p in selected if p in existing_properties]
print(f"frozen selection (original grouping |r| >= {SOIL_GROUP_THRESHOLD}): {len(selected)} properties | "
      f"already on the wheat cells: {len(kept_existing)} | new: {new_properties}")

cell_files = sorted(f for f in os.listdir(CELL_FOLDER) if FILE_PATTERN.match(f))
files_by_county = {}
for file_name in cell_files:
    state_part, county_part, year = FILE_PATTERN.match(file_name).groups()
    files_by_county.setdefault(f"{state_part}_{county_part}", []).append((int(year), file_name))
print(f"{len(cell_files)} county-year cell files, {len(files_by_county)} counties")


def active_cells(cell_table, county_id):
    """The active 1 km cells of a county-year, rebuilt from their stored centres."""
    geometries = [box(x - 500, y - 500, x + 500, y + 500) for x, y in zip(cell_table.x_5070, cell_table.y_5070)]
    return gpd.GeoDataFrame({"county_id": county_id, "x_5070": cell_table.x_5070.to_numpy(),
                             "y_5070": cell_table.y_5070.to_numpy()}, geometry=geometries, crs=GRID_CRS)


def soil_means_for(cells, county_id, soil_property):
    """Mean of the county's SoilGrids raster inside every cell, with the original function."""
    raster = os.path.join(SOIL_FOLDER, f"{county_id}_{soil_property}.nc")
    with xr.open_dataarray(raster) as dataarray:
        result = add_soil_means_fast(grid_5070=cells, dataarray=dataarray.load(), soil_var=soil_property,
                                     soil_crs="EPSG:4326", all_touched=True)
    return result[f"{soil_property}_mean_1km"].to_numpy(), result[f"{soil_property}_soil_pixel_count_1km"].to_numpy()


# 3. check 1: the same method reproduces two existing properties on a sample of county-years
random_generator = np.random.default_rng(0)
check_files = [cell_files[i] for i in random_generator.choice(len(cell_files), N_CHECK_FILES, replace=False)]
largest_difference = 0.0
files_with_counts = 0
older_differences = []
for file_name in check_files:
    state_part, county_part, _ = FILE_PATTERN.match(file_name).groups()
    county_id = f"{state_part}_{county_part}"
    stored = pd.read_csv(os.path.join(CELL_FOLDER, file_name))
    older_version = not any(c.endswith("_soil_pixel_count_1km") for c in stored.columns)
    cells = active_cells(stored, county_id)
    for soil_property in CHECK_PROPERTIES:
        recomputed, pixel_counts = soil_means_for(cells, county_id, soil_property)
        # a few cell files store the column as text, so read it as a number the way pandas would
        stored_values = pd.to_numeric(stored[f"{soil_property}_mean_1km"], errors="coerce").to_numpy(dtype=float)
        finite = ~np.isnan(stored_values) & ~np.isnan(recomputed)
        difference = float(np.max(np.abs(recomputed[finite] - stored_values[finite]), initial=0.0))
        if older_version:                 # the 2020 files come from the older extraction, see the header
            older_differences.append(difference)
            continue
        assert np.array_equal(np.isnan(recomputed), np.isnan(stored_values)), f"missing values differ in {file_name}"
        largest_difference = max(largest_difference, difference)
        count_column = f"{soil_property}_soil_pixel_count_1km"      # not every cell file stores the counts
        if count_column in stored.columns:
            files_with_counts += 1
            assert np.array_equal(pixel_counts, stored[count_column].to_numpy()), f"pixel counts differ in {file_name}"
print(f"check 1 passed: {N_CHECK_FILES} county-years x {CHECK_PROPERTIES}, largest difference {largest_difference:.2e} "
      f"on the files of the current extraction, pixel counts identical in {files_with_counts} comparisons")
if older_differences:
    print(f"  {len(older_differences)} comparisons fell on 2020 files (older extraction); they differ by up to "
          f"{max(older_differences):.3f} per cell, which is why all 15 properties are recomputed here")
assert largest_difference < 1e-6

# 4. all 15 properties for every county-year (one raster load per county and property), so that every
#    season is extracted with the same function
os.makedirs(EXTRA_FOLDER, exist_ok=True)


def target_file(county_id, year):
    return os.path.join(EXTRA_FOLDER, f"county_{county_id}_{year}_soil_cells.csv")


def build_county(county_id):
    """Add the properties a county-year does not have yet; a file that has them all is left alone."""
    todo, outputs, missing_properties = [], {}, set()
    for year, name in files_by_county[county_id]:
        existing = pd.read_csv(target_file(county_id, year)) if os.path.exists(target_file(county_id, year)) else None
        absent = [p for p in selected if existing is None or f"{p}_mean_1km" not in existing.columns]
        if not absent:
            continue
        todo.append((year, name))
        outputs[year] = existing
        missing_properties.update(absent)
    if not todo:
        return 0
    stored_by_year = {year: pd.read_csv(os.path.join(CELL_FOLDER, name), usecols=["x_5070", "y_5070"]) for year, name in todo}
    for year, stored in stored_by_year.items():
        if outputs[year] is None:
            outputs[year] = stored[["x_5070", "y_5070"]].copy()
    for soil_property in sorted(missing_properties):
        raster = os.path.join(SOIL_FOLDER, f"{county_id}_{soil_property}.nc")
        with xr.open_dataarray(raster) as dataarray:
            loaded = dataarray.load()
        for year, stored in stored_by_year.items():
            result = add_soil_means_fast(grid_5070=active_cells(stored, county_id), dataarray=loaded,
                                         soil_var=soil_property, soil_crs="EPSG:4326", all_touched=True)
            outputs[year][f"{soil_property}_mean_1km"] = result[f"{soil_property}_mean_1km"].to_numpy()
            outputs[year][f"{soil_property}_soil_pixel_count_1km"] = result[f"{soil_property}_soil_pixel_count_1km"].to_numpy()
    for year, table in outputs.items():
        if "county_id" not in table.columns:
            table.insert(0, "county_id", county_id)
        target = target_file(county_id, year)
        table.to_csv(target + ".partial", index=False)
        os.replace(target + ".partial", target)
    return len(todo)


start = time.time()
done = 0
with ThreadPoolExecutor(max_workers=8) as executor:
    for position, written in enumerate(executor.map(build_county, sorted(files_by_county)), start=1):
        done += written
        if position % 100 == 0:
            print(f"  {position}/{len(files_by_county)} counties ({time.time() - start:.0f}s)", flush=True)
print(f"extraction: {done} county-year files written ({time.time() - start:.0f}s)")

# 5. county means over the wheat cells, exactly as notebook 00 averages them (plain mean over the cells)
rows = []
for file_name in cell_files:
    state_part, county_part, year = FILE_PATTERN.match(file_name).groups()
    county_id = f"{state_part}_{county_part}"
    cells = pd.read_csv(target_file(county_id, year))
    record = {"county_id": county_id, "year": int(year)}
    for soil_property in selected:
        record[f"soil_{soil_property}_mean_1km"] = float(cells[f"{soil_property}_mean_1km"].mean())
    rows.append(record)
table = pd.DataFrame(rows).drop_duplicates(["county_id", "year"])

# check 2: the ten existing properties equal the county means of notebook 00
existing = pd.read_parquet(EXISTING_TABLE)
existing["county_id"] = existing.county_id.astype(str)
merged = table.merge(existing, on=["county_id", "year"], suffixes=("", "_00"))
assert len(merged) == len(existing) == len(table), f"county-years differ: new {len(table)}, 00 {len(existing)}"
differences = pd.DataFrame({p: (merged[f"soil_{p}_mean_1km"] - merged[f"soil_{p}_mean_1km_00"]).abs()
                            for p in kept_existing})
# 88 county-years have no soil value at all (no raster pixel in any of their cells); notebook 00 has the
# same ones, so the missing pattern is compared instead of the value there
same_missing = all((merged[f"soil_{p}_mean_1km"].isna() == merged[f"soil_{p}_mean_1km_00"].isna()).all()
                   for p in kept_existing)
largest_difference = float(np.nanmax(differences[merged.year != 2020].to_numpy()))
largest_2020 = float(np.nanmax(differences[merged.year == 2020].to_numpy()))
missing_county_years = int(merged[f"soil_{kept_existing[0]}_mean_1km"].isna().sum())
print(f"check 2 passed: the {int((merged.year != 2020).sum())} county-years outside 2020 match notebook 00 "
      f"(largest difference {largest_difference:.2e}, same missing values: {same_missing}, "
      f"{missing_county_years} county-years have no soil at all); the {int((merged.year == 2020).sum())} rows of "
      f"2020 differ by up to {largest_2020:.3f}, because notebook 00 read the older extraction")
assert largest_difference < 1e-9 and same_missing

table.to_parquet(OUTPUT_TABLE, index=False)
missing = table.drop(columns=["county_id", "year"]).isna().sum()
print(f"written {OUTPUT_TABLE}: {len(table)} county-years x {len(selected)} properties")
print("county-years without a value:", {name: int(count) for name, count in missing.items() if count})
