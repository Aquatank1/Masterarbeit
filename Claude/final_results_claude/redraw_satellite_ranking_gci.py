"""Redraw the satellite ranking figure of section 5.1 (08_experiments.ipynb, E17) with the band label GCI.

The thesis renamed GCVI to GCI (2026-09-26). The figure is redrawn from the cached importances with the plotting
code of the notebook cell unchanged except for the two label dictionaries, so nothing is retrained. It writes the
same three files the notebook and refresh_figures_c8.py would write.

    python Claude/final_results_claude/redraw_satellite_ranking_gci.py
"""
import os
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_FOLDER = os.path.join(HERE, "cache", "e17_rf_ranking_satellite")
FIGURES = os.path.join(HERE, "figures")
SELECTION = os.path.join(HERE, "..", "..", "figures_C8")

plt.rcParams.update({"font.size": 14, "axes.labelsize": 15, "xtick.labelsize": 14, "ytick.labelsize": 14,
                     "legend.fontsize": 13})
SATELLITE_RANKING_COLOUR = "#1f7a4d"
BAND_NAMES = {"B2": "Blue (B2)", "B3": "Green (B3)", "B4": "Red (B4)", "B5": "NIR (B5)",
              "B6": "SWIR1 (B6)", "B7": "SWIR2 (B7)", "NDVI": "NDVI", "GCVI": "GCI"}
BAND_SHORT_NAMES = {"B2": "Blue", "B3": "Green", "B4": "Red", "B5": "NIR", "B6": "SWIR1", "B7": "SWIR2",
                    "NDVI": "NDVI", "GCVI": "GCI"}
STATISTIC_SHORT_NAMES = {"mean": "mean", "stdDev": "SD", "skew": "skewness", "kurtosis": "kurtosis"}


def statistic_type_of(statistic_name):
    """Group a statistic suffix: mean, standard deviation, skewness, kurtosis or percentiles."""
    if statistic_name == "mean":
        return "mean"
    elif statistic_name == "stdDev":
        return "standard deviation"
    elif statistic_name == "skew":
        return "skewness"
    elif statistic_name == "kurtosis":
        return "kurtosis"
    else:
        return "percentiles (p10-p90)"


importances = pd.read_csv(os.path.join(CACHE_FOLDER, "feature_importances.csv"))
importances["statistic type"] = importances["statistic"].map(statistic_type_of)
importance_by_statistic_type = importances.groupby("statistic type")["importance"].sum().sort_values(ascending=False)
importance_by_band = importances.groupby("band")["importance"].sum().sort_values(ascending=False)
importance_by_band_statistic = importances.groupby(["band", "statistic"])["importance"].sum().sort_values(ascending=False)
top_25_statistics = importance_by_band_statistic.head(25)
top_25_statistic_labels = [f"{BAND_SHORT_NAMES[band]} {STATISTIC_SHORT_NAMES.get(statistic, statistic)}"
                           for band, statistic in top_25_statistics.index]

figure = plt.figure(figsize=(10, 11), layout="constrained")
grid = figure.add_gridspec(2, 2, height_ratios=[1, 2.3])
axes = [figure.add_subplot(grid[0, 0]), figure.add_subplot(grid[0, 1]), figure.add_subplot(grid[1, :])]
axes[0].barh(importance_by_statistic_type.index, importance_by_statistic_type.values, color=SATELLITE_RANKING_COLOUR)
axes[0].invert_yaxis()
axes[0].set_xlabel("Summed importance,\nby statistic type")
axes[1].barh([BAND_NAMES[band] for band in importance_by_band.index], importance_by_band.values, color=SATELLITE_RANKING_COLOUR)
axes[1].invert_yaxis()
axes[1].set_xlabel("Summed importance,\nby band")
axes[2].barh(top_25_statistic_labels, top_25_statistics.values, color=SATELLITE_RANKING_COLOUR)
axes[2].invert_yaxis()
axes[2].set_xlabel("Summed importance over the seven windows, 25 most important statistics")
for axis in axes:
    axis.grid(axis="x", alpha=0.25)

figure.savefig(os.path.join(CACHE_FOLDER, "satellite_ranking.png"), dpi=200, bbox_inches="tight")
figure.savefig(os.path.join(FIGURES, "5-1_satellite_ranking.png"), dpi=300, bbox_inches="tight")
shutil.copy2(os.path.join(FIGURES, "5-1_satellite_ranking.png"), os.path.join(SELECTION, "5-1_satellite_ranking.png"))
print("written: cache, figures/ and figures_C8/ 5-1_satellite_ranking.png")
