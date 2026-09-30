"""
Soil features at a point, read straight from the SoilGrids cloud-optimized
GeoTIFFs over HTTP (/vsicurl byte-range windowed reads) -- no WCS, no server-side
warp, no mapserver throttling.

Input : longitude, latitude (WGS84)
Logic : for each of the 37 raw coverages, window-read a ~12 km box around the point
        from the native 250 m COG, average valid pixels within 2 km (else 5 km),
        warn if 5 km is empty, then derive the 18 features. The 37 reads run in
        parallel (I/O-bound).

CONSISTENCY NOTE: these are the NATIVE Homolosine pixels. Your existing table was
built from WCS output resampled to ESRI:54052. The values will be very close but
NOT bit-identical (different pixel grid in the radius). Acceptable for patching
<1% gaps; do not mix-and-match methods within a single column elsewhere.

aggregate_window + derive_features are unit-tested; the rasterio windowed-read
path is tested against a local COG. Only the live /vsicurl fetch needs network.
"""

import os
os.environ.setdefault("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")
os.environ.setdefault("CPL_VSIL_CURL_ALLOWED_EXTENSIONS", ".vrt,.tif")
os.environ.setdefault("GDAL_HTTP_MULTIPLEX", "YES")
os.environ.setdefault("GDAL_HTTP_VERSION", "2")
os.environ.setdefault("VSI_CACHE", "TRUE")
os.environ.setdefault("GDAL_HTTP_MAX_RETRY", "3")
os.environ.setdefault("GDAL_HTTP_RETRY_DELAY", "1")

import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import rasterio
from rasterio.windows import from_bounds
from rasterio.transform import Affine
from pyproj import Transformer

# native SoilGrids projection; reproject the query point into it ourselves so the
# (often CRS-less) VRT doesn't matter -- the VRT transform is already in IGH metres.
IGH = "+proj=igh +lat_0=0 +lon_0=0 +datum=WGS84 +units=m +no_defs"
_to_igh = Transformer.from_crs("EPSG:4326", IGH, always_xy=True)

BASE = "/vsicurl/https://files.isric.org/soilgrids/latest/data"
RES_M     = 250
RADII_M   = (2000.0, 5000.0)
THRESHOLD = 1.0                      # value > THRESHOLD == valid (your mean_data convention)
MARGIN_M  = 3 * RES_M
WORKERS   = 8                        # keep modest; ISRIC tolerates this fine

PROP_DEPTHS = {
    "ocd": ["0-5cm", "5-15cm", "15-30cm", "30-60cm"],
    "soc": ["0-5cm", "5-15cm", "15-30cm", "30-60cm", "60-100cm"],
    "nitrogen": ["0-5cm", "5-15cm", "15-30cm", "30-60cm", "60-100cm"],
    "phh2o": ["0-5cm", "5-15cm", "15-30cm", "30-60cm", "60-100cm"],
    "clay": ["0-5cm", "5-15cm", "15-30cm", "30-60cm", "60-100cm"],
    "sand": ["0-5cm", "5-15cm", "15-30cm", "30-60cm", "60-100cm"],
    "bdod": ["0-5cm"],
    "wv0010": ["0-5cm"],
    "wv0033": ["0-5cm", "5-15cm", "15-30cm"],
    "wv1500": ["0-5cm", "5-15cm", "15-30cm"],
}
COVERAGES = [(p, d) for p, ds in PROP_DEPTHS.items() for d in ds]   # 37

W = {"0-5cm": 5, "5-15cm": 10, "15-30cm": 15, "30-60cm": 30, "60-100cm": 40}

# --- pixel math (unit-tested) ------------------------------------------------
def aggregate_window(values, dist, radii, threshold):
    valid = np.isfinite(values) & (values > threshold)
    for r in radii:
        sel = valid & (dist <= r)
        if sel.any():
            return float(values[sel].mean()), r, int(sel.sum())
    return float("nan"), None, 0

def _agg(raw, prop, depths):
    num = sum(W[d] * raw[f"{prop}_{d}_mean"] for d in depths)
    return num / sum(W[d] for d in depths)

def derive_features(raw):
    a030  = lambda p: _agg(raw, p, ["0-5cm", "5-15cm", "15-30cm"])
    a3010 = lambda p: _agg(raw, p, ["30-60cm", "60-100cm"])
    out = {}
    for p, d in [("ocd", "0-5cm"), ("soc", "0-5cm"), ("nitrogen", "0-5cm"),
                 ("bdod", "0-5cm"), ("ocd", "30-60cm"), ("phh2o", "30-60cm"),
                 ("wv0010", "0-5cm"), ("nitrogen", "15-30cm")]:
        out[f"{p}_{d}_mean_mean_1km"] = raw[f"{p}_{d}_mean"]
    out["ocd_0_30_mean_1km"]      = a030("ocd")
    out["soc_0_30_mean_1km"]      = a030("soc")
    out["nitrogen_0_30_mean_1km"] = a030("nitrogen")
    out["phh2o_30_100_mean_1km"]  = a3010("phh2o")
    out["awc_0_30_mean_1km"]      = a030("wv0033") - a030("wv1500")
    for p in ["phh2o", "clay", "sand", "nitrogen", "soc"]:
        out[f"{p}_gradient_30_100_minus_0_30_mean_1km"] = a3010(p) - a030(p)
    return out

# --- COG windowed read -------------------------------------------------------
_cached_transform = None   # once per process; all layers on same 250m IGH grid

def _get_grid_transform():
    """Read the geotransform from a known-good property layer once."""
    global _cached_transform
    if _cached_transform is not None:
        return _cached_transform
    url = f"{BASE}/soc/soc_0-5cm_mean.vrt"
    with rasterio.open(url) as src:
        _cached_transform = src.transform
    return _cached_transform

def _read_coverage(prop, depth, px, py, half):
    """Window-read one COG around (px,py) in IGH; aggregate within the radii.
    All layers share the same geotransform (native 250m IGH grid)."""
    url = f"{BASE}/{prop}/{prop}_{depth}_mean.vrt"
    transform = _get_grid_transform()
    with rasterio.open(url) as src:
        nodata = src.nodata
        # use the cached transform for the window, not src.transform (which may be missing/wrong for wv)
        win = from_bounds(px - half, py - half, px + half, py + half, transform)
        a = src.read(1, window=win, boundless=True,
                     fill_value=(nodata if nodata is not None else 0)).astype(float)
    rows, cols = a.shape
    # window's own transform = grid transform shifted by the window offset
    wt = Affine(transform.a, transform.b, transform.c + win.col_off * transform.a,
                transform.d, transform.e, transform.f + win.row_off * transform.e)
    xs = wt.c + (np.arange(cols) + 0.5) * wt.a
    ys = wt.f + (np.arange(rows) + 0.5) * wt.e
    gx, gy = np.meshgrid(xs, ys)
    dist = np.hypot(gx - px, gy - py)
    if nodata is not None:
        a[a == nodata] = np.nan
    return aggregate_window(a, dist, RADII_M, THRESHOLD)

# --- entry point -------------------------------------------------------------
def soil_features_at_point(lon, lat, workers=WORKERS):
    """lon, lat (WGS84) -> (features_dict[18], diagnostics_dict)."""
    px, py = _to_igh.transform(lon, lat)
    half = RADII_M[-1] + MARGIN_M

    raw, diag, missing = {}, {}, []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_read_coverage, p, d, px, py, half): (p, d)
                for (p, d) in COVERAGES}
        for fut in as_completed(futs):
            p, d = futs[fut]
            cov = f"{p}_{d}_mean"
            try:
                mean, r_used, n = fut.result()
            except Exception as e:                      # network/read failure != gap
                mean, r_used, n = float("nan"), None, 0
                diag.setdefault("errors", {})[cov] = repr(e)
            raw[cov] = mean
            diag[cov] = {"radius_m": r_used, "n_pixels": n}
            if not np.isfinite(mean):
                missing.append(cov)

    if missing:
        warnings.warn(
            f"\n*** No valid SoilGrids pixels within {RADII_M[-1]:.0f} m of "
            f"({lon:.5f}, {lat:.5f}) for: {', '.join(sorted(missing))}.\n"
            f"Very strange for arable land -- inspect before trusting this point. "
            f"(If diag['errors'] is set, it was a network failure, not a real gap.)",
            stacklevel=2)

    feats = derive_features(raw)        # NaN propagates if any input missing
    return feats, {"raw": raw, "per_coverage": diag, "missing": missing}


if __name__ == "__main__":
    import time
    t = time.time()
    feats, diag = soil_features_at_point(-90.26726427598487, 36.1996609058439)
    print(f"\n{time.time()-t:.1f}s")

    if diag.get("errors"):
        print("\nREAD ERRORS (these are NOT real soil gaps -- URL/format/network):")
        for cov, e in diag["errors"].items():
            print(f"  {cov}: {e}")

    radii = sorted({v["radius_m"] for v in diag["per_coverage"].values()
                    if isinstance(v, dict) and v.get("radius_m") is not None})
    print("radii used:", radii or "(none -- everything failed)")

    print("\nfeatures:")
    for k, v in feats.items():
        print(f"  {k:48s} {v:.4f}" if np.isfinite(v) else f"  {k:48s} NaN")