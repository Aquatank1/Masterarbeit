import pandas as pd
import numpy as np
import geopandas as gpd
import xarray
from scipy.ndimage import distance_transform_edt
from soilgrids import SoilGrids
from soilgrids.exceptions import SoilGridsWcsError
from datetime import datetime
import time



SoilGrids.MAP_SERVICES["wv0010"] = {
    "name": "Volumetric water content at -10 kPa",
    "units": "10^-3 cm3/cm3",
    "link": "https://maps.isric.org/mapserv?map=/map/wv0010.map",
}

SoilGrids.MAP_SERVICES["wv0033"] = {
    "name": "Volumetric water content at -33 kPa",
    "units": "10^-3 cm3/cm3",
    "link": "https://maps.isric.org/mapserv?map=/map/wv0033.map",
}

SoilGrids.MAP_SERVICES["wv1500"] = {
    "name": "Volumetric water content at -1500 kPa",
    "units": "10^-3 cm3/cm3",
    "link": "https://maps.isric.org/mapserv?map=/map/wv1500.map",
}
soil_grids = SoilGrids()




def get_shapeid(list_shapes, id_comb):
    us_county = list_shapes[list_shapes["id_combo"] == id_comb]
    return us_county["geometry"]
def h5files_from_shapes(ids, list_shapes):
    in_shapes = []
    for id in ids:
        in_shapes.append(get_shapeid(list_shapes, id))
    return in_shapes


# # Ranking: Soil texture data Static Soil Soil_Clay, Soil_Sand, Soil_Silt, phh20, soc, nitrogen


def nearest_fill_nan(da):
    arr = da.values.copy()
    valid = ~np.isnan(arr)

    _, indices = distance_transform_edt(
        ~valid,
        return_indices=True
    )

    filled = arr[tuple(indices)]
    return xarray.DataArray(filled, coords=da.coords, dims=da.dims, attrs=da.attrs, name=da.name)

def nearest_fill_below_threshold(da, threshold=1.0):
    arr = da.values.copy()

    valid = (~np.isnan(arr)) & (arr > threshold)

    if not valid.any():
        raise ValueError("No valid pixels above threshold.")

    _, indices = distance_transform_edt(~valid, return_indices=True)
    filled = arr[tuple(indices)]

    return xarray.DataArray(
        filled,
        coords=da.coords,
        dims=da.dims,
        attrs=da.attrs,
        name=da.name
    )

def get_coverage_data_retry(retries=5, sleep=15, **kwargs):
    last_error = None

    for attempt in range(1, retries + 1):
        try:
            return soil_grids.get_coverage_data(**kwargs)

        except SoilGridsWcsError as e:
            last_error = e
            print(f"Attempt {attempt}/{retries} failed.")
            print(e)

            if attempt < retries:
                time.sleep(sleep * attempt)

    raise last_error

def get_soilgrids_layer(shape_in, service_id, coverage_id):
    print(coverage_id)
    print(shape_in.crs)
    gdf = shape_in.to_crs("ESRI:54052")
    print(gdf.crs)

    west, south, east, north = map(float, gdf.total_bounds)
    print(f"Bounds: west={west}, south={south}, east={east}, north={north}")
    data = get_coverage_data_retry(
        service_id=service_id,
        coverage_id=coverage_id,
        west=west,
        south=south,
        east=east,
        north=north,
        crs="urn:ogc:def:crs:EPSG::152160",
        output="test.tif",
        resx=250,
        resy=250,
    ).fillna(0)
    if data.size == 0 or 0 in data.shape:
        print(f"-> Bad Data: SoilGrids returned an empty array. Skipping.")
        return None
    data = nearest_fill_below_threshold(data, threshold=0.1)
    data.rio.write_crs("ESRI:54052", inplace=True)
    data = data.rio.clip(
        gdf.geometry.values,
        gdf.crs,
        drop=False
    ).fillna(0)
    return data

def mean_data(data_in):
    # everything outside polygon becomes 0
    valid = data_in.where(data_in > 1)
    return (valid.mean(skipna=True).item())

def save_to_netcdf(data_in, filename):
    data_in.to_netcdf(filename)

def load_from_netcdf(filename):
    return xarray.open_dataarray(filename)



def soil_data_id(id_in, df_in):
    print(f"Processing ID: {id_in}")
    print(datetime.now())
    shapes = get_shapeid(us_list_shapes, id_in)
    soil_features_main = [    "phh2o", "clay", "sand", "silt", "soc", "nitrogen",
    "cec", "bdod", "cfvo", "wv0010", "wv0033", "wv1500", "ocd"]
    depths = ["0-5cm", "5-15cm", "15-30cm", "30-60cm", "60-100cm"]
    soil_coverages = {feature: [f"{feature}_{depth}_mean" for depth in depths] for feature in soil_features_main}
    rows = []
    # special case: ocs only has 0-30cm
    soil_coverages["ocs"] = ["ocs_0-30cm_mean"]
    for service_id, coverage_ids in soil_coverages.items():
        for coverage_id in coverage_ids:
            da = get_soilgrids_layer(
                shapes,
                service_id,
                coverage_id
            )
            rows = []
            rows.append({
                "county_id": id_in,
                "coverage_id": coverage_id,
                "mean": mean_data(da)
            })

            out_file = f"soil_data_others/{id_in}_{coverage_id}.nc"
            save_to_netcdf(da, out_file)

            df_in = pd.concat([df_in, pd.DataFrame(rows)], ignore_index=True)
    return df_in

us_list_shapes = gpd.read_file(r'Data_Ashape/gadm36_USA_2.shp')
us_list_shapes["id_combo"] = us_list_shapes["id_comb"]
Texas =['44_6', '44_7', '44_12', '44_14', '44_15', '44_18', '44_25', '44_30', '44_33', '44_38', '44_39', '44_43', '44_48', 
        '44_49', '44_50', '44_51', '44_56', '44_59', '44_60', '44_61', '44_71', '44_73', '44_74', '44_90', '44_91', '44_95', 
        '44_105', '44_106', '44_109', '44_116', '44_117', '44_126', '44_129', '44_138', '44_140', '44_153', '44_160', '44_161', 
        '44_163', '44_166', '44_169', '44_171', '44_175', '44_177', '44_179', '44_180', '44_191', '44_200', '44_211', '44_226',
        '44_232', '44_243', '44_244', '44_246', '44_247', '44_249', '44_251', '44_252']

Oregon= ['38_2', '38_3', '38_11', '38_22', '38_24', '38_25', '38_27', '38_28', '38_30', '38_32', '38_33', '38_34']

Idaho= ['13_1', '13_3', '13_6', '13_10', '13_11', '13_13', '13_14', '13_15', '13_16', '13_17', '13_21', '13_23', '13_24',
        '13_25', '13_26', '13_27', '13_28', '13_29', '13_31', '13_32', '13_33', '13_35', '13_38', '13_42', '13_44']

Montana= ['27_2', '27_3', '27_4', '27_5', '27_6', '27_7', '27_8', '27_9', '27_10', '27_11', '27_13', '27_14', '27_15',
          '27_16', '27_17', '27_18', '27_21', '27_23', '27_24', '27_25', '27_26', '27_29', '27_30', '27_33', '27_34', 
          '27_36', '27_37', '27_38', '27_40', '27_42', '27_43', '27_44', '27_48', '27_50', '27_51', '27_52', '27_53', '27_56']


others = Texas + Oregon + Idaho + Montana
for id_in in others:
    df_in = pd.DataFrame(columns=["county_id", "coverage_id", "mean"])
    df_in = soil_data_id(id_in, df_in)
    df_in.to_csv(f"soil_data_means{id_in}.csv", index=False)