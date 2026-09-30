import ee
import datetime
import pandas as pd
import time
import os
# 1. Point your entire Python environment to the service account key
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "path/to/your-service-account-key.json"

# 2. Initialize Earth Engine using the service account (no browser popup needed)
try:
    ee.Initialize(project='project-master')
    print("Earth Engine initialized via Service Account successfully.")
except Exception as e:
    print(f"Failed to initialize Earth Engine: {e}")

print(ee.String('Hello from the Earth Engine servers HLSS!').getInfo())
print(ee.String('Hello from the Earth Engine servers HLSS!').getInfo())
print(ee.String('Hello from the Earth Engine servers HLSS!').getInfo())



ASSET_PATH = 'projects/project-master/assets/USshape'
custom_counties_us = ee.FeatureCollection(ASSET_PATH)

def run_scout_pass(county_id, year):
    county_shape = custom_counties_us.filter(ee.Filter.eq("id_comb", county_id)).geometry()

    start_date = datetime.date(year, 4, 7).isoformat()
    end_date = datetime.date(year, 9, 2).isoformat()

    # USDA CDL WHEAT MASK
    cdl = ee.Image(f"USDA/NASS/CDL/{year}")
    crop_band = cdl.select('cropland')
    wheat_mask = crop_band.eq(24).Or(crop_band.eq(23)).Or(crop_band.eq(22))

    # 1km Base Grid
    def add_centroids(feature):
        centroid = feature.geometry().centroid(maxError=1)
        return feature.set({
            'longitude': centroid.coordinates().get(0),
            'latitude': centroid.coordinates().get(1)
        })

    base_grid = county_shape.coveringGrid('EPSG:5070', 1000).map(add_centroids)
    # THE STATIC SPATIAL GATE: Find which cells actually have wheat
    static_wheat_count = wheat_mask.reduceRegions(
        collection=base_grid,
        reducer=ee.Reducer.sum().setOutputs(['static_sum']),
        scale=30,
        crs='EPSG:5070',
        tileScale=4
    )

    # Drop grid cells with zero (or negligible) wheat pixels permanently if less than 1%(11 pixels) of grid is wheat drop it
    active_grid = static_wheat_count.filter(ee.Filter.gt('static_sum', 11) )
    # THE SCOUT FUNCTION: Only fetch and evaluate Fmask
    def scout_clouds(img):
        qa = img.select('Fmask')
        clear_sky = qa.bitwiseAnd(15).eq(0).And(qa.bitwiseAnd(192).neq(192))

        # Create a binary matrix: 1 if clear AND wheat, 0 otherwise
        clear_wheat = clear_sky.updateMask(wheat_mask).rename('clear_wheat_pixels')
        image_id = img.get('system:index')

        # Count the clear wheat pixels in each cell
        stats = clear_wheat.reduceRegions(
            collection=active_grid,
            reducer=ee.Reducer.sum(),
            scale=30,
            crs='EPSG:5070',
            tileScale=4
        )

        # DROP EMPTY CELLS: If sum is 0, the cell is entirely clouded or non-wheat today
        valid_cells = stats.filter(ee.Filter.gt('sum', 0))

        return valid_cells.map(lambda f: f.set({'image_id': image_id}))

    # Apply the 85% metadata filter
    hls_scout = ee.ImageCollection("NASA/HLS/HLSS30/v002") \
                    .filterBounds(county_shape) \
                    .filterDate(start_date, end_date) \
                    .filter(ee.Filter.lt('CLOUD_COVERAGE', 85))

    scout_table = ee.FeatureCollection(hls_scout.map(scout_clouds)).flatten()

    # EXPORT SCOUT DATA
    export_columns = ['image_id', 'longitude', 'latitude', 'sum']

    task = ee.batch.Export.table.toCloudStorage(
        collection=scout_table,
        description=f'Scout_County_HLSS_{county_id}_{year}',
        outputBucket='example-wheat2020',  # <--- THIS IS THE FIX
        fileNamePrefix=f'Scouts/Scout_County_HLSS_{county_id}_{year}',
        fileFormat='CSV',
        selectors=export_columns
    )

    task.start()
    print(f"Scout Task Started for County {county_id} ({year})!")
    return task

# # harvest with percentiles and ndvi


def run_harvester_ndvi(county_id, year, scout_csv_path):
    # 2. Load the Scout CSV locally
    print(f"Loading Scout Data from {scout_csv_path}...")
    df = pd.read_csv(scout_csv_path)

    # 3. Rebuild the static CDL Wheat Mask
    cdl = ee.Image(f"USDA/NASS/CDL/{year}")
    crop_band = cdl.select('cropland')
    wheat_mask = crop_band.eq(24).Or(crop_band.eq(23)).Or(crop_band.eq(22))

    # 4. Group the data by specific satellite image
    grouped_images = df.groupby('image_id')
    print(f"Found {len(grouped_images)} unique satellite overpasses to harvest.")

    tasks = []

    # 5. Dispatch a targeted task for each image
    for i, (image_id, group) in enumerate(grouped_images):

        coords = group[['longitude', 'latitude']].values.tolist()

        ee_features = [
            ee.Feature(
                ee.Geometry.Point([lon, lat], proj='EPSG:4326')
                  .transform('EPSG:5070')
                  .buffer(500, proj='EPSG:5070')
                  .bounds(proj='EPSG:5070'),
                {'longitude': lon, 'latitude': lat}
            )
            for lon, lat in coords
        ]
        target_grid = ee.FeatureCollection(ee_features)

        img = ee.ImageCollection("NASA/HLS/HLSS30/v002") \
                .filter(ee.Filter.eq('system:index', image_id)) \
                .first()
        if img is None or img.getInfo() is None:
            print(f"Warning: Image {image_id} not found. Skipping.")
            continue
        qa = img.select('Fmask')
        clear_sky = qa.bitwiseAnd(15).eq(0).And(qa.bitwiseAnd(192).neq(192))

        # --- ADDITION 1: Calculate Agronomic Indices ---
        # HLS Bands: B5 is NIR, B4 is Red, B3 is Green
        ndvi = img.normalizedDifference(['B8A', 'B4']).rename('NDVI')
        gcvi = img.expression(
            '(NIR / GREEN) - 1',
            {'NIR': img.select('B8A'), 'GREEN': img.select('B3')}
        ).rename('GCVI')

        # Cleaned up masking logic & append new index bands
        spectral = img.select(["B2", "B3", "B4", "B8A", "B11", "B12"]) \
                      .addBands([ndvi, gcvi]) \
                      .updateMask(clear_sky)

        # Safely chain the cloud band, THEN apply the wheat mask
        combined = spectral.updateMask(wheat_mask)

        # Define the percentile list
        percentiles = ee.Reducer.percentile([10, 20, 30, 40, 50, 60, 70, 80, 90])

        # --- ADDITION 2: Add Skew and Kurtosis to Reducer ---
        combined_reducer = ee.Reducer.mean() \
            .combine(reducer2=ee.Reducer.stdDev(), sharedInputs=True) \
            .combine(reducer2=ee.Reducer.count(), sharedInputs=True) \
            .combine(reducer2=ee.Reducer.skew(), sharedInputs=True) \
            .combine(reducer2=ee.Reducer.kurtosis(), sharedInputs=True) \
            .combine(reducer2=percentiles, sharedInputs=True)

        stats = combined.reduceRegions(
            collection=target_grid,
            reducer=combined_reducer,
            scale=30,
            crs='EPSG:5070',
            tileScale=4
        )

        # Restored the missing final_table variable
        final_table = stats.filter(ee.Filter.gt('B2_count', 0)) \
                           .map(lambda f: f.set({'image_id': image_id}))

        # --- Expanded Export Columns ---
        export_columns = ['image_id', 'longitude', 'latitude']

        # --- ADDITION 3: Expand the loops to catch the new bands and metrics ---
        bands = ['B2', 'B3', 'B4', 'B8A', 'B11', 'B12', 'NDVI', 'GCVI']
        metrics = ['mean', 'stdDev', 'count', 'skew', 'kurtosis', 'p10', 'p20', 'p30', 'p40', 'p50', 'p60', 'p70', 'p80', 'p90']

        for b in bands:
            for m in metrics:
                export_columns.append(f"{b}_{m}")


        safe_image_id = str(image_id).replace('.', '_').replace('/', '_')
        safe_image_id += str(i)

        # Launch the mini-task
        task = ee.batch.Export.table.toDrive(
            collection=final_table,
            description=f'Harvester_{year}_{county_id}_{safe_image_id}',
            folder=f'Wheat_Harvester_{year}',
            fileFormat='CSV',
            selectors=export_columns
        )

        task.start()
        tasks.append(task)

    # Unindented to run once at the end
    print(f"Successfully dispatched {len(tasks)} targeted tasks to Earth Engine.")
    return tasks


def wait_for_tasks(task_list, poll_interval=60.0):
    """Pauses the script until all GEE tasks in the list are finished."""
    print(f"Tracking {len(task_list)} tasks...")
    
    while True:
        statuses = [task.status()['state'] for task in task_list]
        
        if all(state in ['COMPLETED', 'FAILED', 'CANCELLED'] for state in statuses):
            print("All tasks finished processing on Earth Engine.")
            
            # Print a quick summary of any failures
            failures = statuses.count('FAILED')
            if failures > 0:
                print(f"WARNING: {failures} tasks failed.")
            break
            
        print(f"Tasks still running... waiting {poll_interval} seconds.")
        time.sleep(poll_interval)

def master_pipeline(county_id, year):
    # ---------------------------------------------------------
    # PHASE 1: DISPATCH THE SCOUT
    # ---------------------------------------------------------
    expected_csv_path =f'gs://example-wheat2020/Scouts/Scout_County_HLSS_{county_id}_{year}.csv'
    print("--- STARTING PHASE 1: SCOUT ---")
    scout_exists = False
    try:
        pd.read_csv(expected_csv_path, nrows=1)
        scout_exists = True
        print("Scout file already exists in Cloud Storage. Skipping Phase 1 and 2.")
        return True
    except Exception:
        scout_exists = False
        
    scout_task = run_scout_pass(county_id, year) # Your Pass 1 function

    # Halt the Python script until Google finishes the Scout
    wait_for_tasks([scout_task], poll_interval=5.0)
    
    # ---------------------------------------------------------
    # THE BRIDGE: WAIT FOR GOOGLE DRIVE TO SYNC
    # ---------------------------------------------------------
    # If running locally, Google Drive Desktop takes a moment to download the CSV.
    # We force Python to wait until the file actually exists on your hard drive.
    
    print("Waiting for Scout CSV to appear")
    while not scout_exists:
        try:
            pd.read_csv(expected_csv_path, nrows=1)
            scout_exists = True
        except Exception:
                print("Scout file doesnt exist in Cloud Storage. Start waiting.")
                time.sleep(1.0)


        
    # ---------------------------------------------------------
    # PHASE 2: DISPATCH THE HARVESTER
    # ---------------------------------------------------------
    print("--- STARTING PHASE 2: HARVESTER ---")
    harvester_tasks = run_harvester_ndvi(county_id, year, expected_csv_path)
    print("--- PIPELINE COMPLETE ---")
    return True

data2025_2 = ['13_1', '13_6', '13_11', '13_14', '13_15', '13_16', '13_21', '13_23', '13_24', '13_25', '13_29', '13_35', '13_44', '27_2', '27_3', '27_4', '27_7', '27_8', '27_10', '27_13', '27_14', '27_15', '27_16', '27_21', '27_23', '27_24', '27_26', '27_29', '27_34', '27_36', '27_37', '27_42', '27_43', '27_50', '27_51', '27_52', '27_53', '27_56', '38_3', '38_11', '38_22', '38_24', '38_25', '38_27', '38_28', '38_30', '38_32', '38_33', '38_34', '44_6', '44_14', '44_18', '44_30', '44_33', '44_48', '44_49', '44_50', '44_56', '44_59', '44_61', '44_71', '44_73', '44_90', '44_91', '44_95', '44_117', '44_126', '44_140', '44_153', '44_163', '44_171', '44_179', '44_180', '44_200', '44_226', '44_232', '44_244']

for county in data2025_2:
    master_pipeline(county, 2025)
