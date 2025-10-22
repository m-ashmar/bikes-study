import pandas as pd
import geopandas as gpd
import numpy as np
from shapely.geometry import Point
from pathlib import Path
from scipy.spatial import cKDTree
import warnings
warnings.filterwarnings('ignore')

def load_cleaned_datasets():
    """
    Load all cleaned datasets from the cleaned_data directory
    """
    print("📂 Loading cleaned datasets...")
    data_path = Path("cleaned_data")
    
    datasets = {}
    
    # Load regular DataFrames
    datasets['daily_bikes_trips'] = pd.read_parquet(data_path / "daily_bikes_trips.parquet")
    datasets['daily_weather_info'] = pd.read_parquet(data_path / "daily_weather_info.parquet")
    datasets['stations_info_dataset'] = pd.read_parquet(data_path / "stations_info_dataset.parquet")
    datasets['stations_table'] = pd.read_parquet(data_path / "stations_table.parquet")
    datasets['metro_bus_stops'] = pd.read_parquet(data_path / "metro_bus_stops.parquet")
    datasets['shuttle_bus_stops'] = pd.read_parquet(data_path / "shuttle_bus_stops.parquet")
    
    # Load GeoDataFrames
    datasets['cbd_polygon'] = gpd.read_file(data_path / "cbd_polygon.geojson")
    datasets['parking_zones'] = gpd.read_file(data_path / "parking_zones.geojson")
    
    print(f"✅ Loaded {len(datasets)} datasets")
    return datasets

def create_ultra_mini_sample(trips_df, sample_size=1000):
    """
    Create a very small sample for testing
    """
    print(f"🎯 Creating ultra-mini sample of {sample_size:,} trips...")
    
    # Sample randomly
    sample_df = trips_df.sample(n=sample_size, random_state=42)
    
    print(f"  Original trips: {len(trips_df):,}")
    print(f"  Sample trips: {len(sample_df):,}")
    
    return sample_df

def ensure_crs_4326(gdf, name):
    """
    Ensure GeoDataFrame is in EPSG:4326 CRS
    """
    if gdf.crs != 'EPSG:4326':
        print(f"  Converting {name} from {gdf.crs} to EPSG:4326")
        gdf = gdf.to_crs('EPSG:4326')
    else:
        print(f"  {name} already in EPSG:4326")
    return gdf

def create_trip_points_gdf(trips_df):
    """
    Create GeoDataFrames for trip start and end points
    """
    print("📍 Creating trip point geometries...")
    
    # Create start points
    start_points = trips_df[['ride_id', 'start_lat', 'start_lng']].copy()
    start_points = start_points.dropna(subset=['start_lat', 'start_lng'])
    start_points['geometry'] = [Point(lng, lat) for lng, lat in zip(start_points['start_lng'], start_points['start_lat'])]
    start_gdf = gpd.GeoDataFrame(start_points, crs='EPSG:4326')
    
    # Create end points
    end_points = trips_df[['ride_id', 'end_lat', 'end_lng']].copy()
    end_points = end_points.dropna(subset=['end_lat', 'end_lng'])
    end_points['geometry'] = [Point(lng, lat) for lng, lat in zip(end_points['end_lng'], end_points['end_lat'])]
    end_gdf = gpd.GeoDataFrame(end_points, crs='EPSG:4326')
    
    print(f"  Start points: {len(start_gdf):,}")
    print(f"  End points: {len(end_gdf):,}")
    
    return start_gdf, end_gdf

def merge_station_data(trips_df, stations_info, stations_table):
    """
    Merge station information with trip data
    """
    print("🏢 Merging station data...")
    
    # Merge start station info
    start_merged = trips_df.merge(
        stations_info[['station_id', 'name', 'station_type', 'capacity', 'region_name']], 
        left_on='start_station_id', 
        right_on='station_id', 
        how='left',
        suffixes=('', '_start')
    )
    
    # Rename start station columns
    start_merged = start_merged.rename(columns={
        'name': 'start_station_name_full',
        'station_type': 'start_station_type',
        'capacity': 'start_station_capacity',
        'region_name': 'start_region_name'
    })
    
    # Merge end station info
    final_merged = start_merged.merge(
        stations_info[['station_id', 'name', 'station_type', 'capacity', 'region_name']], 
        left_on='end_station_id', 
        right_on='station_id', 
        how='left',
        suffixes=('', '_end')
    )
    
    # Rename end station columns
    final_merged = final_merged.rename(columns={
        'name': 'end_station_name_full',
        'station_type': 'end_station_type',
        'capacity': 'end_station_capacity',
        'region_name': 'end_region_name'
    })
    
    # Drop duplicate station_id columns
    final_merged = final_merged.drop(columns=['station_id', 'station_id_end'])
    
    print(f"  Original trips: {len(trips_df):,}")
    print(f"  After station merge: {len(final_merged):,}")
    
    return final_merged

def spatial_join_with_polygons(start_gdf, end_gdf, cbd_polygon, parking_zones):
    """
    Perform spatial joins with polygon data (CBD and parking zones)
    """
    print("🏙️ Performing spatial joins with polygons...")
    
    # Ensure CRS consistency
    cbd_polygon = ensure_crs_4326(cbd_polygon, "CBD Polygon")
    parking_zones = ensure_crs_4326(parking_zones, "Parking Zones")
    
    # Spatial join with CBD
    start_cbd = gpd.sjoin(start_gdf, cbd_polygon[['geometry']], how='left', predicate='within')
    end_cbd = gpd.sjoin(end_gdf, cbd_polygon[['geometry']], how='left', predicate='within')
    
    # Spatial join with parking zones
    start_parking = gpd.sjoin(start_gdf, parking_zones[['name', 'rpp_zone', 'geometry']], how='left', predicate='within')
    end_parking = gpd.sjoin(end_gdf, parking_zones[['name', 'rpp_zone', 'geometry']], how='left', predicate='within')
    
    # Create result DataFrames
    start_results = pd.DataFrame({
        'ride_id': start_gdf['ride_id'],
        'start_in_cbd': start_cbd['index_right'].notna(),
        'start_parking_zone': start_parking['name'],
        'start_rpp_zone': start_parking['rpp_zone']
    })
    
    end_results = pd.DataFrame({
        'ride_id': end_gdf['ride_id'],
        'end_in_cbd': end_cbd['index_right'].notna(),
        'end_parking_zone': end_parking['name'],
        'end_rpp_zone': end_parking['rpp_zone']
    })
    
    print(f"  Trips starting in CBD: {start_results['start_in_cbd'].sum():,}")
    print(f"  Trips ending in CBD: {end_results['end_in_cbd'].sum():,}")
    
    return start_results, end_results

def calculate_nearest_bus_stops_fast(start_gdf, end_gdf, metro_bus_stops, shuttle_bus_stops):
    """
    Calculate distance to nearest bus stops using spatial indexing
    """
    print("🚌 Calculating nearest bus stops (fast version)...")
    
    # Create GeoDataFrames for bus stops
    metro_gdf = gpd.GeoDataFrame(
        metro_bus_stops[['bstp_lon', 'bstp_lat']].dropna(),
        geometry=[Point(lng, lat) for lng, lat in zip(metro_bus_stops['bstp_lon'].dropna(), metro_bus_stops['bstp_lat'].dropna())],
        crs='EPSG:4326'
    )
    
    shuttle_gdf = gpd.GeoDataFrame(
        shuttle_bus_stops[['longitude', 'latitude']].dropna(),
        geometry=[Point(lng, lat) for lng, lat in zip(shuttle_bus_stops['longitude'].dropna(), shuttle_bus_stops['latitude'].dropna())],
        crs='EPSG:4326'
    )
    
    print(f"  Metro bus stops: {len(metro_gdf):,}")
    print(f"  Shuttle bus stops: {len(shuttle_gdf):,}")
    
    # Convert to arrays for faster computation
    def get_nearest_distance_fast(points_gdf, stops_gdf):
        # Extract coordinates
        points_coords = np.array([[p.x, p.y] for p in points_gdf.geometry])
        stops_coords = np.array([[s.x, s.y] for s in stops_gdf.geometry])
        
        # Build spatial index
        tree = cKDTree(stops_coords)
        
        # Find nearest distances
        distances, _ = tree.query(points_coords, k=1)
        
        return distances
    
    # Calculate distances for start points
    print("  Calculating start point distances...")
    start_metro_dist = get_nearest_distance_fast(start_gdf, metro_gdf)
    start_shuttle_dist = get_nearest_distance_fast(start_gdf, shuttle_gdf)
    
    # Calculate distances for end points
    print("  Calculating end point distances...")
    end_metro_dist = get_nearest_distance_fast(end_gdf, metro_gdf)
    end_shuttle_dist = get_nearest_distance_fast(end_gdf, shuttle_gdf)
    
    # Create results
    start_bus_results = pd.DataFrame({
        'ride_id': start_gdf['ride_id'],
        'start_nearest_metro_m': start_metro_dist,
        'start_nearest_shuttle_m': start_shuttle_dist
    })
    
    end_bus_results = pd.DataFrame({
        'ride_id': end_gdf['ride_id'],
        'end_nearest_metro_m': end_metro_dist,
        'end_nearest_shuttle_m': end_shuttle_dist
    })
    
    print(f"  Calculated distances for {len(start_gdf):,} start points")
    print(f"  Calculated distances for {len(end_gdf):,} end points")
    
    return start_bus_results, end_bus_results

def merge_weather_data(trips_df, weather_df):
    """
    Merge weather data by date
    """
    print("🌤️ Merging weather data...")
    
    # Extract date from trip start time
    trips_df['trip_date'] = trips_df['started_at'].dt.date
    
    # Extract date from weather datetime
    weather_df['weather_date'] = weather_df['datetime'].dt.date
    
    # Merge on date
    merged = trips_df.merge(
        weather_df.drop('datetime', axis=1),
        left_on='trip_date',
        right_on='weather_date',
        how='left'
    )
    
    # Drop temporary date columns
    merged = merged.drop(['trip_date', 'weather_date'], axis=1)
    
    print(f"  Original trips: {len(trips_df):,}")
    print(f"  After weather merge: {len(merged):,}")
    
    return merged

def merge_ultra_mini_spatial_data():
    """
    Main function to merge all datasets with ultra-mini sample
    """
    print("🚀 Starting ULTRA-MINI spatial data merging process...")
    print("=" * 60)
    
    # Load all datasets
    datasets = load_cleaned_datasets()
    
    # Get main datasets
    trips_df = datasets['daily_bikes_trips']
    weather_df = datasets['daily_weather_info']
    stations_info = datasets['stations_info_dataset']
    stations_table = datasets['stations_table']
    cbd_polygon = datasets['cbd_polygon']
    parking_zones = datasets['parking_zones']
    metro_bus_stops = datasets['metro_bus_stops']
    shuttle_bus_stops = datasets['shuttle_bus_stops']
    
    # Create ultra-mini sample
    trips_df = create_ultra_mini_sample(trips_df, sample_size=1000)
    
    print(f"\n📊 Sample trip count: {len(trips_df):,}")
    
    # Step 1: Merge station data
    trips_df = merge_station_data(trips_df, stations_info, stations_table)
    
    # Step 2: Create point geometries for spatial operations
    start_gdf, end_gdf = create_trip_points_gdf(trips_df)
    
    # Step 3: Spatial joins with polygons
    start_spatial, end_spatial = spatial_join_with_polygons(start_gdf, end_gdf, cbd_polygon, parking_zones)
    
    # Step 4: Calculate nearest bus stops (fast version)
    start_bus, end_bus = calculate_nearest_bus_stops_fast(start_gdf, end_gdf, metro_bus_stops, shuttle_bus_stops)
    
    # Step 5: Merge spatial results back to main dataset
    print("🔗 Merging spatial results...")
    
    # Merge start spatial data
    trips_df = trips_df.merge(start_spatial, on='ride_id', how='left')
    trips_df = trips_df.merge(start_bus, on='ride_id', how='left')
    
    # Merge end spatial data
    trips_df = trips_df.merge(end_spatial, on='ride_id', how='left')
    trips_df = trips_df.merge(end_bus, on='ride_id', how='left')
    
    # Step 6: Merge weather data
    trips_df = merge_weather_data(trips_df, weather_df)
    
    # Step 7: Save merged dataset
    print("\n💾 Saving ultra-mini merged dataset...")
    output_dir = Path("merged_data")
    output_dir.mkdir(exist_ok=True)
    
    output_path = output_dir / "ultra_mini_merged_sample.parquet"
    trips_df.to_parquet(output_path, index=False)
    
    print(f"✅ Saved ultra-mini merged dataset to: {output_path}")
    print(f"📊 Final dataset shape: {trips_df.shape}")
    
    return trips_df

def validate_ultra_mini_data(merged_df, original_count):
    """
    Validate the ultra-mini merged dataset
    """
    print("\n" + "=" * 60)
    print("🧪 ULTRA-MINI VALIDATION RESULTS")
    print("=" * 60)
    
    # Check row count
    print(f"📊 Row count validation:")
    print(f"  Original sample: {original_count:,}")
    print(f"  Merged sample: {len(merged_df):,}")
    print(f"  Data loss: {original_count - len(merged_df):,} ({((original_count - len(merged_df))/original_count)*100:.2f}%)")
    
    # Check new columns
    new_columns = [
        'start_station_name_full', 'start_station_type', 'start_station_capacity', 'start_region_name',
        'end_station_name_full', 'end_station_type', 'end_station_capacity', 'end_region_name',
        'start_in_cbd', 'start_parking_zone', 'start_rpp_zone',
        'end_in_cbd', 'end_parking_zone', 'end_rpp_zone',
        'start_nearest_metro_m', 'start_nearest_shuttle_m',
        'end_nearest_metro_m', 'end_nearest_shuttle_m',
        'name', 'tempmax', 'tempmin', 'temp', 'humidity', 'windspeed', 'conditions'
    ]
    
    print(f"\n📋 New columns added:")
    for col in new_columns:
        if col in merged_df.columns:
            null_count = merged_df[col].isnull().sum()
            null_pct = (null_count / len(merged_df)) * 100
            print(f"  ✅ {col}: {null_count:,} nulls ({null_pct:.1f}%)")
        else:
            print(f"  ❌ {col}: NOT FOUND")
    
    # Check spatial data quality
    print(f"\n🗺️ Spatial data quality:")
    cbd_trips = merged_df['start_in_cbd'].sum() + merged_df['end_in_cbd'].sum()
    print(f"  CBD trips: {cbd_trips:,}")
    
    parking_trips = merged_df['start_parking_zone'].notna().sum() + merged_df['end_parking_zone'].notna().sum()
    print(f"  Parking zone trips: {parking_trips:,}")
    
    metro_trips = merged_df['start_nearest_metro_m'].notna().sum() + merged_df['end_nearest_metro_m'].notna().sum()
    print(f"  Metro proximity trips: {metro_trips:,}")
    
    # Check weather data
    weather_trips = merged_df['temp'].notna().sum()
    print(f"  Weather data trips: {weather_trips:,} ({weather_trips/len(merged_df)*100:.1f}%)")
    
    # Show sample of results
    print(f"\n📋 Sample of merged data:")
    print(merged_df[['ride_id', 'start_in_cbd', 'end_in_cbd', 'start_nearest_metro_m', 'temp']].head())
    
    return True

if __name__ == "__main__":
    # Load original count for validation
    original_trips = pd.read_parquet("cleaned_data/daily_bikes_trips.parquet")
    original_sample = original_trips.sample(n=1000, random_state=42)
    original_count = len(original_sample)
    
    # Perform merging
    merged_df = merge_ultra_mini_spatial_data()
    
    # Validate results
    validate_ultra_mini_data(merged_df, original_count)
    
    print("\n🎯 Ultra-mini spatial merging completed successfully!")
    print("✅ Ready to scale up to full dataset!") 