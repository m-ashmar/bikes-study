import pandas as pd
import geopandas as gpd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def comprehensive_validation():
    """
    Comprehensive validation of the enriched dataset
    """
    print("🔍 BLOCK 2 FINAL VALIDATION CHECK")
    print("=" * 60)
    
    # 1. Load the enriched dataset
    print("📂 Loading enriched dataset...")
    merged_df = pd.read_parquet("merged_data/ultra_mini_merged_sample.parquet")
    
    # 2. Row count validation
    print(f"\n📊 Row count validation:")
    print(f"  Expected: 1,000")
    print(f"  Actual: {len(merged_df):,}")
    row_check = len(merged_df) == 1000
    print(f"  ✅ Row count check: {'PASS' if row_check else 'FAIL'}")
    
    # 3. Column count and types validation
    print(f"\n📋 Column validation:")
    print(f"  Expected columns: 47")
    print(f"  Actual columns: {len(merged_df.columns)}")
    col_check = len(merged_df.columns) == 47
    print(f"  ✅ Column count check: {'PASS' if col_check else 'FAIL'}")
    
    # 4. Column types analysis
    print(f"\n🔍 Column types analysis:")
    dtype_counts = merged_df.dtypes.value_counts()
    for dtype, count in dtype_counts.items():
        print(f"  {dtype}: {count} columns")
    
    # Check for unexpected object columns
    object_cols = merged_df.select_dtypes(include=['object']).columns.tolist()
    print(f"\n📋 Object columns (should be minimal):")
    for col in object_cols:
        null_pct = merged_df[col].isnull().sum() / len(merged_df) * 100
        print(f"  {col}: {null_pct:.1f}% nulls")
    
    # 5. CRS validation for spatial data
    print(f"\n🗺️ CRS validation:")
    
    # Load original spatial datasets to check CRS
    cbd_polygon = gpd.read_file("cleaned_data/cbd_polygon.geojson")
    parking_zones = gpd.read_file("cleaned_data/parking_zones.geojson")
    
    print(f"  CBD Polygon CRS: {cbd_polygon.crs}")
    print(f"  Parking Zones CRS: {parking_zones.crs}")
    
    crs_check = str(cbd_polygon.crs) == 'EPSG:4326' and str(parking_zones.crs) == 'EPSG:4326'
    print(f"  ✅ CRS check: {'PASS' if crs_check else 'FAIL'}")
    
    # 6. Data quality checks
    print(f"\n🧪 Data quality checks:")
    
    # Trip duration computation
    merged_df['trip_duration'] = (merged_df['ended_at'] - merged_df['started_at']).dt.total_seconds() / 60
    duration_check = merged_df['trip_duration'].notna().all()
    print(f"  ✅ Trip duration computable: {'PASS' if duration_check else 'FAIL'}")
    print(f"    Duration range: {merged_df['trip_duration'].min():.1f} to {merged_df['trip_duration'].max():.1f} minutes")
    
    # Weather data validation
    weather_check = merged_df['temp'].notna().all()
    print(f"  ✅ Weather data complete: {'PASS' if weather_check else 'FAIL'}")
    print(f"    Temperature range: {merged_df['temp'].min():.1f}°C to {merged_df['temp'].max():.1f}°C")
    
    # Distance data validation
    distance_check = merged_df['start_nearest_metro_m'].notna().all()
    print(f"  ✅ Distance data complete: {'PASS' if distance_check else 'FAIL'}")
    print(f"    Metro distance range: {merged_df['start_nearest_metro_m'].min():.6f} to {merged_df['start_nearest_metro_m'].max():.6f} degrees")
    
    # CBD data validation
    cbd_check = merged_df['start_in_cbd'].dtype == 'bool' and merged_df['end_in_cbd'].dtype == 'bool'
    print(f"  ✅ CBD data valid: {'PASS' if cbd_check else 'FAIL'}")
    print(f"    CBD trips: {merged_df['start_in_cbd'].sum() + merged_df['end_in_cbd'].sum()}")
    
    # 7. Sample data verification
    print(f"\n📋 Sample data verification:")
    sample_cols = ['ride_id', 'started_at', 'ended_at', 'start_in_cbd', 'end_in_cbd', 
                  'start_nearest_metro_m', 'temp', 'conditions']
    print(merged_df[sample_cols].head())
    
    # 8. Memory and file validation
    print(f"\n💾 File and memory validation:")
    
    # Check merged_data directory
    merged_dir = Path("merged_data")
    files = list(merged_dir.glob("*.parquet"))
    print(f"  Files in merged_data/: {[f.name for f in files]}")
    
    # File size check
    file_size = merged_dir / "ultra_mini_merged_sample.parquet"
    if file_size.exists():
        size_mb = file_size.stat().st_size / 1024 / 1024
        print(f"  File size: {size_mb:.2f} MB")
    
    # 9. Overall validation summary
    print(f"\n" + "=" * 60)
    print("🎯 OVERALL VALIDATION SUMMARY")
    print("=" * 60)
    
    all_checks = [
        row_check,
        col_check,
        crs_check,
        duration_check,
        weather_check,
        distance_check,
        cbd_check
    ]
    
    passed_checks = sum(all_checks)
    total_checks = len(all_checks)
    
    print(f"✅ Passed: {passed_checks}/{total_checks} checks")
    
    if passed_checks == total_checks:
        print("🎉 ALL VALIDATION CHECKS PASSED!")
        print("✅ Dataset is ready for Block 3 feature engineering!")
        return True
    else:
        print("❌ Some validation checks failed!")
        return False

def save_samples_for_testing():
    """
    Save sample datasets for future testing
    """
    print("\n💾 Saving sample datasets for future testing...")
    
    # Create samples directory
    samples_dir = Path("sample_data")
    samples_dir.mkdir(exist_ok=True)
    
    # Load original cleaned datasets
    print("  Loading original datasets...")
    trips_df = pd.read_parquet("cleaned_data/daily_bikes_trips.parquet")
    weather_df = pd.read_parquet("cleaned_data/daily_weather_info.parquet")
    stations_info = pd.read_parquet("cleaned_data/stations_info_dataset.parquet")
    
    # Create samples
    print("  Creating samples...")
    
    # Sample 1: Small trip sample (1,000 trips)
    trips_sample = trips_df.sample(n=1000, random_state=42)
    trips_sample.to_parquet(samples_dir / "trips_sample_1000.parquet", index=False)
    print(f"    ✅ Saved trips_sample_1000.parquet ({len(trips_sample):,} trips)")
    
    # Sample 2: Medium trip sample (5,000 trips)
    trips_sample_5k = trips_df.sample(n=5000, random_state=42)
    trips_sample_5k.to_parquet(samples_dir / "trips_sample_5000.parquet", index=False)
    print(f"    ✅ Saved trips_sample_5000.parquet ({len(trips_sample_5k):,} trips)")
    
    # Sample 3: Weather sample (all weather data)
    weather_df.to_parquet(samples_dir / "weather_sample.parquet", index=False)
    print(f"    ✅ Saved weather_sample.parquet ({len(weather_df):,} records)")
    
    # Sample 4: Stations sample (all stations)
    stations_info.to_parquet(samples_dir / "stations_sample.parquet", index=False)
    print(f"    ✅ Saved stations_sample.parquet ({len(stations_info):,} stations)")
    
    # Sample 5: Enriched sample (already created)
    enriched_sample = pd.read_parquet("merged_data/ultra_mini_merged_sample.parquet")
    enriched_sample.to_parquet(samples_dir / "enriched_sample_1000.parquet", index=False)
    print(f"    ✅ Saved enriched_sample_1000.parquet ({len(enriched_sample):,} trips)")
    
    print(f"\n📁 Sample data saved to: {samples_dir}")
    print(f"📊 Total sample files: {len(list(samples_dir.glob('*.parquet')))}")
    
    return True

if __name__ == "__main__":
    # Run comprehensive validation
    validation_passed = comprehensive_validation()
    
    # Save samples for future testing
    save_samples_for_testing()
    
    if validation_passed:
        print("\n🚀 READY FOR BLOCK 3!")
    else:
        print("\n❌ Validation failed - please fix issues before proceeding.") 