#!/usr/bin/env python3
"""
🚴 BIKE-SHARING COMPLETE ANALYSIS - SINGLE SCRIPT
=================================================

This is the ONE script that runs everything from data loading to final figures.
It executes all blocks in the correct order:

1. Block 1: Data Loading & Merging (spatial_merger_ultra_mini.py)
2. Block 2: Data Validation (block2_final_validation.py)  
3. Block 4: Spatial Features Engineering (spatial_features_engineering.py)
4. Block 6: Time Series Forecasting (block6_time_series_forecasting.py)
5. Block 7: Clustering Analysis (block7_simple.py)

Author: Data Science Team
Date: 2024
"""

import subprocess
import sys
import time
from pathlib import Path
import os

def run_script(script_name, description, is_data_loading=False):
    """
    Run a Python script and handle errors
    """
    print(f"\n{'='*60}")
    if is_data_loading:
        print(f"🚀 BLOCK 1: DATA LOADING & MERGING")
        print(f"📝 {description}")
    else:
        print(f"🚀 RUNNING: {script_name}")
        print(f"📝 {description}")
    print(f"{'='*60}")
    
    start_time = time.time()
    
    try:
        # Run the script
        result = subprocess.run([
            sys.executable, f"main_scripts/{script_name}"
        ], capture_output=True, text=True, cwd=Path.cwd())
        
        # Print output
        if result.stdout:
            print(result.stdout)
        
        # Check for errors
        if result.returncode != 0:
            print(f"❌ ERROR in {script_name}:")
            print(result.stderr)
            return False
        else:
            elapsed_time = time.time() - start_time
            print(f"✅ {script_name} completed successfully in {elapsed_time:.1f} seconds")
            return True
            
    except Exception as e:
        print(f"❌ Failed to run {script_name}: {e}")
        return False

def check_prerequisites():
    """
    Check if required cleaned data files exist
    """
    print("🔍 Checking prerequisites...")
    
    required_files = [
        "cleaned_data/daily_bikes_trips.parquet",
        "cleaned_data/daily_weather_info.parquet", 
        "cleaned_data/stations_info_dataset.parquet",
        "cleaned_data/stations_table.parquet",
        "cleaned_data/metro_bus_stops.parquet",
        "cleaned_data/shuttle_bus_stops.parquet",
        "cleaned_data/cbd_polygon.geojson",
        "cleaned_data/parking_zones.geojson"
    ]
    
    missing_files = []
    for file_path in required_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)
    
    if missing_files:
        print("❌ Missing required cleaned data files:")
        for file in missing_files:
            print(f"   - {file}")
        print("\nPlease ensure all cleaned data files are in the cleaned_data/ directory.")
        return False
    
    print("✅ All required cleaned data files found!")
    return True

def create_output_directories():
    """
    Create necessary output directories
    """
    print("📁 Creating output directories...")
    
    directories = [
        "merged_data",
        "engineered_data",
        "figures/block6",
        "figures/block7", 
        "sample_data"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"   ✅ Created: {directory}")

def main():
    """
    Main execution function - runs everything in order
    """
    print("🚴 BIKE-SHARING COMPLETE ANALYSIS - SINGLE SCRIPT")
    print("=" * 60)
    print("This script will run ALL blocks from data loading to final figures")
    print("Estimated time: 8-15 minutes")
    print("=" * 60)
    
    # Check prerequisites
    if not check_prerequisites():
        print("\n❌ Prerequisites not met. Exiting.")
        return False
    
    # Create output directories
    create_output_directories()
    
    # Define complete execution plan
    execution_plan = [
        {
            "script": "spatial_merger_ultra_mini.py",
            "description": "Load and merge all datasets (1,000 trip sample)",
            "is_data_loading": True
        },
        {
            "script": "block2_final_validation.py",
            "description": "Validate merged dataset quality",
            "is_data_loading": False
        },
        {
            "script": "spatial_features_engineering.py", 
            "description": "Create spatial features (geohash, weather, distance bins)",
            "is_data_loading": False
        },
        {
            "script": "block6_time_series_forecasting.py",
            "description": "Time series forecasting with Prophet",
            "is_data_loading": False
        },
        {
            "script": "block7_simple.py",
            "description": "Clustering analysis and pattern discovery",
            "is_data_loading": False
        }
    ]
    
    # Track success/failure
    results = []
    
    # Execute each block
    for i, block in enumerate(execution_plan, 1):
        print(f"\n📊 BLOCK {i}/{len(execution_plan)}")
        success = run_script(
            block["script"], 
            block["description"], 
            block["is_data_loading"]
        )
        results.append(success)
        
        if not success:
            print(f"\n❌ Block {i} failed. Stopping execution.")
            break
    
    # Summary
    print(f"\n{'='*60}")
    print("📊 COMPLETE EXECUTION SUMMARY")
    print(f"{'='*60}")
    
    successful_blocks = sum(results)
    total_blocks = len(results)
    
    print(f"✅ Successful blocks: {successful_blocks}/{total_blocks}")
    
    if successful_blocks == total_blocks:
        print("🎉 ALL BLOCKS COMPLETED SUCCESSFULLY!")
        print("\n📁 Generated outputs:")
        print("   📂 merged_data/")
        print("      - ultra_mini_merged_sample.parquet (enriched dataset)")
        print("   📂 engineered_data/")
        print("      - spatial_features_block4.parquet (spatial features)")
        print("      - forecast_block6.csv (forecasting results)")
        print("      - clustered_block7.parquet (clustering results)")
        print("   📂 figures/block6/")
        print("      - forecast_plot.png (revenue forecast)")
        print("      - forecast_components.png (forecast components)")
        print("   📂 figures/block7/")
        print("      - kmeans_pca.png (K-Means clustering)")
        print("      - dbscan_tsne.png (DBSCAN clustering)")
        print("      - agglomerative_dendrogram.png (hierarchical clustering)")
        print("\n🚀 Complete analysis pipeline finished!")
        print("📊 All final figures and datasets are ready!")
    else:
        print("❌ Some blocks failed. Check the output above for errors.")
    
    return successful_blocks == total_blocks

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 