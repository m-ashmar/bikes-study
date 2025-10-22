#!/usr/bin/env python3
"""
Complete Bike-Sharing Analysis Pipeline
=======================================

This script runs all analysis blocks in the correct order to generate
all final figures and outputs for the bike-sharing project.

Execution Order:
1. Block 2: Data Validation
2. Block 4: Spatial Features Engineering  
3. Block 6: Time Series Forecasting
4. Block 7: Clustering Analysis

Author: Data Science Team
Date: 2024
"""

import subprocess
import sys
import time
from pathlib import Path
import os

def run_script(script_name, description):
    """
    Run a Python script and handle errors
    """
    print(f"\n{'='*60}")
    print(f"🚀 RUNNING: {script_name}")
    print(f"📝 Description: {description}")
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
    Check if required data files exist
    """
    print("🔍 Checking prerequisites...")
    
    required_files = [
        "merged_data/ultra_mini_merged_sample.parquet",
        "cleaned_data/cbd_polygon.geojson",
        "cleaned_data/parking_zones.geojson"
    ]
    
    missing_files = []
    for file_path in required_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)
    
    if missing_files:
        print("❌ Missing required files:")
        for file in missing_files:
            print(f"   - {file}")
        print("\nPlease ensure all data files are in place before running the analysis.")
        return False
    
    print("✅ All required files found!")
    return True

def create_output_directories():
    """
    Create necessary output directories
    """
    print("📁 Creating output directories...")
    
    directories = [
        "figures/block6",
        "figures/block7", 
        "engineered_data",
        "sample_data"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"   ✅ Created: {directory}")

def main():
    """
    Main execution function
    """
    print("🚴 BIKE-SHARING COMPLETE ANALYSIS PIPELINE")
    print("=" * 60)
    print("This will run all analysis blocks in the correct order")
    print("Estimated time: 5-10 minutes")
    print("=" * 60)
    
    # Check prerequisites
    if not check_prerequisites():
        print("\n❌ Prerequisites not met. Exiting.")
        return False
    
    # Create output directories
    create_output_directories()
    
    # Define execution order
    execution_plan = [
        {
            "script": "block2_final_validation.py",
            "description": "Data validation and quality checks"
        },
        {
            "script": "spatial_features_engineering.py", 
            "description": "Spatial features engineering (Block 4)"
        },
        {
            "script": "block6_time_series_forecasting.py",
            "description": "Time series forecasting with Prophet"
        },
        {
            "script": "block7_simple.py",
            "description": "Clustering analysis and pattern discovery"
        }
    ]
    
    # Track success/failure
    results = []
    
    # Execute each block
    for i, block in enumerate(execution_plan, 1):
        print(f"\n📊 BLOCK {i}/{len(execution_plan)}")
        success = run_script(block["script"], block["description"])
        results.append(success)
        
        if not success:
            print(f"\n❌ Block {i} failed. Stopping execution.")
            break
    
    # Summary
    print(f"\n{'='*60}")
    print("📊 EXECUTION SUMMARY")
    print(f"{'='*60}")
    
    successful_blocks = sum(results)
    total_blocks = len(results)
    
    print(f"✅ Successful blocks: {successful_blocks}/{total_blocks}")
    
    if successful_blocks == total_blocks:
        print("🎉 ALL BLOCKS COMPLETED SUCCESSFULLY!")
        print("\n📁 Generated outputs:")
        print("   - figures/block6/ (forecasting plots)")
        print("   - figures/block7/ (clustering plots)")
        print("   - engineered_data/ (processed datasets)")
        print("\n🚀 Analysis pipeline complete!")
    else:
        print("❌ Some blocks failed. Check the output above for errors.")
    
    return successful_blocks == total_blocks

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 