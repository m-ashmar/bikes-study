import pandas as pd
import numpy as np
import pygeohash as pgh
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def load_engineered_data():
    """
    Load the engineered dataset from Block 3
    """
    print("📂 Loading engineered dataset...")
    data_path = Path("engineered_data/enriched_block3_fixed.parquet")
    df = pd.read_parquet(data_path)
    print(f"✅ Loaded dataset: {df.shape}")
    return df

def create_geohash_features(df):
    """
    Create geohash encoding with precision 6 and zone classification
    """
    print("🧭 Creating geohash features...")
    
    # Create geohash with precision 6
    df['start_geohash'] = df.apply(
        lambda row: pgh.encode(row['start_lat'], row['start_lng'], precision=6), 
        axis=1
    )
    df['end_geohash'] = df.apply(
        lambda row: pgh.encode(row['end_lat'], row['end_lng'], precision=6), 
        axis=1
    )
    
    # Calculate trip counts per start geohash
    geohash_counts = df['start_geohash'].value_counts()
    
    # Classify geohash zones based on trip volume
    total_geohashes = len(geohash_counts)
    top_25_percent = int(total_geohashes * 0.25)
    bottom_25_percent = int(total_geohashes * 0.25)
    
    # Get thresholds
    high_threshold = geohash_counts.iloc[top_25_percent - 1] if top_25_percent > 0 else geohash_counts.max()
    low_threshold = geohash_counts.iloc[-bottom_25_percent] if bottom_25_percent > 0 else geohash_counts.min()
    
    print(f"  📊 Geohash classification thresholds:")
    print(f"    - High volume (Red): ≥ {high_threshold} trips")
    print(f"    - Low volume (Gray): ≤ {low_threshold} trips")
    
    # Create geohash zone color mapping
    def classify_geohash_zone(geohash):
        count = geohash_counts.get(geohash, 0)
        if count >= high_threshold:
            return 'red'
        elif count <= low_threshold:
            return 'gray'
        else:
            return 'yellow'
    
    df['geohash_zone_color'] = df['start_geohash'].apply(classify_geohash_zone)
    
    # Print distribution
    color_distribution = df['geohash_zone_color'].value_counts()
    print(f"  🎨 Geohash zone distribution:")
    for color, count in color_distribution.items():
        percentage = (count / len(df)) * 100
        print(f"    - {color.capitalize()}: {count:,} trips ({percentage:.1f}%)")
    
    return df

def simplify_weather_categories(df):
    """
    Simplify weather conditions into categories
    """
    print("🌤️ Simplifying weather categories...")
    
    def categorize_weather(condition):
        if pd.isna(condition):
            return 'unknown'
        
        condition_lower = str(condition).lower()
        
        if any(word in condition_lower for word in ['clear', 'sunny', 'fair']):
            return 'sunny'
        elif any(word in condition_lower for word in ['overcast', 'partly', 'cloudy', 'cloud']):
            return 'cloudy'
        elif any(word in condition_lower for word in ['rain', 'storm', 'drizzle', 'shower', 'precipitation']):
            return 'rainy'
        else:
            return 'other'
    
    df['weather_category'] = df['conditions'].apply(categorize_weather)
    
    # Print distribution
    weather_distribution = df['weather_category'].value_counts()
    print(f"  🌤️ Weather category distribution:")
    for category, count in weather_distribution.items():
        percentage = (count / len(df)) * 100
        print(f"    - {category.capitalize()}: {count:,} trips ({percentage:.1f}%)")
    
    return df

def create_distance_bins(df):
    """
    Create distance-based binning for metro and shuttle stops
    """
    print("📏 Creating distance bins...")
    
    # Metro distance binning
    def bin_metro_distance(distance):
        if pd.isna(distance):
            return 'unknown'
        elif distance < 250:
            return '< 250m'
        elif distance <= 500:
            return '250–500m'
        else:
            return '> 500m'
    
    # Shuttle distance binning
    def bin_shuttle_distance(distance):
        if pd.isna(distance):
            return 'unknown'
        elif distance < 250:
            return '< 250m'
        elif distance <= 500:
            return '250–500m'
        else:
            return '> 500m'
    
    df['metro_distance_bin'] = df['start_nearest_metro_m'].apply(bin_metro_distance)
    df['shuttle_distance_bin'] = df['start_nearest_shuttle_m'].apply(bin_shuttle_distance)
    
    # Print distributions
    print(f"  🚇 Metro distance bin distribution:")
    metro_dist = df['metro_distance_bin'].value_counts()
    for bin_name, count in metro_dist.items():
        percentage = (count / len(df)) * 100
        print(f"    - {bin_name}: {count:,} trips ({percentage:.1f}%)")
    
    print(f"  🚌 Shuttle distance bin distribution:")
    shuttle_dist = df['shuttle_distance_bin'].value_counts()
    for bin_name, count in shuttle_dist.items():
        percentage = (count / len(df)) * 100
        print(f"    - {bin_name}: {count:,} trips ({percentage:.1f}%)")
    
    return df

def calculate_cbd_proximity(df):
    """
    Calculate CBD proximity for trips outside CBD
    """
    print("🏙️ Calculating CBD proximity...")
    
    # Use approximate CBD coordinates
    cbd_lat, cbd_lng = 40.7589, -73.9851  # Approximate NYC CBD
    
    def haversine_distance(lat1, lng1, lat2, lng2):
        """
        Calculate Haversine distance between two points
        """
        from math import radians, cos, sin, asin, sqrt
        
        # Convert to radians
        lat1, lng1, lat2, lng2 = map(radians, [lat1, lng1, lat2, lng2])
        
        # Haversine formula
        dlat = lat2 - lat1
        dlng = lng2 - lng1
        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlng/2)**2
        c = 2 * asin(sqrt(a))
        r = 6371  # Radius of earth in kilometers
        return c * r * 1000  # Convert to meters
    
    def classify_cbd_proximity(lat, lng, in_cbd):
        """
        Classify proximity to CBD
        """
        if in_cbd:
            return 'in_cbd'
        
        distance = haversine_distance(lat, lng, cbd_lat, cbd_lng)
        
        if distance <= 500:
            return 'near'
        elif distance <= 1500:
            return 'medium'
        else:
            return 'far'
    
    df['start_to_cbd_proximity'] = df.apply(
        lambda row: classify_cbd_proximity(row['start_lat'], row['start_lng'], row['start_in_cbd']), 
        axis=1
    )
    df['end_to_cbd_proximity'] = df.apply(
        lambda row: classify_cbd_proximity(row['end_lat'], row['end_lng'], row['end_in_cbd']), 
        axis=1
    )
    
    # Print distributions
    print(f"  🏙️ Start to CBD proximity distribution:")
    start_prox_dist = df['start_to_cbd_proximity'].value_counts()
    for proximity, count in start_prox_dist.items():
        percentage = (count / len(df)) * 100
        print(f"    - {proximity}: {count:,} trips ({percentage:.1f}%)")
    
    print(f"  🏙️ End to CBD proximity distribution:")
    end_prox_dist = df['end_to_cbd_proximity'].value_counts()
    for proximity, count in end_prox_dist.items():
        percentage = (count / len(df)) * 100
        print(f"    - {proximity}: {count:,} trips ({percentage:.1f}%)")
    
    return df

def validate_spatial_features(df):
    """
    Validate all spatial features
    """
    print("✅ Validating spatial features...")
    
    tests = []
    
    # Test 1: Geohash precision
    sample_geohash = df['start_geohash'].iloc[0]
    geohash_precision = len(sample_geohash)
    tests.append(('Geohash precision = 6', geohash_precision == 6))
    print(f"  ✅ Geohash precision: {geohash_precision} (expected: 6)")
    
    # Test 2: Geohash zone colors
    zone_colors = set(df['geohash_zone_color'].unique())
    expected_colors = {'red', 'yellow', 'gray'}
    tests.append(('Geohash zone colors', zone_colors == expected_colors))
    print(f"  ✅ Zone colors: {zone_colors}")
    
    # Test 3: Weather categories
    weather_cats = df['weather_category'].unique()
    tests.append(('Weather categories', len(weather_cats) > 0 and df['weather_category'].isnull().sum() == 0))
    print(f"  ✅ Weather categories: {weather_cats}")
    
    # Test 4: Distance bins
    expected_bins = {'< 250m', '250–500m', '> 500m', 'unknown'}
    metro_bins = set(df['metro_distance_bin'].unique())
    shuttle_bins = set(df['shuttle_distance_bin'].unique())
    tests.append(('Distance bins', metro_bins.issubset(expected_bins) and shuttle_bins.issubset(expected_bins)))
    print(f"  ✅ Metro bins: {metro_bins}")
    print(f"  ✅ Shuttle bins: {shuttle_bins}")
    
    # Test 5: CBD proximity
    start_prox = set(df['start_to_cbd_proximity'].unique())
    end_prox = set(df['end_to_cbd_proximity'].unique())
    expected_prox = {'in_cbd', 'near', 'medium', 'far'}
    tests.append(('CBD proximity', start_prox.issubset(expected_prox) and end_prox.issubset(expected_prox)))
    print(f"  ✅ Start proximity: {start_prox}")
    print(f"  ✅ End proximity: {end_prox}")
    
    # Test 6: No nulls in new features
    new_features = ['start_geohash', 'end_geohash', 'geohash_zone_color', 
                   'weather_category', 'metro_distance_bin', 'shuttle_distance_bin',
                   'start_to_cbd_proximity', 'end_to_cbd_proximity']
    
    tests.append(('No nulls in new features', all(df[feature].isnull().sum() == 0 for feature in new_features)))
    print(f"  ✅ Null check: All new features have no null values")
    
    # Print results
    passed = sum(1 for _, result in tests if result)
    print(f"  📋 Validation results: {passed}/{len(tests)} tests passed")
    
    for test_name, result in tests:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"    {test_name}: {status}")
    
    return passed == len(tests)

def save_spatial_features(df):
    """
    Save the spatial features dataset
    """
    print("💾 Saving spatial features dataset...")
    
    # Create output directory
    output_dir = Path("engineered_data")
    output_dir.mkdir(exist_ok=True)
    
    # Save the dataset
    output_path = output_dir / "spatial_features_block4.parquet"
    df.to_parquet(output_path, index=False)
    
    print(f"  ✅ Saved to: {output_path}")
    print(f"  📊 Final shape: {df.shape}")
    print(f"  💾 File size: {output_path.stat().st_size / 1024:.1f} KB")
    
    return output_path

def main():
    """
    Main spatial features engineering pipeline
    """
    print("🚀 BLOCK 4 - Spatial Features Engineering Pipeline")
    print("=" * 60)
    
    # Load data
    df = load_engineered_data()
    
    # Step 1: Geohash encoding
    df = create_geohash_features(df)
    
    # Step 2: Weather simplification
    df = simplify_weather_categories(df)
    
    # Step 3: Distance-based binning
    df = create_distance_bins(df)
    
    # Step 4: CBD proximity classification
    df = calculate_cbd_proximity(df)
    
    # Validate features
    validation_passed = validate_spatial_features(df)
    
    # Save spatial features
    output_path = save_spatial_features(df)
    
    # Final summary
    print("\n" + "=" * 60)
    print("🎯 BLOCK 4 SUMMARY")
    print("=" * 60)
    
    print(f"✅ Spatial features added:")
    print(f"  - Geohash: start_geohash, end_geohash, geohash_zone_color")
    print(f"  - Weather: weather_category")
    print(f"  - Distance bins: metro_distance_bin, shuttle_distance_bin")
    print(f"  - CBD proximity: start_to_cbd_proximity, end_to_cbd_proximity")
    
    print(f"\n📊 Dataset statistics:")
    print(f"  - Shape: {df.shape}")
    print(f"  - New features: 8")
    print(f"  - Total features: {len(df.columns)}")
    
    print(f"\n🧪 Validation results:")
    print(f"  - Feature validation: {'PASS' if validation_passed else 'FAIL'}")
    
    if validation_passed:
        print("\n🎉 BLOCK 4 COMPLETED SUCCESSFULLY!")
        print("✅ Ready for Block 5!")
    else:
        print("\n❌ Some issues found - please review before proceeding.")

if __name__ == "__main__":
    main() 