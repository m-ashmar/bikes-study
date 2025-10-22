# Bike-Sharing Data Science Project
## Final Graduation Project - Data Science

---

## 📋 Project Overview

This project analyzes bike-sharing data to understand usage patterns, weather impacts, and profitability factors. The analysis includes data cleaning, feature engineering, spatial analysis, time series forecasting, and pattern discovery through clustering.

### Key Objectives
- Understand temporal and spatial usage patterns
- Analyze weather impact on trip costs and revenue
- Identify high-usage areas and recurring patterns
- Develop revenue forecasting models
- Discover user segments through clustering

---

## Project Structure

```
final_project/
├── README.md
├── requirements.txt
├── run_everything_single_script.py      # Runs full pipeline (Blocks 1,2,4,6,7)
├── run_complete_analysis.py             # Runs Blocks 2,4,6,7 (requires merged sample)
├── main_scripts/                        # Individual block scripts
│   ├── spatial_merger_ultra_mini.py     # Block 1: merge & sample (creates merged_data/)
│   ├── block2_final_validation.py       # Block 2: data validation
│   ├── spatial_features_engineering.py  # Block 4: spatial features
│   ├── block6_time_series_forecasting.py# Block 6: forecasting
│   ├── block7_simple.py                 # Block 7: clustering (simple)
│   ├── block7_clustering.py             # Block 7: clustering (alt)
│   ├── spatial_merger.py                # Full merge (large)
│   └── spatial_merger_supermini.py      # Smaller merge
├── cleaned_data/                        # Input data (not tracked; place files here)
│   ├── daily_bikes_trips.parquet
│   ├── daily_weather_info.parquet
│   ├── stations_info_dataset.parquet
│   ├── stations_table.parquet
│   ├── metro_bus_stops.parquet
│   ├── shuttle_bus_stops.parquet
│   ├── cbd_polygon.geojson
│   └── parking_zones.geojson
├── merged_data/                         # Created by Block 1 (ignored by git)
├── engineered_data/                     # Processed datasets
│   ├── spatial_features_block4.parquet
│   ├── forecast_block6.csv
│   └── clustered_block7.parquet
├── figures/
│   └── final_report/                    # Selected visualizations
├── report_arabic.md
├── report_arabic.pdf
├── BLOCK1_SUMMARY.md
└── tests.py
```

---

##  Quick Start

### Prerequisites
```bash
# Python 3.10+ recommended
python --version

# Create and activate a virtual environment (macOS/zsh)
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Data
Place the following files under `cleaned_data/` (not tracked by git):
- daily_bikes_trips.parquet
- daily_weather_info.parquet
- stations_info_dataset.parquet
- stations_table.parquet
- metro_bus_stops.parquet
- shuttle_bus_stops.parquet
- cbd_polygon.geojson
- parking_zones.geojson

### Run
- Full pipeline (Blocks 1→7):
```bash
python run_everything_single_script.py
```

- If you already have `merged_data/ultra_mini_merged_sample.parquet`:
```bash
python run_complete_analysis.py
```

- Run blocks individually:
```bash
python main_scripts/spatial_merger_ultra_mini.py
python main_scripts/block2_final_validation.py
python main_scripts/spatial_features_engineering.py
python main_scripts/block6_time_series_forecasting.py
python main_scripts/block7_simple.py
```

---

## 📊 Key Findings

### Temporal Patterns
- Peak usage during rush hours (8-9 AM, 5-6 PM)
- Higher usage on weekdays vs weekends
- Seasonal patterns with summer peaks

### Spatial Patterns
- Concentrated usage in central areas
- Regular patterns in residential areas
- Continuous usage in commercial districts

### Weather Impact
- Reduced usage in rainy/cold conditions
- Strong correlation between weather and revenue
- Temperature sensitivity in trip duration

### User Segments
1. **Regular Users**: Short, frequent trips
2. **Long-distance Travelers**: Long, rare trips
3. **Balanced Users**: Moderate trip patterns

### Revenue Forecasting
- Expected annual growth: 15-20%
- Clear seasonal patterns
- Event and holiday impacts

---

##  Technical Details

### Data Processing Pipeline
1. **Data Loading**: Multiple CSV files merged
2. **Cleaning**: Missing values, outliers, format standardization
3. **Feature Engineering**: Temporal, spatial, weather features
4. **Spatial Analysis**: Geographic clustering and proximity
5. **Time Series**: Prophet forecasting with seasonality
6. **Clustering**: K-Means, DBSCAN, Hierarchical methods

### Performance Optimizations
- Mini-dataset approach (10,000 trips) for development
- Parquet format for efficient storage
- Optimized algorithms for large-scale processing
- Comprehensive testing at each stage

### Validation Strategy
- Cross-validation for machine learning models
- Statistical significance testing
- Comparison with full dataset results
- Automated test suites for each block

---

## 📈 Results Summary

| Metric | Value |
|--------|-------|
| Total Trips | 10,000 |
| Average Trip Duration | 18 minutes |
| Average Cost | 3.50 SAR |
| Average Distance | 2.8 km |
| Annual Members | 65% |
| Electric Bikes | 35% |

---

##  Business Insights

### Short-term Recommendations (3-6 months)
- **Distribution Optimization**: Increase bikes in high-usage areas
- **Dynamic Pricing**: Adjust prices based on demand and weather
- **Targeted Marketing**: Customized campaigns for each user segment

### Medium-term Recommendations (6-12 months)
- **Network Expansion**: Add stations in new areas
- **App Enhancement**: Predictive features for users
- **Partnerships**: Collaborate with public transport

### Long-term Recommendations (1+ years)
- **Advanced AI**: Sophisticated predictive models
- **Real-time Analytics**: Continuous pattern monitoring
- **Geographic Expansion**: Reach new cities

---

##  Methodology

### Data Science Workflow
1. **Problem Definition**: Understanding business objectives
2. **Data Collection**: Gathering relevant datasets
3. **Data Preparation**: Cleaning and feature engineering
4. **Exploratory Analysis**: Understanding patterns and relationships
5. **Modeling**: Applying appropriate algorithms
6. **Evaluation**: Validating results and performance
7. **Deployment**: Implementing insights and recommendations

### Tools and Technologies
- **Python**: Primary programming language
- **Pandas**: Data manipulation and analysis
- **GeoPandas**: Spatial data analysis
- **Prophet**: Time series forecasting
- **Plotly**: Interactive visualizations
- **Scikit-learn**: Machine learning algorithms
- **NumPy**: Numerical computations

---

##  Report

The comprehensive Arabic report (`report_arabic.md`) includes:
- Project introduction and objectives
- Detailed methodology
- Key insights and findings
- Machine learning results
- Challenges and solutions
- Conclusions and recommendations

---

## Validation

All components have been validated through:
- [x] Automated test suites
- [x] Statistical significance testing
- [x] Cross-validation for ML models
- [x] Performance benchmarking
- [x] Data quality checks

---

##  Contact

For questions or collaboration opportunities, please refer to the project documentation and test scripts for implementation details.

---

*This project was completed as part of Data Science graduation requirements - 2024* 
