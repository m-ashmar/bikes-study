# 🚴 Bike-Sharing Data Science Project
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

## 🏗️ Project Structure

```
final_project/
├── README.md                           # This file
├── report_arabic.md                    # Comprehensive Arabic report
├── figures/final_report/               # Selected visualizations
│   ├── 01_top_start_stations.png      # Top starting stations
│   ├── 02_member_bike_sunburst.png    # Member/bike type analysis
│   ├── 06_trip_duration_histogram.png # Trip duration distribution
│   ├── 10_trip_cost_distribution.png  # Cost distribution
│   ├── 11_daily_revenue_trend.png     # Daily revenue trends
│   ├── 13_geohash_heatmap.png         # Geographic usage heatmap
│   ├── 17_cbd_usage_analysis.png      # CBD proximity analysis
│   ├── 19_revenue_by_weather.png      # Weather impact on revenue
│   ├── forecast_plot.png              # Revenue forecasting
│   ├── forecast_components.png        # Forecast components
│   ├── kmeans_pca.png                 # K-Means clustering
│   ├── dbscan_tsne.png                # DBSCAN clustering
│   └── agglomerative_dendrogram.png   # Hierarchical clustering
├── engineered_data/                    # Processed datasets
│   ├── spatial_features_block4.parquet # Spatial features
│   ├── eda_block5_outputs.parquet     # EDA outputs
│   ├── forecast_block6.csv            # Forecasting results
│   └── clustered_block7.parquet       # Clustering results
└── test_scripts/                       # Validation scripts
    ├── test_block4.py                 # Spatial features tests
    ├── test_block5.py                 # EDA tests
    ├── test_block6.py                 # Forecasting tests
    └── test_block7.py                 # Clustering tests
```

---

## 🚀 Quick Start

### Prerequisites
```bash
# Python 3.8+ required
python --version

# Install required packages
pip install pandas geopandas plotly prophet scikit-learn numpy matplotlib seaborn
```

### Running the Analysis

1. **Data Processing** (Blocks 1-4):
```bash
python test_scripts/spatial_features_test.py
```

2. **Exploratory Analysis** (Block 5):
```bash
python test_scripts/test_block5.py
```

3. **Time Series Forecasting** (Block 6):
```bash
python test_scripts/test_block6.py
```

4. **Pattern Discovery** (Block 7):
```bash
python test_scripts/test_block7.py
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

## 🔧 Technical Details

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

## 🎯 Business Insights

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

## 🔍 Methodology

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

## 📝 Report

The comprehensive Arabic report (`report_arabic.md`) includes:
- Project introduction and objectives
- Detailed methodology
- Key insights and findings
- Machine learning results
- Challenges and solutions
- Conclusions and recommendations

---

## ✅ Validation

All components have been validated through:
- [x] Automated test suites
- [x] Statistical significance testing
- [x] Cross-validation for ML models
- [x] Performance benchmarking
- [x] Data quality checks

---

## 📞 Contact

For questions or collaboration opportunities, please refer to the project documentation and test scripts for implementation details.

---

*This project was completed as part of Data Science graduation requirements - 2024* 