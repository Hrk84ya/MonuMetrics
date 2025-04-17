# 🏛️ Heritage Site Degradation Analysis Dashboard

![Dashboard Preview](https://img.shields.io/badge/Status-Active-success)
![Python Version](https://img.shields.io/badge/Python-3.8%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)

## 📊 Overview

The Heritage Site Degradation Analysis Dashboard is a powerful, interactive tool designed for archaeologists, conservationists, and heritage management professionals. It provides comprehensive analysis and visualization of degradation patterns in heritage sites, enabling data-driven conservation strategies and proactive preservation efforts.

This dashboard combines advanced machine learning techniques with intuitive visualizations to monitor, predict, and mitigate the degradation of cultural heritage sites.

## ✨ Key Features

- **Interactive Dashboard**: Multi-page interface with real-time data visualization
- **Predictive Analytics**: Forecast future degradation rates using time series analysis
- **Anomaly Detection**: Identify unusual degradation patterns that require immediate attention
- **Comparative Analysis**: Compare similar monuments to understand degradation factors
- **Monument-Specific Analysis**: Detailed metrics and insights for individual heritage sites
- **Time Series Analysis**: Track degradation trends over time
- **3D Visualization**: Explore monuments with degradation hotspots
- **PDF Report Generation**: Create detailed reports with analysis and recommendations

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- pip (Python package installer)

### Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/heritage-degradation-dashboard.git
   cd heritage-degradation-dashboard
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the dashboard:
   ```bash
   streamlit run heritage_dashboard.py
   ```

4. The dashboard will open in your default web browser at `http://localhost:8501`

## 📁 Project Structure

```
heritage-degradation-dashboard/
├── heritage_dashboard.py       # Main dashboard application
├── heritage_degradation_model.py # Machine learning model for degradation prediction
├── Pollution.csv               # Environmental data
├── monuments.txt               # Monument information
├── requirements.txt            # Project dependencies
└── README.md                   # Project documentation
```

## 🧩 Dashboard Components

### 1. Overview
- Key metrics and overall degradation statistics
- Feature importance visualization
- Risk level distribution

### 2. Monument Analysis
- Select specific monuments to analyze degradation factors
- Environmental impact assessment
- Material composition analysis

### 3. Time Series Analysis
- Historical degradation trends
- Seasonal pattern identification
- Trend forecasting

### 4. 3D Visualization
- Interactive 3D models of monuments
- Degradation hotspot identification
- Structural vulnerability assessment

### 5. Predictive Analytics
- Future degradation rate forecasting
- Risk assessment based on predicted trends
- Conservation planning recommendations

### 6. Anomaly Detection
- Identification of unusual degradation patterns
- Detailed insights for anomalous monuments
- Recommended conservation actions

### 7. Comparative Analysis
- Comparison of similar monuments
- Similarity scoring and clustering
- Insights based on comparative metrics

### 8. Report Generation
- Customizable PDF reports
- Detailed analysis and recommendations
- Conservation strategy suggestions

## 🔬 Technical Details

The dashboard is built using:

- **Streamlit**: For the interactive web interface
- **Pandas & NumPy**: For data manipulation and analysis
- **Scikit-learn**: For machine learning models
- **Plotly**: For interactive visualizations
- **Matplotlib & Seaborn**: For static visualizations
- **ReportLab**: For PDF report generation

## 📈 Advanced Analytics

### Predictive Modeling
- Linear regression for time series forecasting
- Trend analysis and seasonal decomposition
- Confidence intervals for predictions

### Anomaly Detection
- Isolation Forest algorithm for outlier detection
- Multi-dimensional feature analysis
- Contextual anomaly scoring

### Comparative Analysis
- K-means clustering for monument grouping
- Feature-based similarity scoring
- Environmental factor correlation analysis

## 🤝 Contributing

Contributions are highly appreciated to improve this project! Here's how you can help:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📧 Contact

For questions or suggestions, please contact:
- Project Maintainer: [Harsh KUmar](mailto:hrk84ya@gmail.com)
- GitHub Issues: [Create an issue](https://github.com/Hrk84ya/heritage-degradation-dashboard/issues)

---

*"Preserving our heritage is not just about protecting the past; it's about ensuring our cultural identity for future generations."* 