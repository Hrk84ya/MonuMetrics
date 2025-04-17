import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from heritage_degradation_model import HeritageDegradationModel
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import io
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
import base64

# Set page configuration
st.set_page_config(
    page_title="Heritage Site Degradation Dashboard",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #424242;
        margin-top: 1rem;
        margin-bottom: 0.5rem;
    }
    .info-text {
        font-size: 1rem;
        color: #616161;
    }
    .highlight {
        background-color: #E3F2FD;
        padding: 0.5rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .anomaly-metric {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        margin-bottom: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        color: black;
    }
    .anomaly-highlight {
        background-color: #ffebee;
        padding: 15px;
        border-radius: 8px;
        margin-bottom: 15px;
        border-left: 5px solid #f44336;
        color: black;
    }
    .anomaly-insight {
        background-color: #e8f5e9;
        padding: 15px;
        border-radius: 8px;
        margin-bottom: 15px;
        border-left: 5px solid #4caf50;
        color: black;
    }
</style>
""", unsafe_allow_html=True)

# Title and introduction
st.markdown('<h1 class="main-header">Heritage Site Degradation Dashboard</h1>', unsafe_allow_html=True)
st.markdown('<p class="info-text">This dashboard provides advanced visualizations and analysis for heritage site degradation prediction.</p>', unsafe_allow_html=True)

# Initialize the model
@st.cache_resource
def load_model():
    model = HeritageDegradationModel()
    df = model.prepare_data('Pollution.csv', 'monuments.txt')
    df = model.create_synthetic_targets(df)
    metrics = model.train_models(df)
    return model, df, metrics

model, df, metrics = load_model()

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Overview", "Monument Analysis", "Time Series Analysis", "3D Visualization", "Predictive Analytics", "Anomaly Detection", "Comparative Analysis", "Generate Report"])

# Overview page
if page == "Overview":
    st.markdown('<h2 class="sub-header">Dashboard Overview</h2>', unsafe_allow_html=True)
    
    # Display key metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Monuments", len(df))
    with col2:
        st.metric("Average Degradation Score", f"{df['degradation_score'].mean():.2f}")
    with col3:
        risk_counts = df['risk_level'].value_counts()
        st.metric("High Risk Monuments", risk_counts.get('High', 0))
    
    # Feature importance visualization
    st.markdown('<h3 class="sub-header">Feature Importance</h3>', unsafe_allow_html=True)
    fig = px.bar(
        metrics['feature_importance'],
        x='importance',
        y='feature',
        orientation='h',
        title='Feature Importance in Degradation Prediction',
        labels={'importance': 'Importance', 'feature': 'Feature'}
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Risk level distribution
    st.markdown('<h3 class="sub-header">Risk Level Distribution</h3>', unsafe_allow_html=True)
    fig = px.pie(
        df,
        names='risk_level',
        title='Distribution of Risk Levels',
        color='risk_level',
        color_discrete_map={'Low': 'green', 'Medium': 'orange', 'High': 'red'}
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Degradation score distribution
    st.markdown('<h3 class="sub-header">Degradation Score Distribution</h3>', unsafe_allow_html=True)
    fig = px.histogram(
        df,
        x='degradation_score',
        nbins=30,
        title='Distribution of Degradation Scores',
        labels={'degradation_score': 'Degradation Score', 'count': 'Number of Monuments'}
    )
    st.plotly_chart(fig, use_container_width=True)

# Monument Analysis page
elif page == "Monument Analysis":
    st.markdown('<h2 class="sub-header">Monument Analysis</h2>', unsafe_allow_html=True)
    
    # Monument selection
    monument_options = ['All Monuments'] + list(df['monument_name'].unique())
    selected_monument = st.selectbox("Select a Monument", monument_options)
    
    if selected_monument == 'All Monuments':
        # Display all monuments in a table
        st.markdown('<h3 class="sub-header">All Monuments</h3>', unsafe_allow_html=True)
        st.dataframe(df[['monument_name', 'degradation_score', 'risk_level']].sort_values('degradation_score', ascending=False))
        
        # Scatter plot of degradation scores
        st.markdown('<h3 class="sub-header">Degradation Score Distribution</h3>', unsafe_allow_html=True)
        fig = px.scatter(
            df,
            x='age_years',
            y='degradation_score',
            color='risk_level',
            size='visitors_per_year',
            hover_data=['monument_name'],
            title='Degradation Score vs Age',
            labels={'age_years': 'Age (Years)', 'degradation_score': 'Degradation Score', 'visitors_per_year': 'Visitors per Year'}
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        # Display selected monument details
        monument_data = df[df['monument_name'] == selected_monument].iloc[0]
        
        st.markdown(f'<h3 class="sub-header">Details for {selected_monument}</h3>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Degradation Score", f"{monument_data['degradation_score']:.2f}")
            st.metric("Risk Level", monument_data['risk_level'])
            st.metric("Age (Years)", monument_data['age_years'])
            st.metric("Visitors per Year", f"{monument_data['visitors_per_year']:,}")
        
        with col2:
            st.metric("PM2.5 Level", f"{monument_data['pm25_level']:.2f}")
            st.metric("PM10 Level", f"{monument_data['pm10_level']:.2f}")
            st.metric("Humidity", f"{monument_data['humidity']:.2f}%")
            st.metric("Temperature", f"{monument_data['temperature']:.2f}°C")
        
        # Environmental factors radar chart
        st.markdown('<h3 class="sub-header">Environmental Factors</h3>', unsafe_allow_html=True)
        
        # Normalize values for radar chart
        env_factors = {
            'PM2.5': monument_data['pm25_level'] / df['pm25_level'].max(),
            'PM10': monument_data['pm10_level'] / df['pm10_level'].max(),
            'Humidity': monument_data['humidity'] / df['humidity'].max(),
            'Temperature': monument_data['temperature'] / df['temperature'].max(),
            'Rainfall': monument_data['rainfall'] / df['rainfall'].max(),
            'Wind Speed': monument_data['wind_speed'] / df['wind_speed'].max(),
            'UV Index': monument_data['uv_index'] / df['uv_index'].max()
        }
        
        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(
            r=list(env_factors.values()),
            theta=list(env_factors.keys()),
            fill='toself',
            name=selected_monument
        ))
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )
            ),
            title=f"Environmental Factors for {selected_monument}"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Material composition
        st.markdown('<h3 class="sub-header">Material Composition</h3>', unsafe_allow_html=True)
        materials = {
            'Stone': monument_data['stone_composition'],
            'Metal': monument_data['metal_composition'],
            'Wood': monument_data['wood_composition']
        }
        
        fig = px.pie(
            values=list(materials.values()),
            names=list(materials.keys()),
            title=f"Material Composition for {selected_monument}"
        )
        st.plotly_chart(fig, use_container_width=True)

# Time Series Analysis page
elif page == "Time Series Analysis":
    st.markdown('<h2 class="sub-header">Time Series Analysis</h2>', unsafe_allow_html=True)
    
    # Generate synthetic time series data
    st.markdown('<p class="info-text">This section shows simulated degradation trends over time for selected monuments.</p>', unsafe_allow_html=True)
    
    # Monument selection
    selected_monuments = st.multiselect(
        "Select Monuments to Compare",
        options=df['monument_name'].unique(),
        default=[df['monument_name'].iloc[0]]
    )
    
    if selected_monuments:
        # Generate time series data
        dates = pd.date_range(start='2020-01-01', end='2023-12-31', freq='M')
        
        # Create a figure
        fig = go.Figure()
        
        for monument_name in selected_monuments:
            # Get base degradation score
            base_score = df[df['monument_name'] == monument_name]['degradation_score'].iloc[0]
            
            # Generate trend with some randomness
            trend = np.linspace(0, 10, len(dates))  # Increasing trend
            noise = np.random.normal(0, 2, len(dates))  # Random noise
            
            # Calculate degradation scores over time
            scores = base_score + trend + noise
            scores = np.clip(scores, 0, 100)  # Ensure scores stay within 0-100
            
            # Add to plot
            fig.add_trace(go.Scatter(
                x=dates,
                y=scores,
                mode='lines+markers',
                name=monument_name
            ))
        
        fig.update_layout(
            title='Degradation Trends Over Time',
            xaxis_title='Date',
            yaxis_title='Degradation Score',
            legend_title='Monument'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Risk level changes over time
        st.markdown('<h3 class="sub-header">Risk Level Changes Over Time</h3>', unsafe_allow_html=True)
        
        # Create a figure for risk level changes
        fig = go.Figure()
        
        for monument_name in selected_monuments:
            # Get base degradation score
            base_score = df[df['monument_name'] == monument_name]['degradation_score'].iloc[0]
            
            # Generate trend with some randomness
            trend = np.linspace(0, 10, len(dates))  # Increasing trend
            noise = np.random.normal(0, 2, len(dates))  # Random noise
            
            # Calculate degradation scores over time
            scores = base_score + trend + noise
            scores = np.clip(scores, 0, 100)  # Ensure scores stay within 0-100
            
            # Convert scores to risk levels
            risk_levels = pd.cut(
                scores,
                bins=[0, 30, 70, 100],
                labels=['Low', 'Medium', 'High']
            )
            
            # Count risk levels by year
            yearly_risks = pd.DataFrame({
                'date': dates,
                'risk': risk_levels
            })
            yearly_risks['year'] = yearly_risks['date'].dt.year
            yearly_counts = yearly_risks.groupby(['year', 'risk']).size().unstack(fill_value=0)
            
            # Add to plot
            for risk in ['Low', 'Medium', 'High']:
                if risk in yearly_counts.columns:
                    fig.add_trace(go.Bar(
                        x=yearly_counts.index,
                        y=yearly_counts[risk],
                        name=f"{monument_name} - {risk}",
                        text=yearly_counts[risk],
                        textposition='auto',
                    ))
        
        fig.update_layout(
            title='Risk Level Changes Over Time',
            xaxis_title='Year',
            yaxis_title='Number of Months',
            barmode='group',
            legend_title='Monument and Risk Level'
        )
        
        st.plotly_chart(fig, use_container_width=True)

# 3D Visualization page
elif page == "3D Visualization":
    st.markdown('<h2 class="sub-header">3D Monument Visualization</h2>', unsafe_allow_html=True)
    
    st.markdown('<p class="info-text">This section provides a simulated 3D visualization of monuments with degradation hotspots.</p>', unsafe_allow_html=True)
    
    # Monument selection
    selected_monument = st.selectbox("Select a Monument", df['monument_name'].unique())
    
    # Get monument data
    monument_data = df[df['monument_name'] == selected_monument].iloc[0]
    
    # Display monument details
    st.markdown(f'<h3 class="sub-header">3D Visualization for {selected_monument}</h3>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Degradation Score", f"{monument_data['degradation_score']:.2f}")
        st.metric("Risk Level", monument_data['risk_level'])
    with col2:
        st.metric("Age (Years)", monument_data['age_years'])
        st.metric("Material Composition", f"Stone: {monument_data['stone_composition']:.0%}, Metal: {monument_data['metal_composition']:.0%}, Wood: {monument_data['wood_composition']:.0%}")
    
    # Simulated 3D visualization
    st.markdown('<h3 class="sub-header">Degradation Hotspots</h3>', unsafe_allow_html=True)
    
    # Create a simulated 3D surface with degradation hotspots
    x = np.linspace(-5, 5, 100)
    y = np.linspace(-5, 5, 100)
    X, Y = np.meshgrid(x, y)
    
    # Create degradation hotspots based on monument data
    degradation_level = monument_data['degradation_score'] / 100
    
    # Base shape (simplified monument)
    Z = 10 * np.exp(-(X**2 + Y**2) / 10)
    
    # Add degradation hotspots
    if degradation_level > 0.3:
        Z += 2 * np.exp(-((X-2)**2 + (Y-2)**2) / 2) * degradation_level
    if degradation_level > 0.5:
        Z += 1.5 * np.exp(-((X+2)**2 + (Y-1)**2) / 1.5) * degradation_level
    if degradation_level > 0.7:
        Z += 1 * np.exp(-((X-1)**2 + (Y+2)**2) / 1) * degradation_level
    
    # Create 3D surface plot
    fig = go.Figure(data=[go.Surface(x=X, y=Y, z=Z, colorscale='RdYlBu_r')])
    fig.update_layout(
        title=f'3D Visualization of {selected_monument} with Degradation Hotspots',
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Height'
        ),
        width=800,
        height=600
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Degradation factors
    st.markdown('<h3 class="sub-header">Contributing Factors to Degradation</h3>', unsafe_allow_html=True)
    
    # Create a radar chart of contributing factors
    factors = {
        'Air Pollution': (monument_data['pm25_level'] + monument_data['pm10_level']) / 200,
        'Humidity': monument_data['humidity'] / 100,
        'Temperature': monument_data['temperature'] / 50,
        'Visitor Impact': monument_data['visitors_per_year'] / 100000,
        'Age': monument_data['age_years'] / 1000,
        'Material Vulnerability': 1 - monument_data['stone_composition']
    }
    
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=list(factors.values()),
        theta=list(factors.keys()),
        fill='toself',
        name='Contributing Factors'
    ))
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )
        ),
        title=f"Contributing Factors to Degradation for {selected_monument}"
    )
    st.plotly_chart(fig, use_container_width=True)

# Predictive Analytics page
elif page == "Predictive Analytics":
    st.markdown('<h2 class="sub-header">Predictive Analytics</h2>', unsafe_allow_html=True)
    
    st.markdown('<p class="info-text">This section provides forecasts of future degradation rates for monuments.</p>', unsafe_allow_html=True)
    
    # Years to forecast
    years_ahead = st.slider("Years to Forecast", 1, 10, 5)
    
    # Generate forecasts
    forecast_data = model.forecast_future_degradation(df, years_ahead=years_ahead)
    
    # Monument selection
    selected_monument = st.selectbox("Select a Monument", df['monument_name'].unique())
    
    if selected_monument:
        # Filter data for selected monument
        monument_forecast = forecast_data[forecast_data['monument_name'] == selected_monument]
        
        # Create forecast plot
        fig = go.Figure()
        
        # Add forecast line
        fig.add_trace(go.Scatter(
            x=monument_forecast['year'],
            y=monument_forecast['forecasted_degradation_score'],
            mode='lines+markers',
            name='Forecast',
            line=dict(color='red', dash='dash')
        ))
        
        # Add current degradation score
        current_score = df[df['monument_name'] == selected_monument]['degradation_score'].iloc[0]
        current_year = datetime.now().year
        
        fig.add_trace(go.Scatter(
            x=[current_year],
            y=[current_score],
            mode='markers',
            name='Current Score',
            marker=dict(color='blue', size=10)
        ))
        
        fig.update_layout(
            title=f'Degradation Forecast for {selected_monument}',
            xaxis_title='Year',
            yaxis_title='Degradation Score',
            legend_title='Data Type'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Display forecast data
        st.markdown('<h3 class="sub-header">Forecast Data</h3>', unsafe_allow_html=True)
        st.dataframe(monument_forecast[['year', 'forecasted_degradation_score']].rename(
            columns={'forecasted_degradation_score': 'Forecasted Score'}
        ))
        
        # Risk assessment based on forecast
        st.markdown('<h3 class="sub-header">Risk Assessment</h3>', unsafe_allow_html=True)
        
        # Calculate risk based on forecast trend
        forecast_trend = monument_forecast['forecasted_degradation_score'].iloc[-1] - current_score
        years = years_ahead
        
        if forecast_trend > 20:
            risk_level = "HIGH"
            risk_color = "red"
            risk_message = f"The monument is projected to experience significant degradation over the next {years} years. Immediate conservation efforts are recommended."
        elif forecast_trend > 10:
            risk_level = "MEDIUM"
            risk_color = "orange"
            risk_message = f"The monument is projected to experience moderate degradation over the next {years} years. Proactive conservation measures should be planned."
        else:
            risk_level = "LOW"
            risk_color = "green"
            risk_message = f"The monument is projected to experience minimal degradation over the next {years} years. Regular monitoring is recommended."
        
        st.markdown(f'<div class="highlight" style="color: {risk_color};"><strong>Risk Level: {risk_level}</strong><br>{risk_message}</div>', unsafe_allow_html=True)

# Anomaly Detection page
elif page == "Anomaly Detection":
    st.markdown('<h2 class="sub-header">Anomaly Detection</h2>', unsafe_allow_html=True)
    
    st.markdown('<p class="info-text">This section identifies unusual patterns in monument degradation data.</p>', unsafe_allow_html=True)
    
    # Detect anomalies
    df_with_anomalies = model.detect_anomalies(df)
    
    # Count anomalies
    anomaly_count = df_with_anomalies['is_anomaly'].sum()
    total_monuments = len(df_with_anomalies)
    
    # Display anomaly count in a more prominent way
    st.markdown(f'<div class="anomaly-metric"><h3>Anomalies Detected: {anomaly_count} out of {total_monuments} monuments</h3></div>', unsafe_allow_html=True)
    
    # Create anomaly plot
    fig = go.Figure()
    
    # Add normal points
    normal_points = df_with_anomalies[~df_with_anomalies['is_anomaly']]
    fig.add_trace(go.Scatter(
        x=normal_points['degradation_score'],
        y=normal_points['anomaly_score'],
        mode='markers',
        name='Normal',
        marker=dict(color='blue', size=8, opacity=0.6)
    ))
    
    # Add anomalies
    anomalies = df_with_anomalies[df_with_anomalies['is_anomaly']]
    fig.add_trace(go.Scatter(
        x=anomalies['degradation_score'],
        y=anomalies['anomaly_score'],
        mode='markers',
        name='Anomaly',
        marker=dict(color='red', size=10, opacity=0.8)
    ))
    
    fig.update_layout(
        title='Anomaly Detection in Degradation Patterns',
        xaxis_title='Degradation Score',
        yaxis_title='Anomaly Score',
        legend_title='Data Point Type'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Display anomalous monuments
    if anomaly_count > 0:
        st.markdown('<h3 class="sub-header">Anomalous Monuments</h3>', unsafe_allow_html=True)
        
        # Create a table of anomalous monuments
        anomaly_table = anomalies[['monument_name', 'degradation_score', 'anomaly_score']].copy()
        anomaly_table = anomaly_table.sort_values('anomaly_score', ascending=True)
        anomaly_table.columns = ['Monument', 'Degradation Score', 'Anomaly Score']
        
        st.dataframe(anomaly_table)
        
        # Provide insights in a more visible way
        st.markdown('<h3 class="sub-header">Anomaly Insights</h3>', unsafe_allow_html=True)
        
        # Add a summary of anomalies
        st.markdown(f'<div class="anomaly-highlight"><strong>Summary:</strong> {anomaly_count} monuments show unusual degradation patterns that deviate significantly from the expected behavior based on their characteristics.</div>', unsafe_allow_html=True)
        
        # Display individual insights for each anomaly
        for _, row in anomalies.iterrows():
            # Determine the type of anomaly based on degradation score
            if row['degradation_score'] > 80:
                anomaly_type = "severe degradation"
                recommendation = "Immediate conservation intervention is recommended."
            elif row['degradation_score'] > 60:
                anomaly_type = "accelerated degradation"
                recommendation = "Proactive conservation measures should be planned."
            else:
                anomaly_type = "unusual degradation pattern"
                recommendation = "Further investigation is needed to understand the underlying causes."
            
            st.markdown(f'''
            <div class="anomaly-insight">
                <strong>{row["monument_name"]}</strong><br>
                This monument shows {anomaly_type} compared to others with similar characteristics.<br>
                <strong>Recommendation:</strong> {recommendation}
            </div>
            ''', unsafe_allow_html=True)
        
        # Add a section for potential causes
        st.markdown('<h3 class="sub-header">Potential Causes of Anomalies</h3>', unsafe_allow_html=True)
        
        st.markdown('''
        <div class="anomaly-highlight">
            <strong>Possible explanations for anomalous degradation patterns:</strong>
            <ul>
                <li>Unique environmental conditions not captured in the data</li>
                <li>Recent changes in visitor patterns or local development</li>
                <li>Specific structural vulnerabilities of the monument</li>
                <li>Historical conservation interventions that may have accelerated degradation</li>
                <li>Data collection or measurement inconsistencies</li>
            </ul>
        </div>
        ''', unsafe_allow_html=True)
        
        # Add a section for next steps
        st.markdown('<h3 class="sub-header">Recommended Next Steps</h3>', unsafe_allow_html=True)
        
        st.markdown('''
        <div class="anomaly-highlight">
            <ol>
                <li>Conduct detailed on-site inspections of anomalous monuments</li>
                <li>Review historical conservation records for these sites</li>
                <li>Implement enhanced monitoring for these specific monuments</li>
                <li>Consider targeted environmental protection measures</li>
                <li>Update the degradation model with new data from these sites</li>
            </ol>
        </div>
        ''', unsafe_allow_html=True)
    else:
        st.markdown('<div class="anomaly-insight"><strong>No anomalies detected.</strong> All monuments show degradation patterns that are consistent with their characteristics.</div>', unsafe_allow_html=True)

# Comparative Analysis page
elif page == "Comparative Analysis":
    st.markdown('<h2 class="sub-header">Comparative Analysis</h2>', unsafe_allow_html=True)
    
    st.markdown('<p class="info-text">This section compares monuments with similar characteristics to identify patterns and differences.</p>', unsafe_allow_html=True)
    
    # Monument selection
    selected_monument = st.selectbox("Select a Monument to Compare", df['monument_name'].unique())
    
    if selected_monument:
        # Find similar monuments
        similar_monuments = model.compare_similar_monuments(df, selected_monument)
        
        # Display target monument details
        st.markdown(f'<h3 class="sub-header">Target Monument: {selected_monument}</h3>', unsafe_allow_html=True)
        
        target_data = df[df['monument_name'] == selected_monument].iloc[0]
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Degradation Score", f"{target_data['degradation_score']:.2f}")
        with col2:
            st.metric("Age (Years)", target_data['age_years'])
        with col3:
            st.metric("Visitors per Year", f"{target_data['visitors_per_year']:,}")
        
        # Create comparison plot
        fig = go.Figure()
        
        # Add degradation scores
        fig.add_trace(go.Bar(
            x=similar_monuments['monument_name'],
            y=similar_monuments['degradation_score'],
            name='Degradation Score',
            marker_color='skyblue'
        ))
        
        # Add similarity scores
        fig.add_trace(go.Scatter(
            x=similar_monuments['monument_name'],
            y=similar_monuments['similarity_score'] * 100,  # Scale to 0-100
            name='Similarity Score (%)',
            mode='lines+markers',
            line=dict(color='green', dash='dash'),
            yaxis='y2'
        ))
        
        fig.update_layout(
            title=f'Comparison with Similar Monuments',
            xaxis_title='Monument',
            yaxis_title='Degradation Score',
            yaxis2=dict(
                title='Similarity Score (%)',
                overlaying='y',
                side='right',
                range=[0, 100]
            ),
            legend_title='Metric',
            barmode='group'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Display similar monuments
        st.markdown('<h3 class="sub-header">Similar Monuments</h3>', unsafe_allow_html=True)
        
        # Create a table of similar monuments
        comparison_table = similar_monuments[['monument_name', 'degradation_score', 'similarity_score']].copy()
        comparison_table = comparison_table.sort_values('similarity_score', ascending=False)
        comparison_table.columns = ['Monument', 'Degradation Score', 'Similarity Score']
        
        st.dataframe(comparison_table)
        
        # Provide insights
        st.markdown('<h3 class="sub-header">Insights</h3>', unsafe_allow_html=True)
        
        # Calculate average degradation score of similar monuments
        avg_degradation = similar_monuments['degradation_score'].mean()
        
        if target_data['degradation_score'] > avg_degradation * 1.2:
            st.markdown(f'<div class="highlight" style="color: red;"><strong>Higher Degradation:</strong> {selected_monument} has significantly higher degradation than similar monuments. This may indicate unique environmental factors or conservation challenges.</div>', unsafe_allow_html=True)
        elif target_data['degradation_score'] < avg_degradation * 0.8:
            st.markdown(f'<div class="highlight" style="color: green;"><strong>Lower Degradation:</strong> {selected_monument} has significantly lower degradation than similar monuments. This may indicate effective conservation practices that could be applied to other monuments.</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="highlight"><strong>Average Degradation:</strong> {selected_monument} has degradation levels similar to other monuments with comparable characteristics.</div>', unsafe_allow_html=True)

# Generate Report page
elif page == "Generate Report":
    st.markdown('<h2 class="sub-header">Generate PDF Report</h2>', unsafe_allow_html=True)
    
    st.markdown('<p class="info-text">Generate a detailed PDF report with analysis and recommendations for selected monuments.</p>', unsafe_allow_html=True)
    
    # Monument selection
    selected_monuments = st.multiselect(
        "Select Monuments for Report",
        options=df['monument_name'].unique(),
        default=[df['monument_name'].iloc[0]]
    )
    
    if selected_monuments:
        # Report title
        report_title = st.text_input("Report Title", "Heritage Site Degradation Analysis Report")
        
        # Report description
        report_description = st.text_area(
            "Report Description",
            "This report provides a detailed analysis of heritage site degradation and recommendations for conservation."
        )
        
        # Generate report button
        if st.button("Generate PDF Report"):
            # Create PDF
            buffer = io.BytesIO()
            doc = SimpleDocTemplate(buffer, pagesize=letter)
            styles = getSampleStyleSheet()
            title_style = ParagraphStyle(
                'CustomTitle',
                parent=styles['Heading1'],
                fontSize=24,
                spaceAfter=30
            )
            heading_style = ParagraphStyle(
                'CustomHeading',
                parent=styles['Heading2'],
                fontSize=18,
                spaceAfter=20
            )
            normal_style = styles['Normal']
            
            # Build the document
            elements = []
            
            # Title
            elements.append(Paragraph(report_title, title_style))
            elements.append(Paragraph(f"Generated on {datetime.now().strftime('%B %d, %Y')}", normal_style))
            elements.append(Spacer(1, 20))
            
            # Description
            elements.append(Paragraph("Executive Summary", heading_style))
            elements.append(Paragraph(report_description, normal_style))
            elements.append(Spacer(1, 20))
            
            # Overview
            elements.append(Paragraph("Overview", heading_style))
            elements.append(Paragraph(f"This report analyzes {len(selected_monuments)} heritage sites for degradation risks and provides recommendations for conservation efforts.", normal_style))
            elements.append(Spacer(1, 20))
            
            # Monument details
            for monument_name in selected_monuments:
                monument_data = df[df['monument_name'] == monument_name].iloc[0]
                
                elements.append(Paragraph(f"Monument: {monument_name}", heading_style))
                
                # Create a table with monument details
                data = [
                    ["Property", "Value"],
                    ["Degradation Score", f"{monument_data['degradation_score']:.2f}"],
                    ["Risk Level", monument_data['risk_level']],
                    ["Age (Years)", str(monument_data['age_years'])],
                    ["Visitors per Year", f"{monument_data['visitors_per_year']:,}"],
                    ["PM2.5 Level", f"{monument_data['pm25_level']:.2f}"],
                    ["PM10 Level", f"{monument_data['pm10_level']:.2f}"],
                    ["Humidity", f"{monument_data['humidity']:.2f}%"],
                    ["Temperature", f"{monument_data['temperature']:.2f}°C"]
                ]
                
                table = Table(data, colWidths=[200, 300])
                table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, 0), 14),
                    ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                    ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                    ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),
                    ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
                    ('FONTSIZE', (0, 1), (-1, -1), 12),
                    ('GRID', (0, 0), (-1, -1), 1, colors.black)
                ]))
                
                elements.append(table)
                elements.append(Spacer(1, 20))
                
                # Risk assessment
                elements.append(Paragraph("Risk Assessment", heading_style))
                
                if monument_data['risk_level'] == 'High':
                    risk_text = f"The {monument_name} is at HIGH risk of degradation. Immediate conservation efforts are recommended to prevent further damage."
                elif monument_data['risk_level'] == 'Medium':
                    risk_text = f"The {monument_name} is at MEDIUM risk of degradation. Proactive conservation measures should be planned within the next year."
                else:
                    risk_text = f"The {monument_name} is at LOW risk of degradation. Regular monitoring and maintenance are recommended to maintain its current condition."
                
                elements.append(Paragraph(risk_text, normal_style))
                elements.append(Spacer(1, 20))
                
                # Recommendations
                elements.append(Paragraph("Recommendations", heading_style))
                
                recommendations = []
                
                # Air pollution recommendations
                if monument_data['pm25_level'] > 50 or monument_data['pm10_level'] > 100:
                    recommendations.append("Install air filtration systems or barriers to reduce exposure to air pollutants.")
                
                # Humidity recommendations
                if monument_data['humidity'] > 70:
                    recommendations.append("Implement humidity control measures to prevent moisture-related damage.")
                
                # Temperature recommendations
                if monument_data['temperature'] > 30:
                    recommendations.append("Consider shading structures to reduce direct sunlight exposure.")
                
                # Visitor impact recommendations
                if monument_data['visitors_per_year'] > 50000:
                    recommendations.append("Implement visitor flow management and consider timed entry systems to reduce wear and tear.")
                
                # Age-related recommendations
                if monument_data['age_years'] > 500:
                    recommendations.append("Conduct detailed structural assessment to identify age-related vulnerabilities.")
                
                # Material-specific recommendations
                if monument_data['stone_composition'] < 0.5:
                    recommendations.append("Consider protective coatings for non-stone materials to enhance durability.")
                
                # Add default recommendations if none specific
                if not recommendations:
                    recommendations.append("Continue regular monitoring and maintenance.")
                    recommendations.append("Document current condition as baseline for future assessments.")
                
                for rec in recommendations:
                    elements.append(Paragraph(f"• {rec}", normal_style))
                
                elements.append(Spacer(1, 30))
            
            # Build the PDF
            doc.build(elements)
            
            # Get the value of the BytesIO buffer
            pdf = buffer.getvalue()
            
            # Create a download button
            b64 = base64.b64encode(pdf).decode()
            href = f'<a href="data:application/pdf;base64,{b64}" download="heritage_degradation_report.pdf">Download PDF Report</a>'
            st.markdown(href, unsafe_allow_html=True)
            
            st.success("PDF report generated successfully! Click the link above to download.")

# Footer
st.markdown("---")
st.markdown("© 2025 Heritage Site Degradation Prediction System | Developed for Archaeological Conservation") 