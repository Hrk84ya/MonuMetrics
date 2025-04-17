import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, IsolationForest
from sklearn.metrics import mean_squared_error, accuracy_score, classification_report
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.inspection import permutation_importance
from datetime import datetime, timedelta

class HeritageDegradationModel:
    def __init__(self):
        self.regression_model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.classification_model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.anomaly_detector = IsolationForest(contamination=0.1, random_state=42)
        self.time_series_model = LinearRegression()
        self.clustering_model = KMeans(n_clusters=3, random_state=42)
        
    def prepare_data(self, air_quality_data, monuments_data):
        """
        Prepare and merge air quality data with monuments data
        """
        # Convert air quality data to DataFrame
        air_quality_df = pd.read_csv(air_quality_data)
        
        # Read monuments data
        with open(monuments_data, 'r') as f:
            monuments = f.readlines()
        monuments = [m.strip() for m in monuments]
        
        # Create features for degradation prediction
        features = []
        for monument in monuments:
            # Find nearest air quality station
            # In reality, you would need proper geocoding and distance calculations
            nearest_station = air_quality_df.iloc[0]  # Simplified for example
            
            # Generate more realistic environmental factors
            rainfall = np.random.normal(100, 30)  # Annual rainfall in cm
            wind_speed = np.random.normal(10, 3)  # Average wind speed in km/h
            uv_index = np.random.normal(5, 1.5)  # UV index
            soil_ph = np.random.normal(7, 1)  # Soil pH
            seismic_activity = np.random.exponential(0.5)  # Seismic activity index
            
            # Material composition (percentage)
            stone_composition = np.random.uniform(0.3, 0.9)
            metal_composition = np.random.uniform(0.05, 0.3)
            wood_composition = np.random.uniform(0.05, 0.3)
            
            feature_dict = {
                'monument_name': monument,
                'pm25_level': nearest_station['pollutant_avg'] if nearest_station['pollutant_id'] == 'PM2.5' else 0,
                'pm10_level': nearest_station['pollutant_avg'] if nearest_station['pollutant_id'] == 'PM10' else 0,
                'humidity': np.random.normal(60, 10),  # Example feature
                'temperature': np.random.normal(25, 5),  # Example feature
                'age_years': np.random.randint(100, 1000),  # Example feature
                'visitors_per_year': np.random.randint(1000, 100000),  # Example feature
                'rainfall': rainfall,
                'wind_speed': wind_speed,
                'uv_index': uv_index,
                'soil_ph': soil_ph,
                'seismic_activity': seismic_activity,
                'stone_composition': stone_composition,
                'metal_composition': metal_composition,
                'wood_composition': wood_composition,
            }
            features.append(feature_dict)
        
        return pd.DataFrame(features)
    
    def create_synthetic_targets(self, df):
        """
        Create synthetic degradation scores and risk levels for demonstration
        In reality, these would come from actual measurements
        """
        # Create degradation score (0-100) with more realistic weighting
        df['degradation_score'] = (
            df['pm25_level'] * 0.15 +
            df['pm10_level'] * 0.15 +
            df['humidity'] * 0.1 +
            df['temperature'] * 0.1 +
            df['age_years'] * 0.05 +
            df['visitors_per_year'] * 0.05 +
            df['rainfall'] * 0.1 +
            df['wind_speed'] * 0.05 +
            df['uv_index'] * 0.1 +
            (abs(df['soil_ph'] - 7) * 5) * 0.05 +  # Deviation from neutral pH
            df['seismic_activity'] * 0.05 +
            (1 - df['stone_composition']) * 0.05  # Less stone = more degradation
        )
        
        # Add some noise to make the data more realistic
        df['degradation_score'] += np.random.normal(0, 5, size=len(df))
        
        # Normalize degradation score to 0-100 range
        df['degradation_score'] = (df['degradation_score'] - df['degradation_score'].min()) / \
                                 (df['degradation_score'].max() - df['degradation_score'].min()) * 100
        
        # Create risk levels based on degradation score
        df['risk_level'] = pd.cut(
            df['degradation_score'],
            bins=[0, 30, 70, 100],
            labels=['Low', 'Medium', 'High']
        )
        
        return df
    
    def train_models(self, df):
        """
        Train both regression and classification models
        """
        # Prepare features
        feature_columns = [
            'pm25_level', 'pm10_level', 'humidity', 'temperature', 
            'age_years', 'visitors_per_year', 'rainfall', 'wind_speed',
            'uv_index', 'soil_ph', 'seismic_activity', 'stone_composition',
            'metal_composition', 'wood_composition'
        ]
        X = df[feature_columns]
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Prepare targets
        y_regression = df['degradation_score']
        y_classification = df['risk_level'].astype(str)  # Convert to string to avoid float issues
        y_classification = self.label_encoder.fit_transform(y_classification)
        
        # Split data
        X_train, X_test, y_reg_train, y_reg_test = train_test_split(
            X_scaled, y_regression, test_size=0.2, random_state=42
        )
        _, _, y_clf_train, y_clf_test = train_test_split(
            X_scaled, y_classification, test_size=0.2, random_state=42
        )
        
        # Train models
        self.regression_model.fit(X_train, y_reg_train)
        self.classification_model.fit(X_train, y_clf_train)
        
        # Evaluate models
        reg_predictions = self.regression_model.predict(X_test)
        clf_predictions = self.classification_model.predict(X_test)
        
        # Calculate metrics
        regression_mse = mean_squared_error(y_reg_test, reg_predictions)
        classification_accuracy = accuracy_score(y_clf_test, clf_predictions)
        
        # Get feature importance
        feature_importance = pd.DataFrame({
            'feature': feature_columns,
            'importance': self.regression_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        # Get unique classes in the test set
        unique_classes = np.unique(y_clf_test)
        target_names = [self.label_encoder.inverse_transform([i])[0] for i in unique_classes]
        
        # Perform cross-validation
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        reg_cv_scores = cross_val_score(self.regression_model, X_scaled, y_regression, cv=kf, scoring='neg_mean_squared_error')
        clf_cv_scores = cross_val_score(self.classification_model, X_scaled, y_classification, cv=kf, scoring='accuracy')
        
        return {
            'regression_mse': regression_mse,
            'classification_accuracy': classification_accuracy,
            'classification_report': classification_report(
                y_clf_test, clf_predictions,
                labels=unique_classes,
                target_names=target_names,
                zero_division=0
            ),
            'feature_importance': feature_importance,
            'regression_cv_scores': -reg_cv_scores,  # Convert to positive MSE
            'classification_cv_scores': clf_cv_scores,
            'reg_predictions': reg_predictions,
            'y_reg_test': y_reg_test,
            'clf_predictions': clf_predictions,
            'y_clf_test': y_clf_test
        }
    
    def predict_degradation(self, new_data):
        """
        Predict degradation score and risk level for new data
        """
        X_scaled = self.scaler.transform(new_data)
        
        degradation_score = self.regression_model.predict(X_scaled)
        risk_level = self.classification_model.predict(X_scaled)
        risk_level = self.label_encoder.inverse_transform(risk_level)
        
        return degradation_score, risk_level
    
    def plot_feature_importance(self, feature_importance):
        """
        Plot feature importance
        """
        plt.figure(figsize=(12, 8))
        sns.barplot(x='importance', y='feature', data=feature_importance)
        plt.title('Feature Importance in Degradation Prediction')
        plt.xlabel('Importance')
        plt.ylabel('Feature')
        plt.tight_layout()
        plt.savefig('feature_importance.png')
        plt.close()
    
    def plot_regression_results(self, y_true, y_pred):
        """
        Plot regression results
        """
        plt.figure(figsize=(10, 6))
        plt.scatter(y_true, y_pred, alpha=0.5)
        plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
        plt.xlabel('Actual Degradation Score')
        plt.ylabel('Predicted Degradation Score')
        plt.title('Regression Model Performance')
        plt.tight_layout()
        plt.savefig('regression_results.png')
        plt.close()
    
    def plot_classification_results(self, y_true, y_pred, target_names):
        """
        Plot classification results
        """
        # Create confusion matrix
        cm = np.zeros((len(target_names), len(target_names)), dtype=int)
        for i, j in zip(y_true, y_pred):
            cm[i, j] += 1
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=target_names, yticklabels=target_names)
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Classification Confusion Matrix')
        plt.tight_layout()
        plt.savefig('classification_results.png')
        plt.close()
    
    def plot_cv_results(self, reg_cv_scores, clf_cv_scores):
        """
        Plot cross-validation results
        """
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.boxplot(reg_cv_scores)
        plt.title('Regression CV Scores (MSE)')
        plt.ylabel('Mean Squared Error')
        
        plt.subplot(1, 2, 2)
        plt.boxplot(clf_cv_scores)
        plt.title('Classification CV Scores (Accuracy)')
        plt.ylabel('Accuracy')
        
        plt.tight_layout()
        plt.savefig('cv_results.png')
        plt.close()
    
    def forecast_future_degradation(self, df, years_ahead=5):
        """
        Forecast future degradation rates for each monument
        
        Args:
            df: DataFrame with monument data
            years_ahead: Number of years to forecast
            
        Returns:
            DataFrame with forecasted degradation scores
        """
        # Create time series data (simulated historical data)
        current_year = datetime.now().year
        historical_years = list(range(current_year - 10, current_year + 1))
        
        # Generate historical degradation scores with trend
        historical_data = []
        for _, row in df.iterrows():
            base_score = row['degradation_score']
            trend = np.random.normal(0.5, 0.2)  # Random trend factor
            noise = np.random.normal(0, 2, size=len(historical_years))
            
            # Generate historical scores with trend and noise
            scores = base_score - (len(historical_years) - 1) * trend + np.cumsum(noise)
            scores = np.clip(scores, 0, 100)  # Ensure scores stay within 0-100
            
            for year, score in zip(historical_years, scores):
                historical_data.append({
                    'monument_name': row['monument_name'],
                    'year': year,
                    'degradation_score': score
                })
        
        historical_df = pd.DataFrame(historical_data)
        
        # Forecast future degradation
        forecast_data = []
        for monument in df['monument_name'].unique():
            monument_data = historical_df[historical_df['monument_name'] == monument]
            
            # Prepare data for time series model
            X = monument_data['year'].values.reshape(-1, 1)
            y = monument_data['degradation_score'].values
            
            # Fit time series model
            self.time_series_model.fit(X, y)
            
            # Generate future years
            future_years = list(range(current_year + 1, current_year + years_ahead + 1))
            X_future = np.array(future_years).reshape(-1, 1)
            
            # Predict future degradation scores
            future_scores = self.time_series_model.predict(X_future)
            future_scores = np.clip(future_scores, 0, 100)  # Ensure scores stay within 0-100
            
            for year, score in zip(future_years, future_scores):
                forecast_data.append({
                    'monument_name': monument,
                    'year': year,
                    'forecasted_degradation_score': score
                })
        
        return pd.DataFrame(forecast_data)
    
    def detect_anomalies(self, df):
        """
        Detect anomalies in degradation patterns
        
        Args:
            df: DataFrame with monument data
            
        Returns:
            DataFrame with anomaly scores and flags
        """
        # Prepare features for anomaly detection
        feature_columns = [
            'pm25_level', 'pm10_level', 'humidity', 'temperature', 
            'age_years', 'visitors_per_year', 'rainfall', 'wind_speed',
            'uv_index', 'soil_ph', 'seismic_activity', 'stone_composition',
            'metal_composition', 'wood_composition', 'degradation_score'
        ]
        X = df[feature_columns]
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Fit and predict anomalies
        anomaly_scores = self.anomaly_detector.fit_predict(X_scaled)
        
        # Create anomaly DataFrame
        anomaly_df = df.copy()
        anomaly_df['anomaly_score'] = self.anomaly_detector.score_samples(X_scaled)
        anomaly_df['is_anomaly'] = anomaly_scores == -1  # -1 indicates anomaly
        
        return anomaly_df
    
    def compare_similar_monuments(self, df, target_monument):
        """
        Find and compare similar monuments based on characteristics
        
        Args:
            df: DataFrame with monument data
            target_monument: Name of the monument to compare
            
        Returns:
            DataFrame with similar monuments and comparison metrics
        """
        # Prepare features for clustering
        feature_columns = [
            'pm25_level', 'pm10_level', 'humidity', 'temperature', 
            'age_years', 'visitors_per_year', 'rainfall', 'wind_speed',
            'uv_index', 'soil_ph', 'seismic_activity', 'stone_composition',
            'metal_composition', 'wood_composition'
        ]
        X = df[feature_columns]
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Fit clustering model
        clusters = self.clustering_model.fit_predict(X_scaled)
        
        # Add cluster labels to DataFrame
        df_with_clusters = df.copy()
        df_with_clusters['cluster'] = clusters
        
        # Find cluster of target monument
        target_cluster = df_with_clusters[df_with_clusters['monument_name'] == target_monument]['cluster'].iloc[0]
        
        # Get similar monuments (same cluster)
        similar_monuments = df_with_clusters[df_with_clusters['cluster'] == target_cluster]
        
        # Calculate similarity scores
        target_features = X_scaled[df_with_clusters['monument_name'] == target_monument]
        distances = np.linalg.norm(X_scaled - target_features, axis=1)
        similarity_scores = 1 / (1 + distances)  # Convert distance to similarity score
        
        # Add similarity scores to DataFrame
        similar_monuments['similarity_score'] = similarity_scores[similar_monuments.index]
        
        # Sort by similarity score
        similar_monuments = similar_monuments.sort_values('similarity_score', ascending=False)
        
        return similar_monuments
    
    def plot_forecast(self, historical_data, forecast_data, monument_name):
        """
        Plot historical and forecasted degradation scores
        
        Args:
            historical_data: DataFrame with historical degradation scores
            forecast_data: DataFrame with forecasted degradation scores
            monument_name: Name of the monument to plot
        """
        plt.figure(figsize=(12, 6))
        
        # Plot historical data
        monument_historical = historical_data[historical_data['monument_name'] == monument_name]
        plt.plot(monument_historical['year'], monument_historical['degradation_score'], 
                 'b-', label='Historical', linewidth=2)
        
        # Plot forecasted data
        monument_forecast = forecast_data[forecast_data['monument_name'] == monument_name]
        plt.plot(monument_forecast['year'], monument_forecast['forecasted_degradation_score'], 
                 'r--', label='Forecast', linewidth=2)
        
        plt.title(f'Degradation Forecast for {monument_name}')
        plt.xlabel('Year')
        plt.ylabel('Degradation Score')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'forecast_{monument_name}.png')
        plt.close()
    
    def plot_anomalies(self, df_with_anomalies):
        """
        Plot anomalies in degradation patterns
        
        Args:
            df_with_anomalies: DataFrame with anomaly scores and flags
        """
        plt.figure(figsize=(12, 6))
        
        # Plot normal points
        normal_points = df_with_anomalies[~df_with_anomalies['is_anomaly']]
        plt.scatter(normal_points['degradation_score'], normal_points['anomaly_score'], 
                   c='blue', alpha=0.5, label='Normal')
        
        # Plot anomalies
        anomalies = df_with_anomalies[df_with_anomalies['is_anomaly']]
        plt.scatter(anomalies['degradation_score'], anomalies['anomaly_score'], 
                   c='red', alpha=0.7, label='Anomaly')
        
        plt.title('Anomaly Detection in Degradation Patterns')
        plt.xlabel('Degradation Score')
        plt.ylabel('Anomaly Score')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('anomalies.png')
        plt.close()
    
    def plot_similar_monuments(self, similar_monuments, target_monument):
        """
        Plot comparison of similar monuments
        
        Args:
            similar_monuments: DataFrame with similar monuments
            target_monument: Name of the target monument
        """
        plt.figure(figsize=(14, 8))
        
        # Plot degradation scores
        plt.subplot(2, 1, 1)
        plt.bar(similar_monuments['monument_name'], similar_monuments['degradation_score'], 
                alpha=0.7, color='skyblue')
        plt.axhline(y=similar_monuments[similar_monuments['monument_name'] == target_monument]['degradation_score'].iloc[0], 
                   color='red', linestyle='--', label='Target Monument')
        plt.title('Degradation Score Comparison')
        plt.xlabel('Monument')
        plt.ylabel('Degradation Score')
        plt.xticks(rotation=45, ha='right')
        plt.legend()
        
        # Plot similarity scores
        plt.subplot(2, 1, 2)
        plt.bar(similar_monuments['monument_name'], similar_monuments['similarity_score'], 
                alpha=0.7, color='lightgreen')
        plt.title('Similarity to Target Monument')
        plt.xlabel('Monument')
        plt.ylabel('Similarity Score')
        plt.xticks(rotation=45, ha='right')
        
        plt.tight_layout()
        plt.savefig(f'similar_monuments_{target_monument}.png')
        plt.close()

def main():
    # Initialize model
    model = HeritageDegradationModel()
    
    # Prepare data
    df = model.prepare_data('Pollution.csv', 'monuments.txt')
    
    # Create synthetic targets (in reality, these would be real measurements)
    df = model.create_synthetic_targets(df)
    
    # Train models and get metrics
    metrics = model.train_models(df)
    
    print("Model Performance Metrics:")
    print(f"Regression MSE: {metrics['regression_mse']:.2f}")
    print(f"Classification Accuracy: {metrics['classification_accuracy']:.2f}")
    
    print("\nCross-Validation Results:")
    print(f"Regression CV MSE: {metrics['regression_cv_scores'].mean():.2f} ± {metrics['regression_cv_scores'].std():.2f}")
    print(f"Classification CV Accuracy: {metrics['classification_cv_scores'].mean():.2f} ± {metrics['classification_cv_scores'].std():.2f}")
    
    print("\nClassification Report:")
    print(metrics['classification_report'])
    
    print("\nFeature Importance:")
    print(metrics['feature_importance'])
    
    # Generate visualizations
    model.plot_feature_importance(metrics['feature_importance'])
    model.plot_regression_results(metrics['y_reg_test'], metrics['reg_predictions'])
    model.plot_classification_results(
        metrics['y_clf_test'], 
        metrics['clf_predictions'], 
        model.label_encoder.classes_
    )
    model.plot_cv_results(metrics['regression_cv_scores'], metrics['classification_cv_scores'])
    
    print("\nVisualizations have been saved as PNG files.")
    
    # Allow user to check for a specific monument
    while True:
        print("\nOptions:")
        print("1. Check a specific monument")
        print("2. Get a random monument prediction")
        print("3. Exit")
        
        choice = input("\nEnter your choice (1-3): ")
        
        if choice == '1':
            # Get monument name from user
            monument_name = input("Enter the name of the monument to check: ")
            
            # Check if the monument exists in the dataset
            monument_data = df[df['monument_name'].str.contains(monument_name, case=False, na=False)]
            
            if len(monument_data) > 0:
                # If multiple matches, use the first one
                if len(monument_data) > 1:
                    print(f"\nFound {len(monument_data)} monuments matching '{monument_name}'. Using the first match.")
                
                selected_monument = monument_data.iloc[0]
                print(f"\nPrediction for Monument: {selected_monument['monument_name']}")
                
                # Create example data using the actual monument's data
                example_data = pd.DataFrame({
                    'pm25_level': [selected_monument['pm25_level']],
                    'pm10_level': [selected_monument['pm10_level']],
                    'humidity': [selected_monument['humidity']],
                    'temperature': [selected_monument['temperature']],
                    'age_years': [selected_monument['age_years']],
                    'visitors_per_year': [selected_monument['visitors_per_year']],
                    'rainfall': [selected_monument['rainfall']],
                    'wind_speed': [selected_monument['wind_speed']],
                    'uv_index': [selected_monument['uv_index']],
                    'soil_ph': [selected_monument['soil_ph']],
                    'seismic_activity': [selected_monument['seismic_activity']],
                    'stone_composition': [selected_monument['stone_composition']],
                    'metal_composition': [selected_monument['metal_composition']],
                    'wood_composition': [selected_monument['wood_composition']]
                })
                
                degradation_score, risk_level = model.predict_degradation(example_data)
                print(f"Predicted Degradation Score: {degradation_score[0]:.2f}")
                print(f"Predicted Risk Level: {risk_level[0]}")
                
                # Compare with actual values
                print(f"Actual Degradation Score: {selected_monument['degradation_score']:.2f}")
                print(f"Actual Risk Level: {selected_monument['risk_level']}")
            else:
                print(f"\nSorry, the monument '{monument_name}' was not found in our database.")
                print("Please try another monument name or choose a random monument.")
        
        elif choice == '2':
            # Select a random monument from the dataset
            random_monument_idx = np.random.randint(0, len(df))
            random_monument = df.iloc[random_monument_idx]
            
            print(f"\nRandom Monument Selected: {random_monument['monument_name']}")
            
            # Create example data using the actual monument's data
            example_data = pd.DataFrame({
                'pm25_level': [random_monument['pm25_level']],
                'pm10_level': [random_monument['pm10_level']],
                'humidity': [random_monument['humidity']],
                'temperature': [random_monument['temperature']],
                'age_years': [random_monument['age_years']],
                'visitors_per_year': [random_monument['visitors_per_year']],
                'rainfall': [random_monument['rainfall']],
                'wind_speed': [random_monument['wind_speed']],
                'uv_index': [random_monument['uv_index']],
                'soil_ph': [random_monument['soil_ph']],
                'seismic_activity': [random_monument['seismic_activity']],
                'stone_composition': [random_monument['stone_composition']],
                'metal_composition': [random_monument['metal_composition']],
                'wood_composition': [random_monument['wood_composition']]
            })
            
            degradation_score, risk_level = model.predict_degradation(example_data)
            print(f"Predicted Degradation Score: {degradation_score[0]:.2f}")
            print(f"Predicted Risk Level: {risk_level[0]}")
            
            # Compare with actual values
            print(f"Actual Degradation Score: {random_monument['degradation_score']:.2f}")
            print(f"Actual Risk Level: {random_monument['risk_level']}")
        
        elif choice == '3':
            print("\nThank you for using the Heritage Site Degradation Prediction Model!")
            break
        
        else:
            print("\nInvalid choice. Please enter 1, 2, or 3.")

if __name__ == "__main__":
    main() 