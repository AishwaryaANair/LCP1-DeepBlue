import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os

class WaterDemandPredictor:
    def __init__(self, data_path='../../datasets/master_dataset.csv'):
        self.data_path = data_path
        self.master_df = None
        self.model = None
        self.scaler = None
        self.scaler_mean = None
        self.scaler_scale = None
        
        # Models to evaluate
        self.models = {
            'Linear Regression': LinearRegression(),
            'Random Forest': RandomForestRegressor(random_state=42),
            'SVR': SVR()
        }
        
        # Parameters for GridSearch
        self.param_grids = {
            'Random Forest': {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20, 30]
            },
            'SVR': {
                'kernel': ['linear', 'rbf'],
                'C': [0.1, 1, 10]
            }
        }

    def load_data(self):
        """Load the master dataset"""
        self.master_df = pd.read_csv(self.data_path)
        return self.master_df

    def analyze_data(self):
        """Analyze the data and create visualizations"""
        if self.master_df is None:
            self.load_data()
            
        print("\nData Analysis:")
        print("\nBasic Statistics:")
        # Select only numeric columns for statistics
        numeric_cols = self.master_df.select_dtypes(include=[np.number]).columns
        print(self.master_df[numeric_cols].describe())
        
        # Check for missing values
        print("\nMissing Values:")
        print(self.master_df.isnull().sum())
        
        # Correlation analysis
        print("\nCorrelation Matrix:")
        correlation_matrix = self.master_df[numeric_cols].corr()
        print(correlation_matrix['TOTAL DEMAND'].sort_values(ascending=False))
        
        # Create output directory if it doesn't exist
        output_dir = 'analysis'
        os.makedirs(output_dir, exist_ok=True)
        
        # Visualize correlation
        plt.figure(figsize=(12, 8))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
        plt.title('Correlation Matrix')
        plt.savefig(os.path.join(output_dir, 'correlation_matrix.png'))
        plt.close()

    def prepare_data(self, ward):
        """Prepare data for a specific ward"""
        if self.master_df is None:
            self.load_data()
            
        # Filter data for the specific ward
        ward_df = self.master_df[self.master_df['WARD'] == ward].copy()
        
        # Check if we have any data at all
        if len(ward_df) == 0:
            print(f"\nSkipping ward {ward} due to no data found")
            return None, None, None, None
            
        # Check if we have enough data points
        if len(ward_df) < 100:  # Minimum threshold of data points
            print(f"\nSkipping ward {ward} due to insufficient data")
            return None, None, None, None
            
        # Check for problematic columns
        problematic_cols = ['Ward', 'Total Population', 'Slum Population', 'Non Slum Population', 'Estimated_Population']
        for col in problematic_cols:
            if col in ward_df.columns and ward_df[col].isna().sum() > len(ward_df) * 0.5:  # More than 50% NaN
                print(f"\nSkipping ward {ward} due to too many NaN values in {col} ({ward_df[col].isna().sum()} NaNs)")
                return None, None, None, None
        
        # Create features and target
        features = ['Year', 'Month', 'Day', 'DayOfWeek', 
                   'Mean_Temp', 'Max_Temp', 'Min_Temp', 
                   'Precipitation', 'Estimated_Population']
        
        # First check for any NaN values
        if ward_df.isna().any().any():
            print(f"\nWarning: NaN values found in data for ward {ward}")
            print(ward_df.isna().sum())
            
            # For numeric columns, fill with mean
            numeric_cols = ward_df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                if ward_df[col].isna().any():
                    # For population-related columns, use forward fill first
                    if col in ['Slum Population', 'Non Slum Population', 'Estimated_Population']:
                        ward_df[col] = ward_df[col].fillna(method='ffill')
                        ward_df[col] = ward_df[col].fillna(method='bfill')
                    else:
                        ward_df[col] = ward_df[col].fillna(ward_df[col].mean())
            
            # For categorical columns, fill with mode
            cat_cols = ward_df.select_dtypes(include=['object']).columns
            for col in cat_cols:
                if ward_df[col].isna().any():
                    try:
                        mode_val = ward_df[col].mode()[0]
                        ward_df[col] = ward_df[col].fillna(mode_val)
                    except IndexError:
                        # If mode fails, try forward fill
                        ward_df[col] = ward_df[col].fillna(method='ffill')
                        # If forward fill fails, try backward fill
                        ward_df[col] = ward_df[col].fillna(method='bfill')
                        # If both fail, use the first non-null value
                        if ward_df[col].isna().any():
                            first_val = ward_df[col].dropna().iloc[0]
                            ward_df[col] = ward_df[col].fillna(first_val)
            
            # Check if we still have NaN values after imputation
            if ward_df.isna().any().any():
                print(f"\nSkipping ward {ward} due to unresolvable NaN values")
                print(ward_df[ward_df.isna().any(axis=1)])
                return None, None, None, None
            
            # Check if we have any empty categorical columns
            for col in cat_cols:
                if ward_df[col].nunique() == 1 and ward_df[col].iloc[0] == '':
                    print(f"\nSkipping ward {ward} due to empty categorical column {col}")
                    return None, None, None, None
        
        # Create features and target
        X = ward_df[features].copy()
        y = ward_df['TOTAL DEMAND'].copy()
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42)
        
        # Scale features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        self.scaler_mean = self.scaler.mean_
        self.scaler_scale = self.scaler.scale_
        
        return X_train_scaled, X_test_scaled, y_train, y_test

    def train_models(self, X_train, y_train):
        """Train multiple models and return the best one"""
        best_model = None
        best_score = -np.inf
        results = {}
        
        for name, model in self.models.items():
            print(f"\nTraining {name}...")
            
            if name in self.param_grids:
                # Use GridSearchCV for models with parameters
                grid = GridSearchCV(model, self.param_grids[name], cv=5, scoring='r2')
                grid.fit(X_train, y_train)
                model = grid.best_estimator_
                print(f"Best params for {name}:", grid.best_params_)
            else:
                model.fit(X_train, y_train)
                
            # Evaluate model
            train_score = model.score(X_train, y_train)
            results[name] = train_score
            
            if train_score > best_score:
                best_score = train_score
                best_model = model
        
        print("\nModel Performance:")
        for name, score in results.items():
            print(f"{name}: R2 Score = {score:.4f}")
        
        self.model = best_model
        return best_model

    def evaluate_model(self, X_test, y_test):
        """Evaluate the model on test data"""
        if self.model is None:
            raise ValueError("Model not trained yet. Please train the model first.")
            
        predictions = self.model.predict(X_test)
        
        # Calculate metrics
        mse = mean_squared_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)
        
        print("\nTest Set Performance:")
        print(f"Mean Squared Error: {mse:.2f}")
        print(f"R2 Score: {r2:.4f}")
        
        # Create output directory if it doesn't exist
        output_dir = 'predictions'
        os.makedirs(output_dir, exist_ok=True)
        
        # Plot predictions vs actual
        plt.figure(figsize=(10, 6))
        plt.scatter(y_test, predictions, alpha=0.5)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        plt.xlabel('Actual Demand')
        plt.ylabel('Predicted Demand')
        plt.title('Predicted vs Actual Demand')
        plt.savefig(os.path.join(output_dir, 'prediction_vs_actual.png'))
        plt.close()
        
        return predictions

    @classmethod
    def predict_2030(cls, ward, model, master_df, scaler_mean, scaler_scale):
        """Predict water demand for 2030"""
        if model is None:
            raise ValueError("Model not trained yet. Please train the model first.")
            
        if master_df is None:
            raise ValueError("Master dataframe is not loaded. Please load the data first.")
            
        # Create 2030 date range
        dates = pd.date_range(start='2030-01-01', end='2030-12-31', freq='D')
        
        # Create features for 2030
        features_df = pd.DataFrame({
            'Year': dates.year,
            'Month': dates.month,
            'Day': dates.day,
            'DayOfWeek': dates.dayofweek,
            'Mean_Temp': master_df[master_df['WARD'] == ward]['Mean_Temp'].mean(),
            'Max_Temp': master_df[master_df['WARD'] == ward]['Max_Temp'].mean(),
            'Min_Temp': master_df[master_df['WARD'] == ward]['Min_Temp'].mean(),
            'Precipitation': master_df[master_df['WARD'] == ward]['Precipitation'].mean(),
            'Estimated_Population': master_df[master_df['WARD'] == ward]['Estimated_Population'].iloc[-1]
        })
        
        # Handle any remaining NaN values
        features_df = features_df.fillna(features_df.mean())
        
        # Check for NaN values before scaling
        if features_df.isna().any().any():
            print("\nWarning: NaN values found in features_df for ward", ward)
            print(features_df.isna().sum())
            
            # Try different methods to fill NaNs
            features_df = features_df.fillna(0)
            
            # If still NaNs, use forward fill
            if features_df.isna().any().any():
                features_df = features_df.fillna(method='ffill')
            
            # If still NaNs, use backward fill
            if features_df.isna().any().any():
                features_df = features_df.fillna(method='bfill')
            
            # If still NaNs, use constant value
            if features_df.isna().any().any():
                features_df = features_df.fillna(1.0)
        
        # Scale features using the same scaler parameters
        try:
            X_2030_scaled = (features_df - scaler_mean) / scaler_scale
        except Exception as e:
            print(f"\nError scaling features for ward {ward}:", str(e))
            print("Attempting to fix scaling issues...")
            
            # Try to fix scaling issues
            features_df = features_df.apply(pd.to_numeric, errors='coerce')
            features_df = features_df.fillna(features_df.mean())
            
            try:
                X_2030_scaled = (features_df - scaler_mean) / scaler_scale
            except:
                print(f"\nSkipping ward {ward} due to unresolvable scaling issues")
                return
        
        # Make predictions
        try:
            predictions = model.predict(X_2030_scaled)
        except Exception as e:
            print(f"\nError making predictions for ward {ward}:", str(e))
            print("Skipping ward due to prediction error")
            return
        
        # Aggregate monthly predictions
        monthly_pred = pd.Series(predictions, index=dates).resample('M').mean()
        
        # Plot and save figure
        ward_safe = ward.replace('/', '_')
        output_dir = 'predictions'
        os.makedirs(output_dir, exist_ok=True)
        plt.figure(figsize=(12,6))
        sns.barplot(x=monthly_pred.index.month, y=monthly_pred.values)
        plt.title(f'2030 Monthly Water Demand Prediction for {ward}')
        plt.xlabel('Month')
        plt.ylabel('Predicted Demand')
        plt.savefig(os.path.join(output_dir, f'2030_prediction_{ward_safe}.png'))
        plt.close()
        
        # Print monthly predictions
        print("\n2030 Monthly Demand Prediction:")
        print(monthly_pred.to_string())

def main():
    """Main function to run the prediction pipeline"""
    # Create predictor instance
    predictor = WaterDemandPredictor()
    
    # Analyze the data
    predictor.analyze_data()
    
    # Get unique wards
    wards = predictor.master_df['WARD'].unique()
    
    # Process each ward
    for ward in wards:
        print(f"\nProcessing ward: {ward}")
        
        # Prepare data
        X_train, X_test, y_train, y_test = predictor.prepare_data(ward)
        
        # Skip if data preparation failed
        if X_train is None:
            continue
            
        # Train models
        model = predictor.train_models(X_train, y_train)
        
        # Evaluate model
        predictions = predictor.evaluate_model(X_test, y_test)
        
        # Make 2030 predictions
        WaterDemandPredictor.predict_2030(ward, model, predictor.master_df, predictor.scaler_mean, predictor.scaler_scale)

if __name__ == "__main__":
    main()
