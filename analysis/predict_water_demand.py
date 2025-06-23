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

# Load the master dataset
df = pd.read_csv('../datasets/master_dataset.csv')

# Data Analysis
def analyze_data(df):
    print("\nData Analysis:")
    print("\nBasic Statistics:")
    # Select only numeric columns for statistics
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    print(df[numeric_cols].describe())
    
    # Check for missing values
    print("\nMissing Values:")
    print(df.isnull().sum())
    
    # Correlation analysis
    print("\nCorrelation Matrix:")
    correlation_matrix = df[numeric_cols].corr()
    print(correlation_matrix['TOTAL DEMAND'].sort_values(ascending=False))
    
    # Visualize correlation
    plt.figure(figsize=(12, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
    plt.title('Correlation Matrix')
    plt.savefig('correlation_matrix.png')
    plt.close()

def prepare_data(df, ward):
    """Prepare data for a specific ward"""
    # Filter data for the specific ward
    ward_df = df[df['WARD'] == ward].copy()
    
    # Create features and target
    features = ['Year', 'Month', 'Day', 'DayOfWeek', 
               'Mean_Temp', 'Max_Temp', 'Min_Temp', 
               'Precipitation', 'Estimated_Population']
    
    X = ward_df[features].copy()
    y = ward_df['TOTAL DEMAND'].copy()
    
    # Handle missing values
    # For temperature and precipitation, fill with mean
    X['Mean_Temp'] = X['Mean_Temp'].fillna(X['Mean_Temp'].mean())
    X['Max_Temp'] = X['Max_Temp'].fillna(X['Max_Temp'].mean())
    X['Min_Temp'] = X['Min_Temp'].fillna(X['Min_Temp'].mean())
    X['Precipitation'] = X['Precipitation'].fillna(X['Precipitation'].mean())
    
    # For Estimated_Population, fill with forward fill
    X['Estimated_Population'] = X['Estimated_Population'].fillna(method='ffill')
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, scaler.mean_, scaler.scale_

def train_models(X_train, y_train):
    """Train multiple models and return the best one"""
    # Models to evaluate
    models = {
        'Linear Regression': LinearRegression(),
        'Random Forest': RandomForestRegressor(random_state=42),
        'SVR': SVR()
    }
    
    # Parameters for GridSearch
    param_grids = {
        'Random Forest': {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20, 30]
        },
        'SVR': {
            'kernel': ['linear', 'rbf'],
            'C': [0.1, 1, 10]
        }
    }
    
    best_model = None
    best_score = -np.inf
    results = {}
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        
        if name in param_grids:
            # Use GridSearchCV for models with parameters
            grid = GridSearchCV(model, param_grids[name], cv=5, scoring='r2')
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
    
    return best_model

def evaluate_model(model, X_test, y_test):
    """Evaluate the model on test data"""
    predictions = model.predict(X_test)
    
    # Calculate metrics
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    
    print("\nTest Set Performance:")
    print(f"Mean Squared Error: {mse:.2f}")
    print(f"R2 Score: {r2:.4f}")
    
    # Plot predictions vs actual
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, predictions, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('Actual Demand')
    plt.ylabel('Predicted Demand')
    plt.title('Predicted vs Actual Demand')
    plt.savefig('prediction_vs_actual.png')
    plt.close()
    
    return predictions

def predict_2030(model, df, ward, scaler, scaler_mean, scaler_scale):
    """Predict demand for 2030"""
    # Create 2030 date range
    dates = pd.date_range(start='2030-01-01', end='2030-12-31', freq='D')
    
    # Create features for 2030
    features_df = pd.DataFrame({
        'Year': dates.year,
        'Month': dates.month,
        'Day': dates.day,
        'DayOfWeek': dates.dayofweek,
        'Mean_Temp': df[df['WARD'] == ward]['Mean_Temp'].mean(),  # Use historical mean
        'Max_Temp': df[df['WARD'] == ward]['Max_Temp'].mean(),
        'Min_Temp': df[df['WARD'] == ward]['Min_Temp'].mean(),
        'Precipitation': df[df['WARD'] == ward]['Precipitation'].mean(),
        'Estimated_Population': df[df['WARD'] == ward]['Estimated_Population'].iloc[-1]  # Use latest population
    })
    
    # Handle any remaining NaN values
    features_df = features_df.fillna(features_df.mean())
    
    # Check for NaN values before scaling
    if features_df.isna().any().any():
        print("Warning: NaN values found in features_df:")
        print(features_df.isna().sum())
        features_df = features_df.fillna(0)  # Replace remaining NaNs with 0
    
    # Scale features using the same scaler parameters
    X_2030_scaled = (features_df - scaler_mean) / scaler_scale
    
    # Make predictions
    predictions = model.predict(X_2030_scaled)
    
    # Create results dataframe
    results = pd.DataFrame({
        'Date': dates,
        'WARD': ward,
        'Predicted_Demand': predictions
    })
    
    # Calculate monthly averages
    monthly_avg = results.groupby(results['Date'].dt.month)['Predicted_Demand'].mean()
    
    print("\n2030 Monthly Demand Prediction:")
    print(monthly_avg)
    
    # Plot monthly predictions
    plt.figure(figsize=(12, 6))
    monthly_avg.plot(kind='bar')
    plt.xlabel('Month')
    plt.ylabel('Predicted Demand')
    plt.title(f'2030 Monthly Demand Prediction for {ward}')
    # Replace forward slashes with underscores in filename and add predictions directory
    filename = f'predictions/2030_prediction_{ward.replace("/", "_")}.png'
    plt.savefig(filename)
    plt.close()
    
    return results

def main():
    # Load data
    df = pd.read_csv('../datasets/master_dataset.csv')
    
    # Get unique wards
    wards = df['WARD'].unique()
    
    # Process each ward
    for ward in wards:
        if pd.isna(ward):
            continue
            
        print(f"\nProcessing ward: {ward}")
        
        # Prepare data
        X_train, X_test, y_train, y_test, scaler, scaler_mean, scaler_scale = prepare_data(df, ward)
        
        # Train models
        best_model = train_models(X_train, y_train)
        
        # Evaluate on test set
        y_pred = best_model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        print(f"\nTest Set Performance:")
        print(f"Mean Squared Error: {mse:.2f}")
        print(f"R2 Score: {r2:.4f}")
        
        # Predict 2030
        predict_2030(best_model, df, ward, scaler, scaler_mean, scaler_scale)

if __name__ == "__main__":
    main()
