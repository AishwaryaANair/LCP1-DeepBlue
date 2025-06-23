import pandas as pd
import numpy as np
from datetime import datetime
import os

def load_demand_data():
    """Load and merge all demand datasets"""
    demand_files = {
        "01.04.2014 TO 31.03.2015 DEMAND.csv": ("2014-04-01", "2015-03-31"),
        "01-04-2015 to 31-03-2016 DEMAND.csv": ("2015-04-01", "2016-03-31"),
        "Demand 2016-17.csv": ("2016-04-01", "2017-03-31"),
        "01.04.2017 To 31.03.2018 Demand.csv": ("2017-04-01", "2018-03-31"),
        "Demand 2018-19.csv": ("2018-04-01", "2019-03-31")
    }
    
    all_data = []
    for file, (start_date, end_date) in demand_files.items():
        # Read the file with specific column names
        df = pd.read_csv(os.path.join("datasets", file), skiprows=2, usecols=[0,1,2],
                        names=['WARD', 'TOTAL DEMAND', 'CONSUMPTION'])
        
        # Generate date range
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        
        # Create DataFrame with dates for each ward
        for _, row in df.iterrows():
            demand_df = pd.DataFrame({
                'Date': dates,
                'WARD': row['WARD'],
                'TOTAL DEMAND': row['TOTAL DEMAND'],
                'CONSUMPTION': row['CONSUMPTION']
            })
            
            all_data.append(demand_df)
    
    # Concatenate all dataframes
    df_demand = pd.concat(all_data, ignore_index=True)
    
    # Convert numeric columns to appropriate types
    for col in ['TOTAL DEMAND', 'CONSUMPTION']:
        df_demand[col] = pd.to_numeric(df_demand[col], errors='coerce')
    
    return df_demand

def load_climate_data():
    """Load and process climate data"""
    df_climate = pd.read_csv(os.path.join("datasets", "climat_43057_198206-201901.csv"), skiprows=1)
    df_climate.columns = ['Year', 'Month', 'Mean_Temp', 'Max_Temp', 'Min_Temp', 
                        'Precipitation', 'Mean_Temp_Normal', 'Precip_Normal', 
                        'SPI_3', 'SPI_6', 'SPI_12']
    
    # Create date column
    df_climate['Date'] = pd.to_datetime(df_climate[['Year', 'Month']].assign(DAY=1))
    
    # Convert temperature to float
    for col in ['Mean_Temp', 'Max_Temp', 'Min_Temp', 'Mean_Temp_Normal']:
        df_climate[col] = pd.to_numeric(df_climate[col], errors='coerce')
    
    # Convert precipitation to float
    for col in ['Precipitation', 'Precip_Normal']:
        df_climate[col] = pd.to_numeric(df_climate[col], errors='coerce')
    
    return df_climate

def load_population_data():
    """Load population data"""
    df_pop = pd.read_csv(os.path.join("datasets", "2011population.csv"))
    # Calculate total population for all wards
    total_population = df_pop['Total Population'].sum()
    return total_population

def create_features(df):
    """Create additional features for prediction"""
    # Add date features
    df['Date'] = pd.to_datetime(df['Date'])
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['Day'] = df['Date'].dt.day
    df['DayOfWeek'] = df['Date'].dt.dayofweek
    
    # Add season feature
    df['Season'] = pd.cut(df['Month'], 
                         bins=[0, 3, 6, 9, 12],
                         labels=['Winter', 'Spring', 'Summer', 'Autumn'])
    
    return df

def create_master_dataset():
    """Create the master dataset for water demand prediction"""
    # Load all datasets
    df_demand = load_demand_data()
    df_climate = load_climate_data()
    base_population = load_population_data()
    
    # Merge datasets
    # First merge demand with climate data
    df_master = pd.merge(df_demand, df_climate, on='Date', how='left')
    
    # Add population data - we'll use 2011 population as a base
    # and create population growth rate feature
    df_master['Base_Population'] = base_population
    df_master['Year_Diff'] = df_master['Year'] - 2011
    
    # Calculate population growth rate (assuming linear growth)
    population_growth_rate = 0.015  # 1.5% annual growth rate for Mumbai
    df_master['Estimated_Population'] = df_master['Base_Population'] * \
        (1 + population_growth_rate)**df_master['Year_Diff']
    
    # Create additional features
    df_master = create_features(df_master)
    
    # Add future year (2030) data
    future_dates = pd.date_range(start='2030-01-01', end='2030-12-31', freq='D')
    future_df = pd.DataFrame({'Date': future_dates})
    future_df['Year'] = future_df['Date'].dt.year
    future_df['Month'] = future_df['Date'].dt.month
    future_df['Day'] = future_df['Date'].dt.day
    future_df['DayOfWeek'] = future_df['Date'].dt.dayofweek
    future_df['Season'] = pd.cut(future_df['Month'], 
                               bins=[0, 3, 6, 9, 12],
                               labels=['Winter', 'Spring', 'Summer', 'Autumn'])
    
    # Calculate population for 2030
    years_to_2030 = 2030 - 2011
    future_df['Estimated_Population'] = df_master['Base_Population'].iloc[0] * \
        (1 + population_growth_rate)**years_to_2030
    
    # Merge future data with existing data
    df_master = pd.concat([df_master, future_df], ignore_index=True)
    
    # Save the master dataset
    df_master.to_csv('master_dataset_2030.csv', index=False)
    print(f"Master dataset created with {len(df_master)} records")
    
    return df_master

if __name__ == "__main__":
    df_master = create_master_dataset()
    print("\nSample of the master dataset:")
    print(df_master.head())
