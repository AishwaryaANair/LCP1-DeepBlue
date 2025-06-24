import pandas as pd
import numpy as np
from datetime import datetime
import os

class WaterDemandDataset:
    def __init__(self):
        """Initialize the dataset class"""
        # Get the absolute path of the script's directory
        script_dir = os.path.dirname(os.path.abspath(__file__))
        self.data_dir = os.path.join(script_dir, '../datasets')
        self.demand_files = {
            "01.04.2014 TO 31.03.2015 DEMAND.csv": ("2014-04-01", "2015-03-31"),
            "01-04-2015 to 31-03-2016 DEMAND.csv": ("2015-04-01", "2016-03-31"),
            "Demand 2016-17.csv": ("2016-04-01", "2017-03-31"),
            "01.04.2017 To 31.03.2018 Demand.csv": ("2017-04-01", "2018-03-31"),
            "Demand 2018-19.csv": ("2018-04-01", "2019-03-31")
        }
        self.master_df = None

    def load_demand_data(self):
        """Load and merge all demand datasets"""
        all_data = []
        for file, (start_date, end_date) in self.demand_files.items():
            # Read the file with specific column names
            df = pd.read_csv(os.path.join(self.data_dir, file), skiprows=2, usecols=[0,1,2],
                            names=['WARD', 'TOTAL DEMAND', 'CONSUMPTION'])
            
            # Clean ward names: remove hyphens and convert to uppercase
            df['WARD'] = df['WARD'].str.replace('-', '').str.upper()
            
            # Create mapping for special cases
            ward_mapping = {
                'A': 'A',
                'B': 'B',
                'C': 'C',
                'D': 'D',
                'E': 'E',
                'F/SOUTH': 'F/SOUTH',
                'F/NORTH': 'F/NORTH',
                'G/SOUTH': 'G/SOUTH',
                'G/NORTH': 'G/NORTH',
                'H/EAST': 'H/EAST',
                'H/WEST': 'H/WEST',
                'K/EAST': 'K/EAST',
                'K/WEST': 'K/WEST',
                'P/SOUTH': 'P/SOUTH',
                'P/NORTH': 'P/NORTH',
                'R/SOUTH': 'R/SOUTH',
                'R/CENTRAL': 'R/CENTRAL',
                'R/NORTH': 'R/NORTH',
                'L': 'L',
                'M/EAST': 'M/EAST',
                'M/WEST': 'M/WEST',
                'N': 'N',
                'S': 'S',
                'T': 'T'
            }
            
            # Apply mapping to clean ward names
            df['WARD'] = df['WARD'].replace(ward_mapping)
            
            # Remove any remaining WARD suffix
            df['WARD'] = df['WARD'].str.replace('WARD', '').str.strip()
            
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
        
        # Print unique ward names for debugging
        print("\nUnique ward names in demand data after cleaning:", df_demand['WARD'].unique())
        
        return df_demand

    def load_climate_data(self):
        """Load climate data and expand monthly data to daily"""
        # Read climate data
        df_climate = pd.read_csv(os.path.join(self.data_dir, 'climat_43057_198206-201901.csv'), skiprows=1)
        
        # Clean column names
        df_climate.columns = df_climate.columns.str.strip()
        
        # Convert to datetime
        df_climate['Date'] = pd.to_datetime(df_climate[['Year', 'Month']].assign(DAY=1), errors='coerce')
        
        # Remove rows with missing dates
        df_climate = df_climate.dropna(subset=['Date'])
        
        # Rename columns
        df_climate = df_climate.rename(columns={
            'Mean Temp.': 'Mean_Temp',
            'Max.Temp.(Monthly Mean)': 'Max_Temp',
            'Min.Temp.(Monthly Mean)': 'Min_Temp',
            'Precip.': 'Precipitation',
            'Mean Temp. Normal': 'Mean_Temp_Normal',
            'Precip. Normal': 'Precip_Normal',
            'SPI 3-Month': 'SPI_3',
            'SPI 6-Month': 'SPI_6',
            'SPI 12-Month': 'SPI_12'
        })
        
        # Convert numeric columns to float
        for col in ['Mean_Temp', 'Max_Temp', 'Min_Temp', 'Mean_Temp_Normal']:
            df_climate[col] = pd.to_numeric(df_climate[col], errors='coerce')
        
        # Convert precipitation to float
        for col in ['Precipitation', 'Precip_Normal']:
            df_climate[col] = pd.to_numeric(df_climate[col], errors='coerce')
        
        # Expand monthly data to daily
        daily_dfs = []
        for _, row in df_climate.iterrows():
            if pd.notna(row['Date']):  # Ensure date is not NaN
                month_dates = pd.date_range(start=row['Date'], 
                                          end=row['Date'] + pd.offsets.MonthEnd(1),
                                          freq='D')
                daily_df = pd.DataFrame({
                    'Date': month_dates,
                    'Mean_Temp': row['Mean_Temp'],
                    'Max_Temp': row['Max_Temp'],
                    'Min_Temp': row['Min_Temp'],
                    'Precipitation': row['Precipitation'] / len(month_dates),  # Distribute monthly precipitation
                    'Mean_Temp_Normal': row['Mean_Temp_Normal'],
                    'Precip_Normal': row['Precip_Normal'] / len(month_dates),
                    'SPI_3': row['SPI_3'],
                    'SPI_6': row['SPI_6'],
                    'SPI_12': row['SPI_12']
                })
                daily_dfs.append(daily_df)
        
        # Combine all daily data
        if daily_dfs:  # Only concatenate if we have data
            df_climate = pd.concat(daily_dfs, ignore_index=True)
        else:
            df_climate = pd.DataFrame(columns=['Date', 'Mean_Temp', 'Max_Temp', 'Min_Temp', 'Precipitation',
                                           'Mean_Temp_Normal', 'Precip_Normal', 'SPI_3', 'SPI_6', 'SPI_12'])
        
        return df_climate[['Date', 'Mean_Temp', 'Max_Temp', 'Min_Temp', 'Precipitation',
                   'Mean_Temp_Normal', 'Precip_Normal', 'SPI_3', 'SPI_6', 'SPI_12']]

    def load_population_data(self):
        """Load and process population data"""
        df = pd.read_csv(os.path.join(self.data_dir, "2011population.csv"))
        return df

    def create_features(self, df):
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

    def create_master_dataset(self):
        """Create and save the master dataset with all features"""
        # Load and merge datasets
        demand_df = self.load_demand_data()
        climate_df = self.load_climate_data()
        population_df = self.load_population_data()
        
        # Merge demand and climate data on Date
        merged_df = pd.merge(demand_df, climate_df, on='Date', how='left')
        print("\nMerged demand and climate:")
        print(merged_df.head())
        
        # Clean ward names in population data to match demand data
        population_df['Ward'] = population_df['Ward'].str.upper()
        
        # Create mapping for special cases
        ward_mapping = {
            'A': 'A',
            'B': 'B',
            'C': 'C',
            'D': 'D',
            'E': 'E',
            'F/S': 'F/SOUTH',
            'F/N': 'F/NORTH',
            'G/S': 'G/SOUTH',
            'G/N': 'G/NORTH',
            'H/E': 'H/EAST',
            'HW': 'H/WEST',
            'K/E': 'K/EAST',
            'K/W': 'K/WEST',
            'P/S': 'P/SOUTH',
            'P/N': 'P/NORTH',
            'R/S': 'R/SOUTH',
            'R/C': 'R/CENTRAL',
            'R/N': 'R/NORTH',
            'L': 'L',
            'M/E': 'M/EAST',
            'M/W': 'M/WEST',
            'N': 'N',
            'S': 'S',
            'T': 'T'
        }
        
        # Apply mapping
        population_df['Ward'] = population_df['Ward'].replace(ward_mapping)
        
        # Clean ward names in demand data
        demand_df['WARD'] = demand_df['WARD'].str.replace('WARD', '').str.strip()
        demand_df['WARD'] = demand_df['WARD'].str.replace('-', '').str.upper()
        
        # Create mapping for demand data
        demand_ward_mapping = {
            'A': 'A',
            'B': 'B',
            'C': 'C',
            'D': 'D',
            'E': 'E',
            'F/SOUTH': 'F/SOUTH',
            'F/NORTH': 'F/NORTH',
            'G/SOUTH': 'G/SOUTH',
            'G/NORTH': 'G/NORTH',
            'H/EAST': 'H/EAST',
            'H/WEST': 'H/WEST',
            'K/EAST': 'K/EAST',
            'K/WEST': 'K/WEST',
            'P/SOUTH': 'P/SOUTH',
            'P/NORTH': 'P/NORTH',
            'R/SOUTH': 'R/SOUTH',
            'R/CENTRAL': 'R/CENTRAL',
            'R/NORTH': 'R/NORTH',
            'L': 'L',
            'M/EAST': 'M/EAST',
            'M/WEST': 'M/WEST',
            'N': 'N',
            'S': 'S',
            'T': 'T'
        }
        
        # Apply mapping to demand data
        demand_df['WARD'] = demand_df['WARD'].replace(demand_ward_mapping)
        
        # Print unique ward names for debugging
        print("\nUnique ward names in demand data:", demand_df['WARD'].unique())
        print("\nUnique ward names in population data:", population_df['Ward'].unique())
        
        # Merge with population data
        final_df = pd.merge(merged_df, population_df, left_on='WARD', right_on='Ward', how='left')
        print("\nFinal merged dataset:")
        print(final_df.head())
        
        # Add date features
        final_df['Year'] = final_df['Date'].dt.year
        final_df['Month'] = final_df['Date'].dt.month
        final_df['Day'] = final_df['Date'].dt.day
        final_df['DayOfWeek'] = final_df['Date'].dt.dayofweek
        
        # Calculate year difference from 2011
        final_df['Year_Diff'] = final_df['Year'] - 2011
        
        # Calculate estimated population growth
        final_df['Estimated_Population'] = final_df['Slum Population'] + final_df['Non Slum Population']
        final_df['Estimated_Population'] = final_df.groupby('WARD')['Estimated_Population'].transform(
            lambda x: x.iloc[0] * (1 + 0.02)**final_df['Year_Diff'])
        
        # Add seasonality
        final_df['Season'] = final_df['Month'].apply(lambda x: 'Summer' if x in [3,4,5] else 
                                                     'Monsoon' if x in [6,7,8] else 
                                                     'Post-Monsoon' if x in [9,10,11] else 'Winter')
        
        print("\nSample of the master dataset:")
        print(final_df.head())
        
        # Save master dataset
        os.makedirs('../../datasets', exist_ok=True)
        final_df.to_csv('../../datasets/master_dataset.csv', index=False)
        print(f"\nMaster dataset saved to ../../datasets/master_dataset.csv")
        
        return final_df

def get_master_dataset(self):
    """Return the master dataset"""
    if self.master_df is None:
        self.create_master_dataset()
    return self.master_df


if __name__ == "__main__":
    dataset = WaterDemandDataset()
    master_df = dataset.create_master_dataset()
    print("\nSample of the master dataset:")
    print(master_df.head())
