import pandas as pd
import numpy as np

def engineer_features(df):
    """
    Transforms raw sensor data into features highlighting trends.
    Uses rolling averages and rolling standard deviations.
    """
    df = df.sort_values(['machine_id', 'timestamp'])
    
    # Calculate rolling metrics for each machine
    for sensor in ['temperature', 'vibration', 'pressure']:
        # 6-hour rolling mean
        df[f'{sensor}_roll_mean'] = df.groupby('machine_id')[sensor].transform(
            lambda x: x.rolling(window=6, min_periods=1).mean()
        )
        # 6-hour rolling std (captures volatility)
        df[f'{sensor}_roll_std'] = df.groupby('machine_id')[sensor].transform(
            lambda x: x.rolling(window=6, min_periods=1).std().fillna(0)
        )
        
    # Drop raw timestamp and machine_id for ML (id kept for dashboard)
    features = [col for col in df.columns if 'roll' in col or col in ['temperature', 'vibration', 'pressure']]
    
    return df, features

if __name__ == "__main__":
    print("Preprocessing Data and Engineering Features...")
    try:
        df = pd.read_csv('data/sensor_data.csv')
        df_processed, feature_cols = engineer_features(df)
        df_processed.to_csv('data/processed_data.csv', index=False)
        print(f"Success! Processed features: {feature_cols}")
    except FileNotFoundError:
        print("Error: 'data/sensor_data.csv' not found. Run simulator.py first.")
