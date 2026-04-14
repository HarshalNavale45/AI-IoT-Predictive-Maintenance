import pandas as pd
import numpy as np
import os

def generate_iot_data(n_days=30, n_machines=5, frequency='h'):
    """
    Simulates IoT sensor data for industrial motors.
    Features: Temperature, Vibration, Pressure.
    Logic: Machines run normally but occasionally start degrading until failure.
    """
    np.random.seed(42)
    
    timestamps = pd.date_range(start='2024-01-01', periods=n_days * 24, freq=frequency)
    data_list = []

    for machine_id in range(1, n_machines + 1):
        # Base healthy values
        temp_base = 65
        vib_base = 0.05
        press_base = 100
        
        # Simulation parameters
        machine_data = pd.DataFrame({'timestamp': timestamps})
        machine_data['machine_id'] = f'M_{machine_id:03d}'
        
        # Generate Normal Noise
        machine_data['temperature'] = temp_base + np.random.normal(0, 2, len(timestamps))
        machine_data['vibration'] = vib_base + np.random.normal(0, 0.01, len(timestamps))
        machine_data['pressure'] = press_base + np.random.normal(0, 5, len(timestamps))
        
        # Inject Failure Patterns (One failure per machine at random midpoint)
        fail_index = np.random.randint(len(timestamps)//2, len(timestamps)-20)
        
        # Degradation starts 24 cycles before failure
        degradation_start = fail_index - 48
        
        # Gradual increase in temp and vibration
        for i in range(degradation_start, fail_index):
            machine_data.loc[i, 'temperature'] += (i - degradation_start) * 0.5
            machine_data.loc[i, 'vibration'] += (i - degradation_start) * 0.005
            machine_data.loc[i, 'pressure'] -= (i - degradation_start) * 0.2
            
        # Labels
        machine_data['fail'] = 0
        # Label the failure window (e.g., 24 hours before failure)
        machine_data.loc[fail_index-24:fail_index, 'fail'] = 1
        
        data_list.append(machine_data)

    full_data = pd.concat(data_list)
    return full_data

if __name__ == "__main__":
    print("Simulating IoT Sensor Data...")
    df = generate_iot_data()
    
    # Create data directory if not exists
    if not os.path.exists('data'):
        os.makedirs('data')
        
    df.to_csv('data/sensor_data.csv', index=False)
    print(f"Success! Generated {len(df)} rows of data in 'data/sensor_data.csv'")
    print(df.head())
