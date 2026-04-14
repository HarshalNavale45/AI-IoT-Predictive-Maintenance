import os
import subprocess
import sys

def main():
    print("AI-Powered Predictive Maintenance System for IoT")
    print("================================================")
    
    # Check if data exists, if not generate
    if not os.path.exists('data/sensor_data.csv'):
        print("\nStep 1: Generating Synthetic IoT Data...")
        subprocess.run([sys.executable, 'src/simulator.py'])
    
    # Preprocess
    print("\nStep 2: Preprocessing and Feature Engineering...")
    subprocess.run([sys.executable, 'src/preprocess.py'])
    
    # Train
    print("\nStep 3: Training Machine Learning Model...")
    subprocess.run([sys.executable, 'src/train.py'])
    
    print("\nPipeline Complete! You can now launch the dashboard:")
    print("Run: streamlit run dashboard/app.py")

if __name__ == "__main__":
    main()
