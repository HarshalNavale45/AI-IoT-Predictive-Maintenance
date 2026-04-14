import pandas as pd
import numpy as np
import joblib
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def generate_visuals(model, X, y, df):
    """
    Generates a suite of analytical plots for the project.
    """
    if not os.path.exists('outputs'):
        os.makedirs('outputs')
    
    # 1. Feature Importance Plot
    plt.figure(figsize=(10, 6))
    importances = model.feature_importances_
    feat_importances = pd.Series(importances, index=X.columns)
    feat_importances.nlargest(10).plot(kind='barh', color='#58a6ff')
    plt.title('Top AI Decision Drivers (Feature Importance)')
    plt.xlabel('Importance Score')
    plt.tight_layout()
    plt.savefig('outputs/feature_importance.png')
    plt.close()

    # 2. Correlation Heatmap
    plt.figure(figsize=(12, 8))
    corr = X.corr()
    sns.heatmap(corr, annot=True, cmap='RdBu_r', center=0)
    plt.title('Sensor Correlation Matrix')
    plt.tight_layout()
    plt.savefig('outputs/correlation_heatmap.png')
    plt.close()

    # 3. Failure Trend Analysis
    # Pick a random machine that actually has a failure
    fail_machines = df[df['fail'] == 1]['machine_id'].unique()
    if len(fail_machines) > 0:
        target_m = fail_machines[0]
        m_df = df[df['machine_id'] == target_m].sort_values('timestamp').tail(100)
        
        fig, ax1 = plt.subplots(figsize=(12, 6))
        ax1.set_xlabel('Time Steps')
        ax1.set_ylabel('Temperature / Pressure', color='tab:blue')
        ax1.plot(m_df.index, m_df['temperature'], color='tab:red', label='Temp', alpha=0.8)
        ax1.plot(m_df.index, m_df['pressure'], color='tab:blue', label='Pressure', alpha=0.5)
        
        ax2 = ax1.twinx()
        ax2.set_ylabel('Failure State', color='black')
        ax2.fill_between(m_df.index, 0, m_df['fail'], color='red', alpha=0.2, label='Degradation Zone')
        
        plt.title(f'Degradation Pattern Analysis: Asset {target_m}')
        fig.tight_layout()
        plt.savefig('outputs/failure_trend.png')
        plt.close()

def train_model():
    print("Training Predictive Maintenance Model...")
    
    # Load data
    df = pd.read_csv('data/processed_data.csv')
    
    # Define features and target
    X = df.drop(['timestamp', 'machine_id', 'fail'], axis=1)
    y = df['fail']
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Train Random Forest
    model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    print("\nModel Performance:")
    print(classification_report(y_test, y_pred))
    
    # Generate Advanced Visuals
    print("Generating Advanced Analytical Visuals...")
    generate_visuals(model, X, y, df)
    
    # Save Model
    if not os.path.exists('models'):
        os.makedirs('models')
    joblib.dump(model, 'models/model.pkl')
    print("Model saved to 'models/model.pkl'")
    joblib.dump(X.columns.tolist(), 'models/features.pkl')
    
    # Visualization: Confusion Matrix
    plt.figure(figsize=(6,4))
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix: Failure Prediction')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.savefig('outputs/confusion_matrix.png')
    plt.close()
    
    print("All visualizations saved to 'outputs/' directory.")

if __name__ == "__main__":
    try:
        train_model()
    except Exception as e:
        print(f"Error during training: {e}")
