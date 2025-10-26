import os 
import pandas as pd
from datetime import datetime
from sklearn.ensemble import IsolationForest
import joblib

# ----------------------------------------------------------------------
# A. CONFIGURATION AND PATH SETUP
# ----------------------------------------------------------------------
# Fixed variable name from timestamp_srt to timestamp_str
timestamp_str = datetime.now().strftime('%Y%m%d')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Updated CSV_DIR to point to the directory containing processed_scaled_features
CSV_DIR = os.path.join(BASE_DIR, '..','CSV_FILE','OUTPUT_CSVFILE','TRAIN_AI') 
MODEL_DIR = os.path.join(BASE_DIR,'trained_model')

# Define input/output file names
INPUT_SCALED_DATA_FILE = f'processed_scaled_features-{timestamp_str}.csv'
MODEL_FILE_NAME = f'isolation_forest_model-{timestamp_str}.pkl'

INPUT_SCALED_DATA_PATH = os.path.join(CSV_DIR, INPUT_SCALED_DATA_FILE)
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_FILE_NAME)

# Model parameters
CONTAMINATION_RATE = 0.01

# ----------------------------------------------------------------------
# B. DATA LOADING FUNCTION
# ----------------------------------------------------------------------
def load_scaled_data(file_path):
    """Loads the scaled feature data from CSV."""
    if not os.path.exists(file_path):
        print(f"ERROR: Scaled data file not found at: {file_path}")
        return None
        
    print(f"Loading data from: {file_path}") 
    try:
        # Read data, set 'timestamp' as index
        df = pd.read_csv(file_path, index_col='timestamp', parse_dates=True)
        print(f"Loading successful. Data shape: {df.shape}") 
        return df
    except Exception as e:
        print(f"ERROR reading CSV file: {e}")
        return None
# ----------------------------------------------------------------------
# C. MODEL TRAINING FUNCTION
# ----------------------------------------------------------------------
def train_anomaly_model(data_df):
    """
    Trains the Isolation Forest model for anomaly detection.
    """
    print(f"\nStarting Isolation Forest training (Contamination={CONTAMINATION_RATE})...")

    # Initialize Isolation Forest model
    model = IsolationForest(
        contamination=CONTAMINATION_RATE,
        random_state=42,
        n_estimators=100,
        n_jobs=-1          
    )
    # Train the model
    model.fit(data_df)

    print("Training complete.")
    return model
# ----------------------------------------------------------------------
# D. MODEL SAVING FUNCTION
# ----------------------------------------------------------------------
def save_model(model, model_path):
    """
    Saves the trained model to a file.
    """
    if not os.path.exists(os.path.dirname(model_path)):
        os.makedirs(os.path.dirname(model_path))
        print(f"Created model saving directory: {os.path.dirname(model_path)}")

    joblib.dump(model, model_path)
    print(f"Model saved to: {model_path}")

# ----------------------------------------------------------------------
# E. MAIN EXECUTION LOGIC
# ----------------------------------------------------------------------
if __name__ == "__main__":
    
    # 1. Load input data
    scaled_data_df = load_scaled_data(INPUT_SCALED_DATA_PATH)

    if scaled_data_df is None or scaled_data_df.empty:
        print("Invalid or empty input data. Stopping training process.")
    else:
        # Create feature dataframe copy
        X_features = scaled_data_df.copy()
        
        # 2. Train the model
        anomaly_model = train_anomaly_model(X_features) 
        
        # 3. Save the trained model
        save_model(anomaly_model, MODEL_PATH)
        
        # 4. Model check and anomaly extraction
        print("\n--- Quick check of classification results on training data ---")
        
        # Calculate anomaly score and prediction
        scaled_data_df['anomaly_score'] = anomaly_model.decision_function(X_features)
        scaled_data_df['anomaly'] = anomaly_model.predict(X_features)

        # Calculate results
        num_anomalies = (scaled_data_df['anomaly'] == -1).sum()
        total_samples = len(scaled_data_df)

        print(f"Total data samples (30s)): {total_samples}")
        print(f"Number of anomalies found: {num_anomalies}")
        print(f"Anomaly ratio (Model): {num_anomalies/total_samples:.2%}")
        print(f"Anomaly ratio (Config): {CONTAMINATION_RATE:.2%}")

        # 5. SAVE ANOMALY RECORDS TO CSV
        print(f"Save to the report then item have anomaly_score < -0.15")
        anomalies_df = scaled_data_df[(scaled_data_df['anomaly'] == -1) & (scaled_data_df['anomaly_score'] < -0.15) ].copy()
        
        if not anomalies_df.empty:
            # Output path for anomaly records (relative to CSV_DIR)
            ANOMALY_OUTPUT_PATH = os.path.join(os.path.dirname(INPUT_SCALED_DATA_PATH), f'anomaly_records-{timestamp_str}.csv')
            
            anomalies_df.to_csv(ANOMALY_OUTPUT_PATH)
            
            print(f"\nSuccessfully saved {len(anomalies_df)} anomaly records to: {ANOMALY_OUTPUT_PATH}")
            
            # --- OPTIMIZED DISPLAY ---
            base_cols = ['avg_session_duration', 'max_session_duration', 'ratio_fatal_to_total', 'anomaly_score']
            count_cols = [col for col in anomalies_df.columns if col.startswith('count_') and ('error' in col or 'fatal' in col or 'connect' in col)]
            
            display_cols = base_cols + count_cols
            available_cols = [col for col in display_cols if col in anomalies_df.columns]
            
            print("\n5 most severe anomaly records:")
            
            # Sort and select top anomalies
            top_anomalies = anomalies_df.sort_values(by='anomaly_score').head(5)
            
            # ⚠️ FIX: Đặt lại index để 'timestamp' trở thành cột, cho phép truy cập nó
            top_anomalies = top_anomalies.reset_index()
            
            # ⚠️ FIX: Thay thế 'timestamp' bằng 'index' trong list cột nếu index ban đầu không tên
            final_display_cols = ['timestamp'] + available_cols
            
            # In ra console
            print(top_anomalies[final_display_cols].to_string())
        
        print("\nModel training process complete.")
