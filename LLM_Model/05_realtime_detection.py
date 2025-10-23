import pandas as pd
import os
import joblib
from datetime import datetime, timedelta
import numpy as np

# ----------------------------------------------------------------------
# A. CONFIGURATION AND PATH SETUP
# ----------------------------------------------------------------------
timestamp_str = datetime.now().strftime('%Y%m%d')
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Adjusted MODEL_DIR path: assumes it's one level up from LLM_Model
MODEL_DIR = os.path.join(BASE_DIR, 'trained_model') 

# File names
MODEL_FILE_NAME = f'isolation_forest_model-{timestamp_str}.pkl'
SCALER_FILE_NAME = f'scaler-{timestamp_str}.pkl'

MODEL_PATH = os.path.join(MODEL_DIR, MODEL_FILE_NAME)
SCALER_PATH = os.path.join(MODEL_DIR, SCALER_FILE_NAME)

# Resample frequency (must match across all modules)
RESAMPLE_FREQUENCY = '5T' 

# Time window duration for simulation
window_duration = pd.Timedelta(RESAMPLE_FREQUENCY)

# ----------------------------------------------------------------------
# B. MODEL AND SCALER LOADING
# ----------------------------------------------------------------------
def load_detection_assets():
    """Loads the trained Isolation Forest Model and Standard Scaler."""
    try:
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        # Replaced Unicode/Vietnamese with ASCII
        print(f"SUCCESS: Loaded Model from: {MODEL_PATH}")
        print(f"SUCCESS: Loaded Scaler from: {SCALER_PATH}")
        return model, scaler
    except FileNotFoundError:
        # Replaced Unicode/Vietnamese with ASCII
        print("ERROR: Model or scaler file not found. Ensure 02 and 03 scripts ran successfully.")
        return None, None
    except Exception as e:
        # Replaced Unicode/Vietnamese with ASCII
        print(f"ERROR loading assets: {e}")
        return None, None

# ----------------------------------------------------------------------
# C. SIMULATION DATA GENERATION (Real-time/Next Batch)
# ----------------------------------------------------------------------
def generate_new_log_batch(reference_df, start_time, end_time, scenario="normal"):
    """
    Generates a simulated batch of raw log data for a time window.
    """
    sample_df = reference_df[
        (reference_df['timestamp'] >= start_time) & (reference_df['timestamp'] < end_time)
    ].copy()

    if sample_df.empty:
        # Create minimal log data if no real samples are found
        data = {
            'pid': [0], 'user': [None], 'database': [None], 'event_type': ['LOG'],
            'session_duration_sec': [0.0], 'query_command': [None], 
            'query_text': [None], 'timestamp': [start_time]
        }
        sample_df = pd.DataFrame(data)
        
    if scenario == "stress":
        # Simulate Anomaly: Increase event count and inject severe errors (FATAL)
        
        # 1. Significantly increase event volume (10x)
        sample_df = pd.concat([sample_df] * 10, ignore_index=True)
        
        # 2. Inject FATAL/ERRORs (30% of new events)
        error_events = []
        error_count = int(len(sample_df) * 0.30)
        for i in range(error_count):
            error_time = start_time + timedelta(seconds=np.random.rand() * window_duration.total_seconds())
            error_events.append({
                'pid': 9999, 'user': 'attack', 'database': 'critical_db', 
                'event_type': 'FATAL', 'session_duration_sec': 0.0, 
                'query_command': 'SELECT', 'query_text': 'UNAUTHORIZED ACCESS', 
                'timestamp': error_time
            })
        sample_df = pd.concat([sample_df, pd.DataFrame(error_events)], ignore_index=True)
        
        print("INFO: 'stress' scenario (FATAL/Connect spike) simulated.")
        
    # Set index and sort
    sample_df = sample_df.sort_values(by='timestamp').set_index('timestamp', drop=False)
    return sample_df

# ----------------------------------------------------------------------
# D. FEATURE CREATION (Must match 02_preprocessing.py exactly)
# ----------------------------------------------------------------------
def create_time_series_features(df, freq=RESAMPLE_FREQUENCY):
    """
    Creates time series features by resampling log data.
    """
    # Drop non-feature ID columns
    df_temp = df.copy().set_index('timestamp', drop=True) 

    features_df = df_temp.drop(columns=['pid', 'user', 'database', 'query_command', 'query_text'], errors='ignore')
    
    # 1. Count Features
    count_features = features_df['event_type'].groupby(level=0).value_counts().unstack(fill_value=0)
    count_features.columns = [f'count_{col.lower().replace(" ", "_")}' for col in count_features.columns]
    count_features['count_total_events'] = count_features.sum(axis=1)

    # 2. Time Features
    time_features = features_df['session_duration_sec'].resample(freq).agg({
        'avg_session_duration': 'mean',
        'max_session_duration': 'max',
        'total_session_time': 'sum'
    }).fillna(0)

    # 3. Combine Features
    final_features_df = count_features.resample(freq).sum().fillna(0)
    final_features_df = final_features_df.join(time_features)

    # 4. Ratio Features
    final_features_df['ratio_fatal_to_total'] = (
        final_features_df.get('count_fatal', 0) / final_features_df['count_total_events']
    ).fillna(0)
    
    # Drop the auxiliary 'count_total_events' column
    final_features_df = final_features_df.drop(columns=['count_total_events'], errors='ignore')

    return final_features_df

# ----------------------------------------------------------------------
# E. REAL-TIME DETECTION FUNCTION
# ----------------------------------------------------------------------
def detect_anomalies_realtime(model, scaler, new_log_data_df):
    """
    Applies the ML pipeline (preprocessing, scaling, prediction) to new data.
    """
    if new_log_data_df.empty:
        print("No new log data to process.")
        return None

    # 1. Feature Engineering (5T aggregation)
    features_df = create_time_series_features(new_log_data_df)
    
    # 2. Feature Alignment
    if not hasattr(scaler, 'feature_names_in_') or scaler.feature_names_in_ is None:
        print("ERROR: Scaler feature names missing.")
        return None
        
    all_feature_names = list(scaler.feature_names_in_)
    
    # Pad missing features with 0.0
    for col in all_feature_names:
        if col not in features_df.columns:
            features_df[col] = 0.0
            
    # Reorder columns to match the trained model
    features_df = features_df[all_feature_names]

    # 3. Scaling (using loaded scaler.transform)
    scaled_data = scaler.transform(features_df)
    scaled_df = pd.DataFrame(scaled_data, index=features_df.index, columns=features_df.columns)

    # 4. Prediction
    X_predict_features = scaled_df.copy()

    scaled_df['anomaly_score'] = model.decision_function(X_predict_features)
    scaled_df['anomaly'] = model.predict(X_predict_features)

    # Filter out anomalies
    anomalies = scaled_df[scaled_df['anomaly'] == -1].copy()

    return anomalies

# ----------------------------------------------------------------------
# F. MAIN EXECUTION LOGIC
# ----------------------------------------------------------------------
if __name__ == "__main__":
    
    # 1. Load Model and Scaler
    anomaly_model, scaler = load_detection_assets()
    
    if anomaly_model is None or scaler is None:
        exit()

    # 2. Load original event data for simulation
    CSV_DIR_EVENT = os.path.join(BASE_DIR, '..', 'CSV_FILE', 'OUTPUT_CSVFILE', 'LOG_EVENT')
    EVENTS_CSV_PATH = os.path.join(CSV_DIR_EVENT, f'postgresql_events-{timestamp_str}.csv') 
    
    if not os.path.exists(EVENTS_CSV_PATH):
        print(f"WARNING: Original event file not found at {EVENTS_CSV_PATH}. Cannot simulate.")
        exit()

    full_events_df = pd.read_csv(EVENTS_CSV_PATH, parse_dates=['timestamp'])
    full_events_df = full_events_df.set_index('timestamp', drop=False)
    
    
    # --- SIMULATION SCENARIO SETUP ---
    last_log_time = full_events_df.index.max()
    
    sim_start_time = last_log_time + window_duration
    sim_end_time = sim_start_time + window_duration 

    print("\n--- Starting Real-Time Anomaly Detection Simulation ---")
    
    # Example 1: NORMAL Scenario
    print(f"\n[1] Monitoring NORMAL window ({sim_start_time} - {sim_end_time})")
    sim_normal_data = generate_new_log_batch(full_events_df, sim_start_time, sim_end_time, scenario="normal")
    anomalies_normal = detect_anomalies_realtime(anomaly_model, scaler, sim_normal_data)
    
    if anomalies_normal is None or anomalies_normal.empty:
        print("RESULT (NORMAL): No anomalies detected.")
    else:
        print(f"WARNING (NORMAL): Detected {len(anomalies_normal)} anomalies.")

    # Example 2: STRESS/ATTACK Scenario
    sim_start_time_stress = sim_end_time + window_duration
    sim_end_time_stress = sim_start_time_stress + window_duration

    print(f"\n[2] Monitoring STRESS window ({sim_start_time_stress} - {sim_end_time_stress})")
    sim_stress_data = generate_new_log_batch(full_events_df, sim_start_time_stress, sim_end_time_stress, scenario="stress")
    anomalies_stress = detect_anomalies_realtime(anomaly_model, scaler, sim_stress_data)

    if anomalies_stress is None or anomalies_stress.empty:
        print("RESULT (STRESS): No anomalies detected.")
    else:
        # Replaced Unicode with ASCII
        print(f"--- ALERT --- Detected {len(anomalies_stress)} anomalies! --- ALERT ---")
        
        # Display features of the detected anomalies
        display_cols = [col for col in anomalies_stress.columns if col not in ['anomaly_score', 'anomaly']]
        
        top_anomalies = anomalies_stress.sort_values(by='anomaly_score', ascending=True).head(3)
        
        print("\nFeatures of the 3 most severe anomalies (Scaled Values):")
        print(top_anomalies[['anomaly_score'] + display_cols].to_string())
