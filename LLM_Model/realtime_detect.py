# A. IMPORTS
import os
import sys
import time
import pandas as pd
from datetime import datetime
import joblib

# Add parser directory to sys.path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

from data_extraction import parse_postgresql_log  # reuse the parser function

# B. CONFIGURATION
timestamp_str = datetime.now().strftime('%Y%m%d')
MODEL_DIR = os.path.join(BASE_DIR, 'trained_model')
LOG_FILE_PATH = os.path.join(BASE_DIR, '..', 'Log_Example', 'postgresql-official.log')

# File paths
SCALER_PATH = os.path.join(MODEL_DIR, f'scaler-{timestamp_str}.pkl')
MODEL_PATH = os.path.join(MODEL_DIR, f'isolation_forest_model-{timestamp_str}.pkl')

# Monitoring interval
POLL_INTERVAL = 5  # seconds

# C. LOAD MODEL & SCALER
try:
    scaler = joblib.load(SCALER_PATH)
    model = joblib.load(MODEL_PATH)
    EXPECTED_FEATURES = scaler.feature_names_in_.tolist()
    print("Model and scaler loaded successfully.")
except Exception as e:
    print(f"Error loading model or scaler: {e}")
    sys.exit(1)

# D. MONITORING LOOP
def monitor_log():
    print("Starting real-time log monitoring. Press Ctrl + C to stop.")
    last_position = 0

    try:
        while True:
            with open(LOG_FILE_PATH, 'r', encoding='utf-8') as f:
                f.seek(last_position)
                new_lines = f.readlines()
                last_position = f.tell()

            if new_lines:
                temp_log_path = os.path.join(BASE_DIR, 'temp_log_chunk.log')
                with open(temp_log_path, 'w', encoding='utf-8') as temp_f:
                    temp_f.writelines(new_lines)

                parsed_df = parse_postgresql_log(temp_log_path)

                if not parsed_df.empty:
                    parsed_df = parsed_df.set_index('timestamp')
                    features_df = parsed_df.drop(columns=['pid', 'user', 'database', 'query_command', 'query_text'], errors='ignore')

                    count_features = features_df['event_type'].value_counts().to_frame().T
                    count_features.columns = [f'count_{col.lower().replace(" ", "_")}' for col in count_features.columns]
                    count_features['count_total_events'] = count_features.sum(axis=1)

                    count_features['ratio_fatal_to_total'] = (
                        count_features.get('count_fatal', 0) / count_features['count_total_events']
                    ).fillna(0)

                    if 'session_duration_sec' in features_df.columns:
                        count_features['avg_session_duration'] = features_df['session_duration_sec'].mean()
                        count_features['max_session_duration'] = features_df['session_duration_sec'].max()
                        count_features['total_session_time'] = features_df['session_duration_sec'].sum()
                    else:
                        count_features['avg_session_duration'] = 0.0
                        count_features['max_session_duration'] = 0.0
                        count_features['total_session_time'] = 0.0

                    # Add missing expected features
                    for col in EXPECTED_FEATURES:
                        if col not in count_features.columns:
                            count_features[col] = 0.0

                    # Ensure correct column order
                    count_features = count_features[EXPECTED_FEATURES]

                    # Scale and predict
                    scaled = scaler.transform(count_features)
                    score = model.decision_function(scaled)[0]
                    prediction = model.predict(scaled)[0]

                    if prediction == -1 and score < -0.8:
                        print(f"\nCRITICAL RISK Anomaly detected at {datetime.now().strftime('%H:%M:%S')} | Score: {score:.4f}")
                        print(parsed_df[['pid', 'user', 'database', 'query_command']].to_string(index=False))
                    elif prediction == -1 and score < -0.5:
                        print(f"\nHIGH RISK Anomaly detected at {datetime.now().strftime('%H:%M:%S')} | Score: {score:.4f}")
                        print(parsed_df[['pid', 'user', 'database', 'query_command']].to_string(index=False))
                    elif prediction == -1 and score < -0.2:
                        print(f"\nMEDIMUM RISK Anomaly detected at {datetime.now().strftime('%H:%M:%S')} | Score: {score:.4f}")
                        print(parsed_df[['pid', 'user', 'database', 'query_command']].to_string(index=False))
                    elif prediction == -1: 
                        print(f"\n NORMAL RISK Anomaly detected at {datetime.now().strftime('%H:%M:%S')} | Score: {score:.4f}")
                        print(parsed_df[['pid', 'user', 'database', 'query_command']].to_string(index=False))

            time.sleep(POLL_INTERVAL)

    except KeyboardInterrupt:
        print("\nMonitoring stopped by user (Ctrl + C). Goodbye!")

# E. MAIN EXECUTION 
if __name__ == "__main__":
    print('Start Realtime Detection')
    monitor_log()
