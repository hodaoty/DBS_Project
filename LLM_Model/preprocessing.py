import pandas as pd
import os
import joblib
from datetime import datetime
from sklearn.preprocessing import StandardScaler

# ----------------------------------------------------------------------
# A. CONFIGURATION AND PATH SETUP
# ----------------------------------------------------------------------
# Resample frequency (must match across all modules)
RESAMPLE_FREQUENCY = '30s'

base_dir = os.path.dirname(os.path.abspath(__file__))
timestamp_str = datetime.now().strftime("%Y%m%d")

# Define base paths
CSV_DIR = os.path.join(base_dir, '..', 'CSV_FILE', 'OUTPUT_CSVFILE')
MODEL_DIR = os.path.join(base_dir, 'trained_model') 

# Define input/output paths (Must match the directory structure used in main.py)
INPUT_EVENTS_FILE = f'postgresql_events-{timestamp_str}.csv'
INPUT_EVENTS_PATH = os.path.join(CSV_DIR, 'LOG_EVENT', INPUT_EVENTS_FILE)

OUTPUT_SCALED_DATA_PATH = os.path.join(CSV_DIR, 'TRAIN_AI', f'processed_scaled_features-{timestamp_str}.csv')
OUTPUT_SCALER_PATH = os.path.join(MODEL_DIR, f'scaler-{timestamp_str}.pkl')

# Ensure Model and Train directories exist
if not os.path.exists(MODEL_DIR): os.makedirs(MODEL_DIR)
if not os.path.exists(os.path.join(CSV_DIR, 'TRAIN_AI')): os.makedirs(os.path.join(CSV_DIR, 'TRAIN_AI'))

# ----------------------------------------------------------------------
# B. DATA LOADING AND PREPARATION
# ----------------------------------------------------------------------
def load_and_prepare_data(filepath):
    """Loads event data and sets the timestamp column as the index."""
    if not os.path.exists(filepath):
        print(f"ERROR: Event file not found at: {filepath}")
        return None
        
    print(f"Loading data from: {filepath}")
    
    try:
        # 1. Read CSV and Parse Dates (essential for time series)
        # Assuming 01_data_extraction.py outputs timestamp in a parsable format
        df = pd.read_csv(filepath, parse_dates=['timestamp'])
        
        # 2. Set Index
        df = df.set_index('timestamp')
        
        print(f"Loading successful. Data shape: {df.shape}")
        return df
    except Exception as e:
        print(f"ERROR reading or preparing CSV: {e}")
        return None

# ----------------------------------------------------------------------
# C. TIME SERIES FEATURE ENGINEERING
# ----------------------------------------------------------------------
def create_time_series_features(df, freq=RESAMPLE_FREQUENCY):
    """
    Creates time series features by resampling log data into aggregate 
    windows (e.g., 5-minute windows).
    """
    # Drop non-feature ID columns that were retained from 01_data_extraction.py
    features_df = df.drop(columns=['pid', 'user', 'database', 'query_command', 'query_text'], errors='ignore')

    # 1. Create Count Features
    # Group by event_type to count different log levels/actions
    count_features = features_df['event_type'].groupby(level=0).value_counts().unstack(fill_value=0)
    
    # Rename columns (e.g., FATAL -> count_fatal)
    count_features.columns = [f'count_{col.lower().replace(" ", "_")}' for col in count_features.columns]
    
    # Add total events (for ratio calculation)
    count_features['count_total_events'] = count_features.sum(axis=1)

    # 2. Create Time Features
    # Aggregate statistics on session_duration_sec
    time_features = features_df['session_duration_sec'].resample(freq).agg({
        'avg_session_duration': 'mean',
        'max_session_duration': 'max',
        'total_session_time': 'sum'
    }).fillna(0)

    # 3. Combine Features
    # Resample and combine count features (summing up counts within the window)
    final_features_df = count_features.resample(freq).sum().fillna(0)
    final_features_df = final_features_df.join(time_features)

    # 4. Create Ratio Features
    # Ratio of fatal errors to total events
    final_features_df['ratio_fatal_to_total'] = (
        final_features_df.get('count_fatal', 0) / final_features_df['count_total_events']
    ).fillna(0)
    
    # Drop the auxiliary 'count_total_events' column
    final_features_df = final_features_df.drop(columns=['count_total_events'], errors='ignore')

    print(f"Feature Engineering complete. Number of features: {len(final_features_df.columns)}")
    return final_features_df

# ----------------------------------------------------------------------
# D. DATA SCALING
# ----------------------------------------------------------------------
def scale_features(df):
    """Scales features using StandardScaler and saves the fitted scaler."""
    scaler = StandardScaler()
    
    # Fit and transform the data
    scaled_data = scaler.fit_transform(df)
    
    # Convert NumPy array back to DataFrame
    scaled_df = pd.DataFrame(scaled_data, index=df.index, columns=df.columns)
    
    # Save the scaler for use in real-time detection
    joblib.dump(scaler, OUTPUT_SCALER_PATH)
    print(f"Standard Scaler saved to: {OUTPUT_SCALER_PATH}")
    
    return scaled_df

# ----------------------------------------------------------------------
# E. MAIN EXECUTION LOGIC
# ----------------------------------------------------------------------
if __name__ == "__main__":
    
    # 1. Load and prepare data
    events_df = load_and_prepare_data(INPUT_EVENTS_PATH)
    
    if events_df is None or events_df.empty:
        print("Invalid or empty event data. Stopping preprocessing.")
    else:
        # 2. Feature Engineering
        features_df = create_time_series_features(events_df)
        
        # Check if any features were created
        if features_df.empty:
            print("No features were created after resampling. Stopping preprocessing.")
        else:
            # 3. Scale data and save scaler
            scaled_features_df = scale_features(features_df)
            
            # 4. Save scaled data for model training
            scaled_features_df.to_csv(OUTPUT_SCALED_DATA_PATH)
            print(f"Scaled data saved to: {OUTPUT_SCALED_DATA_PATH}")
            
            print("\nPreprocessing complete. Ready for 03_model_training.py")
