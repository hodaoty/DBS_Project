import pandas as pd
import os
from datetime import datetime
import re

# ----------------------------------------------------------------------
# A. CONFIGURATION AND PATH SETUP
# ----------------------------------------------------------------------
timestamp_str = datetime.now().strftime('%Y%m%d')
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_DIR = os.path.join(BASE_DIR, '..', 'CSV_FILE', 'OUTPUT_CSVFILE')

# Anomaly file name (OUTPUT from 03_model_training.py)
ANOMALY_FILE = f'anomaly_records-{timestamp_str}.csv'
# Path: TRAIN_AI subdirectory
ANOMALY_PATH = os.path.join(CSV_DIR, 'TRAIN_AI', ANOMALY_FILE) 

# Detailed event file name (OUTPUT from 01_data_extraction.py) - CONTAINS PID
EVENTS_FILE = f'postgresql_events-{timestamp_str}.csv'
# Path: LOG_EVENT subdirectory
EVENTS_PATH = os.path.join(CSV_DIR, 'LOG_EVENT', EVENTS_FILE) 

# Output report file name
OUTPUT_PID_REPORT_FILE = f'anomalous_pid_report-{timestamp_str}.csv'
# Path: REPORT subdirectory
OUTPUT_PID_REPORT_PATH = os.path.join(CSV_DIR, 'REPORT', OUTPUT_PID_REPORT_FILE) 

# Time window configuration (must match 02_preprocessing.py)
RESAMPLE_FREQUENCY = '5T' # 5 minutes

# ----------------------------------------------------------------------
# B. LOOK-BACK FUNCTION
# ----------------------------------------------------------------------
# List of critical log event types to filter (excluding common LOG messages)
CRITICAL_EVENT_TYPES = [
    'FATAL',        # Most severe errors
    'ERROR',        # System/Query errors
    'DISCONNECT',   # Disconnections
    'CONNECT_RECEIVED', # Connection attempts received
    'CONNECT_AUTHORIZED' # Successful connections
]

def look_back_and_report_pids(anomaly_path, events_path, output_path, freq):
    """
    Looks back into detailed event logs (including PID, User, Query) 
    for time windows flagged as anomalous by the model.
    """
    try:
        # 1. Load anomaly data (only need timestamps and score)
        print(f"Loading anomaly data from: {anomaly_path}")
        anomaly_df = pd.read_csv(anomaly_path, index_col=0, parse_dates=True)
        anomaly_timestamps = anomaly_df.index
        
        # 2. Load detailed event data (CONTAINS PID)
        print(f"Loading detailed event data from: {events_path}")
        events_df = pd.read_csv(events_path, parse_dates=['timestamp'])
        
        # 3. Prepare time window
        window_delta = pd.Timedelta(freq)
        anomalous_events_list = []
        
        print(f"Starting look-back on {len(anomaly_timestamps)} anomalous time windows...")
        
        for start_time in anomaly_timestamps:
            end_time = start_time + window_delta
            
            # 3a. Filter all detailed log events within the 5-minute window
            current_window_events = events_df[
                (events_df['timestamp'] >= start_time) & (events_df['timestamp'] < end_time)
            ].copy()
            
            if not current_window_events.empty:
                # Filter for critical/error logs only
                critical_events = current_window_events[
                    current_window_events['event_type'].isin(CRITICAL_EVENT_TYPES)
                ].copy()
                
                if not critical_events.empty:
                    # 3b. Add time window info and anomaly score to the report
                    critical_events['Anomaly_Time_Window'] = start_time
                    critical_events['Anomaly_Score'] = anomaly_df.loc[start_time, 'anomaly_score']
                    
                    anomalous_events_list.append(critical_events)
                
        # 4. Combine results and generate report
        if anomalous_events_list:
            final_report_df = pd.concat(anomalous_events_list, ignore_index=True)
            
            # Columns to include in the report (New columns + Original columns)
            new_cols = ['Anomaly_Time_Window', 'Anomaly_Score']
            report_cols_ordered = new_cols + [col for col in final_report_df.columns if col not in new_cols]
            
            # Save the report
            if not os.path.exists(os.path.dirname(output_path)):
                os.makedirs(os.path.dirname(output_path))
                print(f"Created output directory: {os.path.dirname(output_path)}")

            final_report_df[report_cols_ordered].to_csv(output_path, index=False)
            
            print("-" * 60)
            print(f"SUCCESS: Detailed PID/Event report saved to: {output_path}")
            print(f"Total critical log events found: {len(final_report_df)}")
            print("-" * 60)
            
            # 5. Display the report summary
            print("\n--- CRITICAL ANOMALOUS LOGS ---")
            
            # Sort by anomaly score (most severe first) and then by time
            final_report_df = final_report_df.sort_values(by=['Anomaly_Score', 'timestamp'], ascending=[True, True])

            # Display the DataFrame
            pd.set_option('display.max_rows', None)
            pd.set_option('display.max_columns', None)
            pd.set_option('display.width', 1000)
            
            # Columns to display in console
            display_cols_console = ['Anomaly_Score', 'pid', 'user', 'event_type', 'query_command', 'timestamp']

            print(final_report_df[display_cols_console].to_string())
            
            print("-" * 60)

            # Display summary of top suspicious PIDs/Users
            print("\nTop 5 suspicious PID/User combinations:")
            suspicious_pids = final_report_df.groupby(['pid', 'user']).size().sort_values(ascending=False)
            print(suspicious_pids.head(5).to_string())
            
            return final_report_df
        else:
            print("No detailed critical events found within anomalous time windows.")
            return None

    except FileNotFoundError as e:
        # Replaced Vietnamese error message with ASCII
        print(f"FATAL ERROR: One of the required input files was not found: {e}")
        return None
    except Exception as e:
        # Replaced Vietnamese error message with ASCII
        print(f"CRITICAL ERROR during look-back processing: {e}")
        return None

# ----------------------------------------------------------------------
# C. MAIN EXECUTION LOGIC
# ----------------------------------------------------------------------
if __name__ == "__main__":
    look_back_and_report_pids(ANOMALY_PATH, EVENTS_PATH, OUTPUT_PID_REPORT_PATH, RESAMPLE_FREQUENCY)
