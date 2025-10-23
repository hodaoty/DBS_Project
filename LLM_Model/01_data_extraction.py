from datetime import datetime
import os
import re 
import pandas as pd 
import sys 

# ----------------------------------------------------------------------
# A. REGEX PATTERNS
# ----------------------------------------------------------------------
LOG_PATTERN = re.compile(
    r'(?P<timestamp_base>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{3})\s+' 
    r'(?P<tz_offset>[+-]\d{2})\s+'                                      
    r'\[(?P<pid>\d+)\] '                                                
    r'(?:(?P<user_db>[^ ]+@[^ ]+) )?'                                   
    r'(?P<level>[A-Z]+): (?P<message>.*)'                               
)
DISCONNECT_PATTERN = re.compile(
    r'disconnection: session time: (?P<session_time>.*?) user=(?P<d_user>.*?) database=(?P<d_db>.*?) host=(?P<d_host>.*)'
)
AUDIT_PATTERN = re.compile(
    r'AUDIT: SESSION,\d+,\d+,(?P<audit_class>[^,]+),(?P<audit_type>[^,]+),.*?,.*?"(?P<audit_query>.*?)"'
)
# ----------------------------------------------------------------------
# B. PARSING FUNCTION
# ----------------------------------------------------------------------
def parse_postgresql_log(filepath):
    """
    Reads the raw log file, parses each line using Regex, and extracts 
    key fields, including PID.
    """
    parsed_data = []
    
    try:
        # Use encoding='utf-8' when opening the raw log file
        with open(filepath, 'r', encoding='utf-8') as f: 
            for line in f:
                line = line.strip()
                if not line:
                    continue
                    
                match = LOG_PATTERN.match(line)
                if match:
                    data = match.groupdict()
                    
                    data['tz'] = data.pop('tz_offset', None)
                    
                    # 1. Separate USER@DB
                    user_db = data.pop('user_db', None)
                    if user_db and '@' in user_db:
                        data['user'] = user_db.split('@')[0]
                        data['database'] = user_db.split('@')[1]
                    else:
                        data['user'] = '[unknown]' if data['level'] == 'LOG' and data.get('pid') else None
                        data['database'] = '[unknown]' if data['level'] == 'LOG' and data.get('pid') else None

                    # Initialize feature fields
                    data['event_type'] = data['level']
                    data['session_duration_sec'] = 0.0
                    data['query_command'] = None
                    data['query_text'] = None
                    
                    message = data['message']
                    
                    # 2. Process specific event types
                    
                    # a) Disconnection
                    if 'disconnection: session time:' in message:
                        d_match = DISCONNECT_PATTERN.search(message)
                        if d_match:
                            data.update(d_match.groupdict())
                            data['event_type'] = 'DISCONNECT'
                            
                            # Convert session_time (0:00:00.xxx format) to seconds
                            try:
                                time_str = data.get('session_time')
                                if ' ' in time_str and time_str.count(':') == 3: 
                                    days, h, m, s = re.split(r'[: ]', time_str)
                                    data['session_duration_sec'] = float(s) + int(m) * 60 + int(h) * 3600 + int(days) * 86400
                                elif time_str.count(':') == 2:
                                    h, m, s = time_str.split(':')
                                    data['session_duration_sec'] = float(s) + int(m) * 60 + int(h) * 3600
                                else:
                                    data['session_duration_sec'] = 0.0
                            except Exception:
                                data['session_duration_sec'] = 0.0
                            
                    # b) Audit
                    elif 'AUDIT: SESSION,' in message:
                        a_match = AUDIT_PATTERN.search(message)
                        if a_match:
                            audit_data = a_match.groupdict()
                            data['event_type'] = f"AUDIT_{audit_data['audit_type']}"
                            data['query_command'] = audit_data['audit_type']
                            data['query_text'] = audit_data['audit_query'].strip()
                    
                    # c) Fatal/Error
                    elif data['level'] in ['FATAL', 'ERROR']:
                        data['event_type'] = data['level']
                    
                    # d) Connection
                    elif 'connection received:' in message:
                        data['event_type'] = 'CONNECT_RECEIVED'
                        if data['user'] is None: data['user'] = '[unknown]'
                        if data['database'] is None: data['database'] = '[unknown]'

                    elif 'connection authorized:' in message:
                        data['event_type'] = 'CONNECT_AUTHORIZED'
                    
                    # Remove unnecessary fields from dict (before DataFrame creation)
                    data.pop('message', None)
                    data.pop('d_user', None)
                    data.pop('d_db', None)
                    data.pop('d_host', None)
                    data.pop('session_time', None)
                    
                    parsed_data.append(data)
    except FileNotFoundError:
        print(f"ERROR: Log file not found at: {filepath}")
        return pd.DataFrame()
                
    # 3. Create and normalize DataFrame
    df = pd.DataFrame(parsed_data)
    
    if df.empty:
        return df

    # --- TIMESTAMP CREATION (KEEPING ORIGINAL TZ OFFSET) ---
    
    # 1. Concatenate timestamp_base and tz into full ISO string (e.g., +07:00)
    df['timestamp'] = df['timestamp_base'].astype(str) + df['tz'].astype(str) 
    
    # 2. Remove auxiliary columns
    df.drop(columns=['timestamp_base', 'tz', 'level'], inplace=True, errors='ignore')
    
    # 3. Convert pid to integer
    df['pid'] = pd.to_numeric(df['pid'], errors='coerce').fillna(0).astype(int)

    # 4. Reorder columns as requested
    final_cols = [
        'pid', 'user', 'database', 'event_type', 'session_duration_sec', 
        'query_command', 'query_text', 'timestamp'
    ]
    
    for col in final_cols:
        if col not in df.columns:
            df[col] = None

    return df[final_cols]
# ----------------------------------------------------------------------
# C. MAIN EXECUTION
# ----------------------------------------------------------------------

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    LOG_FILE_PATH = os.path.join(base_dir, '..', 'Log_Example', 'postgresql.log')
    
    time_str = datetime.now().strftime('%Y%m%d')
    # Output directory 
    # Adjusted path structure to avoid potential issues
    CSV_DIR = os.path.join(base_dir, '..', 'CSV_FILE','OUTPUT_CSVFILE','LOG_EVENT') 

    # Ensure output directory exists
    if not os.path.exists(CSV_DIR):
        os.makedirs(CSV_DIR)
        
    OUTPUT_CSV_PATH = os.path.join(CSV_DIR, f'postgresql_events-{time_str}.csv') 

    print(f"Starting log parsing: {LOG_FILE_PATH}")
    
    # 1. Parse and create Event DataFrame
    events_df = parse_postgresql_log(LOG_FILE_PATH)
    
    if events_df.empty:
        print("No valid events extracted. Stopping process.")
    else:
        print(f"Parsing complete. Total events: {len(events_df)}")
        
        # 2. Save result
        events_df.to_csv(OUTPUT_CSV_PATH, index=False)
        
        print(f"\nEvent DataFrame saved to: {OUTPUT_CSV_PATH}")
        print("\nChecking first 5 data rows:")
        print(events_df.head())
