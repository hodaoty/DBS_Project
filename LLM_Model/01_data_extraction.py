from datetime import datetime
import os
import re 
import pandas as pd 

# ----------------------------------------------------------------------
# A. ĐỊNH NGHĨA BIỂU THỨC CHÍNH QUY (REGEX PATTERNS)
# ----------------------------------------------------------------------
LOG_PATTERN = re.compile(
    r'(?P<timestamp_base>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{3})\s+' # Base time (có khoảng trắng)
    r'(?P<tz_offset>[+-]\d{2})\s+'                                      # Múi giờ (+07) và khoảng trắng
    r'\[(?P<pid>\d+)\] '                                                # Ví dụ: [7002] - GIỮ LẠI CỘT NÀY
    r'(?:(?P<user_db>[^ ]+@[^ ]+) )?'                                   # Ví dụ: postgres@template1 (Tùy chọn)
    r'(?P<level>[A-Z]+): (?P<message>.*)'                               # Ví dụ: LOG: message / FATAL: message
)
DISCONNECT_PATTERN = re.compile(
    r'disconnection: session time: (?P<session_time>.*?) user=(?P<d_user>.*?) database=(?P<d_db>.*?) host=(?P<d_host>.*)'
)
AUDIT_PATTERN = re.compile(
    r'AUDIT: SESSION,\d+,\d+,(?P<audit_class>[^,]+),(?P<audit_type>[^,]+),.*?,.*?"(?P<audit_query>.*?)"'
)
# ----------------------------------------------------------------------
# B. HÀM PHÂN TÍCH CÚ PHÁP (PARSING FUNCTION)
# ----------------------------------------------------------------------
def parse_postgresql_log(filepath):
    """
    Đọc file log thô, phân tích cú pháp từng dòng bằng Regex và trích xuất
    các trường dữ liệu quan trọng, bao gồm PID.
    """
    parsed_data = []
    
    # 1. Đọc file và trích xuất dữ liệu thô
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                    
                match = LOG_PATTERN.match(line)
                if match:
                    data = match.groupdict()
                    
                    data['tz'] = data.pop('tz_offset', None)
                    
                    # 1. Phân tách USER@DB
                    user_db = data.pop('user_db', None)
                    if user_db and '@' in user_db:
                        data['user'] = user_db.split('@')[0]
                        data['database'] = user_db.split('@')[1]
                    else:
                        # Gán giá trị mặc định cho user/database nếu không có
                        data['user'] = '[unknown]' if data['level'] == 'LOG' and data.get('pid') else None
                        data['database'] = '[unknown]' if data['level'] == 'LOG' and data.get('pid') else None

                    # Khởi tạo các trường đặc trưng
                    data['event_type'] = data['level']
                    data['session_duration_sec'] = 0.0
                    data['query_command'] = None
                    data['query_text'] = None
                    
                    message = data['message']
                    
                    # 2. Xử lý các loại sự kiện cụ thể
                    
                    # a) Ngắt kết nối (Disconnection)
                    if 'disconnection: session time:' in message:
                        d_match = DISCONNECT_PATTERN.search(message)
                        if d_match:
                            data.update(d_match.groupdict())
                            data['event_type'] = 'DISCONNECT'
                            
                            # Chuyển đổi session_time (dạng 0:00:00.xxx) sang giây
                            try:
                                time_str = data.get('session_time')
                                # Xử lý dạng D HH:MM:SS.ms (chứa ngày)
                                if ' ' in time_str and time_str.count(':') == 3: 
                                    days, h, m, s = re.split(r'[: ]', time_str)
                                    data['session_duration_sec'] = float(s) + int(m) * 60 + int(h) * 3600 + int(days) * 86400
                                # Xử lý dạng H:M:S.ms
                                elif time_str.count(':') == 2:
                                    h, m, s = time_str.split(':')
                                    data['session_duration_sec'] = float(s) + int(m) * 60 + int(h) * 3600
                                else:
                                    data['session_duration_sec'] = 0.0
                            except Exception:
                                data['session_duration_sec'] = 0.0
                            
                    # b) Kiểm toán (AUDIT)
                    elif 'AUDIT: SESSION,' in message:
                        a_match = AUDIT_PATTERN.search(message)
                        if a_match:
                            audit_data = a_match.groupdict()
                            data['event_type'] = f"AUDIT_{audit_data['audit_type']}"
                            data['query_command'] = audit_data['audit_type']
                            data['query_text'] = audit_data['audit_query'].strip()
                    
                    # c) Lỗi/Fatal
                    elif data['level'] in ['FATAL', 'ERROR']:
                        data['event_type'] = data['level']
                    
                    # d) Kết nối
                    elif 'connection received:' in message:
                        data['event_type'] = 'CONNECT_RECEIVED'
                        # Gán [unknown] cho user/db nếu thiếu để khớp định dạng đầu ra mong muốn
                        if data['user'] is None: data['user'] = '[unknown]'
                        if data['database'] is None: data['database'] = '[unknown]'

                    elif 'connection authorized:' in message:
                        data['event_type'] = 'CONNECT_AUTHORIZED'
                    
                    # Loại bỏ các trường không cần thiết từ dict (trước khi tạo DataFrame)
                    data.pop('message', None)
                    data.pop('d_user', None)
                    data.pop('d_db', None)
                    data.pop('d_host', None)
                    data.pop('session_time', None)
                    
                    parsed_data.append(data)
    except FileNotFoundError:
        print(f"LỖI: Không tìm thấy file log tại đường dẫn: {filepath}")
        return pd.DataFrame()
                
    # 3. Tạo và chuẩn hóa DataFrame
    df = pd.DataFrame(parsed_data)
    
    if df.empty:
        return df

    # --- CHUẨN HÓA VÀ TẠO CỘT TIMESTAMP (GIỮ NGUYÊN MÚI GIỜ LOG) ---
    
    # 1. Ghép timestamp_base và tz lại thành chuỗi ISO đầy đủ (vd: +07:00)
    df['timestamp'] = df['timestamp_base'].astype(str) + df['tz'].astype(str) 
    
    # 2. Loại bỏ các cột phụ trợ
    df.drop(columns=['timestamp_base', 'tz', 'level'], inplace=True, errors='ignore')
    
    # 3. Chuyển đổi pid sang kiểu số nguyên
    df['pid'] = pd.to_numeric(df['pid'], errors='coerce').fillna(0).astype(int)

    # 4. Sắp xếp lại cột theo thứ tự yêu cầu
    final_cols = [
        'pid', 'user', 'database', 'event_type', 'session_duration_sec', 
        'query_command', 'query_text', 'timestamp'
    ]
    
    # Đảm bảo các cột thiếu được thêm vào với giá trị None
    for col in final_cols:
        if col not in df.columns:
            df[col] = None

    return df[final_cols] # Trả về DataFrame đã sắp xếp
# ----------------------------------------------------------------------
# C. PHẦN THỰC THI CHÍNH
# ----------------------------------------------------------------------

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    # Đảm bảo đường dẫn này trỏ đến file log thô thực tế
    LOG_FILE_PATH = os.path.join(base_dir, '..', 'Log_Example', 'postgresql.log')
    
    time_str = datetime.now().strftime('%Y%m%d')
    # Thư mục đầu ra
    CSV_DIR = os.path.join(base_dir, '..', 'CSV_FILE','OUTPUT_CSVFILE','LOG_EVENT')
    
    OUTPUT_CSV_PATH = os.path.join(CSV_DIR, f'postgresql_events-{time_str}.csv') 

    if not os.path.exists(CSV_DIR):
        os.makedirs(CSV_DIR)
        print(f"Đã tạo thư mục đầu ra: {CSV_DIR}")

    print(f"Bắt đầu phân tích cú pháp file log: {LOG_FILE_PATH}")
    
    # 1. Phân tích cú pháp và tạo DataFrame sự kiện
    events_df = parse_postgresql_log(LOG_FILE_PATH)
    
    if events_df.empty:
        print("Không có sự kiện hợp lệ nào được trích xuất. Dừng xử lý.")
    else:
        print(f"Hoàn thành phân tích. Tổng số sự kiện: {len(events_df)}")
        
        # 2. Lưu kết quả
        events_df.to_csv(OUTPUT_CSV_PATH, index=False)
        
        print(f"\nDataFrame sự kiện đã được lưu vào: {OUTPUT_CSV_PATH}")
        print("\nKiểm tra 5 dòng dữ liệu đầu tiên:")
        print(events_df.head())