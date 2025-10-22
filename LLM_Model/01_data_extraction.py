from datetime import datetime
import os
import re 
import pandas as pd 

# ----------------------------------------------------------------------
# A. ĐỊNH NGHĨA BIỂU THỨC CHÍNH QUY (REGEX PATTERNS)
# ----------------------------------------------------------------------
# Regex cơ bản để trích xuất các trường cố định ở đầu mỗi dòng log.
# Regex này phù hợp với định dạng log thô bạn cung cấp.
LOG_PATTERN = re.compile(
    r'(?P<timestamp_base>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{3}) '  # Ví dụ: 2025-10-04 20:59:26.037
    r'(?P<tz_offset>[+-]\d{2}) '                                        # Ví dụ: +07
    r'\[(?P<pid>\d+)\] '                                                # Ví dụ: [7002]
    r'(?:(?P<user_db>[^ ]+@[^ ]+) )?'                                   # Ví dụ: postgres@template1 (Tùy chọn)
    r'(?P<level>[A-Z]+): (?P<message>.*)'                               # Ví dụ: LOG: message / FATAL: message
)
# Regex để trích xuất thông tin chi tiết từ thông báo ngắt kết nối (Disconnection).
DISCONNECT_PATTERN = re.compile(
    r'disconnection: session time: (?P<session_time>.*?) user=(?P<d_user>.*?) database=(?P<d_db>.*?) host=(?P<d_host>.*)'
)
# Regex để trích xuất thông tin từ thông báo kiểm toán (AUDIT) - có thể là từ pgaudit.
# Đây là nguồn chính để lấy loại hành động (SELECT, INSERT, DDL, v.v.).
AUDIT_PATTERN = re.compile(
    r'AUDIT: SESSION,\d+,\d+,(?P<audit_class>[^,]+),(?P<audit_type>[^,]+),.*?,.*?"(?P<audit_query>.*?)"'
)
# ----------------------------------------------------------------------
# B. HÀM PHÂN TÍCH CÚ PHÁP (PARSING FUNCTION)
# ----------------------------------------------------------------------
def parse_postgresql_log(filepath):
    """
    Đọc file log thô, phân tích cú pháp từng dòng bằng Regex và trích xuất
    các trường dữ liệu quan trọng vào danh sách các từ điển.
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
                    
                    # Gán múi giờ vào data
                    data['tz'] = data.pop('tz_offset', None)
                    
                    # 1. Phân tách USER@DB
                    user_db = data.pop('user_db', None)
                    if user_db and '@' in user_db:
                        data['user'] = user_db.split('@')[0]
                        data['database'] = user_db.split('@')[1]
                    else:
                        data['user'] = data['database'] = None
                    
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
                    elif 'connection authorized:' in message:
                        data['event_type'] = 'CONNECT_AUTHORIZED'
                    
                    # Loại bỏ các trường không cần thiết
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

    # --- KHẮC PHỤC LỖI VÀ CHUYỂN ĐỔI TIMESTAMP ---
    
    # 1. Ghép timestamp_base và tz lại với định dạng chuẩn ISO (+HH:MM)
    # df['tz'] là chuỗi '+07'. Ta thêm ':00' để được '+07:00'
    # Kết quả: 2025-10-04 20:59:26.037+07:00
    df['iso_timestamp_str'] = df['timestamp_base'].astype(str) + df['tz'].astype(str) + ':00'

    # 2. Chuyển đổi thành datetime có múi giờ
    # Sử dụng .str.replace(' ', '') để loại bỏ khoảng trắng dư thừa
    # Sử dụng format='%Y-%m-%d%H:%M:%S.%f%z'
    df['timestamp'] = pd.to_datetime(
        df['iso_timestamp_str'].str.replace(' ', ''), 
        format='%Y-%m-%d%H:%M:%S.%f%z', 
        errors='coerce' # Thay thế giá trị không hợp lệ bằng NaT
    )
    
    # 3. Chuyển đổi sang múi giờ địa phương (Asia/Ho_Chi_Minh)
    df['timestamp'] = df['timestamp'].dt.tz_convert('Asia/Ho_Chi_Minh')

    # Loại bỏ các cột phụ trợ
    df.drop(columns=['timestamp_base', 'tz', 'iso_timestamp_str', 'pid', 'level'], inplace=True, errors='ignore')
    
    return df
# ----------------------------------------------------------------------
# ----------------------------------------------------------------------
# C. PHẦN THỰC THI CHÍNH
# ----------------------------------------------------------------------

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    LOG_FILE_PATH = os.path.join(base_dir, '..', 'Log_Example', 'postgresql.log')
    
    # Hàm thời gian lấy ngày tháng hiện tại để đặt tên file output
    time_str = datetime.now().strftime('%Y%m%d')
    CSV_DIR = os.path.join(base_dir, '..', 'CSV_FILE','OUTPUT_CSVFILE')
    
    # Sửa lỗi: Ghép đường dẫn thư mục và tên file output
    OUTPUT_CSV_PATH = os.path.join(CSV_DIR, f'postgresql_events-{time_str}.csv') 

    # ⚠️ Bổ sung: Kiểm tra và tạo thư mục CSV_FILE nếu chưa tồn tại
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
        print("\nCác loại sự kiện chính được phát hiện (Top 10):")
        print(events_df['event_type'].value_counts().head(10))
        
        # 2. Lưu kết quả
        events_df.to_csv(OUTPUT_CSV_PATH, index=False)
        
        print(f"\nDataFrame sự kiện đã được lưu vào: {OUTPUT_CSV_PATH}")
        print("\nKiểm tra 5 dòng dữ liệu đầu tiên:")
        print(events_df.head())
                    