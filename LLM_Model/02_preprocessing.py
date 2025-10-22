# ----------------------------------------------------------------------
import pandas as pd 
import numpy as np
import os 
from sklearn.preprocessing import StandardScaler
import joblib   
from datetime import datetime

# ----------------------------------------------------------------------
# A. CẤU HÌNH VÀ THIẾT LẬP ĐƯỜNG DẪN
# ----------------------------------------------------------------------
# Thiết lập đường dẫn cơ bản
timestamp_str = datetime.now().strftime("%Y%m%d")
csv_file_name = f'processed_scaled_features-{timestamp_str}.csv'
scaler_file_name = f'scaler-{timestamp_str}.pkl'

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_DIR = os.path.join(BASE_DIR, '..','CSV_FILE','OUTPUT_CSVFILE')
MODEL_DIR = os.path.join(BASE_DIR,'trained_model')
OUTPUT_SCALED_DATA_PATH = os.path.join(CSV_DIR, csv_file_name)
SCALER_PATH = os.path.join(MODEL_DIR, scaler_file_name)

# Tần suất tổng hợp: Chọn cửa sổ thời gian (ví dụ: 5 phút)
RESAMPLE_FREQUENCY = '5T'

# ----------------------------------------------------------------------
# B. HÀM TẠO ĐẶC TRƯNG CHUỖI THỜI GIAN
# ----------------------------------------------------------------------
def create_time_series_features(events_df, freq):
    """
    Tổng hợp các sự kiện rời rạc thành các đặc trưng chuỗi thời gian (5 phút/điểm).
    """
    if events_df.empty:
        return pd.DataFrame()

    # Đảm bảo cột timestamp là index
    events_df.set_index('timestamp', inplace=True)

    print(f"Bắt đầu tổng hợp dữ liệu theo chu kỳ: {freq}...")

    # 1. TẠO CÁC ĐẶC TRƯNG TỪ SỐ LƯỢNG SỰ KIỆN (EVENTS COUNTS)
    
    # SỬA LỖI: Sử dụng groupby(pd.Grouper) để đếm các giá trị phân loại
    event_counts = events_df.groupby(pd.Grouper(freq=freq))['event_type'].value_counts().unstack(fill_value=0)
    
    event_counts.columns = [f'count_{col.lower()}' for col in event_counts.columns]
    
    # 2. TẠO CÁC ĐẶC TRƯNG TỪ THỜI GIAN PHIÊN (SESSION DURATION)
    
    duration_stats = events_df[events_df['event_type'] == 'DISCONNECT']['session_duration_sec'].resample(freq).agg(
        avg_session_duration='mean',
        max_session_duration='max',
        total_session_time='sum'
    ).fillna(0)
    
    # 3. TẠO CÁC ĐẶC TRƯNG LỖI VÀ TỶ LỆ (ERROR & RATIOS)

    # Lỗi xác thực thất bại FATAL 
    fatal_counts = event_counts.get('count_fatal', pd.Series(0, index=event_counts.index)).rename('count_fatal_errors')
        
    # Tổng truy vấn trong cửa sổ thời gian
    total_queries = events_df['event_type'].resample(freq).count().rename('count_total_events').fillna(0)

    # 4. GHÉP CÁC ĐẶC TRƯNG LẠI
    
    features_to_concat = [total_queries.to_frame(), fatal_counts.to_frame(), duration_stats, event_counts]
    features_df = pd.concat([f for f in features_to_concat if not f.empty], axis=1).fillna(0)
    
    # Tính toán tỷ lệ lỗi
    features_df['ratio_fatal_to_total'] = features_df['count_fatal_errors'] / (features_df['count_total_events'] + 1e-6)

    print("Hoàn thành tổng hợp đặc trưng.")
    
    return features_df

# ----------------------------------------------------------------------
# C. CHUẨN HÓA DỮ LIỆU (SCALING)
# ----------------------------------------------------------------------
def scale_features(features_df):
    """
    Sử dụng StandardScaler để chuẩn hóa dữ liệu.
    """
    if features_df.empty:
        return pd.DataFrame(), None
        
    print("Bắt đầu chuẩn hóa dữ liệu...")
    
    # Khởi tạo Scaler
    scaler = StandardScaler()
    
    # Chỉ chuẩn hóa các cột số (Không cần chuẩn hóa timestamp)
    X_scaled = scaler.fit_transform(features_df)
    
    # Chuyển đổi trở lại DataFrame
    scaled_df = pd.DataFrame(X_scaled, columns=features_df.columns, index=features_df.index)
    
    print("Hoàn thành chuẩn hóa.")
    
    return scaled_df, scaler

# ----------------------------------------------------------------------
# D. PHẦN THỰC THI CHÍNH
# ----------------------------------------------------------------------

if __name__ == "__main__":
    
    # 1. TÌM KIẾM VÀ TẢI DỮ LIỆU ĐẦU VÀO
    try:
        if not os.path.exists(CSV_DIR):
            print(f"LỖI: Thư mục CSV_FILE không tồn tại tại {CSV_DIR}.")
            exit()
            
        csv_files = [f for f in os.listdir(CSV_DIR) if f.startswith('postgresql_events-') and f.endswith('.csv')]
        if not csv_files:
            print(f"LỖI: Không tìm thấy file sự kiện CSV trong thư mục {CSV_DIR}. Hãy chạy 01_data_extraction.py trước.")
            exit()
            
        # Chọn file gần nhất (Mới nhất)
        latest_csv_file = os.path.join(CSV_DIR, sorted(csv_files, reverse=True)[0])
        print(f"Sử dụng file dữ liệu đầu vào: {latest_csv_file}")
        
        # Đọc dữ liệu. Cột 'timestamp' phải được đọc đúng kiểu.
        # Đọc mà không dùng parse_dates trước, để chuyển đổi thủ công sau đó.
        events_df = pd.read_csv(latest_csv_file) 
        
        # ⚠️ KHẮC PHỤC LỖI TẠI ĐÂY: Chuyển đổi cưỡng bức cột timestamp
        
        # 1. Chuyển đổi cưỡng bức sang datetime. errors='coerce' sẽ biến lỗi thành NaT.
        events_df['timestamp'] = pd.to_datetime(events_df['timestamp'], errors='coerce')
        
        # 2. Loại bỏ các hàng có timestamp không hợp lệ (NaT)
        events_df.dropna(subset=['timestamp'], inplace=True)
        
        # 3. Xử lý múi giờ: Đảm bảo nó là 'aware' (có múi giờ)
        if events_df['timestamp'].dt.tz is None:
             # Gán múi giờ vì biết rằng dữ liệu log là Asia/Ho_Chi_Minh
             events_df['timestamp'] = events_df['timestamp'].dt.tz_localize('Asia/Ho_Chi_Minh')
        else:
             # Nếu đã có múi giờ, chuyển về múi giờ chuẩn của project
             events_df['timestamp'] = events_df['timestamp'].dt.tz_convert('Asia/Ho_Chi_Minh')
        
    except Exception as e:
        print(f"LỖI khi tải hoặc tiền xử lý dữ liệu: {e}")
        # In thêm lỗi để dễ debug nếu lỗi không phải do múi giờ
        # raise 
        exit()

    # 2. TẠO ĐẶC TRƯNG VÀ CHUẨN HÓA
    if events_df.empty:
        print("DataFrame sự kiện trống sau khi làm sạch. Dừng xử lý.")
        exit()
        
    features_df = create_time_series_features(events_df, RESAMPLE_FREQUENCY)
    
    if features_df.empty:
        print("Không có đặc trưng nào được tạo ra. Dừng xử lý.")
        exit()
        
    scaled_features_df, scaler = scale_features(features_df)
    
    # 3. LƯU KẾT QUẢ ĐẦU RA CHO MÔ HÌNH
    
    # Đảm bảo thư mục model tồn tại
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
        
    # Lưu Scaler
    joblib.dump(scaler, SCALER_PATH)
    
    # Lưu dữ liệu đã chuẩn hóa
    scaled_features_df.to_csv(OUTPUT_SCALED_DATA_PATH, index=True)
    
    print("-" * 50)
    print(f"✅ Scaler (scaler.pkl) đã được lưu vào: {SCALER_PATH}")
    print(f"✅ Đặc trưng đã chuẩn hóa được lưu vào: {OUTPUT_SCALED_DATA_PATH}")
    print("-" * 50)
    print("Kiểm tra 5 dòng đặc trưng đã chuẩn hóa đầu tiên:")
    print(scaled_features_df.head())
# ----------------------------------------------------------------------