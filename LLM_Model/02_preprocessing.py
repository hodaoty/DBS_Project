import pandas as pd
import os
import joblib
from datetime import datetime
from sklearn.preprocessing import StandardScaler

# ----------------------------------------------------------------------
# A. CẤU HÌNH VÀ THIẾT LẬP ĐƯỜNG DẪN
# ----------------------------------------------------------------------
# Tần suất gom nhóm (5 phút)
RESAMPLE_FREQUENCY = '5T'

base_dir = os.path.dirname(os.path.abspath(__file__))
timestamp_str = datetime.now().strftime("%Y%m%d")

# Đường dẫn đến file sự kiện thô (từ bước 01)
CSV_DIR = os.path.join(base_dir, '..', 'CSV_FILE', 'OUTPUT_CSVFILE')
INPUT_EVENTS_FILE = f'postgresql_events-{timestamp_str}.csv'
INPUT_EVENTS_PATH = os.path.join(CSV_DIR,'LOG_EVENT', INPUT_EVENTS_FILE)

# Thư mục lưu scaler và output
MODEL_DIR = os.path.join(base_dir, '..', 'trained_model') # Nơi lưu scaler
if not os.path.exists(MODEL_DIR): os.makedirs(MODEL_DIR)

OUTPUT_SCALED_DATA_PATH = os.path.join(CSV_DIR,'TRAIN_AI', f'processed_scaled_features-{timestamp_str}.csv')
OUTPUT_SCALER_PATH = os.path.join(MODEL_DIR, f'scaler-{timestamp_str}.pkl')

# ----------------------------------------------------------------------
# B. HÀM TẢI DỮ LIỆU VÀ CHUẨN BỊ INDEX
# ----------------------------------------------------------------------
def load_and_prepare_data(filepath):
    """Tải dữ liệu sự kiện và đặt cột timestamp làm index."""
    if not os.path.exists(filepath):
        print(f"LỖI: Không tìm thấy file sự kiện tại: {filepath}")
        return None
        
    print(f"Đang tải dữ liệu từ: {filepath}")
    
    # 1. Đọc CSV và Parse Dates
    # Đọc cột 'timestamp' dưới dạng datetime object.
    df = pd.read_csv(filepath, parse_dates=['timestamp'])
    
    # 2. Đặt Index
    # Đảm bảo cột timestamp được đặt làm index cho Resampling.
    df = df.set_index('timestamp')
    
    print(f"Tải thành công. Kích thước dữ liệu: {df.shape}")
    return df

# ----------------------------------------------------------------------
# C. HÀM KỸ THUẬT ĐẶC TRƯNG CHUỖI THỜI GIAN (FEATURE ENGINEERING)
# ----------------------------------------------------------------------
def create_time_series_features(df):
    """
    Tạo các đặc trưng chuỗi thời gian bằng cách gom nhóm dữ liệu log 
    theo tần suất RESAMPLE_FREQUENCY (5 phút).
    """
    # ⚠️ BƯỚC SỬA LỖI: Loại bỏ các cột ID không phải đặc trưng khỏi quá trình thống kê
    # user, database, pid là các ID/nhãn, không nên được tính toán thống kê.
    # Ta chỉ giữ lại các cột số (session_duration_sec) và các cột đếm (event_type).
    features_df = df.drop(columns=['pid', 'user', 'database', 'query_command', 'query_text'], errors='ignore')

    # 1. Tạo Đặc trưng Đếm (Count Features)
    # Gom nhóm theo event_type để đếm các loại sự kiện
    count_features = features_df['event_type'].groupby(level=0).value_counts().unstack(fill_value=0)
    
    # Đổi tên cột để dễ đọc hơn (ví dụ: FATAL -> count_fatal_errors)
    count_features.columns = [f'count_{col.lower().replace(" ", "_")}' for col in count_features.columns]
    
    # Thêm tổng số sự kiện
    count_features['count_total_events'] = count_features.sum(axis=1)

    # 2. Tạo Đặc trưng Thời gian (Time Features)
    # Tính toán thống kê trên session_duration_sec
    time_features = features_df['session_duration_sec'].resample(RESAMPLE_FREQUENCY).agg({
        'avg_session_duration': 'mean',
        'max_session_duration': 'max',
        'total_session_time': 'sum'
    }).fillna(0) # Điền 0 cho các cửa sổ không có sự kiện.

    # 3. Gom nhóm Đặc trưng
    # Nối các đặc trưng đếm và thời gian
    final_features_df = count_features.resample(RESAMPLE_FREQUENCY).sum().fillna(0)
    final_features_df = final_features_df.join(time_features)

    # 4. Tạo Đặc trưng Tỷ lệ (Ratio Features)
    # Tỷ lệ lỗi fatal trên tổng số sự kiện (chỉ cho các cửa sổ có sự kiện)
    final_features_df['ratio_fatal_to_total'] = (
        final_features_df.get('count_fatal', 0) / final_features_df['count_total_events']
    ).fillna(0)
    
    # Loại bỏ cột tổng số sự kiện ban đầu khỏi danh sách các cột đếm cụ thể 
    # (Để mô hình chỉ học trên 20 cột đặc trưng mà bạn đã sử dụng trước đó)
    final_features_df = final_features_df.drop(columns=['count_total_events'], errors='ignore')

    print(f"Hoàn tất Kỹ thuật Đặc trưng. Số lượng đặc trưng: {len(final_features_df.columns)}")
    return final_features_df

# ----------------------------------------------------------------------
# D. CHUẨN HÓA DỮ LIỆU (SCALING)
# ----------------------------------------------------------------------
def scale_features(df):
    """Chuẩn hóa các đặc trưng bằng Standard Scaler và lưu scaler."""
    # Khởi tạo Standard Scaler
    scaler = StandardScaler()
    
    # Huấn luyện và chuyển đổi dữ liệu
    scaled_data = scaler.fit_transform(df)
    
    # Chuyển đổi mảng NumPy trở lại thành DataFrame
    scaled_df = pd.DataFrame(scaled_data, index=df.index, columns=df.columns)
    
    # Lưu scaler để sử dụng lại cho dữ liệu thời gian thực
    joblib.dump(scaler, OUTPUT_SCALER_PATH)
    print(f"✅ Standard Scaler đã được lưu tại: {OUTPUT_SCALER_PATH}")
    
    return scaled_df

# ----------------------------------------------------------------------
# E. LOGIC CHÍNH
# ----------------------------------------------------------------------
if __name__ == "__main__":
    
    # 1. Tải và chuẩn bị dữ liệu
    events_df = load_and_prepare_data(INPUT_EVENTS_PATH)
    
    if events_df is None or events_df.empty:
        print("Dữ liệu sự kiện không hợp lệ. Dừng tiền xử lý.")
    else:
        # 2. Kỹ thuật Đặc trưng
        features_df = create_time_series_features(events_df)
        
        # 3. Chuẩn hóa dữ liệu và lưu scaler
        scaled_features_df = scale_features(features_df)
        
        # 4. Lưu dữ liệu đã chuẩn hóa
        scaled_features_df.to_csv(OUTPUT_SCALED_DATA_PATH)
        print(f"✅ Dữ liệu đã chuẩn hóa được lưu tại: {OUTPUT_SCALED_DATA_PATH}")
        
        print("\nTiền xử lý hoàn tất. Sẵn sàng cho 03_model_training.py")