import pandas as pd
import os
import joblib
from datetime import datetime, timedelta
import numpy as np

# ----------------------------------------------------------------------
# A. CẤU HÌNH ĐƯỜNG DẪN VÀ THAM SỐ
# ----------------------------------------------------------------------
timestamp_str = datetime.now().strftime('%Y%m%d')
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# ⚠️ ĐIỀU CHỈNH: Cần tìm thư mục trained_model từ BASE_DIR (thường là 05_realtime_detection.py)
MODEL_DIR = os.path.join(BASE_DIR, 'trained_model') 

# Tên file model và scaler (Sử dụng ngày tháng mới nhất để tìm)
MODEL_FILE_NAME = f'isolation_forest_model-{timestamp_str}.pkl'
SCALER_FILE_NAME = f'scaler-{timestamp_str}.pkl'

MODEL_PATH = os.path.join(MODEL_DIR, MODEL_FILE_NAME)
SCALER_PATH = os.path.join(MODEL_DIR, SCALER_FILE_NAME)

# Tần suất gom nhóm phải khớp với bước huấn luyện (02_preprocessing.py)
RESAMPLE_FREQUENCY = '5T' 

# ----------------------------------------------------------------------
# B. HÀM TẢI MODEL VÀ SCALER
# ----------------------------------------------------------------------
def load_detection_assets():
    """Tải Isolation Forest Model và Standard Scaler đã huấn luyện."""
    try:
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        print(f"✅ Tải thành công Model từ: {MODEL_PATH}")
        print(f"✅ Tải thành công Scaler từ: {SCALER_PATH}")
        return model, scaler
    except FileNotFoundError:
        print("❌ LỖI: Không tìm thấy file mô hình hoặc scaler.")
        print("Vui lòng đảm bảo đã chạy 02_preprocessing.py và 03_model_training.py thành công.")
        return None, None
    except Exception as e:
        print(f"❌ LỖI khi tải assets: {e}")
        return None, None

# ----------------------------------------------------------------------
# C. HÀM MÔ PHỎNG TẠO DỮ LIỆU LOG MỚI (Real-time/Next Batch)
# ----------------------------------------------------------------------
# Lấy độ dài cửa sổ thời gian
window_duration = pd.Timedelta(RESAMPLE_FREQUENCY)

def generate_new_log_batch(reference_df, start_time, end_time, scenario="normal"):
    """
    Tạo một batch dữ liệu log giả định (chưa được xử lý) cho một cửa sổ thời gian.
    """
    # Lấy các sự kiện log thực tế trong khung thời gian tham chiếu
    sample_df = reference_df[
        (reference_df['timestamp'] >= start_time) & (reference_df['timestamp'] < end_time)
    ].copy()

    if sample_df.empty:
        # Tạo dữ liệu log tối thiểu nếu không có mẫu thực tế
        data = {
            'pid': [0], 'user': [None], 'database': [None], 'event_type': ['LOG'],
            'session_duration_sec': [0.0], 'query_command': [None], 
            'query_text': [None], 'timestamp': [start_time]
        }
        sample_df = pd.DataFrame(data)
        
    if scenario == "stress":
        # Mô phỏng bất thường: Tăng đột biến lỗi FATAL và kết nối
        
        # 1. Tăng số lượng kết nối/sự kiện
        sample_df = pd.concat([sample_df] * 5, ignore_index=True)
        
        # 2. Thêm lỗi FATAL/ERROR
        error_events = []
        error_count = int(len(sample_df) * 0.1) # 10% sự kiện là lỗi
        for i in range(error_count):
            error_time = start_time + timedelta(seconds=np.random.rand() * window_duration.total_seconds())
            error_events.append({
                'pid': 9999, 'user': 'attack', 'database': 'critical_db', 
                'event_type': 'FATAL', 'session_duration_sec': 0.0, 
                'query_command': 'SELECT', 'query_text': 'UNAUTHORIZED ACCESS', 
                'timestamp': error_time
            })
        sample_df = pd.concat([sample_df, pd.DataFrame(error_events)], ignore_index=True)
        
        print("⚠️ Kịch bản 'stress' (FATAL/Kết nối) đã được mô phỏng.")
        
    # Đặt lại index theo timestamp và sắp xếp
    sample_df = sample_df.sort_values(by='timestamp').set_index('timestamp', drop=False)
    return sample_df

# ----------------------------------------------------------------------
# D. HÀM TẠO ĐẶC TRƯNG CHUỖI THỜI GIAN (Phỏng theo 02_preprocessing.py)
# ----------------------------------------------------------------------
# ⚠️ Lưu ý: Hàm này phải giống hệt hàm trong 02_preprocessing.py để đảm bảo tính nhất quán
def create_time_series_features(df, freq=RESAMPLE_FREQUENCY):
    """
    Tạo các đặc trưng chuỗi thời gian bằng cách gom nhóm dữ liệu log.
    (Giống hệt logic trong 02_preprocessing.py)
    """
    
    # Loại bỏ các cột ID không phải đặc trưng khỏi quá trình thống kê
    # Đảm bảo index là timestamp trước khi drop
    df_temp = df.copy().set_index('timestamp', drop=True) 

    features_df = df_temp.drop(columns=['pid', 'user', 'database', 'query_command', 'query_text'], errors='ignore')
    
    # 1. Tạo Đặc trưng Đếm (Count Features)
    count_features = features_df['event_type'].groupby(level=0).value_counts().unstack(fill_value=0)
    count_features.columns = [f'count_{col.lower().replace(" ", "_")}' for col in count_features.columns]
    
    # Thêm tổng số sự kiện (dùng cho ratio)
    count_features['count_total_events'] = count_features.sum(axis=1)

    # 2. Tạo Đặc trưng Thời gian (Time Features)
    time_features = features_df['session_duration_sec'].resample(freq).agg({
        'avg_session_duration': 'mean',
        'max_session_duration': 'max',
        'total_session_time': 'sum'
    }).fillna(0)

    # 3. Gom nhóm Đặc trưng
    final_features_df = count_features.resample(freq).sum().fillna(0)
    final_features_df = final_features_df.join(time_features)

    # 4. Tạo Đặc trưng Tỷ lệ (Ratio Features)
    final_features_df['ratio_fatal_to_total'] = (
        final_features_df.get('count_fatal', 0) / final_features_df['count_total_events']
    ).fillna(0)
    
    # Loại bỏ cột 'count_total_events' và các cột không cần thiết khác
    final_features_df = final_features_df.drop(columns=['count_total_events'], errors='ignore')

    return final_features_df

# ----------------------------------------------------------------------
# E. HÀM PHÁT HIỆN BẤT THƯỜNG (CHÍNH)
# ----------------------------------------------------------------------
def detect_anomalies_realtime(model, scaler, new_log_data_df):
    """
    Áp dụng pipeline tiền xử lý và mô hình để phát hiện bất thường trên dữ liệu mới.
    """
    if new_log_data_df.empty:
        print("Không có dữ liệu log mới để xử lý.")
        return None

    # 1. Kỹ thuật Đặc trưng (Gom nhóm 5T)
    features_df = create_time_series_features(new_log_data_df)
    
    # 2. Sắp xếp lại cột và điền 0 cho các đặc trưng bị thiếu
    if not hasattr(scaler, 'feature_names_in_') or scaler.feature_names_in_ is None:
        print("❌ LỖI: Scaler không có danh sách tên đặc trưng. Vui lòng kiểm tra 02_preprocessing.py.")
        return None
        
    all_feature_names = list(scaler.feature_names_in_)
    
    # Thêm các cột còn thiếu vào features_df với giá trị 0
    for col in all_feature_names:
        if col not in features_df.columns:
            features_df[col] = 0.0
            
    # Sắp xếp lại cột theo thứ tự đã huấn luyện
    features_df = features_df[all_feature_names]

    # 3. Chuẩn hóa dữ liệu (Áp dụng Scaler đã tải - KHÔNG dùng fit_transform)
    scaled_data = scaler.transform(features_df)
    scaled_df = pd.DataFrame(scaled_data, index=features_df.index, columns=features_df.columns)

    # 4. Dự đoán
    # ⚠️ FIX: Sử dụng bản sao sạch (18 cột) cho cả hai hàm dự đoán
    X_predict_features = scaled_df.copy()

    scaled_df['anomaly_score'] = model.decision_function(X_predict_features)
    scaled_df['anomaly'] = model.predict(X_predict_features) # <--- Đã sửa lỗi: Dùng X_predict_features

    # Lọc ra các bản ghi bất thường
    anomalies = scaled_df[scaled_df['anomaly'] == -1].copy()

    return anomalies

# ----------------------------------------------------------------------
# F. LOGIC THỰC THI CHÍNH
# ----------------------------------------------------------------------
if __name__ == "__main__":
    
    # 1. Tải Model và Scaler
    anomaly_model, scaler = load_detection_assets()
    
    if anomaly_model is None or scaler is None:
        exit()

    # 2. Tải dữ liệu sự kiện chi tiết gốc để làm mẫu
    CSV_DIR_EVENT = os.path.join(BASE_DIR, '..', 'CSV_FILE', 'OUTPUT_CSVFILE', 'LOG_EVENT')
    EVENTS_CSV_PATH = os.path.join(CSV_DIR_EVENT, f'postgresql_events-{timestamp_str}.csv') 
    
    if not os.path.exists(EVENTS_CSV_PATH):
        print(f"❌ CẢNH BÁO: Không tìm thấy file sự kiện gốc tại {EVENTS_CSV_PATH}. Không thể mô phỏng.")
        exit()
    
    full_events_df = pd.read_csv(EVENTS_CSV_PATH, parse_dates=['timestamp'])
    # Đặt index là timestamp và giữ lại cột timestamp
    full_events_df = full_events_df.set_index('timestamp', drop=False)
    
    window_duration = pd.Timedelta(RESAMPLE_FREQUENCY)

    # --- KỊCH BẢN MÔ PHỎNG ---
    
    # Lấy thời điểm log mới nhất từ dữ liệu gốc
    last_log_time = full_events_df.index.max()
    
    # Cửa sổ thời gian mô phỏng (ví dụ: 5 phút sau log cuối cùng)
    sim_start_time = last_log_time + window_duration
    sim_end_time = sim_start_time + window_duration 

    print("\n--- Bắt đầu Mô Phỏng Phát Hiện Bất Thường Thời Gian Thực ---")
    
    # Ví dụ 1: Kịch bản BÌNH THƯỜNG
    print(f"\n[1] Giám sát cửa sổ BÌNH THƯỜNG ({sim_start_time} - {sim_end_time})")
    sim_normal_data = generate_new_log_batch(full_events_df, sim_start_time, sim_end_time, scenario="normal")
    anomalies_normal = detect_anomalies_realtime(anomaly_model, scaler, sim_normal_data)
    
    if anomalies_normal is None or anomalies_normal.empty:
        print("✅ Kết quả (BÌNH THƯỜNG): Không phát hiện bất thường.")
    else:
        print(f"⚠️ Kết quả (BÌNH THƯỜNG): Phát hiện {len(anomalies_normal)} bất thường.")

    # Ví dụ 2: Kịch bản CĂNG THẲNG/TẤN CÔNG (STRESS)
    sim_start_time_stress = sim_end_time + window_duration # Cửa sổ tiếp theo
    sim_end_time_stress = sim_start_time_stress + window_duration

    print(f"\n[2] Giám sát cửa sổ CĂNG THẲNG ({sim_start_time_stress} - {sim_end_time_stress})")
    sim_stress_data = generate_new_log_batch(full_events_df, sim_start_time_stress, sim_end_time_stress, scenario="stress")
    anomalies_stress = detect_anomalies_realtime(anomaly_model, scaler, sim_stress_data)

    if anomalies_stress is None or anomalies_stress.empty:
        print("✅ Kết quả (STRESS): Không phát hiện bất thường.")
    else:
        print(f"🚨🚨 KẾT QUẢ (STRESS): Phát hiện {len(anomalies_stress)} bất thường! 🚨🚨")
        print("Dữ liệu bất thường được chuẩn hóa (Top 3):")
        
        # Hiển thị 3 hàng đầu tiên của các đặc trưng bất thường
        display_cols = [col for col in anomalies_stress.columns if col not in ['anomaly_score', 'anomaly']]
        
        # Sắp xếp và in ra
        top_anomalies = anomalies_stress.sort_values(by='anomaly_score', ascending=True).head(3)
        print(top_anomalies[['anomaly_score'] + display_cols].to_string())
