#import library
import os 
import pandas as pd
from datetime import datetime
from sklearn.ensemble import IsolationForest
import joblib

# ----------------------------------------------------------------------
# A. CẤU HÌNH VÀ THIẾT LẬP ĐƯỜNG DẪN
# ----------------------------------------------------------------------
# Đã sửa lỗi chính tả trong tên biến: timestamp_srt -> timestamp_str
timestamp_str = datetime.now().strftime('%Y%m%d')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# ⚠️ Cập nhật CSV_DIR để trỏ đúng đến thư mục chứa processed_scaled_features
CSV_DIR = os.path.join(BASE_DIR, '..','CSV_FILE','OUTPUT_CSVFILE','TRAIN_AI') 
MODEL_DIR = os.path.join(BASE_DIR,'trained_model')

# Đặt tên file input (đã chuẩn hóa) và file output (mô hình đã huấn luyện)
INPUT_SCALED_DATA_FILE = f'processed_scaled_features-{timestamp_str}.csv'
MODEL_FILE_NAME = f'isolation_forest_model-{timestamp_str}.pkl'

INPUT_SCALED_DATA_PATH = os.path.join(CSV_DIR, INPUT_SCALED_DATA_FILE)
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_FILE_NAME)

# Tham số mô hình
# contamination: Tỷ lệ bất thường dự kiến trong dữ liệu.
# Đây là giá trị quan trọng cần điều chỉnh dựa trên hiểu biết về dữ liệu.
CONTAMINATION_RATE = 0.01
# ----------------------------------------------------------------------
# B. HÀM TẢI DỮ LIỆU
# ----------------------------------------------------------------------
def load_scaled_data(file_path):
    """Tải dữ liệu đặc trưng đã chuẩn hóa từ file CSV."""
    if not os.path.exists(file_path):
        print(f"LỖI: Không tìm thấy file dữ liệu đầu vào tại: {file_path}")
        return None
        
    print(f"Đang tải dữ liệu từ: {file_path}")
    try:
        # Đọc dữ liệu, đặt cột 'timestamp' làm chỉ mục (index)
        df = pd.read_csv(file_path, index_col='timestamp', parse_dates=True)
        print(f"Tải thành công. Kích thước dữ liệu: {df.shape}")
        return df
    except Exception as e:
        print(f"LỖI khi đọc file CSV: {e}")
        return None
# ----------------------------------------------------------------------
# C. HUẤN LUYỆN MÔ HÌNH
# ----------------------------------------------------------------------
# Đã sửa lỗi chính tả trong tên hàm: train_anomoly_model -> train_anomaly_model
def train_anomaly_model(data_df):
    """
    Huấn luyện mô hình Isolation Forest cho việc phát hiện bất thường.
    """
    print(f"\nBắt đầu huấn luyện mô hình Isolation Forest (Contamination={CONTAMINATION_RATE})...")

    # Khởi tạo mô hình Isolation Forest
    model = IsolationForest(
        contamination=CONTAMINATION_RATE,
        random_state=42,
        n_estimators=100,
        n_jobs=-1          
    )
    #Huấn luyện mô hình
    model.fit(data_df)

    print("Hoàn thành huấn luyện mô hình.")
    return model
# ----------------------------------------------------------------------
# D. LƯU MÔ HÌNH
# ----------------------------------------------------------------------
def save_model(model, model_path):
    """
    Lưu mô hình đã huấn luyện vào file.
    """
    if not os.path.exists(os.path.dirname(model_path)):
        os.makedirs(os.path.dirname(model_path))
        print(f"Đã tạo thư mục lưu mô hình: {os.path.dirname(model_path)}")

    joblib.dump(model, model_path)
    print(f"Mô hình đã được lưu tại: {model_path}")

# ----------------------------------------------------------------------
# E. LOGIC CHÍNH
# ----------------------------------------------------------------------
if __name__ == "__main__":
    
    # 1. Tải dữ liệu đầu vào
    scaled_data_df = load_scaled_data(INPUT_SCALED_DATA_PATH)

    if scaled_data_df is None or scaled_data_df.empty:
        print("Dữ liệu đầu vào không hợp lệ hoặc trống. Dừng quá trình huấn luyện.")
    else:
        # TẠO DATAFRAME CHỈ CHỨA ĐẶC TRƯNG CHO MÔ HÌNH
        X_features = scaled_data_df.copy()
        
        # 2. Huấn luyện mô hình (Đã sửa tên hàm)
        anomaly_model = train_anomaly_model(X_features) 
        
        # 3. Lưu mô hình đã huấn luyện
        save_model(anomaly_model, MODEL_PATH)
        
        # 4. Kiểm tra mô hình và trích xuất bất thường
        print("\n--- Kiểm tra nhanh kết quả phân loại trên dữ liệu huấn luyện ---")
        
        # Tính điểm số và phân loại trên X_features, sau đó gán kết quả về scaled_data_df
        scaled_data_df['anomaly_score'] = anomaly_model.decision_function(X_features)
        scaled_data_df['anomaly'] = anomaly_model.predict(X_features)

        # Tính toán kết quả
        num_anomalies = (scaled_data_df['anomaly'] == -1).sum()
        total_samples = len(scaled_data_df)

        print(f"Tổng số mẫu dữ liệu (5T): {total_samples}")
        print(f"Số lượng bất thường tìm thấy: {num_anomalies}")
        print(f"Tỷ lệ bất thường (Model): {num_anomalies/total_samples:.2%}")
        print(f"Tỷ lệ bất thường (Cấu hình): {CONTAMINATION_RATE:.2%}")

        # ----------------------------------------------------------------------
        # 5. LƯU VÀ HIỂN THỊ CÁC BẢN GHI BẤT THƯỜNG
        
        # Lọc ra các bản ghi bất thường.
        anomalies_df = scaled_data_df[scaled_data_df['anomaly'] == -1].copy()
        
        if not anomalies_df.empty:
            # ⚠️ Đã sửa lại đường dẫn ANOMALY_OUTPUT_PATH để sử dụng CSV_DIR đã cập nhật
            ANOMALY_OUTPUT_PATH = os.path.join(CSV_DIR, f'anomaly_records-{timestamp_str}.csv')
            
            anomalies_df.to_csv(ANOMALY_OUTPUT_PATH)
            
            print(f"\n✅ Đã lưu {len(anomalies_df)} bản ghi bất thường vào: {ANOMALY_OUTPUT_PATH}")
            
            # --- TỐI ƯU HÓA HIỂN THỊ ĐỂ TRÁNH KEYERROR ---
            
            # 1. Xác định các cột thống kê quan trọng (bao gồm score)
            base_cols = ['avg_session_duration', 'max_session_duration', 'ratio_fatal_to_total', 'anomaly_score']
            
            # 2. Tìm các cột đếm liên quan đến lỗi/kết nối trong DataFrame
            count_cols = [col for col in anomalies_df.columns if col.startswith('count_') and ('error' in col or 'fatal' in col or 'connect' in col)]
            
            # 3. Kết hợp danh sách cột hiển thị và chỉ giữ lại những cột THỰC SỰ có
            display_cols = base_cols + count_cols
            available_cols = [col for col in display_cols if col in anomalies_df.columns]
            
            print("\n5 bản ghi bất thường nghiêm trọng nhất:")
            
            # Sắp xếp, chọn 5 hàng đầu, và chỉ in ra các cột quan trọng
            top_anomalies = anomalies_df.sort_values(by='anomaly_score').head(5)
            print(top_anomalies[available_cols])
        
        print("\nQuá trình huấn luyện mô hình hoàn tất.")
