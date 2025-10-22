#import library
import os 
import pandas as pd
from datetime import datetime
from sklearn.ensemble import IsolationForest
import joblib

# ----------------------------------------------------------------------
# A. CẤU HÌNH VÀ THIẾT LẬP ĐƯỜNG DẪN
# ----------------------------------------------------------------------
timestamp_srt = datetime.now().strftime('%Y%m%d')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_DIR =  os.path.join(BASE_DIR, '..','CSV_FILE','OUTPUT_CSVFILE','TRAIN_AI')
MODEL_DIR = os.path.join(BASE_DIR,'trained_model')

# Đặt tên file input (đã chuẩn hóa) và file output (mô hình đã huấn luyện)
INPUT_SCALED_DATA_FILE = f'processed_scaled_features-{timestamp_srt}.csv'
MODEL_FILE_NAME = f'isolation_forest_model-{timestamp_srt}.pkl'

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
def train_anomoly_model(data_df):
    """
    Huấn luyện mô hình Isolation Forest cho việc phát hiện bất thường.
    
    Args:
        data_df (pd.DataFrame): DataFrame chứa các đặc trưng đã chuẩn hóa.
        
    Returns:
        IsolationForest: Mô hình đã được huấn luyện.
    """
    print(f"\nBắt đầu huấn luyện mô hình Isolation Forest (Contamination={CONTAMINATION_RATE})...")

    # Khởi tạo mô hình Isolation Forest
    # random_state được đặt để đảm bảo kết quả có thể tái lập
    model = IsolationForest(
        contamination=CONTAMINATION_RATE,
        random_state=42,
        n_estimators=100,
        n_jobs=-1           
    )
    #Huấn luyện mô hình
    # Isolation Forest là mô hình không giám sát (unsupervised), chỉ cần truyền dữ liệu đầu vào.
    model.fit(data_df)

    print("Hoàn thành huấn luyện mô hình.")
    return model
# ----------------------------------------------------------------------
# D. LƯU MÔ HÌNH
# ----------------------------------------------------------------------
def save_model(model, model_path):
    """
    Lưu mô hình đã huấn luyện vào file.
    
    Args:
        model (IsolationForest): Mô hình đã được huấn luyện.
        model_path (str): Đường dẫn để lưu mô hình.
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
        # Đảm bảo chỉ có 20 cột features
        X_features = scaled_data_df.copy()
        
        # 2. Huấn luyện mô hình
        anomaly_model = train_anomoly_model(X_features) 
        
        # 3. Lưu mô hình đã huấn luyện
        save_model(anomaly_model, MODEL_PATH)
        
        # 4. Kiểm tra mô hình và trích xuất bất thường
        print("\n--- Kiểm tra nhanh kết quả phân loại trên dữ liệu huấn luyện ---")
        
        # ⚠️ BƯỚC KHẮC PHỤC: Tính và gán trực tiếp vào scaled_data_df
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

        # 5. LƯU CÁC BẢN GHI BẤT THƯỜNG VÀO FILE CSV
        # Lọc ra các bản ghi bất thường. anomalies_df LÚC NÀY ĐÃ CÓ CỘT 'anomaly_score'
        anomalies_df = scaled_data_df[scaled_data_df['anomaly'] == -1].copy()
        
        if not anomalies_df.empty:
            ANOMALY_OUTPUT_PATH = os.path.join(CSV_DIR, f'anomaly_records-{timestamp_srt}.csv')
            
            anomalies_df.to_csv(ANOMALY_OUTPUT_PATH)
            
            print(f"\n✅ Đã lưu {len(anomalies_df)} bản ghi bất thường vào: {ANOMALY_OUTPUT_PATH}")
            
            # Hiển thị 5 bản ghi bất thường có điểm số thấp nhất (nghiêm trọng nhất)
            display_cols = [
                'count_total_events', 'count_fatal_errors', 'avg_session_duration', 
                'count_connect_authorized', 'count_connect_received', 'ratio_fatal_to_total', 
                'anomaly_score'
            ]
            
            print("\n5 bản ghi bất thường nghiêm trọng nhất:")
            
            # Sắp xếp, chọn 5 hàng đầu, và chỉ in ra các cột quan trọng
            top_anomalies = anomalies_df.sort_values(by='anomaly_score').head(5)
            print(top_anomalies[display_cols])
        
        print("\nQuá trình huấn luyện mô hình hoàn tất.")
