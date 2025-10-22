import pandas as pd
import os
from datetime import datetime
import re

# ----------------------------------------------------------------------
# A. CẤU HÌNH VÀ THIẾT LẬP ĐƯỜNG DẪN
# ----------------------------------------------------------------------
timestamp_str = datetime.now().strftime('%Y%m%d')
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_DIR = os.path.join(BASE_DIR, '..', 'CSV_FILE', 'OUTPUT_CSVFILE')

# Tên file bất thường (OUTPUT từ 03_model_training.py)
ANOMALY_FILE = f'anomaly_records-{timestamp_str}.csv'
# ⚠️ ĐÃ SỬA: Thêm thư mục con 'TRAIN_AI'
ANOMALY_PATH = os.path.join(CSV_DIR, 'TRAIN_AI', ANOMALY_FILE) 

# Tên file sự kiện chi tiết (OUTPUT từ 01_data_extraction.py) - CHỨA CỘT PID
EVENTS_FILE = f'postgresql_events-{timestamp_str}.csv'
# ⚠️ ĐÃ SỬA: Thêm thư mục con 'LOG_EVENT'
EVENTS_PATH = os.path.join(CSV_DIR, 'LOG_EVENT', EVENTS_FILE) 

# Tên file đầu ra chi tiết (Không sử dụng file này, nhưng vẫn giữ đường dẫn)
OUTPUT_PID_REPORT_FILE = f'anomalous_pid_report-{timestamp_str}.csv'
# ⚠️ ĐÃ SỬA: Thêm thư mục con 'REPORT'
OUTPUT_PID_REPORT_PATH = os.path.join(CSV_DIR, 'REPORT', OUTPUT_PID_REPORT_FILE) 

# Cấu hình cửa sổ thời gian (Phải khớp với 02_preprocessing.py)
RESAMPLE_FREQUENCY = '5T' # 5 phút

# ----------------------------------------------------------------------
# B. HÀM TRUY TÌM NGƯỢC
# ----------------------------------------------------------------------
# Danh sách các loại sự kiện log quan trọng mà ta muốn truy tìm ngược
# Đã loại bỏ các sự kiện AUDIT và LOG thông thường để tập trung vào lỗi/kết nối đột biến
CRITICAL_EVENT_TYPES = [
    'FATAL',        # Lỗi nghiêm trọng nhất
    'ERROR',        # Lỗi hệ thống/truy vấn
    'DISCONNECT',   # Ngắt kết nối
    'CONNECT_RECEIVED', # Kết nối nhận được
    'CONNECT_AUTHORIZED' # Kết nối được ủy quyền
]

def look_back_and_report_pids(anomaly_path, events_path, output_path, freq):
    """
    Truy tìm ngược các sự kiện log chi tiết (bao gồm PID, User, Query) 
    trong các khung thời gian đã được mô hình phân loại là bất thường, 
    chỉ lọc các sự kiện log có mức độ quan trọng cao và HIỂN THỊ TRỰC TIẾP.
    """
    try:
        # 1. Tải dữ liệu bất thường (chỉ cần các timestamp)
        print(f"Đang tải dữ liệu bất thường từ: {anomaly_path}")
        anomaly_df = pd.read_csv(anomaly_path, index_col=0, parse_dates=True)
        anomaly_timestamps = anomaly_df.index
        
        # 2. Tải dữ liệu sự kiện chi tiết (CHỨA PID)
        print(f"Đang tải dữ liệu sự kiện chi tiết từ: {events_path}")
        events_df = pd.read_csv(events_path, parse_dates=['timestamp'])
        
        # 3. Chuẩn bị khoảng thời gian
        window_delta = pd.Timedelta(freq)
        anomalous_events_list = []
        
        print(f"Bắt đầu truy tìm ngược trên {len(anomaly_timestamps)} khung thời gian bất thường...")
        
        for start_time in anomaly_timestamps:
            end_time = start_time + window_delta
            
            # 3a. Lọc tất cả các sự kiện log chi tiết nằm trong khung 5 phút này
            current_window_events = events_df[
                (events_df['timestamp'] >= start_time) & (events_df['timestamp'] < end_time)
            ].copy()
            
            if not current_window_events.empty:
                # ⚠️ BƯỚC LỌC: Chỉ giữ lại các sự kiện có event_type nằm trong danh sách CRITICAL_EVENT_TYPES
                critical_events = current_window_events[
                    current_window_events['event_type'].isin(CRITICAL_EVENT_TYPES)
                ].copy()
                
                if not critical_events.empty:
                    # 3b. Thêm thông tin khung thời gian và điểm bất thường vào báo cáo
                    critical_events['Anomaly_Time_Window'] = start_time
                    critical_events['Anomaly_Score'] = anomaly_df.loc[start_time, 'anomaly_score']
                    
                    anomalous_events_list.append(critical_events)
                
        # 4. Gom kết quả và tạo báo cáo
        if anomalous_events_list:
            final_report_df = pd.concat(anomalous_events_list, ignore_index=True)
            
            # Các cột mới được thêm vào
            new_cols = ['Anomaly_Time_Window', 'Anomaly_Score']
            
            # Thứ tự cột: Cột mới + Cột gốc
            report_cols_ordered = new_cols + [col for col in final_report_df.columns if col not in new_cols]
            
            # ⚠️ KÍCH HOẠT LẠI LƯU FILE CSV (Theo yêu cầu người dùng)
            
            # Tạo thư mục OUTPUT nếu chưa tồn tại
            if not os.path.exists(os.path.dirname(output_path)):
                os.makedirs(os.path.dirname(output_path))
                print(f"Đã tạo thư mục đầu ra: {os.path.dirname(output_path)}")


            final_report_df[report_cols_ordered].to_csv(output_path, index=False) # <--- ĐÃ BỎ COMMENT VÀ KÍCH HOẠT
            
            print("-" * 60)
            print(f"✅ HOÀN TẤT TRUY TÌM NGƯỢC VÀ LƯU FILE. Tổng số sự kiện log quan trọng được tìm thấy: {len(final_report_df)}")
            print(f"✅ File báo cáo đã được lưu tại: {output_path}")
            print("-" * 60)

            # 5. HIỂN THỊ TRỰC TIẾP
            print("\n----------------------------------------------------------------------")
            print("🚨 BÁO CÁO LOG BẤT THƯỜNG QUAN TRỌNG (PID/ERROR/CONNECT) 🚨")
            print("----------------------------------------------------------------------")
            
            # Sắp xếp theo điểm số bất thường (càng âm càng nghiêm trọng)
            # Sau đó sắp xếp theo timestamp để dễ theo dõi
            final_report_df = final_report_df.sort_values(by=['Anomaly_Score', 'timestamp'], ascending=[True, True])

            # Hiển thị DataFrame
            pd.set_option('display.max_rows', 100) # Giới hạn lại số hàng hiển thị
            pd.set_option('display.max_columns', None)
            pd.set_option('display.width', 1000)

            # Chỉ hiển thị các cột quan trọng nhất cho phân tích log
            display_cols = ['Anomaly_Time_Window', 'Anomaly_Score', 'pid', 'user', 'event_type', 'query_command', 'query_text', 'timestamp']

            print(final_report_df[display_cols].to_string())
            
            print("-" * 60)

            # Hiển thị tóm tắt các PID/User nổi bật
            print("\n🚨 Top 5 PID/User đáng ngờ trong các khung bất thường:")
            suspicious_pids = final_report_df.groupby(['pid', 'user']).size().sort_values(ascending=False)
            print(suspicious_pids.head(5).to_string())
            
            return final_report_df
        else:
            print("Không tìm thấy sự kiện chi tiết nào trong các khung thời gian bất thường.")
            return None

    except FileNotFoundError as e:
        print(f"❌ LỖI: Không tìm thấy một trong các file đầu vào: {e}")
        print("Vui lòng đảm bảo đã chạy thành công 01_data_extraction.py và 03_model_training.py.")
        return None
    except Exception as e:
        print(f"❌ LỖI trong quá trình xử lý truy tìm ngược: {e}")
        return None

# ----------------------------------------------------------------------
# C. PHẦN THỰC THI CHÍNH
# ----------------------------------------------------------------------
if __name__ == "__main__":
    look_back_and_report_pids(ANOMALY_PATH, EVENTS_PATH, OUTPUT_PID_REPORT_PATH, RESAMPLE_FREQUENCY)
