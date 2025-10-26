from collections import defaultdict, Counter
import csv
import os
import re
import time
from rich.console import Console
from rich.panel import Panel
from rich.style import Style
from rich.prompt import Prompt
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt 
import subprocess
import sys
import requests

#API_TOKEN_TELEGRAM
TOKEN_TELEGRAM = '8090593461:AAFPvUWUzWKpj1Xsgk_wmZWCeMJGC_eQwHE'
CHAT_ID = 6271854528
#Setup Rich
# --- Define Other Custom RGB Styles ---
H1_RGB = "#3498DB"  # A vibrant cyan-blue color
ITEM_RGB = "#F1C40F"    # Vibrant Gold/Yellow (Item Numbers)
TEXT_RGB = "#9B59B6"    # Deep Purple (Menu Text)
EXIT_RGB = "#E74C3C"    # Bright Red (Exit/Error)
LINE_RGB = "#95A5A6"    # Soft Gray (Lines/Separators)
console = Console()
# Styles for menu items and general text
H1 = Style(color=H1_RGB, bold=True)
ITEM = Style(color=ITEM_RGB)
ERROR = Style(color=EXIT_RGB, bold=True)
EXIT = Style(color=EXIT_RGB,bold=True)
LINE = Style(color=LINE_RGB, bold=True)
######################################
# --- TẠM THỜI ĐỊNH NGHĨA ĐƯỜNG DẪN CÁC FILE SCRIPT (CẦN ĐIỀU CHỈNH) ---
BASE_DIR_MAIN = os.path.dirname(os.path.abspath(__file__))
ML_SCRIPT_DIR = os.path.join(BASE_DIR_MAIN,'..', 'LLM_Model') 
ML_SCRIPT_DIR_SCRIPT05 = os.path.join(BASE_DIR_MAIN,'..', 'LLM_Model','realtime_detect.py') 

SCRIPT_01 = os.path.join(ML_SCRIPT_DIR, 'data_extraction.py')
SCRIPT_02 = os.path.join(ML_SCRIPT_DIR, 'preprocessing.py')
SCRIPT_03 = os.path.join(ML_SCRIPT_DIR, 'model_training.py')
SCRIPT_04 = os.path.join(ML_SCRIPT_DIR, 'anomaly_reporting.py')
SCRIPT_05 = os.path.join(ML_SCRIPT_DIR, 'realtime_detect.py')
# Hàm chạy file Python bên ngoài
def run_python_script(script_path):
    """Chạy một file Python bên ngoài bằng subprocess."""
    try:
        # Sử dụng sys.executable để đảm bảo dùng đúng interpreter
        process = subprocess.run([sys.executable, script_path], capture_output=True, text=True,encoding='utf-8',errors='replace', check=True)
        console.print(f"[{H1_RGB}]--- Kết quả chạy {os.path.basename(script_path)} ---[/]", style=LINE)
        console.print(process.stdout)
        if process.stderr:
            console.print(f"[{EXIT_RGB}]LỖI KHI CHẠY SCRIPT:[/]", style=ERROR)
            console.print(process.stderr)
    except subprocess.CalledProcessError as e:
        console.print(f"[{EXIT_RGB}]LỖI HỆ THỐNG khi chạy {os.path.basename(script_path)}:[/]", style=ERROR)
        console.print(e.stderr)
    except FileNotFoundError:
        console.print(f"[{EXIT_RGB}]LỖI: Không tìm thấy file script tại đường dẫn: {script_path}[/]", style=ERROR)


# Regex Pattern để trích xuất các trường thông tin chính
# Nhóm 1: Timestamp
# Nhóm 2: PID
# Nhóm 3: User
# Nhóm 4: Database
# Nhóm 5: CẤP ĐỘ LOG (LOG|ERROR|FATAL|STATEMENT) <--- ĐƯỢC TRÍCH XUẤT LÀM GROUP
# Nhóm 6: Log Content (phần còn lại của dòng log)
POSTGRES_LOG_PATTERN = re.compile(
    r'(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}\.\d{3}\s+[+\-]\d{2})\s+\[(\d+)\]\s+'  # Group 1, 2: Timestamp and PID
    r'(?:(\w+))?@(\w+)\s+'                                                     # Group 3, 4: User@Database
    r'((?:LOG|ERROR|FATAL|STATEMENT)(?::))?\s*'                                 # Group 5: Cấp độ Log (LOG:|ERROR:|...)
    r'(.*)'                                                                   # Group 6: The rest of the log message (Content)
)

def readFile(Path) -> list:
    """Đọc file log và trả về list các dòng log (strings). File được đóng tự động."""
    logCollection = []
    try:
        # File sẽ tự động đóng khi khối 'with' kết thúc
        with open(Path, 'r', encoding='utf-8') as file:
            # print(f"Đã mở file thành công: {os.path.basename(Path)}")
            for line in file:
                text = line.strip()
                if text:
                    logCollection.append(text)
        return logCollection
    except FileNotFoundError:
        print(f"Lỗi: Không tìm thấy file '{Path}'. Vui lòng kiểm tra lại đường dẫn.")
        return []
    except Exception as e:
        print(f"Đã xảy ra lỗi: {e}")
        return []


def filter_and_parse_logs(logs: list) -> list:
    """Phân tích cú pháp mỗi dòng log và trích xuất các trường chính."""
    parsed_data = []
    for log_line in logs:
        # Sửa lỗi: Sử dụng re.match() hoặc re.search() với regex chính
        match = POSTGRES_LOG_PATTERN.search(log_line)
        if match:
            # Trích xuất 6 nhóm từ regex (Timestamp, PID, User, DB, Level_raw, Content)
            timestamp, pid, user, db, level_raw, content = match.groups()

            # --- LOGIC PHÂN LOẠI CẤP ĐỘ MỚI ---
            
            # Xử lý level_raw: chỉ lấy từ khóa (ví dụ: 'LOG:' -> 'LOG')
            level_1 = level_raw.strip(':') if level_raw else 'UNKNOWN'
            level_2 = ''
            
            # Tìm kiếm AUDIT (Cấp độ 2)
            if 'AUDIT:' in content:
                level_2 = 'AUDIT'
            
            # Nếu có Level 2 (AUDIT) thì Final là AUDIT, ngược lại là Level 1
            if level_2 == 'AUDIT':
                final_level = 'AUDIT'
            else:
                final_level = level_1


            # Cố gắng trích xuất loại hành động SQL chi tiết từ nội dung AUDIT
            # 1. Trích xuất loại hành động chung (DDL, READ, WRITE, ROLE)
            action_match_general = re.search(r'AUDIT:\s+SESSION,\d+,\d+,(\w+),', content)
            action_type_general = action_match_general.group(1) if action_match_general else ''

            # 2. Cố gắng trích xuất tên lệnh SQL thực tế (SELECT, INSERT, CREATE, DELETE, GRANT, v.v.)
            # Tên lệnh SQL luôn nằm ngay sau loại hành động chung trong log AUDIT
            action_match_specific = re.search(r'AUDIT:\s+SESSION,\d+,\d+,\w+,(\w+),', content)
            action_type_specific = action_match_specific.group(1) if action_match_specific else action_type_general

            # Cố gắng trích xuất Nội dung SQL (nằm trong dấu "")
            query_match = re.search(r'"(.*?)"', content)
            query_text = query_match.group(1) if query_match else ''
            
            # Cố gắng trích xuất lệnh SQL từ STATEMENT hoặc ERROR/FATAL
            if final_level in ['STATEMENT', 'ERROR', 'FATAL'] and query_text:
                # Lấy từ khóa đầu tiên của câu lệnh SQL
                specific_command_match = re.match(r'(\w+)', query_text.strip())
                action_type = specific_command_match.group(1).upper() if specific_command_match else action_type_general
            else:
                # Nếu là AUDIT, ưu tiên tên lệnh SQL cụ thể (ví dụ: CREATE TABLE, SELECT)
                action_type = action_type_specific


            parsed_data.append({
                'timestamp': timestamp,
                'pid': pid,
                'user': user or 'N/A',
                'database': db or 'N/A',
                'level_final': final_level, 
                'level_1': level_1,
                'level_2': level_2,
                'action_type': action_type, # Sử dụng tên lệnh SQL cụ thể
                'query': query_text,
                'raw_content': content.strip()
            })
        else:
            # Xử lý các dòng log không khớp (ví dụ: log system đơn giản không có user@db)
            parsed_data.append({'raw_log': log_line, 'level_final': 'SYSTEM'})

    return parsed_data

# ----------------------------------------------------
# Official File End
# ----------------------------------------------------
def initial_parse() -> list:
    """Hàm khởi tạo và phân tích file log ban đầu."""
    parsed_data = []
    base_dir = os.path.dirname(os.path.abspath(__file__))
    log_file_path = os.path.join(base_dir, '..', 'Log_Example', 'postgresql-official.log')
    Log_lines = readFile(log_file_path)
    if Log_lines: 
        parsed_data = filter_and_parse_logs(Log_lines)
    return parsed_data
# ----------------------------------------------------
# Function List All Logs base on PID: 
# ----------------------------------------------------
def logs_baseon_pid() -> list:
    logs = defaultdict(list)
    base_dir = os.path.dirname(os.path.abspath(__file__))
    log_file_path = os.path.join(base_dir, '..', 'Log_Example', 'postgresql-official.log')
    Log_lines = readFile(log_file_path)
    pattern = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{3}) .*?\[\s*(\d+)\s*\].*?\s(LOG|FATAL|ERROR|DETAIL):\s+(.*)")
    for line in Log_lines:
        match = pattern.search(line)
        if match:
            timestamp, pid, level, message = match.groups()
            logs[pid].append(f"{timestamp} | {level}: {message}") #pid la key 
    return logs 
    

def print_logs_by_pid(logs: dict):
    print("KẾT QUẢ PHÂN TÍCH CÁC DÒNG LOG THEO PID")
    print("=" * 60)
    for pid, actions in logs.items():
        console.print(f"\n[{ITEM}]PID[/]: {pid}")
        for action in actions:
            console.print(f"[{ITEM}]*[/]:  {action}")
    
def export_logs_to_csv():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    log_file_path = os.path.join(base_dir, '..', 'Log_Example', 'postgresql-official.log')
    timestamp_str = datetime.now().strftime("%Y%m%d")
    csv_file_name = f'logs-{timestamp_str}.csv'
    output_file_path = os.path.join(base_dir, '..', 'CSV_FILE', csv_file_name)
    Log_lines = readFile(log_file_path) 
    # Nếu file đã tồn tại thì xóa
    if os.path.exists(output_file_path):
        os.remove(output_file_path)

    pattern = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{3}) .*?\[\s*(\d+)\s*\].*?\s(LOG|FATAL|ERROR|DETAIL):\s+(.*)")
    extraced_logs = []
    for line in Log_lines:
        match = pattern.search(line)
        if match:
            timestamp, pid, level, message = match.groups()
            extraced_logs.append([timestamp, pid, level, message])

        with open(output_file_path, 'w', encoding='utf-8') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(['Timestamp', 'PID', 'Level', 'Message'])
            writer.writerows(extraced_logs)
    print(f"Logs đã được xuất thành công vào file: {output_file_path}")


# ----------------------------------------------------
# Function List All Logs
# ----------------------------------------------------
def list_all_logs(parsed_data: list):
    """Liệt kê tất cả các dòng log đã phân tích."""
    print("\n" + "=" * 60)
    print("KẾT QUẢ PHÂN TÍCH CÁC DÒNG LOG")
    print("=" * 60)  
    for item in parsed_data:
                #time.sleep(0.1)  # Thêm độ trễ nhỏ để dễ quan sát khi in ra
        if item.get('level_final') != 'SYSTEM':
                # In ra chi tiết nếu final_level không phải là SYSTEM
            print(f"Thời gian: {item['timestamp']}")
            print(f"Người dùng: {item['user']} | Database: {item['database']}")
            print(f"CẤP ĐỘ FINAL: {item['level_final']} (L1: {item['level_1']} | L2: {item['level_2']})")
                
            # Logic hiển thị chi tiết (đã sửa)
        if item['level_final'] == 'AUDIT':
            print(f"Loại hành động: {item['action_type']}")
            print(f"Query: {item['query'][:70]}...")
        elif item['level_final'] in ['ERROR', 'FATAL', 'STATEMENT']:
            print(f"Nội dung: {item['raw_content']}")
        elif item['level_final'] == 'LOG':
            print(f"Nội dung: {item['raw_content']}")
        else:
            # Trường hợp UNKNOWN
            print(f"Log thô: {item.get('raw_content', item.get('raw_log', 'N/A'))}")

        print(f"This is the end of list log")        
        print("-" * 30)
# ----------------------------------------------------
# Function List Connect Log
# ----------------------------------------------------
def list_connect_logs(parsed_data: list):
    """Liệt kê tất cả các dòng log kết nối."""
    time.sleep(0.5) # Thêm độ trễ nhỏ để dễ quan sát khi in ra
    list_connect = []
    for item in parsed_data:
        if item.get('final_level') != 'SYSTEM':
            raw_data = item.get('raw_content')
            if raw_data and raw_data.startswith('connection authorized: user'):
                list_connect.append(item)
    for item in list_connect:
        time.sleep(0.05)
        console.print(f'\n[{ITEM}]PID[/] :\r{item['pid']}')
        console.print(f'[{ITEM}]Time[/] :\r{item['timestamp']}')
        console.print(f'[{ITEM}]Raw data[/]:\r{item['raw_content']}') 
    console.print(f"[{LINE}]#####[/]" * 30)
        # Đếm số lượng connect theo database
    db_counts = Counter(item['user'] for item in list_connect)
    # Tạo DataFrame từ dữ liệu đếm
    df = pd.DataFrame.from_dict(db_counts, orient='index', columns=['connect_count'])
    df = df.sort_values(by='connect_count', ascending=False)
    # Tạo biểu đồ area chart
    plt.figure(figsize=(10, 6))
    plt.fill_between(df.index, df['connect_count'], color='skyblue', alpha=0.5)
    plt.plot(df.index, df['connect_count'], color='Slateblue', alpha=0.6, linewidth=2)
    plt.title('Số lượng User connect')
    plt.xlabel('Tên User')
    plt.ylabel('Số lần connect')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.grid(True)
    # Hiển thị số lượng connect trên từng điểm
    for i, value in enumerate(df['connect_count']):
        plt.text(i, value + 0.5, str(value), ha='center', va='bottom', fontsize=9)

    plt.show()
# ----------------------------------------------------
# Function List Disconnect Log
# ----------------------------------------------------
def list_disconnect_logs(parsed_data: list):
    """Liệt kê tất cả các dòng log ngắt kết nối."""
    time.sleep(0.5) # Thêm độ trễ nhỏ để dễ quan sát khi in ra
    list_disconnect = []
    for item in parsed_data:
        if item.get('final_level') != 'SYSTEM':
            raw_data = item.get('raw_content')
            if raw_data and raw_data.startswith('disconnection'):
                list_disconnect.append(item)
    for item in list_disconnect:
        time.sleep(0.05)
        console.print(f'\n[{ITEM}]PID[/] :\r{item['pid']}')
        console.print(f'[{ITEM}]Time[/]:{item['timestamp']}')
        console.print(f'[{ITEM}]Raw data[/]:{item['raw_content']}') 
    console.print(f"[{LINE}]#####[/]" * 30)
    # Đếm số lượng disconnect theo database
    db_counts = Counter(item['database'] for item in list_disconnect)
    # Tạo DataFrame từ dữ liệu đếm
    df = pd.DataFrame.from_dict(db_counts, orient='index', columns=['disconnect_count'])
    df = df.sort_values(by='disconnect_count', ascending=False)
    # Tạo biểu đồ area chart
    plt.figure(figsize=(10, 6))
    plt.fill_between(df.index, df['disconnect_count'], color='skyblue', alpha=0.5)
    plt.plot(df.index, df['disconnect_count'], color='Slateblue', alpha=0.6, linewidth=2)
    plt.title('Số lượng disconnect theo Database')
    plt.xlabel('Tên Database')
    plt.ylabel('Số lần disconnect')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.grid(True)
    # Hiển thị số lượng disconnect trên từng điểm
    for i, value in enumerate(df['disconnect_count']):
        plt.text(i, value + 0.5, str(value), ha='center', va='bottom', fontsize=9)

    plt.show()
# ----------------------------------------------------
# HÀM CẢNH BÁO Permission Denied
# ----------------------------------------------------
def alertPermission(parsed_data: list):
    
    """Tìm kiếm và in ra cảnh báo khi phát hiện lỗi "permission denied"."""
    
    permission_denied_found = False
    count = 0
    print("\n" + "=" * 60)
    print("CẢNH BÁO BẢO MẬT: LỖI QUYỀN TRUY CẬP")
    print("=" * 60)
    
    for item in parsed_data:
        
        # Lọc nhanh chỉ các dòng có cấp độ ERROR hoặc FATAL
        if item.get('level_1') in ['ERROR', 'FATAL']:
            # Kiểm tra xem chuỗi "permission denied" có tồn tại trong raw_content không
            if "permission denied" in item.get('raw_content', ''):
                count+=1
                permission_denied_found = True
                
                console.print(f"[{H1}][PERMISSION DENIED DETECTED][/]")
                console.print(f"[{ITEM}]PID[/]: {item['pid']}")
                console.print(f"[{ITEM}]Time[/]: {item['timestamp']}")
                console.print(f"[{ITEM}]User[/]: {item['user']} @ {item['database']}")
                console.print(f"[{ITEM}]Lỗi[/]: {item['raw_content']}")
                
                # Nếu có query, in ra lệnh SQL đã cố gắng chạy
                if item.get('query'):
                     print(f"Lệnh SQL: {item['query']}")
                print("-" * 30)
    print(f'Tong cong {count} permission denied')
    if not permission_denied_found:
        print("Không tìm thấy lỗi 'permission denied' nào trong log được phân tích.")
    print("=" * 60)

#-----------------------------------------------------
#SEND REPORT TO TELEGRAM
#-----------------------------------------------------
def send_report(parsed_data: list):
    """Tìm kiếm và in ra cảnh báo khi phát hiện lỗi "permission denied"."""
    NOW = datetime.now().strftime('%Y-%m-%d')
    permission_denied_found = False
    base_url = f'https://api.telegram.org/bot{TOKEN_TELEGRAM}/sendMessage'
    print("\n" + "=" * 60)
    print("CẢNH BÁO BẢO MẬT: LỖI QUYỀN TRUY CẬP")
    print("=" * 60)
    
    for item in parsed_data:
        
        # Lọc nhanh chỉ các dòng có cấp độ ERROR hoặc FATAL
        if item.get('level_1') in ['ERROR', 'FATAL']:
            # Kiểm tra xem chuỗi "permission denied" có tồn tại trong raw_content không
            if "permission denied" in item.get('raw_content', '') and NOW in item['timestamp'] :
                PID = item['pid']
                TIMESTAMP = item['timestamp']
                USER = item['user']
                DATABASE = item['database']
                CONTENT = item['raw_content']
                if item.get('query'):
                    QUERY = item['query']
                    payload = {
                        'chat_id': CHAT_ID,
                        'text': f"PID:{PID}\nTIMESTAMP:{TIMESTAMP}\nUSER:{USER}\nDATABASE:{DATABASE}\nCONTENT:{CONTENT}\nQUERY:{QUERY}" 
                    }
                else: 
                    payload = {
                        'chat_id': CHAT_ID,
                        'text': f"PID:{PID}\nTIMESTAMP:{TIMESTAMP}\nUSER:{USER}\nDATABASE:{DATABASE}\nCONTENT:{CONTENT}" 
                    }
                response = requests.post(base_url, data=payload)
                
                # Nếu có query, in ra lệnh SQL đã cố gắng chạy
    
    if not permission_denied_found:
        print("Không tìm thấy lỗi 'permission denied' nào trong log được phân tích.")
    print("=" * 60)
    
#-----------------------------------------------------
#Menu
#-----------------------------------------------------

def display_menu():
    """Displays the menu options."""
    menu_text = (
        f"[{ITEM}]1 Xem LOG dựa trên PID và xuất ra file CSV[/]\n"
        f"[{ITEM}]2 Xem cảnh báo quyền không được phép[/]\n"
        f"[{ITEM}]3 Xem danh sách USER kết nối với DB[/]\n"
        f"[{ITEM}]4 Xem danh sách USER ngắt kết nối với DB[/]\n"
        f"{'':^40}\n"
        f"[{H1_RGB}]PHÁT HIỆN BẤT THƯỜNG (ML PIPELINE):[/]\n"
        f"[{ITEM}]5[/] - CHẠY TOÀN BỘ PIPELINE ML (1 -> 4)\n"
        f"[{ITEM}]6[/] - [TRAIN] 01. Trích xuất/Làm sạch Log (Tạo Events CSV)\n"
        f"[{ITEM}]7[/] - [TRAIN] 02. Tiền xử lý/Tạo đặc trưng (Tạo Scaler/Features)\n"
        f"[{ITEM}]8[/] - [TRAIN] 03. Huấn luyện Mô hình (Tạo Isolation Forest Model)\n"
        f"[{ITEM}]9[/] - [REPORT] 04. Truy tìm ngược Báo cáo PID bất thường\n"
        f"{'':-^40}\n"
        f"[{H1_RGB}]GIÁM SÁT REAL-TIME:[/]\n"
        f"[{ITEM}]R[/] - SEND REPORT TO TELEGRAM\n"
        f"[{ITEM}]X[/] - Khác để THOÁT\n"
        f"[{ITEM}]Khác để THOÁT[/]\n"
    )
    panel = Panel(
        menu_text,
        title = "[bold red]✨ DBS401-MENU ✨[/bold red]",
        border_style="green",
        padding=(1,2)
    )
    console.print(panel)


def menu_choice():
    parsed_data = initial_parse()

    while True:
        display_menu()
        choice = Prompt.ask('Enter your choice, other to exit')
        console.print("\n" + "="*20)

        if choice == '1':
            console.print(f'[{H1}]>>>You choose option [1]: Monitor full logs base on PID & Export to CSV')
            time.sleep(1)
            #list_all_logs(parsed_data)
            logs = logs_baseon_pid()
            print_logs_by_pid(logs)
            export_logs_to_csv()

        elif choice == '2':
            console.print(f'[{H1}]>>>You choose option [2]: Unauthorized use alert ')
            alertPermission(parsed_data)
            print("-" * 30)
        elif choice == '3':
            console.print(f'[{H1}]>>>You choose option [3]: List connection')
            list_connect_logs(parsed_data)
        elif choice == '4':
            console.print(f'[{H1}]>>>You choose option [4]: List disconnection')
            #List Disconnect
            list_disconnect_logs(parsed_data)
        #-- ML PIPLINE --
        elif choice == '5':
            console.print(f'[{H1}]>>>Bạn chọn [5]: CHẠY TOÀN BỘ PIPELINE ML[/]')
            run_python_script(SCRIPT_01)
            run_python_script(SCRIPT_02)
            run_python_script(SCRIPT_03)
            run_python_script(SCRIPT_04)
        elif choice == '6':
            console.print(f'[{H1}]>>>Bạn chọn [6]: 01. Trích xuất/Làm sạch Log[/]')
            run_python_script(SCRIPT_01)
        elif choice == '7': 
            console.print(f'[{H1}]>>>Bạn chọn [7]: 02. Tiền xử lý/Tạo đặc trưng[/]')
            run_python_script(SCRIPT_02)
        elif choice == '8':
            console.print(f'[{H1}]>>>Bạn chọn [8]: 03. Huấn luyện Mô hình[/]')
            run_python_script(SCRIPT_03)
        elif choice == '9':
            console.print(f'[{H1}]>>>Bạn chọn [9]: 04. Truy tìm ngược Báo cáo PID bất thường[/]')
            run_python_script(SCRIPT_04)

        #---REAL TIME MONITOR---#
        elif choice == 'R':
            send_report(parsed_data)
            console.print(f"[{EXIT}]>>>Send Done!")

        else: 
            console.print(f"[{EXIT}]>>>Exiting. Goodbye!")
            break




def main():
    menu_choice()


if __name__ == "__main__":
    main()
