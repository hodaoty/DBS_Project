import os
import re
import time
from rich.console import Console
from rich.panel import Panel
from rich.style import Style
from rich.prompt import Prompt

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
    log_file_path = os.path.join(base_dir, '..', 'Log_Example', 'postgresql.log')
    Log_lines = readFile(log_file_path)
    if Log_lines: 
        parsed_data = filter_and_parse_logs(Log_lines)
    return parsed_data
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
        console.print(f'[{ITEM}]PID[/] :\r{item['pid']}')
        console.print(f'[{ITEM}]Time[/] :\r{item['timestamp']}')
        console.print(f'[{ITEM}]Raw data[/]:\r{item['raw_content']}') 
    console.print(f"[{LINE}]#####[/]" * 30)
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
        console.print(f'[{ITEM}]PID[/] :\r{item['pid']}')
        console.print(f'[{ITEM}]Time[/]:{item['timestamp']}')
        console.print(f'[{ITEM}]Raw data[/]:{item['raw_content']}') 
    console.print(f"[{LINE}]#####[/]" * 30)
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
#Menu
#-----------------------------------------------------
def display_menu():
    """Displays the menu options."""
    menu_text = (
        f"[{ITEM}]1 Monitor full logs[/]\n"
        f"[{ITEM}]2 Unauthorized use alert[/]\n"
        f"[{ITEM}]3 List connection[/]\n"
        f"[{ITEM}]4 List disconnection[/]\n"
        f"[{ITEM}]Other Exit application[/]\n"
    )
    panel = Panel(
        menu_text,
        title = "[bold red]✨ APPLICATION MAIN MENU ✨[/bold red]",
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
            console.print(f'[{H1}]>>>You choose option [1]: Monitor full logs ')
                # Lấy đường dẫn thư mục chứa file Python hiện tại (/api)

            list_all_logs(parsed_data)

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
        else: 
            console.print(f"[{EXIT}]>>>Exiting. Goodbye!")
            break




def main():
    menu_choice()


if __name__ == "__main__":
    main()
