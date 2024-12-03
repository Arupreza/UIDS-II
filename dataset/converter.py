import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import csv
import re
from enum import Enum
from pathlib import Path
import os

class AttackType(Enum):
    LEGITIMATE = 0
    DOS = 1
    FUZZING = 2
    REPLAY = 3
    MALFUNCTION = 4
    SPOOFING = 5
    MASQUERADE = 6
    FABRICATION = 7

class TRCConverterGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("TRC to CSV Converter")
        self.root.geometry("600x400")
        
        # 스타일 설정
        style = ttk.Style()
        style.configure("TButton", padding=5)
        style.configure("TLabel", padding=5)
        
        self.create_widgets()
        
    def create_widgets(self):
        # 파일 선택 프레임
        file_frame = ttk.LabelFrame(self.root, text="File Selection", padding=10)
        file_frame.pack(fill="x", padx=10, pady=5)
        
        # 입력 파일
        ttk.Label(file_frame, text="Input TRC File:").pack(anchor="w")
        input_frame = ttk.Frame(file_frame)
        input_frame.pack(fill="x", pady=2)
        
        self.input_path = tk.StringVar()
        self.input_entry = ttk.Entry(input_frame, textvariable=self.input_path)
        self.input_entry.pack(side="left", fill="x", expand=True, padx=(0, 5))
        
        ttk.Button(input_frame, text="Browse", command=self.browse_input).pack(side="right")
        
        # 출력 파일
        ttk.Label(file_frame, text="Output CSV File:").pack(anchor="w")
        output_frame = ttk.Frame(file_frame)
        output_frame.pack(fill="x", pady=2)
        
        self.output_path = tk.StringVar()
        self.output_entry = ttk.Entry(output_frame, textvariable=self.output_path)
        self.output_entry.pack(side="left", fill="x", expand=True, padx=(0, 5))
        
        ttk.Button(output_frame, text="Browse", command=self.browse_output).pack(side="right")
        
        # 옵션 프레임
        option_frame = ttk.LabelFrame(self.root, text="Options", padding=10)
        option_frame.pack(fill="x", padx=10, pady=5)
        
        # 진행 상황
        progress_frame = ttk.LabelFrame(self.root, text="Progress", padding=10)
        progress_frame.pack(fill="x", padx=10, pady=5)
        
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(
            progress_frame, 
            variable=self.progress_var,
            maximum=100,
            mode='determinate'
        )
        self.progress_bar.pack(fill="x", pady=5)
        
        self.status_var = tk.StringVar(value="Ready")
        self.status_label = ttk.Label(progress_frame, textvariable=self.status_var)
        self.status_label.pack(anchor="w")
        
        # 변환 버튼
        button_frame = ttk.Frame(self.root)
        button_frame.pack(fill="x", padx=10, pady=5)
        
        ttk.Button(
            button_frame, 
            text="Convert", 
            command=self.convert_file,
            style="TButton"
        ).pack(side="right")

    def browse_input(self):
        filename = filedialog.askopenfilename(
            title="Select TRC file",
            filetypes=[("TRC files", "*.trc"), ("All files", "*.*")]
        )
        if filename:
            self.input_path.set(filename)
            # 자동으로 출력 파일명 생성
            output_path = os.path.splitext(filename)[0] + ".csv"
            self.output_path.set(output_path)

    def browse_output(self):
        filename = filedialog.asksaveasfilename(
            title="Save CSV file",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
            defaultextension=".csv"
        )
        if filename:
            self.output_path.set(filename)

    def parse_trc_line(self, line):
        """TRC 파일의 데이터 라인을 파싱합니다."""
        pattern = r'(\d+)\)\s+(\d+\.\d+)\s+(\w+)\s+(\w+)\s+(\d+)\s+((?:[0-9A-F]{2}\s*)+)'
        match = re.match(pattern, line.strip())
        
        if match:
            number = match.group(1)
            time_offset = match.group(2)
            msg_type = match.group(3)
            msg_id = match.group(4)
            dlc = match.group(5)
            data = match.group(6).strip().split()
            
            # DLC 길이만큼 데이터 패딩
            data.extend(['00'] * (8 - len(data)))
            
            return {
                'number': number,
                'time_offset': time_offset,
                'type': msg_type,
                'id': msg_id,
                'dlc': dlc,
                'data': data
            }
        return None

    def convert_file(self):
        input_file = self.input_path.get()
        output_file = self.output_path.get()
        
        if not input_file or not output_file:
            messagebox.showerror("Error", "Please select both input and output files")
            return
            
        try:
            self.status_var.set("Converting...")
            self.progress_var.set(0)
            self.root.update()
            
            headers = [
                'NUMBER', 'TIME_OFFSET', 'TYPE', 'ID', 'DLC',
                'DATA_1', 'DATA_2', 'DATA_3', 'DATA_4',
                'DATA_5', 'DATA_6', 'DATA_7', 'DATA_8',
                'DELTA_TIME', 'ATTACK_TYPE'
            ]
            
            parsed_data = []
            prev_time = 0.0
            
            # 전체 라인 수 계산
            with open(input_file, 'r') as f:
                total_lines = sum(1 for line in f)
            
            with open(input_file, 'r') as f:
                lines = f.readlines()
                processed_lines = 0
                
                for line in lines:
                    if line.strip() and not line.startswith(';') and not line.startswith('---'):
                        parsed = self.parse_trc_line(line)
                        if parsed:
                            current_time = float(parsed['time_offset'])
                            delta_time = current_time - prev_time if prev_time != 0 else 0
                            
                            row = [
                                parsed['number'],
                                parsed['time_offset'],
                                parsed['type'],
                                parsed['id'],
                                parsed['dlc'],
                                *parsed['data'],
                                f"{delta_time:.3f}",
                                AttackType.LEGITIMATE.value
                            ]
                            parsed_data.append(row)
                            prev_time = current_time
                    
                    processed_lines += 1
                    progress = (processed_lines / total_lines) * 100
                    self.progress_var.set(progress)
                    self.root.update()
            
            with open(output_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(headers)
                writer.writerows(parsed_data)
            
            self.status_var.set("Conversion completed successfully!")
            messagebox.showinfo("Success", "File converted successfully!")
            
        except Exception as e:
            self.status_var.set("Error occurred during conversion")
            messagebox.showerror("Error", f"An error occurred: {str(e)}")
            
        finally:
            self.progress_var.set(100)

if __name__ == "__main__":
    root = tk.Tk()
    app = TRCConverterGUI(root)
    root.mainloop()