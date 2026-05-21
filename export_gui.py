import tkinter as tk
from tkinter import filedialog, messagebox
import threading
import runpy
import io
import queue
import os
import sys
from contextlib import redirect_stdout, redirect_stderr


class ExportGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("YOLOv7 Export Tool")
        self.root.resizable(False, False)

        pad = {"padx": 10, "pady": 5}

        tk.Label(root, text="PT 파일 경로:").grid(row=0, column=0, sticky="w", **pad)
        self.pt_var = tk.StringVar()
        tk.Entry(root, textvariable=self.pt_var, width=50).grid(row=0, column=1, **pad)
        tk.Button(root, text="찾기", command=self.browse_pt).grid(row=0, column=2, **pad)

        tk.Label(root, text="저장 경로:").grid(row=1, column=0, sticky="w", **pad)
        self.out_var = tk.StringVar()
        tk.Entry(root, textvariable=self.out_var, width=50).grid(row=1, column=1, **pad)
        tk.Button(root, text="찾기", command=self.browse_out).grid(row=1, column=2, **pad)

        res_frame = tk.LabelFrame(root, text="해상도 (img-size)", padx=8, pady=5)
        res_frame.grid(row=2, column=0, columnspan=3, sticky="ew", padx=10, pady=5)

        tk.Label(res_frame, text="Width:").grid(row=0, column=0, sticky="w")
        self.width_var = tk.StringVar(value="640")
        tk.Entry(res_frame, textvariable=self.width_var, width=8).grid(row=0, column=1, padx=5)

        tk.Label(res_frame, text="Height:").grid(row=0, column=2, sticky="w", padx=(15, 0))
        self.height_var = tk.StringVar(value="640")
        tk.Entry(res_frame, textvariable=self.height_var, width=8).grid(row=0, column=3, padx=5)

        tk.Label(root, text="고정 옵션: --grid  --end2end  --device 0",
                 fg="gray", font=("Consolas", 9)).grid(row=3, column=0, columnspan=3, pady=(0, 5))

        self.run_btn = tk.Button(root, text="Export", command=self.run_export,
                                 bg="#2d7dd2", fg="white", font=("Arial", 10, "bold"),
                                 padx=20, pady=6)
        self.run_btn.grid(row=4, column=0, columnspan=3, pady=8)

        tk.Label(root, text="로그:").grid(row=5, column=0, sticky="w", padx=10)
        self.log = tk.Text(root, height=12, width=70, state="disabled",
                           bg="#1e1e1e", fg="#d4d4d4", font=("Consolas", 9))
        self.log.grid(row=6, column=0, columnspan=3, padx=10, pady=(0, 10))

        sb = tk.Scrollbar(root, command=self.log.yview)
        sb.grid(row=6, column=3, sticky="ns", pady=(0, 10))
        self.log.config(yscrollcommand=sb.set)

        self._log_queue = queue.Queue()
        self._poll_log()

    def browse_pt(self):
        path = filedialog.askopenfilename(filetypes=[("PyTorch weights", "*.pt")])
        if path:
            self.pt_var.set(path)

    def browse_out(self):
        path = filedialog.askdirectory()
        if path:
            self.out_var.set(path)

    def log_write(self, text):
        self.log.config(state="normal")
        self.log.insert("end", text)
        self.log.see("end")
        self.log.config(state="disabled")

    def _poll_log(self):
        """큐에서 로그를 꺼내 UI에 표시 (100ms 주기)"""
        try:
            while True:
                msg = self._log_queue.get_nowait()
                self.log_write(msg)
        except queue.Empty:
            pass
        self.root.after(100, self._poll_log)

    def validate(self):
        pt  = self.pt_var.get().strip()
        out = self.out_var.get().strip()
        w   = self.width_var.get().strip()
        h   = self.height_var.get().strip()

        if not pt or not os.path.isfile(pt):
            messagebox.showerror("오류", "유효한 PT 파일을 선택하세요.")
            return False
        if not out or not os.path.isdir(out):
            messagebox.showerror("오류", "유효한 저장 경로를 선택하세요.")
            return False
        try:
            int(w); int(h)
        except ValueError:
            messagebox.showerror("오류", "해상도는 정수여야 합니다.")
            return False
        return True

    def run_export(self):
        if not self.validate():
            return

        pt  = self.pt_var.get().strip()
        out = self.out_var.get().strip()
        w   = self.width_var.get().strip()
        h   = self.height_var.get().strip()

        script_dir = os.path.dirname(os.path.abspath(sys.argv[0]))
        export_script = os.path.join(script_dir, "export.py")

        if not os.path.isfile(export_script):
            messagebox.showerror("Error",
                f"export.py not found:\n{export_script}\n"
                "Place this EXE in the YOLOv7 repo root.")
            return

        self.log_write(f"[export.py] {export_script}\n")
        self.log_write(f"[weights]  {pt}\n")
        self.log_write(f"[img-size] {w} {h}\n\n")
        self.run_btn.config(state="disabled", text="실행 중...")

        def worker():
            original_argv = sys.argv[:]
            original_dir  = os.getcwd()
            sys.argv = [
                export_script,
                "--weights", pt,
                "--img-size", w, h,
                "--grid",
                "--end2end",
                "--device", "0",
            ]
            buf = io.StringIO()
            success = False
            try:
                os.chdir(out)
                with redirect_stdout(buf), redirect_stderr(buf):
                    runpy.run_path(export_script, run_name="__main__")
                success = True
            except SystemExit as e:
                success = (e.code == 0)
                if not success:
                    buf.write(f"\n[SystemExit] code={e.code}\n")
            except Exception as e:
                buf.write(f"\n[Exception] {e}\n")
            finally:
                sys.argv = original_argv
                os.chdir(original_dir)

            # 로그 큐로 전달
            output = buf.getvalue()
            if output:
                self._log_queue.put(output)

            if success:
                self._log_queue.put("\n[완료] Export 성공!\n")
                self.root.after(0, messagebox.showinfo, "완료", "Export 성공!")
            else:
                self._log_queue.put("\n[오류] Export 실패. 로그를 확인하세요.\n")

            self.root.after(0, self.run_btn.config,
                            {"state": "normal", "text": "Export"})

        threading.Thread(target=worker, daemon=True).start()


if __name__ == "__main__":
    root = tk.Tk()
    app = ExportGUI(root)
    root.mainloop()
