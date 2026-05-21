import argparse
import contextlib
import queue
import sys
import threading
import traceback
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk


class QueueWriter:
    """worker thread의 stdout/stderr를 GUI 로그 창으로 전달합니다."""

    def __init__(self, log_queue):
        self.log_queue = log_queue

    def write(self, text):
        if text:
            self.log_queue.put(text)

    def flush(self):
        return None


class DetectionGui(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Detect Label 22")
        self.geometry("980x720")
        self.minsize(860, 640)

        self.log_queue = queue.Queue()
        self.worker = None
        self.stop_event = threading.Event()
        self.device_values = {}

        self._build_variables()
        self._build_layout()
        self._load_devices()
        self.after(100, self._drain_log_queue)

    def _build_variables(self):
        self.source_var = tk.StringVar()
        self.weights_var = tk.StringVar()
        self.savedirs_var = tk.StringVar(value=str(Path("outputs") / "sorted"))
        self.project_var = tk.StringVar(value=str(Path("runs") / "detect"))
        self.name_var = tk.StringVar(value="exp")
        self.img_size_var = tk.StringVar(value="1280")
        self.conf_thres_var = tk.StringVar(value="0.25")
        self.iou_thres_var = tk.StringVar(value="0.45")
        self.min_recall_var = tk.StringVar(value="0.8")
        self.classes_var = tk.StringVar(value="0,1,3,4")
        self.device_var = tk.StringVar(value="자동(GPU 우선, 실패 시 CPU)")
        self.view_img_var = tk.BooleanVar(value=False)
        self.save_debug_var = tk.BooleanVar(value=False)
        self.exist_ok_var = tk.BooleanVar(value=False)
        self.no_trace_var = tk.BooleanVar(value=False)
        self.status_var = tk.StringVar(value="대기 중")

    def _build_layout(self):
        container = ttk.Frame(self, padding=16)
        container.pack(fill=tk.BOTH, expand=True)

        container.columnconfigure(0, weight=0)
        container.columnconfigure(1, weight=1)
        container.columnconfigure(2, weight=0)

        row = 0
        row = self._add_source_row(container, row)
        row = self._add_file_row(container, row, "weights pt", self.weights_var, [("PyTorch weights", "*.pt"), ("All files", "*.*")])
        row = self._add_dir_row(container, row, "savedirs", self.savedirs_var)
        row = self._add_dir_row(container, row, "project", self.project_var)
        row = self._add_entry_row(container, row, "name", self.name_var)

        numeric_frame = ttk.Frame(container)
        numeric_frame.grid(row=row, column=0, columnspan=3, sticky="ew", pady=(8, 0))
        for col in range(8):
            numeric_frame.columnconfigure(col, weight=1)

        self._add_compact_entry(numeric_frame, 0, "img-size", self.img_size_var)
        self._add_compact_entry(numeric_frame, 2, "conf-thres", self.conf_thres_var)
        self._add_compact_entry(numeric_frame, 4, "iou-thres", self.iou_thres_var)
        self._add_compact_entry(numeric_frame, 6, "min-recall", self.min_recall_var)
        row += 1

        ttk.Label(container, text="classes").grid(row=row, column=0, sticky="w", pady=(8, 0))
        ttk.Entry(container, textvariable=self.classes_var).grid(row=row, column=1, sticky="ew", pady=(8, 0), padx=(8, 8))
        ttk.Label(container, text="예: 0,1,3,4 / 비우면 전체 class").grid(row=row, column=2, sticky="w", pady=(8, 0))
        row += 1

        ttk.Label(container, text="device").grid(row=row, column=0, sticky="w", pady=(8, 0))
        self.device_combo = ttk.Combobox(container, textvariable=self.device_var, state="readonly")
        self.device_combo.grid(row=row, column=1, sticky="ew", pady=(8, 0), padx=(8, 8))
        ttk.Button(container, text="새로고침", command=self._load_devices).grid(row=row, column=2, sticky="ew", pady=(8, 0))
        row += 1

        options_frame = ttk.Frame(container)
        options_frame.grid(row=row, column=0, columnspan=3, sticky="ew", pady=(10, 0))
        ttk.Checkbutton(options_frame, text="view image", variable=self.view_img_var).pack(side=tk.LEFT)
        ttk.Checkbutton(options_frame, text="save debug images", variable=self.save_debug_var).pack(side=tk.LEFT, padx=(14, 0))
        ttk.Checkbutton(options_frame, text="exist ok", variable=self.exist_ok_var).pack(side=tk.LEFT, padx=(14, 0))
        ttk.Checkbutton(options_frame, text="no trace", variable=self.no_trace_var).pack(side=tk.LEFT, padx=(14, 0))
        row += 1

        action_frame = ttk.Frame(container)
        action_frame.grid(row=row, column=0, columnspan=3, sticky="ew", pady=(14, 0))
        action_frame.columnconfigure(0, weight=1)
        action_frame.columnconfigure(1, weight=1)
        action_frame.columnconfigure(2, weight=4)
        self.start_button = ttk.Button(action_frame, text="실행", command=self._start_detection)
        self.start_button.grid(row=0, column=0, sticky="w")
        self.stop_button = ttk.Button(action_frame, text="중지", command=self._stop_detection, state=tk.DISABLED)
        self.stop_button.grid(row=0, column=1, sticky="w", padx=(8, 0))
        ttk.Label(action_frame, textvariable=self.status_var).grid(row=0, column=2, sticky="e")
        row += 1

        ttk.Label(container, text="실행 로그").grid(row=row, column=0, columnspan=3, sticky="w", pady=(14, 0))
        row += 1

        log_frame = ttk.Frame(container)
        log_frame.grid(row=row, column=0, columnspan=3, sticky="nsew", pady=(6, 0))
        container.rowconfigure(row, weight=1)
        log_frame.rowconfigure(0, weight=1)
        log_frame.columnconfigure(0, weight=1)

        self.log_text = tk.Text(log_frame, wrap="word", height=18)
        self.log_text.grid(row=0, column=0, sticky="nsew")
        scrollbar = ttk.Scrollbar(log_frame, orient="vertical", command=self.log_text.yview)
        scrollbar.grid(row=0, column=1, sticky="ns")
        self.log_text.configure(yscrollcommand=scrollbar.set)

    def _add_file_row(self, parent, row, label, variable, filetypes):
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", pady=(0, 6))
        ttk.Entry(parent, textvariable=variable).grid(row=row, column=1, sticky="ew", padx=(8, 8), pady=(0, 6))
        ttk.Button(parent, text="찾기", command=lambda: self._browse_file(variable, filetypes)).grid(row=row, column=2, sticky="ew", pady=(0, 6))
        return row + 1

    def _add_source_row(self, parent, row):
        ttk.Label(parent, text="source").grid(row=row, column=0, sticky="w", pady=(0, 6))
        ttk.Entry(parent, textvariable=self.source_var).grid(row=row, column=1, sticky="ew", padx=(8, 8), pady=(0, 6))
        button_frame = ttk.Frame(parent)
        button_frame.grid(row=row, column=2, sticky="ew", pady=(0, 6))
        ttk.Button(
            button_frame,
            text="txt",
            command=lambda: self._browse_file(self.source_var, [("Text files", "*.txt"), ("All files", "*.*")]),
        ).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(button_frame, text="폴더", command=lambda: self._browse_dir(self.source_var)).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(6, 0))
        return row + 1

    def _add_dir_row(self, parent, row, label, variable):
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", pady=(0, 6))
        ttk.Entry(parent, textvariable=variable).grid(row=row, column=1, sticky="ew", padx=(8, 8), pady=(0, 6))
        ttk.Button(parent, text="찾기", command=lambda: self._browse_dir(variable)).grid(row=row, column=2, sticky="ew", pady=(0, 6))
        return row + 1

    def _add_entry_row(self, parent, row, label, variable):
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", pady=(0, 6))
        ttk.Entry(parent, textvariable=variable).grid(row=row, column=1, sticky="ew", padx=(8, 8), pady=(0, 6))
        return row + 1

    def _add_compact_entry(self, parent, col, label, variable):
        ttk.Label(parent, text=label).grid(row=0, column=col, sticky="w")
        ttk.Entry(parent, textvariable=variable, width=12).grid(row=0, column=col + 1, sticky="ew", padx=(6, 12))

    def _browse_file(self, variable, filetypes):
        selected = filedialog.askopenfilename(filetypes=filetypes)
        if selected:
            variable.set(selected)

    def _browse_dir(self, variable):
        selected = filedialog.askdirectory()
        if selected:
            variable.set(selected)

    def _load_devices(self):
        options = {"자동(GPU 우선, 실패 시 CPU)": "", "CPU": "cpu"}
        try:
            import torch

            if torch.cuda.is_available():
                for idx in range(torch.cuda.device_count()):
                    name = torch.cuda.get_device_name(idx)
                    try:
                        x = torch.ones((1,), device=f"cuda:{idx}")
                        _ = (x + 1).sum().item()
                        torch.cuda.synchronize(idx)
                        options[f"GPU {idx}: {name}"] = str(idx)
                    except Exception as exc:
                        self._append_log(f"GPU {idx} 사용 불가, CPU fallback 대상: {name} / {exc}\n")
        except Exception as exc:
            self._append_log(f"device 조회 실패: {exc}\n")

        self.device_values = options
        self.device_combo.configure(values=list(options.keys()))
        if self.device_var.get() not in options:
            self.device_var.set("자동(GPU 우선, 실패 시 CPU)")

    def _parse_classes(self):
        raw = self.classes_var.get().strip()
        if not raw:
            return None
        tokens = raw.replace(",", " ").split()
        try:
            return [int(token) for token in tokens]
        except ValueError as exc:
            raise ValueError("classes는 쉼표 또는 공백으로 구분된 정수여야 합니다.") from exc

    def _build_options(self):
        source_text = self.source_var.get().strip()
        weights_text = self.weights_var.get().strip()
        savedirs = self.savedirs_var.get().strip()
        project = self.project_var.get().strip()
        name = self.name_var.get().strip()

        if not source_text:
            raise ValueError("source를 입력해 주세요.")
        if not weights_text:
            raise ValueError("weights pt 파일을 입력해 주세요.")

        source = Path(source_text)
        weights = Path(weights_text)

        if not source.exists() or not (source.is_file() or source.is_dir()):
            raise ValueError("source는 이미지 폴더 또는 txt 파일이어야 합니다.")
        if not weights.is_file():
            raise ValueError("weights pt 파일을 확인해 주세요.")
        if not savedirs:
            raise ValueError("savedirs를 입력해 주세요.")
        if not project:
            raise ValueError("project를 입력해 주세요.")
        if not name:
            raise ValueError("name을 입력해 주세요.")

        return argparse.Namespace(
            weights=str(weights),
            source=str(source),
            img_size=int(self.img_size_var.get()),
            conf_thres=float(self.conf_thres_var.get()),
            iou_thres=float(self.iou_thres_var.get()),
            min_recall=float(self.min_recall_var.get()),
            device=self.device_values.get(self.device_var.get(), ""),
            view_img=self.view_img_var.get(),
            save_debug_images=self.save_debug_var.get(),
            save_txt=False,
            nosave=False,
            classes=self._parse_classes(),
            augment=False,
            project=project,
            name=name,
            exist_ok=self.exist_ok_var.get(),
            no_trace=self.no_trace_var.get(),
            agnostic_nms=False,
            savedirs=savedirs,
        )

    def _start_detection(self):
        if self.worker and self.worker.is_alive():
            messagebox.showinfo("실행 중", "이미 detection이 실행 중입니다.")
            return

        try:
            opt = self._build_options()
        except Exception as exc:
            messagebox.showerror("입력 오류", str(exc))
            return

        self.log_text.delete("1.0", tk.END)
        self.stop_event.clear()
        self.status_var.set("실행 중")
        self.start_button.configure(state=tk.DISABLED)
        self.stop_button.configure(state=tk.NORMAL)
        self.worker = threading.Thread(target=self._run_detection, args=(opt,), daemon=True)
        self.worker.start()

    def _stop_detection(self):
        if self.worker and self.worker.is_alive():
            self.stop_event.set()
            self.status_var.set("중지 요청됨")
            self._append_log("\n중지 요청됨: 현재 처리 중인 이미지가 끝난 뒤 중지합니다.\n")
            self.stop_button.configure(state=tk.DISABLED)

    def _run_detection(self, opt):
        writer = QueueWriter(self.log_queue)
        try:
            with contextlib.redirect_stdout(writer), contextlib.redirect_stderr(writer):
                import torch
                from detect_label_22 import detect

                with torch.no_grad():
                    save_dir = detect(opt, stop_event=self.stop_event)
            self.log_queue.put(f"\n완료: {save_dir}\n")
        except Exception:
            self.log_queue.put("\n실행 실패\n")
            self.log_queue.put(traceback.format_exc())
        finally:
            self.log_queue.put(("__STATUS__", "대기 중"))

    def _drain_log_queue(self):
        try:
            while True:
                item = self.log_queue.get_nowait()
                if isinstance(item, tuple) and item[0] == "__STATUS__":
                    self.status_var.set(item[1])
                    self.start_button.configure(state=tk.NORMAL)
                    self.stop_button.configure(state=tk.DISABLED)
                else:
                    self._append_log(str(item))
        except queue.Empty:
            pass
        self.after(100, self._drain_log_queue)

    def _append_log(self, text):
        self.log_text.insert(tk.END, text)
        self.log_text.see(tk.END)


def main():
    app = DetectionGui()
    app.mainloop()


if __name__ == "__main__":
    sys.exit(main())
