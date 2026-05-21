"""
detect_label_22_gui.py  —  YOLOv7 Detection Evaluation  GUI  v1.0.0
=====================================================================
• 모던 다크 테마 (GitHub Dark 스타일)
• 실시간 GOOD / MISS / FAIL 통계 카드
• 커스텀 진행 바 + ETA 표시
• 컬러 코딩 로그 (INFO / WARNING / ERROR)
• HTML 리포트 / 결과 폴더 바로 열기
• GPU / CPU 자동 선택 (smoke test 포함)
• 백그라운드 스레드로 UI 응답성 유지
"""

import argparse
import contextlib
import datetime
import os
import platform
import queue
import subprocess
import sys
import threading
import traceback
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

APP_VERSION = '1.0.0'

# ── 색상 팔레트 (GitHub Dark 계열) ────────────────────────────────
C_BG        = '#0d1117'   # 메인 배경
C_SURFACE   = '#161b22'   # 카드 / 패널
C_SURFACE2  = '#21262d'   # 입력 필드 배경
C_BORDER    = '#30363d'   # 테두리
C_ACCENT    = '#388bfd'   # 파란 강조 (실행 버튼)
C_ACCENT_H  = '#58a6ff'   # hover 시 밝은 파랑
C_TEXT      = '#e6edf3'   # 기본 텍스트
C_TEXT_DIM  = '#8b949e'   # 보조 텍스트
C_SUCCESS   = '#3fb950'   # GOOD  (초록)
C_WARNING   = '#d29922'   # MISS  (황금)
C_DANGER    = '#f85149'   # FAIL  (빨강)
C_BTN       = '#21262d'   # 일반 버튼 배경
C_BTN_H     = '#30363d'   # 일반 버튼 hover
C_STOP_BG   = '#3a1a1a'   # 중지 버튼 배경
C_STOP_H    = '#5a2a2a'   # 중지 버튼 hover

# ── 폰트 ────────────────────────────────────────────────────────
_WIN = platform.system() == 'Windows'
FF       = 'Segoe UI'    if _WIN else 'Helvetica'
FF_MONO  = 'Consolas'    if _WIN else 'Courier New'

FONT_UI   = (FF, 9)
FONT_BOLD = (FF, 9, 'bold')
FONT_SM   = (FF, 8)
FONT_LG   = (FF, 11)
FONT_TITL = (FF, 13, 'bold')
FONT_NUM  = (FF, 28, 'bold')
FONT_MONO = (FF_MONO, 8)


# ══════════════════════════════════════════════════════════════════
#  헬퍼 위젯
# ══════════════════════════════════════════════════════════════════

def _btn(parent, text, command, bg=C_BTN, fg=C_TEXT, hover=C_BTN_H, **kw):
    """평평한 커스텀 버튼 (disabled 상태에서는 hover 효과 없음)."""
    b = tk.Button(
        parent, text=text, command=command,
        bg=bg, fg=fg,
        activebackground=hover, activeforeground=fg,
        relief='flat', bd=0, padx=14, pady=7,
        cursor='hand2', font=FONT_UI, **kw,
    )

    def _on_enter(_):
        if b.cget('state') != 'disabled':
            b.configure(bg=hover)

    def _on_leave(_):
        if b.cget('state') != 'disabled':
            b.configure(bg=bg)

    b.bind('<Enter>', _on_enter)
    b.bind('<Leave>', _on_leave)
    return b


def _lbl(parent, text='', fg=C_TEXT, font=FONT_UI, bg=None, **kw):
    return tk.Label(parent, text=text, fg=fg, font=font,
                    bg=bg if bg else C_BG, **kw)


def _entry(parent, textvariable, width=None, **kw):
    return tk.Entry(
        parent, textvariable=textvariable,
        bg=C_SURFACE2, fg=C_TEXT, insertbackground=C_TEXT,
        relief='flat', bd=0,
        highlightbackground=C_BORDER, highlightthickness=1,
        font=FONT_UI, **({'width': width} if width else {}), **kw,
    )


class StatCard(tk.Frame):
    """GOOD / MISS / FAIL / 합계 실시간 통계 카드."""

    def __init__(self, parent, title, color, compact=False, **kw):
        super().__init__(parent, bg=C_BORDER, **kw)
        inner = tk.Frame(self, bg=C_SURFACE)
        inner.pack(fill='both', expand=True, padx=1, pady=1)

        self._count_var = tk.StringVar(value='0')
        self._pct_var   = tk.StringVar(value='0.0%')

        num_font = (FF, 16, 'bold') if compact else FONT_NUM
        pad_v    = (5, 1) if compact else (10, 2)
        pad_b    = (1, 5) if compact else (2, 10)

        tk.Label(inner, text=title, bg=C_SURFACE, fg=color,
                 font=(FF, 8, 'bold')).pack(pady=pad_v)
        tk.Label(inner, textvariable=self._count_var, bg=C_SURFACE, fg=C_TEXT,
                 font=num_font).pack()
        tk.Label(inner, textvariable=self._pct_var, bg=C_SURFACE, fg=C_TEXT_DIM,
                 font=FONT_SM).pack(pady=pad_b)

    def update(self, count, total):
        self._count_var.set(str(count))
        pct = count / total * 100 if total else 0.0
        self._pct_var.set(f'{pct:.1f}%')


class ProgressBar(tk.Canvas):
    """커스텀 슬림 진행 바."""
    H = 6

    def __init__(self, parent, color=C_ACCENT, **kw):
        super().__init__(parent, height=self.H, bg=C_SURFACE2,
                         highlightthickness=0, bd=0, **kw)
        self._color = color
        self._rect  = self.create_rectangle(0, 0, 0, self.H, fill=color, outline='')

    def set(self, ratio: float):
        ratio = max(0.0, min(1.0, ratio))
        w = self.winfo_width() or 1
        self.coords(self._rect, 0, 0, w * ratio, self.H)


class ClassStatsTable(tk.Frame):
    """클래스 × 어노테이션 레벨 실시간 통계 테이블.

    class_stats 구조: {cls_id: {'gt': N, 'matched_gt': M, 'miss': K, 'false': F}}
    """

    _COLS = [
        # (key,        헤더,       width_chars, anchor, normal_fg,  highlight_fn)
        ('cls',      '클래스',    7,   'center', None,       None),
        ('gt',       'GT 수',     6,   'e',      C_TEXT_DIM, None),
        ('matched',  '매칭',      6,   'e',      C_SUCCESS,  None),
        ('miss',     '미검출',    7,   'e',      None,       lambda v: C_WARNING if v > 0 else C_TEXT_DIM),
        ('false',    '오검출',    7,   'e',      None,       lambda v: C_DANGER  if v > 0 else C_TEXT_DIM),
        ('recall',   '재현율',    8,   'e',      None,       lambda v: (C_SUCCESS if v >= 80 else
                                                                         C_WARNING if v >= 50 else C_DANGER)),
    ]
    _MAX_ROWS = 16   # 클래스 수 상한

    def __init__(self, parent, **kw):
        super().__init__(parent, bg=C_SURFACE, **kw)
        self._row_data: dict  = {}   # cls_id  → row_idx
        self._cells:    dict  = {}   # (row, col) → tk.Label
        self._next_row: int   = 0
        self._build()

    # ── 헤더 ──────────────────────────────────────────────────
    def _build(self):
        hdr = tk.Frame(self, bg=C_SURFACE2)
        hdr.pack(fill='x')
        for col_i, (_, label, w, anc, _, _hfn) in enumerate(self._COLS):
            sep = tk.Frame(hdr, bg=C_BORDER, width=1)
            sep.pack(side='left', fill='y')
            tk.Label(
                hdr, text=label, bg=C_SURFACE2, fg=C_TEXT_DIM,
                font=(FF, 8, 'bold'), width=w, anchor=anc, padx=4, pady=4,
            ).pack(side='left')
        tk.Frame(hdr, bg=C_BORDER, width=1).pack(side='left', fill='y')

        self._body = tk.Frame(self, bg=C_SURFACE)
        self._body.pack(fill='x')

        # 빈 안내 레이블 (데이터 없을 때)
        self._empty_lbl = tk.Label(
            self._body, text='검출 시작 후 클래스별 통계가 여기에 표시됩니다.',
            bg=C_SURFACE, fg=C_TEXT_DIM, font=FONT_SM, pady=8,
        )
        self._empty_lbl.pack()

    # ── 행 생성 ───────────────────────────────────────────────
    def _add_row(self, cls_id: int) -> int:
        row_idx = self._next_row
        bg = C_BG if row_idx % 2 == 0 else '#13181f'
        row_frame = tk.Frame(self._body, bg=bg)
        row_frame.pack(fill='x')

        for col_i, (_, _, w, anc, _, _) in enumerate(self._COLS):
            sep = tk.Frame(row_frame, bg=C_BORDER, width=1)
            sep.pack(side='left', fill='y')
            lbl = tk.Label(
                row_frame, text='', bg=bg, fg=C_TEXT,
                font=FONT_MONO, width=w, anchor=anc, padx=4, pady=2,
            )
            lbl.pack(side='left')
            self._cells[(row_idx, col_i)] = lbl

        tk.Frame(row_frame, bg=C_BORDER, width=1).pack(side='left', fill='y')
        self._row_data[cls_id] = row_idx
        self._next_row += 1
        return row_idx

    # ── 데이터 갱신 (메인 스레드에서 호출) ───────────────────
    def update_stats(self, class_stats: dict):
        if not class_stats:
            return
        if self._empty_lbl.winfo_ismapped():
            self._empty_lbl.pack_forget()

        for cls_id in sorted(class_stats.keys()):
            if cls_id not in self._row_data:
                if self._next_row >= self._MAX_ROWS:
                    continue
                self._add_row(cls_id)
            row_idx = self._row_data[cls_id]
            cs      = class_stats[cls_id]
            gt      = cs['gt']
            matched = cs['matched_gt']
            miss    = cs['miss']
            false_  = cs['false']
            recall  = matched / gt * 100.0 if gt > 0 else 100.0

            raw_vals = [cls_id, gt, matched, miss, false_, recall]
            disp = [
                f'cls {cls_id}',
                str(gt),
                str(matched),
                str(miss)   if miss   > 0 else '-',
                str(false_) if false_ > 0 else '-',
                f'{recall:.1f}%',
            ]

            for col_i, (_, _, _, _, base_fg, hfn) in enumerate(self._COLS):
                lbl = self._cells[(row_idx, col_i)]
                lbl.configure(text=disp[col_i])
                if hfn is not None:
                    lbl.configure(fg=hfn(raw_vals[col_i]))
                elif base_fg:
                    lbl.configure(fg=base_fg)

    def reset(self):
        """새 검출 실행 시 테이블 초기화."""
        for w in self._body.winfo_children():
            if w is not self._empty_lbl:
                w.destroy()
        self._row_data.clear()
        self._cells.clear()
        self._next_row = 0
        self._empty_lbl.pack()


# ══════════════════════════════════════════════════════════════════
#  큐 라이터  (worker → GUI)
# ══════════════════════════════════════════════════════════════════

class QueueWriter:
    """worker thread의 print / logging 출력을 GUI 로그로 전달합니다."""

    def __init__(self, log_queue):
        self.q = log_queue

    def write(self, text):
        if text:
            try:
                # [FIX] put_nowait: 큐가 꽉 찼을 때 worker 스레드 블로킹 방지
                self.q.put_nowait(('__LOG__', text))
            except queue.Full:
                pass  # 로그 드롭 (GUI 렌더링이 일시적으로 느릴 때만 발생)

    def flush(self):
        pass


# ══════════════════════════════════════════════════════════════════
#  메인 GUI
# ══════════════════════════════════════════════════════════════════

class DetectionGui(tk.Tk):

    # ── 초기화 ──────────────────────────────────────────────────
    def __init__(self):
        super().__init__()
        self.title(f'객체 검출 평가 Tool    v{APP_VERSION}')
        self.geometry('1000x820')
        self.minsize(860, 700)
        self.configure(bg=C_BG)

        # maxsize=2000: 큐 무제한 증가 방지 (초당 수천 장 처리 시에도 여유)
        self.log_queue      = queue.Queue(maxsize=2000)
        self.worker         = None
        self.stop_event     = threading.Event()
        self.device_values  = {}
        self._save_dir      = None
        self._report_path   = None
        self._start_time    = None

        self._build_vars()
        self._build_ui()
        self._load_devices()
        self.after(100, self._drain_queue)

        # [FIX] 창 닫기(X) 버튼 핸들러 — worker 실행 중 강제 종료 방지
        self.protocol('WM_DELETE_WINDOW', self._on_close)

    # ── tkinter 변수 ────────────────────────────────────────────
    def _build_vars(self):
        self.v_source    = tk.StringVar()
        self.v_weights   = tk.StringVar()
        self.v_savedirs  = tk.StringVar(value=str(Path('outputs') / 'sorted'))
        self.v_project   = tk.StringVar(value=str(Path('runs') / 'detect'))
        self.v_name      = tk.StringVar(value='exp')
        self.v_imgsize   = tk.StringVar(value='1280')
        self.v_conf      = tk.StringVar(value='0.25')
        self.v_iou       = tk.StringVar(value='0.45')
        self.v_recall    = tk.StringVar(value='0.8')
        self.v_classes   = tk.StringVar(value='0,1,3,4')
        self.v_device    = tk.StringVar(value='자동(GPU 우선, 실패 시 CPU)')
        self.v_debug     = tk.BooleanVar(value=False)
        self.v_viewimg   = tk.BooleanVar(value=False)
        self.v_existok   = tk.BooleanVar(value=False)
        self.v_notrace   = tk.BooleanVar(value=False)
        self.v_status    = tk.StringVar(value='대기 중')
        self.v_progress  = tk.StringVar(value='')
        self.v_elapsed   = tk.StringVar(value='')
        self.v_eta       = tk.StringVar(value='')

    # ══════════════════════════════════════════════════════════════
    #  UI 빌드
    # ══════════════════════════════════════════════════════════════

    def _build_ui(self):
        # ── 헤더 ─────────────────────────────────────────────────
        hdr = tk.Frame(self, bg=C_SURFACE, pady=14, padx=20)
        hdr.pack(fill='x')
        tk.Label(hdr, text='🔍  Detect Label 22', bg=C_SURFACE,
                 fg=C_TEXT, font=FONT_TITL).pack(side='left')
        tk.Label(hdr, text=f'v{APP_VERSION}', bg=C_SURFACE,
                 fg=C_TEXT_DIM, font=FONT_SM).pack(side='left', padx=(8, 0), anchor='s', pady=(0,2))

        # ── 스크롤 영역 ──────────────────────────────────────────
        main = tk.Frame(self, bg=C_BG, padx=16, pady=12)
        main.pack(fill='both', expand=True)

        # 설정
        self._build_settings(_card_frame(main, '⚙  설정'))
        tk.Frame(main, bg=C_BG, height=10).pack(fill='x')

        # 액션 + 진행
        self._build_action(main)
        tk.Frame(main, bg=C_BG, height=8).pack(fill='x')

        # 통계 카드
        self._build_stat_cards(main)
        tk.Frame(main, bg=C_BG, height=8).pack(fill='x')

        # 로그
        self._build_log(_card_frame(main, '📋  실행 로그', expand=True))

    # ── 설정 패널 ────────────────────────────────────────────────
    def _build_settings(self, card):
        body = tk.Frame(card, bg=C_SURFACE, padx=14, pady=10)
        body.pack(fill='x')
        body.columnconfigure(1, weight=1)

        r = 0
        # source (txt / 폴더 2개 버튼)
        _lbl(body, '소스 경로', fg=C_TEXT_DIM, font=FONT_SM, bg=C_SURFACE,
             width=11, anchor='w').grid(row=r, column=0, sticky='w', pady=3)
        _entry(body, self.v_source).grid(row=r, column=1, sticky='ew', padx=(8,6), pady=3)
        bf = tk.Frame(body, bg=C_SURFACE)
        bf.grid(row=r, column=2, sticky='ew', pady=3)
        _btn(bf, 'TXT', lambda: self._browse_file(
             self.v_source, [('Text', '*.txt'), ('All', '*.*')])).pack(side='left')
        _btn(bf, '폴더', lambda: self._browse_dir(self.v_source)).pack(side='left', padx=(4,0))
        r += 1

        # weights
        self._row_file(body, r, '가중치 파일', self.v_weights,
                       [('PyTorch weights', '*.pt'), ('All', '*.*')]); r += 1

        # savedirs
        self._row_dir(body, r, '저장 경로', self.v_savedirs); r += 1

        # project + name
        pn = tk.Frame(body, bg=C_SURFACE)
        pn.grid(row=r, column=0, columnspan=3, sticky='ew', pady=(4, 0))
        _lbl(pn, '프로젝트', fg=C_TEXT_DIM, font=FONT_SM, bg=C_SURFACE,
             width=11, anchor='w').pack(side='left')
        _entry(pn, self.v_project).pack(side='left', fill='x', expand=True, padx=(8,8))
        _lbl(pn, '이름', fg=C_TEXT_DIM, font=FONT_SM, bg=C_SURFACE,
             width=5, anchor='w').pack(side='left')
        _entry(pn, self.v_name, width=16).pack(side='left', padx=(4, 0))
        r += 1

        # 수치 파라미터
        num = tk.Frame(body, bg=C_SURFACE)
        num.grid(row=r, column=0, columnspan=3, sticky='ew', pady=(8, 0))
        for col, (lbl, var) in enumerate([
            ('이미지 크기',  self.v_imgsize),
            ('신뢰도',       self.v_conf),
            ('IOU 임계값',   self.v_iou),
            ('최소 재현율',  self.v_recall),
        ]):
            pad = (16 if col else 0, 0)
            _lbl(num, lbl, fg=C_TEXT_DIM, font=FONT_SM, bg=C_SURFACE).grid(
                row=0, column=col*2, sticky='w', padx=(pad[0], 0))
            _entry(num, var, width=9).grid(
                row=0, column=col*2+1, sticky='ew', padx=(4, 0))
        r += 1

        # classes + device
        cd = tk.Frame(body, bg=C_SURFACE)
        cd.grid(row=r, column=0, columnspan=3, sticky='ew', pady=(8, 0))
        _lbl(cd, '클래스', fg=C_TEXT_DIM, font=FONT_SM, bg=C_SURFACE).pack(side='left')
        _entry(cd, self.v_classes, width=18).pack(side='left', padx=(4, 20))
        _lbl(cd, '장치', fg=C_TEXT_DIM, font=FONT_SM, bg=C_SURFACE).pack(side='left')
        self._device_combo = ttk.Combobox(cd, textvariable=self.v_device,
                                          state='readonly', width=30)
        self._device_combo.pack(side='left', padx=(4, 6))
        _btn(cd, '↺', self._load_devices, bg=C_BTN).pack(side='left')
        r += 1

        # 체크박스
        chk = tk.Frame(body, bg=C_SURFACE)
        chk.grid(row=r, column=0, columnspan=3, sticky='ew', pady=(8, 4))
        for text, var in [
            ('디버그 이미지 저장',     self.v_debug),
            ('결과 화면 표시',         self.v_viewimg),
            ('결과 폴더 덮어쓰기',     self.v_existok),
            ('TracedModel 비활성화',   self.v_notrace),
        ]:
            tk.Checkbutton(
                chk, text=text, variable=var,
                bg=C_SURFACE, fg=C_TEXT, selectcolor=C_SURFACE2,
                activebackground=C_SURFACE, activeforeground=C_TEXT,
                font=FONT_SM, bd=0,
            ).pack(side='left', padx=(0, 16))

    def _row_file(self, parent, row, label, var, ftypes):
        _lbl(parent, label, fg=C_TEXT_DIM, font=FONT_SM, bg=C_SURFACE,
             width=11, anchor='w').grid(row=row, column=0, sticky='w', pady=3)
        _entry(parent, var).grid(row=row, column=1, sticky='ew', padx=(8,6), pady=3)
        _btn(parent, '찾기', lambda: self._browse_file(var, ftypes)).grid(
            row=row, column=2, sticky='ew', pady=3)

    def _row_dir(self, parent, row, label, var):
        _lbl(parent, label, fg=C_TEXT_DIM, font=FONT_SM, bg=C_SURFACE,
             width=11, anchor='w').grid(row=row, column=0, sticky='w', pady=3)
        _entry(parent, var).grid(row=row, column=1, sticky='ew', padx=(8,6), pady=3)
        _btn(parent, '찾기', lambda: self._browse_dir(var)).grid(
            row=row, column=2, sticky='ew', pady=3)

    # ── 액션 바 + 진행 바 ────────────────────────────────────────
    def _build_action(self, parent):
        frame = tk.Frame(parent, bg=C_BG)
        frame.pack(fill='x')

        # 버튼 영역
        btns = tk.Frame(frame, bg=C_BG)
        btns.pack(side='left')

        self._btn_start = _btn(btns, '▶  실행', self._start,
                                bg=C_ACCENT, fg='#ffffff', hover=C_ACCENT_H)
        self._btn_start.pack(side='left', padx=(0, 8))

        self._btn_stop = _btn(btns, '■  중지', self._stop,
                               bg=C_STOP_BG, fg=C_DANGER, hover=C_STOP_H)
        self._btn_stop.configure(state='disabled')
        self._btn_stop.pack(side='left')

        # 상태 / 시간
        right = tk.Frame(frame, bg=C_BG)
        right.pack(side='right')
        tk.Label(right, textvariable=self.v_elapsed,
                 bg=C_BG, fg=C_TEXT_DIM, font=FONT_SM).pack(side='right', padx=(6,0))
        tk.Label(right, textvariable=self.v_eta,
                 bg=C_BG, fg=C_WARNING, font=FONT_UI).pack(side='right', padx=(12,0))
        tk.Label(right, textvariable=self.v_status,
                 bg=C_BG, fg=C_TEXT, font=FONT_UI).pack(side='right')

        # 진행 바
        pb_wrap = tk.Frame(parent, bg=C_BG)
        pb_wrap.pack(fill='x', pady=(8, 0))

        self._pb = ProgressBar(pb_wrap, color=C_ACCENT)
        self._pb.pack(fill='x', pady=(0, 4))

        pb_txt = tk.Frame(pb_wrap, bg=C_BG)
        pb_txt.pack(fill='x')
        tk.Label(pb_txt, textvariable=self.v_progress,
                 bg=C_BG, fg=C_TEXT_DIM, font=FONT_SM).pack(side='left')

    # ── 통계 카드 + 클래스 통계 테이블 ──────────────────────────
    def _build_stat_cards(self, parent):
        # ── 소형 통계 카드 4개 (compact=True → 절반 높이) ──────
        frame = tk.Frame(parent, bg=C_BG)
        frame.pack(fill='x')
        frame.columnconfigure((0, 1, 2, 3), weight=1)

        self._card_good  = StatCard(frame, '정상',   C_SUCCESS,  compact=True)
        self._card_miss  = StatCard(frame, '미검출', C_WARNING,  compact=True)
        self._card_fail  = StatCard(frame, '오검출', C_DANGER,   compact=True)
        self._card_total = StatCard(frame, '처리 수', C_TEXT_DIM, compact=True)
        for col, card in enumerate((self._card_good, self._card_miss,
                                     self._card_fail, self._card_total)):
            card.grid(row=0, column=col, sticky='nsew',
                      padx=(0 if col == 0 else 8, 0))

        tk.Frame(parent, bg=C_BG, height=8).pack(fill='x')

        # ── 클래스별 어노테이션 통계 테이블 ────────────────────
        tbl_card = _card_frame(parent, '📊  클래스별 어노테이션 통계')
        tbl_wrap = tk.Frame(tbl_card, bg=C_SURFACE, padx=14, pady=8)
        tbl_wrap.pack(fill='x')
        self._class_stats_table = ClassStatsTable(tbl_wrap)
        self._class_stats_table.pack(fill='x')

    # ── 로그 패널 ────────────────────────────────────────────────
    def _build_log(self, card):
        # 버튼 행
        btn_row = tk.Frame(card, bg=C_SURFACE, padx=14, pady=6)
        btn_row.pack(fill='x')
        _btn(btn_row, '📊 리포트 열기', self._open_report,
             bg=C_BTN, fg=C_TEXT_DIM).pack(side='right', padx=(4,0))
        _btn(btn_row, '📂 결과 폴더', self._open_folder,
             bg=C_BTN, fg=C_TEXT_DIM).pack(side='right', padx=(4,0))
        _btn(btn_row, '📁 저장 경로', self._open_savedirs,
             bg=C_BTN, fg=C_TEXT_DIM).pack(side='right', padx=(4,0))
        _btn(btn_row, '🗑 지우기', self._clear_log,
             bg=C_BTN, fg=C_TEXT_DIM).pack(side='right')

        # 구분선
        tk.Frame(card, bg=C_BORDER, height=1).pack(fill='x')

        # 텍스트 영역
        log_wrap = tk.Frame(card, bg=C_SURFACE, padx=14, pady=10)
        log_wrap.pack(fill='both', expand=True)
        log_wrap.rowconfigure(0, weight=1)
        log_wrap.columnconfigure(0, weight=1)

        self._log = tk.Text(
            log_wrap, wrap='word', height=12,
            bg=C_BG, fg=C_TEXT, insertbackground=C_TEXT,
            font=FONT_MONO, relief='flat', bd=0, state='disabled',
        )
        self._log.grid(row=0, column=0, sticky='nsew')
        sb = ttk.Scrollbar(log_wrap, orient='vertical', command=self._log.yview)
        sb.grid(row=0, column=1, sticky='ns')
        self._log.configure(yscrollcommand=sb.set)

        # 텍스트 컬러 태그
        self._log.tag_configure('info',    foreground=C_TEXT)
        self._log.tag_configure('warn',    foreground=C_WARNING)
        self._log.tag_configure('error',   foreground=C_DANGER)
        self._log.tag_configure('success', foreground=C_SUCCESS)
        self._log.tag_configure('dim',     foreground=C_TEXT_DIM)
        self._log.tag_configure('accent',  foreground=C_ACCENT_H)

    # ══════════════════════════════════════════════════════════════
    #  장치 탐색 (백그라운드 스레드로 실행 — GUI 블로킹 방지)
    # ══════════════════════════════════════════════════════════════

    def _load_devices(self):
        """GPU 목록 탐색을 백그라운드 스레드에서 실행합니다."""
        self._device_combo.configure(values=['탐색 중…'], state='disabled')
        self.v_device.set('탐색 중…')
        # [FIX] 탐색 중 실행 버튼 비활성화 (device_values가 비어있을 때 실행 방지)
        self._btn_start.configure(state='disabled')
        t = threading.Thread(target=self._scan_devices, daemon=True)
        t.start()

    def _scan_devices(self):
        """실제 장치 탐색 (worker thread)."""
        opts = {'자동(GPU 우선, 실패 시 CPU)': '', 'CPU': 'cpu'}
        try:
            import torch
            if torch.cuda.is_available():
                for idx in range(torch.cuda.device_count()):
                    name = torch.cuda.get_device_name(idx)
                    try:
                        x = torch.ones((1,), device=f'cuda:{idx}')
                        _ = (x + 1).sum().item()
                        torch.cuda.synchronize(idx)
                        opts[f'GPU {idx}: {name}'] = str(idx)
                    except Exception as exc:
                        self.log_queue.put(('__LOG__',
                            f'WARNING: GPU {idx} 사용 불가 ({name}): {exc}\n'))
        except Exception as exc:
            self.log_queue.put(('__LOG__', f'WARNING: device 조회 실패: {exc}\n'))

        # UI 업데이트는 메인 스레드 스케줄러에 위임
        self.after(0, self._apply_devices, opts)

    def _apply_devices(self, opts: dict):
        """탐색 결과를 UI에 반영합니다 (메인 스레드)."""
        self.device_values = opts
        self._device_combo.configure(values=list(opts.keys()), state='readonly')
        current = self.v_device.get()
        if current not in opts:
            self.v_device.set('자동(GPU 우선, 실패 시 CPU)')
        # [FIX] 탐색 완료 후 실행 버튼 다시 활성화
        self._btn_start.configure(state='normal')

    # ══════════════════════════════════════════════════════════════
    #  파일 / 폴더 선택
    # ══════════════════════════════════════════════════════════════

    def _browse_file(self, var, ftypes):
        p = filedialog.askopenfilename(filetypes=ftypes)
        if p:
            var.set(p)

    def _browse_dir(self, var):
        p = filedialog.askdirectory()
        if p:
            var.set(p)

    # ══════════════════════════════════════════════════════════════
    #  옵션 빌드 및 검증
    # ══════════════════════════════════════════════════════════════

    def _parse_classes(self):
        raw = self.v_classes.get().strip()
        if not raw:
            return None
        tokens = raw.replace(',', ' ').split()
        try:
            return [int(t) for t in tokens]
        except ValueError:
            raise ValueError('classes는 쉼표/공백으로 구분된 정수여야 합니다. 예: 0,1,3,4')

    def _build_opt(self):
        src  = self.v_source.get().strip()
        wt   = self.v_weights.get().strip()
        sd   = self.v_savedirs.get().strip()
        proj = self.v_project.get().strip()
        name = self.v_name.get().strip()

        if not src:  raise ValueError('source를 입력해 주세요.')
        if not wt:   raise ValueError('weights 파일을 선택해 주세요.')
        if not sd:   raise ValueError('savedirs를 입력해 주세요.')
        if not proj: raise ValueError('project를 입력해 주세요.')
        if not name: raise ValueError('name을 입력해 주세요.')

        sp, wp = Path(src), Path(wt)
        if not sp.exists():    raise ValueError(f'source가 존재하지 않습니다:\n{src}')
        if not wp.is_file():   raise ValueError(f'weights 파일을 확인해 주세요:\n{wt}')

        try:    img_size = int(self.v_imgsize.get())
        except: raise ValueError('이미지 크기는 정수여야 합니다.')
        try:    conf = float(self.v_conf.get())
        except: raise ValueError('신뢰도는 0~1 사이 실수여야 합니다.')
        try:    iou  = float(self.v_iou.get())
        except: raise ValueError('IOU 임계값은 0~1 사이 실수여야 합니다.')
        try:    recall = float(self.v_recall.get())
        except: raise ValueError('최소 재현율은 0~1 사이 실수여야 합니다.')

        return argparse.Namespace(
            weights=str(wp), source=str(sp),
            img_size=img_size, conf_thres=conf, iou_thres=iou, min_recall=recall,
            device=self.device_values.get(self.v_device.get(), ''),
            view_img=self.v_viewimg.get(),
            save_debug_images=self.v_debug.get(),
            save_txt=False, nosave=False,
            classes=self._parse_classes(),
            augment=False,
            project=proj, name=name,
            exist_ok=self.v_existok.get(),
            no_trace=self.v_notrace.get(),
            agnostic_nms=False,
            savedirs=sd,
        )

    # ══════════════════════════════════════════════════════════════
    #  실행 / 중지
    # ══════════════════════════════════════════════════════════════

    def _start(self):
        if self.worker and self.worker.is_alive():
            messagebox.showinfo('실행 중', 'Detection이 이미 실행 중입니다.')
            return

        try:
            opt = self._build_opt()
        except ValueError as exc:
            messagebox.showerror('입력 오류', str(exc))
            return

        # UI 초기화
        self._clear_log()
        self.stop_event.clear()
        self._save_dir     = None
        self._report_path  = None
        self._start_time   = datetime.datetime.now()

        self.v_status.set('실행 중…')
        self.v_elapsed.set('')
        self.v_progress.set('')
        self.v_eta.set('')
        self._pb.set(0)
        for card in (self._card_good, self._card_miss, self._card_fail, self._card_total):
            card.update(0, 1)

        self._btn_start.configure(state='disabled')
        self._btn_stop.configure(state='normal')
        self._class_stats_table.reset()

        callback = self._make_callback()
        self.worker = threading.Thread(
            target=self._worker, args=(opt, callback), daemon=True)
        self.worker.start()
        self._tick_elapsed()

    def _on_close(self):
        """창 닫기(X) 버튼 핸들러 — worker 실행 중이면 확인 후 종료."""
        if self.worker and self.worker.is_alive():
            from tkinter import messagebox as _mb
            if not _mb.askyesno(
                '종료 확인',
                'Detection이 실행 중입니다.\n중지하고 종료하시겠습니까?\n\n'
                '(현재 이미지 처리 후 종료됩니다)',
            ):
                return
            self.stop_event.set()
        self.destroy()

    def _stop(self):
        if self.worker and self.worker.is_alive():
            self.stop_event.set()
            self.v_status.set('중지 요청됨…')
            self._log_gui('\n⚠  중지 요청: 현재 이미지 완료 후 정지합니다.\n', 'warn')
            self._btn_stop.configure(state='disabled')

    def _tick_elapsed(self):
        """경과 시간을 1초마다 갱신합니다."""
        if self.worker and self.worker.is_alive():
            if self._start_time:
                elapsed = (datetime.datetime.now() - self._start_time).total_seconds()
                self.v_elapsed.set(f'⏱ {_fmt_time(elapsed)}')
            self.after(1000, self._tick_elapsed)

    def _make_callback(self):
        def cb(current, total, category, stats, group_stats, elapsed, eta,
               class_stats=None):
            self.log_queue.put(('__PROGRESS__', {
                'current': current, 'total': total,
                'category': category,
                'stats': stats, 'group_stats': group_stats,
                'elapsed': elapsed, 'eta': eta,
                'class_stats': class_stats or {},
            }))
        return cb

    def _worker(self, opt, progress_callback):
        writer = QueueWriter(self.log_queue)
        try:
            with contextlib.redirect_stdout(writer), contextlib.redirect_stderr(writer):
                import torch
                from detect_label_22 import detect
                with torch.no_grad():
                    save_dir = detect(opt,
                                       stop_event=self.stop_event,
                                       progress_callback=progress_callback)
            self.log_queue.put(('__DONE__', str(save_dir) if save_dir else ''))
        except Exception:
            self.log_queue.put(('__ERROR__', traceback.format_exc()))
        finally:
            self.log_queue.put(('__STATUS__', '대기 중'))

    # ══════════════════════════════════════════════════════════════
    #  큐 드레인 (100ms 폴링)
    # ══════════════════════════════════════════════════════════════

    def _drain_queue(self):
        try:
            while True:
                item = self.log_queue.get_nowait()
                if not isinstance(item, tuple):
                    self._write(str(item), 'info')
                    continue
                kind, data = item

                if kind == '__STATUS__':
                    self.v_status.set(data)
                    self._btn_start.configure(state='normal')
                    self._btn_stop.configure(state='disabled')

                elif kind == '__PROGRESS__':
                    self._handle_progress(data)

                elif kind == '__DONE__':
                    self._save_dir = data
                    if data:
                        rp = Path(data) / 'report.html'
                        if rp.exists():
                            self._report_path = str(rp)
                    self._log_gui(f'\n✅  완료: {data}\n', 'success')
                    self._pb.set(1.0)
                    self.v_progress.set('완료')
                    self.v_eta.set('')

                elif kind == '__ERROR__':
                    self._log_gui(f'\n❌  실행 실패\n{data}\n', 'error')

                elif kind == '__LOG__':
                    tag = ('warn'  if 'WARNING' in data else
                           'error' if 'ERROR'   in data else 'info')
                    self._write(data.replace('\r', ''), tag)

        except queue.Empty:
            pass
        self.after(100, self._drain_queue)

    def _handle_progress(self, d):
        cur, tot = d['current'], d['total']
        group       = d['group_stats']
        elapsed     = d['elapsed']
        eta         = d['eta']
        class_stats = d.get('class_stats', {})

        ratio = cur / tot if tot else 0
        self._pb.set(ratio)
        self.v_progress.set(f'{cur} / {tot}   ({ratio*100:.1f}%)')
        self.v_eta.set(f'⏳ {_fmt_time(eta)}' if eta > 0 else '')

        self._card_good.update(group.get('GOOD', 0), tot)
        self._card_miss.update(group.get('MISS', 0), tot)
        self._card_fail.update(group.get('FAIL', 0), tot)
        self._card_total.update(cur, tot)

        if class_stats:
            self._class_stats_table.update_stats(class_stats)

    # ══════════════════════════════════════════════════════════════
    #  로그 헬퍼
    # ══════════════════════════════════════════════════════════════

    def _log_gui(self, text, tag='info'):
        """GUI 자체 메시지 (타임스탬프 포함)."""
        ts = datetime.datetime.now().strftime('%H:%M:%S')
        self._write(f'[{ts}] ', 'dim')
        self._write(text, tag)

    def _write(self, text, tag='info'):
        self._log.configure(state='normal')
        self._log.insert('end', text, tag)
        self._log.see('end')
        self._log.configure(state='disabled')

    def _clear_log(self):
        self._log.configure(state='normal')
        self._log.delete('1.0', 'end')
        self._log.configure(state='disabled')

    # ══════════════════════════════════════════════════════════════
    #  결과 폴더 / 리포트 열기
    # ══════════════════════════════════════════════════════════════

    def _open_folder(self):
        path = self._save_dir
        if not path:
            messagebox.showinfo('알림', 'Detection을 먼저 실행해 주세요.')
            return
        p = Path(path)
        if not p.exists():
            messagebox.showwarning('알림', f'폴더가 존재하지 않습니다:\n{path}')
            return
        _open_path(str(p))

    def _open_savedirs(self):
        """저장 경로(GOOD/MISS/FAIL 분류 폴더)를 탐색기로 엽니다."""
        path = self.v_savedirs.get().strip()
        if not path:
            messagebox.showinfo('알림', '저장 경로를 먼저 설정해 주세요.')
            return
        p = Path(path)
        if not p.exists():
            if messagebox.askyesno('폴더 없음',
                                   f'저장 경로가 아직 존재하지 않습니다:\n{p}\n\n'
                                   '폴더를 생성하고 열까요?'):
                try:
                    p.mkdir(parents=True, exist_ok=True)
                except Exception as exc:
                    messagebox.showerror('생성 실패', str(exc))
                    return
            else:
                return
        _open_path(str(p))

    def _open_report(self):
        path = self._report_path
        if not path:
            messagebox.showinfo('알림', 'Detection 완료 후 리포트가 생성됩니다.')
            return
        _open_path(path)


# ══════════════════════════════════════════════════════════════════
#  유틸리티 함수
# ══════════════════════════════════════════════════════════════════

def _fmt_time(sec: float) -> str:
    """초를 MM:SS 또는 HH:MM:SS 형태로 변환합니다."""
    sec = max(0, int(sec))
    m, s = divmod(sec, 60)
    h, m = divmod(m, 60)
    return f'{h:02d}:{m:02d}:{s:02d}' if h else f'{m:02d}:{s:02d}'


def _open_path(path: str):
    """OS에 맞게 파일/폴더를 기본 앱으로 엽니다."""
    system = platform.system()
    try:
        if system == 'Windows':
            os.startfile(path)
        elif system == 'Darwin':
            subprocess.Popen(['open', path])
        else:
            subprocess.Popen(['xdg-open', path])
    except Exception as exc:
        messagebox.showerror('열기 실패', str(exc))


def _card_frame(parent, title, expand=False):
    """제목이 있는 카드 프레임을 parent에 배치하고 inner를 반환합니다."""
    outer = tk.Frame(parent, bg=C_BORDER, padx=1, pady=1)
    if expand:
        outer.pack(fill='both', expand=True)
    else:
        outer.pack(fill='x')
    inner = tk.Frame(outer, bg=C_SURFACE)
    inner.pack(fill='both', expand=True)
    if title:
        tk.Label(inner, text=title, bg=C_SURFACE, fg=C_TEXT_DIM,
                 font=(FF, 8, 'bold'), padx=14, pady=8).pack(side='top', anchor='w')
        tk.Frame(inner, bg=C_BORDER, height=1).pack(fill='x')
    return inner


# ══════════════════════════════════════════════════════════════════
#  진입점
# ══════════════════════════════════════════════════════════════════

def main():
    app = DetectionGui()
    app.mainloop()


if __name__ == '__main__':
    sys.exit(main())
