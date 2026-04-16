"""
Ham Radio Phone Patch Daemon
============================
Listens on a virtual audio cable (from SDRAngel RX),
detects any sustained tone > 1200 Hz for >= 3 seconds.
- If idle: Places a SIP call.
- If active: Hangs up the SIP call.

Requirements:
    pip install sounddevice numpy scipy

External:
    - VB-Audio Virtual Cable
    - MicroSIP
    - SDRAngel for RX/TX
"""

import tkinter as tk
from tkinter import font as tkfont
import threading
import queue
import time
import subprocess
import os
import collections
import math

import numpy as np
import sounddevice as sd

# ─────────────────────────────────────────────
#  USER CONFIGURATION  ← edit these
# ─────────────────────────────────────────────
SIP_DIAL_URI      = "1234567890"     # Number to dial via MicroSIP
TONE_MIN_HZ       = 1200          # minimum trigger frequency (Hz)
TONE_HOLD_SECS    = 3.0           # how long tone must be present
SILENCE_HANG_SECS = 5.0           # silence → auto hang-up
SQUELCH_DB        = -75.0         # MINIMUM volume (dBFS) to look for a tone
SAMPLE_RATE       = 48000         # must match SDRAngel audio output
BLOCK_SIZE        = 2048          # FFT block size

RX_DEVICE_EXACT    = "CABLE Output"   
RX_DEVICE_FALLBACK = "CABLE"          

# NOTE: Edit this path! Replace YOUR_USERNAME with your actual Windows username.
MICROSIP_EXE = r"C:\Users\YOUR_USERNAME\AppData\Local\MicroSIP\microsip.exe"
# ─────────────────────────────────────────────

# ── colour palette ───────────────────────────
C_BG      = "#0a0e14"
C_PANEL   = "#111820"
C_BORDER  = "#1e2d3d"
C_ACCENT  = "#00d4ff"
C_GREEN   = "#39ff14"
C_AMBER   = "#ffb800"
C_RED     = "#ff3c3c"
C_TEXT    = "#c8d8e8"
C_DIMTEXT = "#4a6070"

FFT_BINS     = 128
LOG_LINES    = 8
NOISE_SMOOTH = 30


# ══════════════════════════════════════════════
#  AUDIO + TONE DETECTION ENGINE
# ══════════════════════════════════════════════
class ToneEngine:
    def __init__(self, ui_queue: queue.Queue):
        self.ui_q          = ui_queue         
        self.audio_q       = queue.Queue()    
        
        self.running       = False
        self._stream       = None
        self._tone_start   = None
        self._tone_last_seen = 0.0            
        self._tone_locked  = False   # Prevents re-triggering if PTT is held
        self._last_voice   = time.time()
        self.call_active   = False
        self.device_name   = "—"

        self.fft_data      = np.zeros(FFT_BINS)
        self.peak_freq     = 0.0
        self.tone_power    = 0.0
        self.tone_progress = 0.0   
        self.countdown_sec = 0     

        self.noise_floor_db  = -80.0
        self.signal_db       = -80.0
        self._rms_history    = collections.deque(maxlen=NOISE_SMOOTH)

    def _find_device(self):
        devs = sd.query_devices()
        for i, d in enumerate(devs):
            if (d['name'].lower() == RX_DEVICE_EXACT.lower()
                    and d['max_input_channels'] > 0):
                self.device_name = d['name']
                return i
        for i, d in enumerate(devs):
            if (RX_DEVICE_FALLBACK.lower() in d['name'].lower()
                    and d['max_input_channels'] > 0):
                self.device_name = d['name']
                return i
        self.device_name = "default"
        return None

    def _callback(self, indata, frames, time_info, status):
        self.audio_q.put(("audio", indata[:, 0].copy()))

    def start(self):
        self.running = True
        dev = self._find_device()
        self._stream = sd.InputStream(
            device=dev,
            channels=1,
            samplerate=SAMPLE_RATE,
            blocksize=BLOCK_SIZE,
            dtype='float32',
            callback=self._callback,
        )
        self._stream.start()
        self.ui_q.put(("event", f"INPUT_DEVICE:{self.device_name}"))
        threading.Thread(target=self._process_loop, daemon=True).start()

    def stop(self):
        self.running = False
        if self._stream:
            self._stream.stop()
            self._stream.close()

    def _process_loop(self):
        while self.running:
            try:
                tag, data = self.audio_q.get(timeout=0.1)
            except queue.Empty:
                if self.call_active and (time.time() - self._last_voice) > SILENCE_HANG_SECS:
                    self._hangup(reason="silence timeout")
                continue

            if tag != "audio":
                continue

            rms = float(np.sqrt(np.mean(data ** 2)))
            rms = max(rms, 1e-9)
            rms_db = 20 * math.log10(rms)
            self._rms_history.append(rms_db)
            self.signal_db = rms_db

            if len(self._rms_history) >= 4:
                sorted_h = sorted(self._rms_history)
                cutoff = max(1, int(len(sorted_h) * 0.6))
                self.noise_floor_db = float(np.mean(sorted_h[:cutoff]))

            win   = data * np.hanning(len(data))
            spec  = np.abs(np.fft.rfft(win, n=BLOCK_SIZE * 2))
            freqs = np.fft.rfftfreq(BLOCK_SIZE * 2, 1 / SAMPLE_RATE)

            edges   = np.logspace(np.log10(20), np.log10(SAMPLE_RATE / 2), FFT_BINS + 1)
            display = np.zeros(FFT_BINS)
            for i in range(FFT_BINS):
                lo = np.searchsorted(freqs, edges[i])
                hi = np.searchsorted(freqs, edges[i + 1])
                if hi > lo:
                    display[i] = np.max(spec[lo:hi])
            mx = display.max()
            if mx > 0:
                display /= mx
            self.fft_data = display

            self.peak_freq = freqs[np.argmax(spec)]

            mask            = freqs >= TONE_MIN_HZ
            self.tone_power = float(spec[mask].max()) if mask.any() else 0.0
            total_power     = float(spec.max()) if spec.max() > 0 else 1.0
            ratio           = self.tone_power / total_power
            POWER_THRESH    = 0.15

            tone_present = (
                ratio > POWER_THRESH 
                and self.peak_freq >= TONE_MIN_HZ 
                and self.signal_db > SQUELCH_DB
            )

            if tone_present:
                self._last_voice = time.time()
                self._tone_last_seen = time.time()  
                
                # If we've already triggered an action, wait until the tone goes away
                if self._tone_locked:
                    pass
                else:
                    if self._tone_start is None:
                        self._tone_start = time.time()
                        self.ui_q.put(("event", "TONE_START"))
                        
                    elapsed = time.time() - self._tone_start
                    self.tone_progress = min(elapsed / TONE_HOLD_SECS, 1.0)
                    remaining = max(0.0, TONE_HOLD_SECS - elapsed)
                    self.countdown_sec = math.ceil(remaining)
                    
                    if elapsed >= TONE_HOLD_SECS:
                        self._tone_locked = True
                        self.countdown_sec = 0
                        
                        if self.call_active:
                            self._hangup(reason="3-second tone command")
                        else:
                            self._dial()
            else:
                if time.time() - self._tone_last_seen > 1.0:
                    if self._tone_start is not None or self._tone_locked:
                        if not self._tone_locked:
                            self.ui_q.put(("event", "TONE_LOST"))
                        
                        self._tone_start   = None
                        self._tone_locked  = False
                        self.tone_progress = 0.0
                        self.countdown_sec = 0

    def _dial(self):
        self.ui_q.put(("event", "DIALING"))
        if os.path.isfile(MICROSIP_EXE):
            subprocess.Popen([MICROSIP_EXE, SIP_DIAL_URI])
        else:
            self.ui_q.put(("event", "APP_NOT_FOUND"))
        self.call_active = True

    def _hangup(self, reason="manual request"):
        self.ui_q.put(("event", f"HANGUP:{reason}"))
        if os.path.isfile(MICROSIP_EXE):
            subprocess.Popen([MICROSIP_EXE, "/hangupall"])
        self.call_active   = False
        self.tone_progress = 0.0
        self.countdown_sec = 0


# ══════════════════════════════════════════════
#  GUI
# ══════════════════════════════════════════════
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Ham Radio Phone Patch")
        self.configure(bg=C_BG)
        self.resizable(False, False)
        
        self._q          = queue.Queue()
        self._engine     = ToneEngine(self._q)
        
        self._call_state = "IDLE"
        self._blink      = False
        self._overlay_visible = False

        self._build_ui()
        self._engine.start()
        self._log("System started — listening for tone")
        self.after(50, self._tick)
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    def _build_ui(self):
        PAD = 14

        try:
            title_font = tkfont.Font(family="Courier New", size=16, weight="bold")
            sub_font   = tkfont.Font(family="Courier New", size=8)
            mono_font  = tkfont.Font(family="Courier New", size=9)
            label_font = tkfont.Font(family="Courier New", size=8,  weight="bold")
            big_font   = tkfont.Font(family="Courier New", size=20, weight="bold")
            log_font   = tkfont.Font(family="Courier New", size=8)
            count_font = tkfont.Font(family="Courier New", size=64, weight="bold")
            count_sub  = tkfont.Font(family="Courier New", size=11)
        except Exception:
            title_font = sub_font = mono_font = label_font = \
            big_font = log_font = count_font = count_sub = None

        hdr = tk.Frame(self, bg=C_BG)
        hdr.pack(fill="x", padx=PAD, pady=(PAD, 0))
        tk.Label(hdr, text="◈ PHONE PATCH CONTROLLER", bg=C_BG,
                 fg=C_ACCENT, font=title_font).pack(side="left")
        tk.Label(hdr, text="HAM RADIO EMERGENCY DEMO", bg=C_BG,
                 fg=C_DIMTEXT, font=sub_font).pack(side="right", anchor="s", pady=4)

        self._sep(PAD)

        row1 = tk.Frame(self, bg=C_BG)
        row1.pack(fill="x", padx=PAD)

        spec_wrap = self._panel(row1, "RX SPECTRUM", width=480, height=130)
        spec_wrap.pack(side="left", padx=(0, PAD // 2))
        self._spec_canvas = tk.Canvas(spec_wrap, width=460, height=100,
                                      bg=C_PANEL, highlightthickness=0)
        self._spec_canvas.pack(padx=6, pady=(0, 6))

        info_col = tk.Frame(row1, bg=C_BG)
        info_col.pack(side="left", fill="y")

        pf = self._panel(info_col, "PEAK FREQ", width=150, height=56)
        pf.pack(pady=(0, PAD // 2))
        self._peak_var = tk.StringVar(value="--- Hz")
        tk.Label(pf, textvariable=self._peak_var, bg=C_PANEL,
                 fg=C_ACCENT, font=big_font).pack(pady=2)

        th = self._panel(info_col, f"THRESHOLD  {TONE_MIN_HZ} Hz", width=150, height=56)
        th.pack()
        self._thresh_var = tk.StringVar(value="BELOW")
        self._thresh_lbl = tk.Label(th, textvariable=self._thresh_var,
                                    bg=C_PANEL, fg=C_DIMTEXT, font=big_font)
        self._thresh_lbl.pack(pady=2)

        self._sep(PAD)

        nf_wrap = self._panel(self, "NOISE FLOOR  /  SIGNAL LEVEL", width=660, height=68)
        nf_wrap.pack(padx=PAD)

        nf_inner = tk.Frame(nf_wrap, bg=C_PANEL)
        nf_inner.pack(fill="x", padx=6, pady=(2, 4))

        self._dev_var = tk.StringVar(value="input device: detecting…")
        tk.Label(nf_inner, textvariable=self._dev_var, bg=C_PANEL,
                 fg=C_DIMTEXT, font=log_font).pack(anchor="w")

        meter_row = tk.Frame(nf_inner, bg=C_PANEL)
        meter_row.pack(fill="x", pady=(2, 0))
        tk.Label(meter_row, text="-80", bg=C_PANEL, fg=C_DIMTEXT,
                 font=log_font, width=4).pack(side="left")
        self._noise_canvas = tk.Canvas(meter_row, width=570, height=22,
                                       bg=C_BORDER, highlightthickness=0)
        self._noise_canvas.pack(side="left", padx=4)
        tk.Label(meter_row, text="0 dBFS", bg=C_PANEL, fg=C_DIMTEXT,
                 font=log_font).pack(side="left")

        self._sep(PAD)

        row3 = tk.Frame(self, bg=C_BG)
        row3.pack(fill="x", padx=PAD)

        tone_wrap = self._panel(row3, f"TONE HOLD  (need {TONE_HOLD_SECS:.0f}s)",
                                width=310, height=72)
        tone_wrap.pack(side="left", padx=(0, PAD // 2))
        self._prog_canvas = tk.Canvas(tone_wrap, width=290, height=32,
                                      bg=C_PANEL, highlightthickness=0)
        self._prog_canvas.pack(padx=6, pady=(0, 4))
        self._prog_time_var = tk.StringVar(value="0.0 s")
        tk.Label(tone_wrap, textvariable=self._prog_time_var, bg=C_PANEL,
                 fg=C_DIMTEXT, font=label_font).pack()

        cs = self._panel(row3, "CALL STATUS", width=318, height=72)
        cs.pack(side="left")
        self._status_var = tk.StringVar(value="IDLE")
        self._status_lbl = tk.Label(cs, textvariable=self._status_var,
                                    bg=C_PANEL, fg=C_DIMTEXT, font=big_font)
        self._status_lbl.pack(pady=2)

        self._sep(PAD)

        cfg_wrap = self._panel(self, "SIP TARGET", width=660, height=40)
        cfg_wrap.pack(padx=PAD)
        uri_frame = tk.Frame(cfg_wrap, bg=C_PANEL)
        uri_frame.pack(fill="x", padx=8, pady=4)
        tk.Label(uri_frame, text="URI:", bg=C_PANEL,
                 fg=C_DIMTEXT, font=label_font).pack(side="left")
        self._uri_var = tk.StringVar(value=SIP_DIAL_URI)
        tk.Entry(uri_frame, textvariable=self._uri_var,
                 bg=C_BORDER, fg=C_TEXT, insertbackground=C_ACCENT,
                 relief="flat", font=mono_font, width=52).pack(side="left", padx=6)
        def _update_uri(*_):
            global SIP_DIAL_URI
            SIP_DIAL_URI = self._uri_var.get()
        self._uri_var.trace_add("write", _update_uri)

        self._sep(PAD)

        log_wrap = self._panel(self, "ACTIVITY LOG", width=660, height=110)
        log_wrap.pack(padx=PAD, pady=(0, PAD))
        self._log_text = tk.Text(log_wrap, width=78, height=LOG_LINES,
                                 bg=C_PANEL, fg=C_DIMTEXT, relief="flat",
                                 state="disabled", font=log_font,
                                 wrap="none", cursor="arrow")
        self._log_text.pack(padx=6, pady=(0, 6))
        self._log_text.tag_config("accent", foreground=C_ACCENT)
        self._log_text.tag_config("green",  foreground=C_GREEN)
        self._log_text.tag_config("amber",  foreground=C_AMBER)
        self._log_text.tag_config("red",    foreground=C_RED)

        ctrl = tk.Frame(self, bg=C_BG)
        ctrl.pack(padx=PAD, pady=(0, PAD))
        self._btn_dial = self._btn(ctrl, "▶  MANUAL DIAL", self._manual_dial, C_GREEN)
        self._btn_dial.pack(side="left", padx=(0, 8))
        self._btn_hang = self._btn(ctrl, "■  HANG UP", self._manual_hangup, C_RED)
        self._btn_hang.pack(side="left")

        self._overlay = tk.Frame(self, bg="#000000")

        self._count_sub_lbl = tk.Label(
            self._overlay, text="",
            bg="#000000", fg=C_DIMTEXT, font=count_sub)
        self._count_sub_lbl.place(relx=0.5, rely=0.30, anchor="center")

        self._count_lbl = tk.Label(
            self._overlay, text="",
            bg="#000000", fg=C_AMBER, font=count_font)
        self._count_lbl.place(relx=0.5, rely=0.48, anchor="center")

        self._count_uri_lbl = tk.Label(
            self._overlay, text=SIP_DIAL_URI,
            bg="#000000", fg=C_ACCENT, font=count_sub)
        self._count_uri_lbl.place(relx=0.5, rely=0.68, anchor="center")

        self._overlay_visible = False

    def _panel(self, parent, title, width=200, height=60):
        outer = tk.Frame(parent, bg=C_BORDER, width=width, height=height)
        outer.pack_propagate(False)
        inner = tk.Frame(outer, bg=C_PANEL)
        inner.place(x=1, y=1, width=width - 2, height=height - 2)
        try:
            lf = tkfont.Font(family="Courier New", size=7, weight="bold")
        except Exception:
            lf = None
        tk.Label(inner, text=f" {title} ", bg=C_BORDER,
                 fg=C_DIMTEXT, font=lf).place(x=6, y=-1)
        content = tk.Frame(inner, bg=C_PANEL)
        content.place(x=0, y=14, width=width - 2, height=height - 16)
        return content

    def _sep(self, pad):
        tk.Frame(self, bg=C_BORDER, height=1).pack(fill="x", padx=pad, pady=pad // 2)

    def _btn(self, parent, text, cmd, color):
        b = tk.Button(parent, text=text, command=cmd,
                      bg=C_PANEL, fg=color, activebackground=C_BORDER,
                      activeforeground=color, relief="flat",
                      padx=16, pady=6, cursor="hand2",
                      highlightthickness=1, highlightbackground=color)
        b.bind("<Enter>", lambda e: b.config(bg=C_BORDER))
        b.bind("<Leave>", lambda e: b.config(bg=C_PANEL))
        return b

    def _log(self, msg, tag=None):
        ts = time.strftime("%H:%M:%S")
        self._log_text.config(state="normal")
        self._log_text.insert("end", f"[{ts}]  {msg}\n", tag or "")
        lines = int(self._log_text.index("end-1c").split(".")[0])
        if lines > LOG_LINES:
            self._log_text.delete("1.0", f"{lines - LOG_LINES}.0")
        self._log_text.see("end")
        self._log_text.config(state="disabled")

    def _show_overlay(self, countdown_int, is_hanging_up=False):
        self._count_uri_lbl.config(text=SIP_DIAL_URI)
        
        if countdown_int > 0:
            self._count_lbl.config(text=str(countdown_int), fg=C_RED if is_hanging_up else C_AMBER)
            sub_txt = "TONE DETECTED — DISCONNECTING IN" if is_hanging_up else "TONE DETECTED — DIALING IN"
            self._count_sub_lbl.config(text=sub_txt)
        else:
            self._count_lbl.config(text="■" if is_hanging_up else "▶", fg=C_RED if is_hanging_up else C_GREEN)
            sub_txt = "HANGING UP NOW" if is_hanging_up else "DIALING NOW"
            self._count_sub_lbl.config(text=sub_txt)

        if not self._overlay_visible:
            self.update_idletasks()
            self._overlay.place(x=0, y=0,
                                width=self.winfo_width(),
                                height=self.winfo_height())
            self._overlay.lift()
            self._overlay_visible = True

    def _hide_overlay(self):
        if self._overlay_visible:
            self._overlay.place_forget()
            self._overlay_visible = False

    def _tick(self):
        try:
            while True:
                tag, val = self._q.get_nowait()
                if tag == "event":
                    self._handle_event(val)
        except queue.Empty:
            pass

        eng = self._engine

        self._draw_spectrum(eng.fft_data, eng.peak_freq)
        self._draw_progress(eng.tone_progress)
        self._draw_noise_meter(eng.noise_floor_db, eng.signal_db)
        self._prog_time_var.set(f"{eng.tone_progress * TONE_HOLD_SECS:.1f} s")
        self._peak_var.set(f"{eng.peak_freq:.0f} Hz")

        above = eng.peak_freq >= TONE_MIN_HZ and eng.tone_power > 0 and eng.signal_db > SQUELCH_DB
        self._thresh_var.set("ABOVE" if above else "BELOW")
        self._thresh_lbl.config(fg=C_GREEN if above else C_DIMTEXT)

        # Show overlay while counting down, but hide it the moment it locks to prevent it getting stuck on screen
        if eng.tone_progress > 0 and not eng._tone_locked:
            self._show_overlay(eng.countdown_sec, is_hanging_up=eng.call_active)
        else:
            self._hide_overlay()

        if self._call_state == "DIALING":
            self._blink = not self._blink
            self._status_lbl.config(fg=C_AMBER if self._blink else C_DIMTEXT)

        self.after(50, self._tick)

    def _handle_event(self, val):
        if val == "DIALING":
            self._call_state = "DIALING"
            self._status_var.set("DIALING")
            self._status_lbl.config(fg=C_AMBER)
            self._log(f"Tone trigger! Dialing {SIP_DIAL_URI}", "amber")
        elif val.startswith("HANGUP:"):
            reason = val.split(":", 1)[1]
            self._call_state = "IDLE"
            self._status_var.set("IDLE")
            self._status_lbl.config(fg=C_DIMTEXT)
            self._hide_overlay()
            self._log(f"Call ended — {reason}", "accent")
        elif val == "TONE_START":
            self._log(f"Tone detected — counting down…", "green")
        elif val == "TONE_LOST":
            self._log("Tone lost before trigger — reset", "accent")
            self._hide_overlay()
        elif val == "APP_NOT_FOUND":
            self._log(f"WARNING: microsip.exe not found at {MICROSIP_EXE}", "red")
            self._log("Edit MICROSIP_EXE path at top of script", "red")
        elif val.startswith("INPUT_DEVICE:"):
            name = val.split(":", 1)[1]
            self._dev_var.set(f"input device: {name}")
            self._log(f"Audio input: {name}", "accent")

    def _draw_spectrum(self, data, peak_freq):
        c = self._spec_canvas
        c.delete("all")
        W, H = 460, 100
        n    = len(data)
        bw   = W / n
        lo   = np.log10(20)
        hi   = np.log10(SAMPLE_RATE / 2)

        tx = int(((np.log10(max(TONE_MIN_HZ, 21)) - lo) / (hi - lo)) * W)

        for gf in [100, 500, 1000, 2000, 5000, 10000, 20000]:
            gx = int(((np.log10(gf) - lo) / (hi - lo)) * W)
            c.create_line(gx, 0, gx, H, fill=C_BORDER, width=1)

        for i, v in enumerate(data):
            x0 = int(i * bw)
            x1 = max(x0 + 1, int((i + 1) * bw) - 1)
            bh = int(v * H)
            freq_lo = 10 ** (lo + (i / n) * (hi - lo))
            col = C_ACCENT if freq_lo < TONE_MIN_HZ else C_GREEN
            if bh > 0:
                c.create_rectangle(x0, H - bh, x1, H, fill=col, outline="")

        c.create_line(tx, 0, tx, H, fill=C_RED, width=2, dash=(4, 3))
        c.create_text(tx + 3, 4, text=f"{TONE_MIN_HZ}Hz", fill=C_RED,
                      anchor="nw", font=("Courier New", 7))

        for gf, lbl in [(500, "500"), (1000, "1k"), (2000, "2k"),
                        (5000, "5k"), (10000, "10k")]:
            gx = int(((np.log10(gf) - lo) / (hi - lo)) * W)
            c.create_text(gx, H - 2, text=lbl, fill=C_DIMTEXT,
                          anchor="s", font=("Courier New", 7))

    def _draw_noise_meter(self, noise_db, signal_db):
        c = self._noise_canvas
        c.delete("all")
        W, H   = 570, 22
        DB_MIN = -80.0
        DB_MAX =   0.0

        def db_x(db):
            return int(((db - DB_MIN) / (DB_MAX - DB_MIN)) * W)

        for z_lo, z_hi, col in [
            (DB_MIN, -40, "#0d2b1a"),   
            (-40,    -20, "#2b2000"),   
            (-20,  DB_MAX,"#2b0000"),   
        ]:
            c.create_rectangle(db_x(z_lo), 0, db_x(z_hi), H, fill=col, outline="")

        nf_x = db_x(max(DB_MIN, noise_db))
        if nf_x > 0:
            c.create_rectangle(0, 5, nf_x, H - 5, fill=C_DIMTEXT, outline="")

        sig_x = db_x(max(DB_MIN, signal_db))
        if sig_x > 0:
            bar_col = C_RED if signal_db > -20 else (C_AMBER if signal_db > -40 else C_GREEN)
            c.create_rectangle(0, 7, sig_x, H - 7, fill=bar_col, outline="")

        c.create_line(nf_x, 0, nf_x, H, fill="#666666", width=1, dash=(2, 2))
        lbl_x = max(nf_x - 2, 2)
        c.create_text(lbl_x, H // 2, text=f"NF {noise_db:.0f}dB",
                      fill="#888888", anchor="e", font=("Courier New", 7))

        c.create_text(min(sig_x + 3, W - 2), H // 2,
                      text=f"SIG {signal_db:.0f}dB",
                      fill=C_TEXT, anchor="w", font=("Courier New", 7))

        for db in range(-80, 1, 10):
            tx = db_x(db)
            c.create_line(tx, H - 4, tx, H, fill=C_DIMTEXT, width=1)

    def _draw_progress(self, progress):
        c = self._prog_canvas
        c.delete("all")
        W, H = 290, 32
        c.create_rectangle(0, 8, W, H - 8, fill=C_BORDER, outline="")
        if progress > 0:
            pw  = int(progress * W)
            col = C_GREEN if progress >= 1.0 else C_AMBER
            c.create_rectangle(0, 8, pw, H - 8, fill=col, outline="")
            seg = max(1, pw // 6)
            for sx in range(0, pw, seg):
                c.create_rectangle(sx, 8, sx + max(1, seg - 2), H - 8,
                                   fill=col, outline="", stipple="gray50")
        for i in range(1, int(TONE_HOLD_SECS)):
            tx = int((i / TONE_HOLD_SECS) * W)
            c.create_line(tx, 6, tx, H - 6, fill=C_DIMTEXT, width=1)

    def _manual_dial(self):
        self._engine._dial()

    def _manual_hangup(self):
        self._engine._hangup(reason="manual GUI button")

    def _on_close(self):
        self._engine.stop()
        self.destroy()

if __name__ == "__main__":
    app = App()
    app.mainloop()