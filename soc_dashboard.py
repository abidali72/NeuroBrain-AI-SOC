"""
🛡️ Neuro Brain — Live AI SOC Dashboard
Real-time Security Operations Center interface for visualizing 
AI detections across Network, Application, and Endpoint layers.

Layout:
• L3/L4 IDS: Monitoring Network Telemetry
• L7 WAF: Monitoring HTTP Payloads
• Endpoint: Monitoring RAM/Memory Forensics
"""

import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import threading
import time
import random
import numpy as np
from datetime import datetime
from src.model_manager import QuickPredictor
from src.defensive_ops import DefensiveOpsEngine
import argparse
import csv
import re
from pathlib import Path

# Visualization backend
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

# ═══════════════════════════════════════════════════════════════
#  DASHBOARD CONSTANTS & STYLE
# ═══════════════════════════════════════════════════════════════
BG_COLOR = "#0A0A0A"  # Deep black
FG_COLOR = "#00FF41"  # Matrix Green
ACCENT_COLOR = "#008F11"
CRITICAL_COLOR = "#FF3131"
WARN_COLOR = "#FFD700"
PANEL_BG = "#151515"
FONT_MAIN = "Consolas"
FONT_MONO = "Consolas"

class SOCDashboard:
    def __init__(self, root):
        self.root = root
        self.root.title("NEURO BRAIN — ADVANCED AI SOC")
        self.root.geometry("1100x750")
        self.root.configure(bg=BG_COLOR)
        
        self.predictor = QuickPredictor()
        self.defensive_engine = DefensiveOpsEngine()
        self.is_running = True
        
        # Statistics counters
        self.stats = {
            "Total Events": 0,
            "AI Threats": 0,
            "Rule Threats": 0,
            "Blocked": 0
        }
        
        # FIM Setup: Monitoring critical project files
        self.defensive_engine.initialize_fim(["soc_dashboard.py", "src/defensive_ops.py"])
        
        # Interactive State
        self.failed_logins_db = {} # {IP: Count}
        self.open_ports_db = []    # List of open ports for graph
        self.event_history = []    # For CSV export
        
        self._setup_ui()
        
        # Start simulation thread
        self.sim_thread = threading.Thread(target=self._threat_simulation_loop, daemon=True)
        self.sim_thread.start()
        
        # UI Refresh loop
        self.root.protocol("WM_DELETE_WINDOW", self._on_closing)
        self._update_graphs() # Initial graph draw

    def _setup_ui(self):
        # Header
        header = tk.Frame(self.root, bg=BG_COLOR, height=60)
        header.pack(fill=tk.X, padx=20, pady=10)
        
        tk.Label(header, text="🛡️ NEURO BRAIN — LIVE AI SOC CONSOLE", 
                 font=(FONT_MAIN, 18, "bold"), bg=BG_COLOR, fg=FG_COLOR).pack(side=tk.LEFT)
        
        self.timer_label = tk.Label(header, text="", font=(FONT_MAIN, 12), bg=BG_COLOR, fg=FG_COLOR)
        self.timer_label.pack(side=tk.RIGHT)
        self._update_time()

        # Top Stats Panel
        stats_frame = tk.Frame(self.root, bg=BG_COLOR)
        stats_frame.pack(fill=tk.X, padx=20, pady=5)
        
        self.stat_labels = {}
        for i, (key, value) in enumerate(self.stats.items()):
            frame = tk.Frame(stats_frame, bg=PANEL_BG, highlightbackground=ACCENT_COLOR, highlightthickness=1)
            frame.pack(side=tk.LEFT, expand=True, fill=tk.BOTH, padx=5)
            
            tk.Label(frame, text=key.upper(), font=(FONT_MAIN, 8), bg=PANEL_BG, fg=ACCENT_COLOR).pack(pady=(5,0))
            lbl = tk.Label(frame, text=str(value), font=(FONT_MAIN, 16, "bold"), bg=PANEL_BG, fg=FG_COLOR)
            lbl.pack(pady=(0,5))
            self.stat_labels[key] = lbl

        # Main Layout: 2 Columns
        main_container = tk.Frame(self.root, bg=BG_COLOR)
        main_container.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        # ─── LEFT PANEL: THREAT STATUS & BLOCKLIST ───
        left_panel = tk.Frame(main_container, bg=BG_COLOR, width=350)
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, padx=(0,10))

        # Detection Indicators
        tk.Label(left_panel, text="ENGINE STATUS", font=(FONT_MAIN, 10, "bold"), bg=BG_COLOR, fg=ACCENT_COLOR).pack(anchor="w")
        
        self.indicators = {}
        for engine, layer in [("NIDS CORE", "Network L3/L4"), ("WAF CORE", "Application L7"), ("DFIR CORE", "Endpoint Memory")]:
            frame = tk.Frame(left_panel, bg=PANEL_BG, pady=10, highlightbackground="#333", highlightthickness=1)
            frame.pack(fill=tk.X, pady=5)
            
            tk.Label(frame, text=engine, font=(FONT_MAIN, 11, "bold"), bg=PANEL_BG, fg=FG_COLOR).pack(anchor="w", padx=10)
            tk.Label(frame, text=layer, font=(FONT_MAIN, 8), bg=PANEL_BG, fg="#888").pack(anchor="w", padx=10)
            
            status_lbl = tk.Label(frame, text="SCANNING", font=(FONT_MAIN, 9, "bold"), bg="#003300", fg=FG_COLOR, padx=10)
            status_lbl.pack(side=tk.RIGHT, padx=10, pady=(0, 20)) # Place it right
            self.indicators[engine] = status_lbl

        # Active Blocklist
        tk.Label(left_panel, text="ACTIVE RESPONSE: BLOCKLIST", font=(FONT_MAIN, 10, "bold"), bg=BG_COLOR, fg=ACCENT_COLOR).pack(anchor="w", pady=(20,0))
        self.blocklist = tk.Listbox(left_panel, bg=PANEL_BG, fg=CRITICAL_COLOR, font=(FONT_MONO, 9), borderwidth=0)
        self.blocklist.pack(fill=tk.BOTH, expand=True)
        
        # ─── RIGHT PANEL: ALERT FEED & SIDEBAR ───
        right_panel = tk.Frame(main_container, bg=BG_COLOR)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10)

        # Alert Feed
        feed_label = tk.Label(right_panel, text="LIVE THREAT INTELLIGENCE FEED", font=(FONT_MAIN, 12, "bold"),
                             fg=ACCENT_COLOR, bg=BG_COLOR)
        feed_label.pack(anchor=tk.W, pady=(0, 5))

        self.feed_text = tk.Text(right_panel, bg=PANEL_BG, fg="#FFFFFF", font=(FONT_MONO, 10),
                                height=20, borderwidth=0, padx=10, pady=10)
        self.feed_text.pack(fill=tk.BOTH, expand=True)
        self.feed_text.tag_configure("time", foreground="#888888")
        
        # Interactive Controls
        ctrl_frame = tk.LabelFrame(right_panel, text="INTERACTIVE CONTROLS", font=(FONT_MAIN, 10, "bold"),
                                  fg=ACCENT_COLOR, bg=BG_COLOR, padx=10, pady=10)
        ctrl_frame.pack(fill=tk.X, pady=10)
        
        # Input: Target IP
        tk.Label(ctrl_frame, text="Target IP:", bg=BG_COLOR, fg="#AAAAAA").grid(row=0, column=0, sticky=tk.W)
        self.ip_entry = tk.Entry(ctrl_frame, bg=PANEL_BG, fg="#FFFFFF", insertbackground="white", width=15)
        self.ip_entry.insert(0, "192.168.1.10")
        self.ip_entry.grid(row=0, column=1, padx=5, pady=2)
        
        # Input: File Path
        tk.Label(ctrl_frame, text="FIM Path:", bg=BG_COLOR, fg="#AAAAAA").grid(row=1, column=0, sticky=tk.W)
        self.path_entry = tk.Entry(ctrl_frame, bg=PANEL_BG, fg="#FFFFFF", insertbackground="white", width=15)
        self.path_entry.insert(0, "soc_dashboard.py")
        self.path_entry.grid(row=1, column=1, padx=5, pady=2)
        
        btn_frame = tk.Frame(ctrl_frame, bg=BG_COLOR)
        btn_frame.grid(row=2, column=0, columnspan=2, pady=10)
        
        tk.Button(btn_frame, text="SCAN PORTS", command=self._run_manual_scan, bg=ACCENT_COLOR, fg=BG_COLOR, 
                  font=(FONT_MAIN, 9, "bold"), width=12).pack(side=tk.LEFT, padx=2)
        tk.Button(btn_frame, text="CHECK FIM", command=self._run_manual_fim, bg=ACCENT_COLOR, fg=BG_COLOR, 
                  font=(FONT_MAIN, 9, "bold"), width=12).pack(side=tk.LEFT, padx=2)
        tk.Button(btn_frame, text="EXPORT CSV", command=self._export_logs, bg="#555555", fg="#FFFFFF", 
                  font=(FONT_MAIN, 9, "bold"), width=12).pack(side=tk.LEFT, padx=2)

        # ─── BOTTOM PANEL: GRAPHS ───
        graph_frame = tk.Frame(self.root, bg=BG_COLOR, height=200)
        graph_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=10)
        
        self.fig = Figure(figsize=(10, 2.5), dpi=80, facecolor=BG_COLOR)
        self.ax1 = self.fig.add_subplot(121, facecolor=PANEL_BG)
        self.ax2 = self.fig.add_subplot(122, facecolor=PANEL_BG)
        self.fig.tight_layout()
        
        self.canvas = FigureCanvasTkAgg(self.fig, master=graph_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def _update_time(self):
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.timer_label.config(text=f"SYS_TIME: {now}")
        self.root.after(1000, self._update_time)

    def _log_event(self, message, severity="INFO"):
        """Helper to log events with specific formatting & severity tracking."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.feed_text.insert(tk.END, f"[{timestamp}] ", "time")
        
        color = FG_COLOR
        if severity == "CRITICAL": color = CRITICAL_COLOR
        elif severity == "WARN": color = WARN_COLOR
        
        tag_name = f"msg_{random.randint(0,10000)}"
        self.feed_text.tag_configure(tag_name, foreground=color)
        self.feed_text.insert(tk.END, f"{message}\n", tag_name)
        self.feed_text.see(tk.END)
        
        # Add to history for CSV
        self.event_history.append({
            "Timestamp": datetime.now().isoformat(),
            "Event Type": "SIMULATION",
            "Source/Target": "SYSTEM",
            "Severity": severity,
            "Detail": message
        })

    def _update_stats(self):
        for key, val in self.stats.items():
            self.stat_labels[key].config(text=str(val))

    def _update_graphs(self):
        """Refreshes the dynamic visualizations."""
        # Graph 1: Failed Logins per IP
        self.ax1.clear()
        if self.failed_logins_db:
            ips = list(self.failed_logins_db.keys())
            counts = list(self.failed_logins_db.values())
            self.ax1.bar(ips, counts, color=CRITICAL_COLOR)
        self.ax1.set_title("FAILED LOGINS PER IP", color=ACCENT_COLOR, fontsize=10)
        self.ax1.tick_params(axis='x', colors='#AAAAAA', labelsize=8)
        self.ax1.tick_params(axis='y', colors='#AAAAAA')
        self.ax1.set_facecolor(PANEL_BG)
        
        # Graph 2: Open Ports (Manual Scan Results)
        self.ax2.clear()
        if self.open_ports_db:
            ports = [str(p['port']) for p in self.open_ports_db if p['status'] == "Open"]
            if ports:
                self.ax2.bar(ports, [1]*len(ports), color=ACCENT_COLOR)
        self.ax2.set_title("OPEN PORTS DETECTED", color=ACCENT_COLOR, fontsize=10)
        self.ax2.tick_params(axis='x', colors='#AAAAAA', labelsize=8)
        self.ax2.set_yticks([]) # Hide Y axis
        self.ax2.set_facecolor(PANEL_BG)
        
        self.canvas.draw()
        self.root.after(5000, self._update_graphs) # Refresh graphs every 5 seconds

    def _threat_simulation_loop(self):
        """Generates random events and passes them through AI models & Rule engines."""
        time.sleep(2) # Wait for models to initialize
        self._log_event("SOC Dashboard initialized. AI & Rule Engines online.", "INFO")
        
        while self.is_running:
            event_type = random.choice(["AI_NET", "AI_WEB", "AI_MEM", "RULE_LOG", "RULE_FIM", "BENIGN"])
            self.stats["Total Events"] += 1
            
            if event_type == "BENIGN":
                msg = random.choice([
                    "Standard TCP handshake from 192.168.1.15",
                    "GET /index.html HTTP/1.1 from 203.0.113.5",
                    "Memory page allocation for pid 4410",
                    "DNS Query: google.com from interior-node-02"
                ])
                self._log_event(msg, "INFO")
                
            elif event_type == "AI_NET":
                # AI-driven NIDS Telemetry
                features = [0, 54, 0, 0, 0, 0, 0, 0, 0, 0, 0, random.randint(400, 600)]
                res = self.predictor.predict_intrusion(features)
                if res["label"] == "Malicious":
                    self.stats["AI Threats"] += 1
                    self._log_event(f"AI ALERT [NIDS]: DDoS Trend Detected (Conf: {res['confidence']:.2%})", "CRITICAL")
                    self._trigger_indicator("NIDS CORE", "DDoS")
                    self.blocklist.insert(0, f"BLOCK_IP: 185.234.{random.randint(1,254)}.{random.randint(1,254)}")
                    self.stats["Blocked"] += 1

            elif event_type == "AI_WEB":
                # AI-driven WAF Query
                query = random.choice(["id=1' OR 1=1--", "<script>alert(1)</script>", "user=root"])
                res = self.predictor.predict_waf(query)
                if res["label"] in ["SQLi", "XSS"]:
                    self.stats["AI Threats"] += 1
                    self._log_event(f"AI ALERT [WAF]: {res['label']} Payload Detected", "WARN")
                    self._trigger_indicator("WAF CORE", "FILTER")

            elif event_type == "AI_MEM":
                # AI-driven Memory Forensics
                page = np.random.randint(100, 255, (64, 64)) / 255.0 # High entropy
                res = self.predictor.predict_forensics(page)
                if res["label"] == "Malicious":
                    self.stats["AI Threats"] += 1
                    self._log_event(f"AI ALERT [DFIR]: Shellcode Signature in RAM", "CRITICAL")
                    self._trigger_indicator("DFIR CORE", "INJECT")
                    self.blocklist.insert(0, f"KILL_PID: {random.randint(1000, 9999)}")
                    self.stats["Blocked"] += 1

            elif event_type == "RULE_LOG":
                # Traditional Rule-Based Brute Force
                ip = f"192.168.1.{random.randint(100, 200)}"
                self.failed_logins_db[ip] = self.failed_logins_db.get(ip, 0) + 1
                
                mock_log = [f"Failed password for user 'admin' from {ip}"] * 6
                alerts = self.defensive_engine.audit_auth_logs(mock_log)
                if alerts:
                    self.stats["Rule Threats"] += 1
                    self._log_event(f"RULE ALERT [LOG]: Brute Force from {ip}", "WARN")
                    self.blocklist.insert(0, f"BLOCK_IP: {ip}")
                    self.stats["Blocked"] += 1
                    self.event_history.extend(self.defensive_engine.generate_event_summary(alerts))

            elif event_type == "RULE_FIM":
                # Traditional File Integrity Violation
                self.stats["Rule Threats"] += 1
                self._log_event("RULE ALERT [FIM]: Modification detected in 'soc_dashboard.py'", "CRITICAL")
                self._trigger_indicator("DFIR CORE", "FIM")

            self._update_stats()
            time.sleep(random.uniform(0.5, 1.5))

    def _trigger_indicator(self, key, text):
        orig_text = self.indicators[key].cget("text")
        self.indicators[key].config(text=text, bg=CRITICAL_COLOR, fg="white")
        self.root.after(2000, lambda: self.indicators[key].config(text="SCANNING", bg="#003300", fg=FG_COLOR))

    def _on_closing(self):
        self.is_running = False
        self.root.destroy()

    # ─── INTERACTIVE ACTIONS ───
    def _run_manual_scan(self):
        target = self.ip_entry.get()
        if not re.match(r"^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$", target):
            messagebox.showerror("Error", "Invalid IP Address format!")
            return
            
        self._log_event(f"INTERACTIVE: Initiating Port Scan on {target}...", "INFO")
        results = self.defensive_engine.simulate_port_scan(target)
        self.open_ports_db = results
        open_ports = [p['port'] for p in results if p['status'] == "Open"]
        self._log_event(f"RESULTS: {len(open_ports)} ports open on {target}", "WARN")
        self._update_graphs()

    def _run_manual_fim(self):
        target_path = self.path_entry.get()
        if not Path(target_path).exists():
            messagebox.showerror("Error", "Invalid File Path!")
            return
            
        self._log_event(f"INTERACTIVE: Checking integrity for {target_path}...", "INFO")
        # For simulation, randomly trigger a modification if it's a critical file
        if "soc_dashboard" in target_path and random.random() > 0.5:
             self._log_event(f"FIM CRITICAL: Unauthorized change in {target_path}!", "CRITICAL")
        else:
             self._log_event(f"FIM SECURE: {target_path} hash matches baseline.", "INFO")

    def _export_logs(self):
        save_path = "data/logs/soc_audit.csv"
        Path("data/logs").mkdir(parents=True, exist_ok=True)
        try:
            with open(save_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=["Timestamp", "Event Type", "Source/Target", "Severity", "Detail"])
                writer.writeheader()
                writer.writerows(self.event_history)
            messagebox.showinfo("Success", f"Logs exported to {save_path}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to export logs: {e}")

if __name__ == "__main__":
    root = tk.Tk()
    app = SOCDashboard(root)
    root.mainloop()
