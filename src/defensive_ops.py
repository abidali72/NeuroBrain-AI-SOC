"""
🛡️ Neuro Brain — Defensive Operations Engine
Traditional Rule-Based Detection & Enterprise Monitoring.

Features:
1. Log Monitor (Brute Force Detection)
2. File Integrity Monitor (FIM)
3. Network Traffic Inspector (Scapy)
4. Enterprise Analytics (Pandas + JSON Logging)
"""

import re
import hashlib
import time
import json
import os
from datetime import datetime
from pathlib import Path
import pandas as pd

# Mocking Scapy for systems without it installed
try:
    from scapy.all import sniff, IP, TCP
    HAS_SCAPY = True
except ImportError:
    HAS_SCAPY = False

class DefensiveOpsEngine:
    def __init__(self, log_dir="data/logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.alert_log = self.log_dir / "threat_alerts.json"
        self.fim_database = {} # Store initial hashes

    # ═══════════════════════════════════════════════════════════════
    #  1. LOG MONITORING (Brute Force Detection)
    # ═══════════════════════════════════════════════════════════════
    def audit_auth_logs(self, log_lines, threshold=5):
        """
        Analyzes authentication logs for multiple failed attempts from the same IP.
        """
        failed_attempts = {}
        alerts = []
        
        for line in log_lines:
            if "Failed password" in line or "AUTH_FAILURE" in line:
                # Regex to extract IP address
                ip_match = re.search(r'(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})', line)
                if ip_match:
                    ip = ip_match.group(1)
                    failed_attempts[ip] = failed_attempts.get(ip, 0) + 1
                    
                    if failed_attempts[ip] >= threshold:
                        alerts.append({
                            "timestamp": datetime.now().isoformat(),
                            "type": "BRUTE_FORCE",
                            "source_ip": ip,
                            "count": failed_attempts[ip],
                            "severity": "HIGH",
                            "message": f"Threshold exceeded: {failed_attempts[ip]} failed attempts from {ip}"
                        })
        return alerts

    # ═══════════════════════════════════════════════════════════════
    #  2. FILE INTEGRITY MONITOR (FIM)
    # ═══════════════════════════════════════════════════════════════
    def calculate_file_hash(self, file_path):
        """Computes SHA-256 hash of a file."""
        sha256 = hashlib.sha256()
        try:
            with open(file_path, "rb") as f:
                while chunk := f.read(4096):
                    sha256.update(chunk)
            return sha256.hexdigest()
        except FileNotFoundError:
            return None

    def initialize_fim(self, files_to_watch):
        """Baseline the files to watch."""
        for file in files_to_watch:
            file_hash = self.calculate_file_hash(file)
            if file_hash:
                self.fim_database[file] = file_hash
        print(f"[*] FIM: Baselined {len(self.fim_database)} critical files.")

    def check_integrity(self):
        """Checks current hashes against the baseline."""
        alerts = []
        for file, original_hash in self.fim_database.items():
            current_hash = self.calculate_file_hash(file)
            if current_hash != original_hash:
                alerts.append({
                    "timestamp": datetime.now().isoformat(),
                    "type": "FIM_VIOLATION",
                    "file": file,
                    "severity": "CRITICAL",
                    "message": f"Integrity compromised for {file}! Hash mismatch."
                })
        return alerts

    # ═══════════════════════════════════════════════════════════════
    #  3. NETWORK TRAFFIC INSPECTION (Scapy)
    # ═══════════════════════════════════════════════════════════════
    def inspect_packet(self, packet):
        """
        Behavioral anomaly detection via packet inspection.
        """
        if not HAS_SCAPY:
            return None
            
        if packet.haslayer(TCP):
            # Example: Catching potential scanning or unusual flag combinations
            if packet[TCP].flags == 0x02: # SYN only (Potential scan)
                return {
                    "type": "TRAFFIC_ANOMALY",
                    "source": packet[IP].src,
                    "dest_port": packet[TCP].dport,
                    "severity": "MEDIUM",
                    "message": "Potential Stealth SYN Scan detected"
                }
        return None

    def simulate_port_scan(self, target_ip):
        """Simulates a port scan on a target IP and returns results."""
        common_ports = [21, 22, 23, 25, 53, 80, 110, 443, 3306, 3389, 8080]
        results = []
        for port in common_ports:
            # Randomly determine if port is open for simulation
            status = "Open" if np.random.rand() > 0.7 else "Closed"
            results.append({"port": port, "status": status})
        return results

    # ═══════════════════════════════════════════════════════════════
    #  4. ENTERPRISE ANALYTICS & SIEM INTEGRATION
    # ═══════════════════════════════════════════════════════════════
    def generate_event_summary(self, alerts):
        """Converts alerts into a CSV-ready list of ornaments."""
        summary = []
        for alert in alerts:
            summary.append({
                "Timestamp": alert.get("timestamp", datetime.now().isoformat()),
                "Event Type": alert.get("type", "UNKNOWN"),
                "Source/Target": alert.get("source_ip") or alert.get("file") or alert.get("source", "SYSTEM"),
                "Severity": alert.get("severity", "INFO"),
                "Detail": alert.get("message", "N/A")
            })
        return summary
    def generate_siem_report(self, alerts):
        """
        Converts raw alerts into a Pandas DataFrame for trend analysis
        and saves as a JSON log for SIEM ingestion.
        """
        if not alerts:
            return None

        # 1. JSON Logging (Persistence)
        with open(self.alert_log, 'a' if self.alert_log.exists() else 'w') as f:
            for alert in alerts:
                f.write(json.dumps(alert) + "\n")

        # 2. Pandas Analytics
        df = pd.DataFrame(alerts)
        
        # Example: Calculate severity distribution
        severity_stats = df['severity'].value_counts().to_dict()
        print(f"\n[📊] SIEM Analytics: {severity_stats}")
        
        return df

    def trigger_webhook(self, alert):
        """Simulates automated alert notification via Webhook/SMTP."""
        print(f"[📤] WEBHOOK TRIGGERED: Sending alert for {alert['type']} to SOC group.")

# ── Self-Test Block ─────────────────────────────────────────────
if __name__ == "__main__":
    engine = DefensiveOpsEngine()
    
    # Test Log Audit
    mock_logs = [
        "2026-03-04 03:00:01 Failed password for root from 192.168.1.100 port 22",
        "2026-03-04 03:00:05 Failed password for root from 192.168.1.100 port 22",
        "2026-03-04 03:00:10 Failed password for root from 192.168.1.100 port 22",
        "2026-03-04 03:00:15 Failed password for root from 192.168.1.100 port 22",
        "2026-03-04 03:00:20 Failed password for root from 192.168.1.100 port 22",
    ]
    log_alerts = engine.audit_auth_logs(mock_logs, threshold=5)
    
    # Test FIM
    test_file = "important_data.txt"
    with open(test_file, "w") as f: f.write("Initial state")
    
    engine.initialize_fim([test_file])
    
    # Modify file to trigger FIM alert
    with open(test_file, "a") as f: f.write(" - Tampered!")
    fim_alerts = engine.check_integrity()
    
    # Unified Reporting
    all_alerts = log_alerts + fim_alerts
    if all_alerts:
        engine.generate_siem_report(all_alerts)
        for a in all_alerts:
            engine.trigger_webhook(a)
    
    # Cleanup test file
    if os.path.exists(test_file): os.remove(test_file)
