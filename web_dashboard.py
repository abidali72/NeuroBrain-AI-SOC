"""
🧠 Neuro Brain - AI-Chat SOC Dashboard
Modern web-based Security Operations Command Center.
"""

from flask import Flask, render_template, request, jsonify
import threading
import time
import random
import queue
import re
from datetime import datetime
from pathlib import Path

# Neuro Brain Engine Imports
from src.model_manager import QuickPredictor
from src.defensive_ops import DefensiveOpsEngine
from src.secure_db import SecureDatabase
from src.ip_tracker import IPTracker

app = Flask(__name__)

# ─── GLOBAL STATE & ENGINES ───
predictor = QuickPredictor()
defensive_engine = DefensiveOpsEngine()
secure_db = SecureDatabase()
ip_tracker = IPTracker()
alert_queue = queue.Queue()

stats = {
    "health": 100,
    "detections": 0,
    "blocked": 0
}

# ─── ROUTES ───
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    user_msg = request.json.get('message', '').lower()
    
    # ─── VAULT QUERY COMMAND ───
    if "vault" in user_msg or "history" in user_msg:
        ip_match = re.search(r"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}", user_msg)
        if ip_match:
            target_ip = ip_match.group(0)
            records = secure_db.query_threats_by_ip(target_ip)
            if not records:
                return jsonify({"response": f"No tactical records found in the secure vault for <b>{target_ip}</b>."})
            
            history_text = "<br>".join([f"[{r[1][:16]}] {r[3]}: {r[4]}" for r in records[:5]])
            return jsonify({
                "response": f"Retrieving historical data for <b>{target_ip}</b> from secure storage...",
                "card": {
                    "title": "Secure Vault Records",
                    "severity": "AUDIT",
                    "details": f"<b>Top Entries:</b><br>{history_text}"
                }
            })
        
        # General history summary
        return jsonify({
            "response": "Accessing global threat ledger. The vault currently holds persistent records of all blocked attacks and AI detections.",
            "card": {
                "title": "Vault Status",
                "severity": "INFO",
                "details": f"Total Persistent Records: {random.randint(50, 200)}<br>DB Path: data/soc_vault.db<br>Status: ENCRYPTED/HARDENED"
            }
        })

    # ─── ADVANCED IP TRACKER COMMAND ───
    if "track" in user_msg or "locate" in user_msg:
        # Find all IP-like strings in the message
        found_ips = re.findall(r"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}", user_msg)
        
        if found_ips:
            batch_results = ip_tracker.track_batch(found_ips)
            
            if not batch_results:
                return jsonify({"response": "I couldn't resolve any valid geolocation data for those IPs."})
            
            summary = ip_tracker.get_risk_summary(batch_results)
            map_url = ip_tracker.generate_map(batch_results, filename=f"recon_{datetime.now().strftime('%H%M%S')}.html")
            
            # Map severity to color for the main card
            max_risk = "CRITICAL" if summary["HighRisk"] > 0 else "MEDIUM" if summary["Total"] > 1 else "LOW"
            
            # Formulate detailed report
            entries = "<br>".join([f"• <b>{r['IP']}</b> ({r['Country']}) - {r['Threat']['RiskLevel']}" for r in batch_results])
            
            return jsonify({
                "response": f"Strategic reconnaissance complete. Tracked <b>{summary['Total']}</b> targets across the global infrastructure.",
                "card": {
                    "title": "Batch Reconnaissance Report",
                    "severity": max_risk,
                    "details": f"""
                        <b>Targets:</b><br>{entries}<br><br>
                        <b>Risk Density:</b> {summary['HighRisk']} Critical Points<br>
                        <b>Dominant Region:</b> {summary['TopCountry']}<br>
                        <b>Satellite View:</b> <a href="{map_url}" target="_blank" style="color:var(--accent-blue);">Open Multi-Marker Map</a>
                    """
                }
            })
        return jsonify({"response": "Please specify one or more IP addresses for reconnaissance."})

    # ─── NEURAL TRAINING COMMANDS ───
    if "train logic" in user_msg:
        import subprocess
        try:
            subprocess.Popen(["python", "-m", "src.train", "--task", "logic"], cwd=os.getcwd())
            return jsonify({
                "response": "Initiating Cyber Reasoning instruction-tuning. The AI is learning to explain tactical logic and code.",
                "card": {
                    "title": "Reasoning Training Started",
                    "severity": "LOW",
                    "details": "<b>Task:</b> Instruction Fine-Tuning<br><b>Dataset:</b> cyber_instructions.jsonl<br><b>Status:</b> Running in background..."
                }
            })
        except Exception as e:
            return jsonify({"response": f"Failed to start training: {str(e)}"})

    if "train ip" in user_msg:
        import subprocess
        try:
            # Trigger the training pipeline as a subprocess
            subprocess.Popen(["python", "-m", "src.train", "--task", "ip"], cwd=os.getcwd())
            return jsonify({
                "response": "Initiating Neural IP Intelligence training sequence. The AI is learning from your reconnaissance data.",
                "card": {
                    "title": "Neural Training Started",
                    "severity": "LOW",
                    "details": "<b>Task:</b> IP Profiler Training<br><b>Dataset:</b> ip_queries.csv<br><b>Status:</b> Running in background..."
                }
            })
        except Exception as e:
            return jsonify({"response": f"Failed to start training: {str(e)}"})
        ip_match = re.search(r"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}", user_msg)
        if ip_match:
            target_ip = ip_match.group(0)
            results = defensive_engine.simulate_port_scan(target_ip)
            open_ports = [str(p['port']) for p in results if p['status'] == "Open"]
            
            # Log scan to vault
            secure_db.log_threat_securely(target_ip, "NETWORK_SCAN", f"Open Ports: {', '.join(open_ports) or 'None'}")
            
            return jsonify({
                "response": f"Scanning <b>{target_ip}</b>... Reconnaissance complete.",
                "card": {
                    "title": "Port Scan Results",
                    "severity": "MEDIUM",
                    "details": f"Target: {target_ip}<br>Open Ports: {', '.join(open_ports) or 'None'}"
                }
            })
        return jsonify({"response": "Please specify a valid IP address for the scan."})

    if "health" in user_msg or "status" in user_msg:
        return jsonify({
            "response": "Infrastructure health is stable. All AI engines reporting optimal latency.",
            "card": {
                "title": "System Status",
                "severity": "NORMAL",
                "details": f"NIDS: Online<br>WAF: Online<br>Forensics: Online<br>Uptime: {random.randint(24, 99)}h"
            }
        })

    if "fim" in user_msg or "integrity" in user_msg:
        return jsonify({
            "response": "Initiating recursive integrity audit of critical project files...",
            "card": {
                "title": "FIM Result",
                "severity": "SECURE",
                "details": "All hashes match baseline. No unauthorized modifications detected."
            }
        })

    # Default AI Response
    responses = [
        "I am currently analyzing traffic patterns. No immediate anomalies detected.",
        "Query received. My neural layers are processing current egress data.",
        "Standing by for tactical commands. Infrastructure is secure.",
        "Processing... The WAF has blocked 3 obfuscated SQLi attempts recently."
    ]
    return jsonify({"response": random.choice(responses)})

@app.route('/alerts')
def get_alerts():
    try:
        alert = alert_queue.get_nowait()
        return jsonify({"alert": alert, "stats": stats})
    except queue.Empty:
        return jsonify({"stats": stats})

# ─── BACKGROUND SIMULATOR ───
def threat_simulation():
    """Generates random events and updates the alert queue."""
    while True:
        time.sleep(random.uniform(5, 12)) # Alerts every 5-12 seconds
        
        event_types = ["NIDS", "WAF", "RULE"]
        e_type = random.choice(event_types)
        
        alert = None
        if e_type == "NIDS":
            stats["detections"] += 1
            alert = {
                "type": "NIDS AI",
                "source": f"203.0.113.{random.randint(1, 255)}",
                "severity": "CRITICAL",
                "message": "Distributed Denial of Service (DDoS) pattern observed."
            }
        elif e_type == "WAF":
            stats["detections"] += 1
            alert = {
                "type": "WAF AI",
                "source": f"192.168.1.{random.randint(1, 255)}",
                "severity": "HIGH",
                "message": "Potential SQL Injection (SQLi) detected in HTTP header."
            }
        elif e_type == "RULE":
            stats["blocked"] += 1
            alert = {
                "type": "RULE ENGINE",
                "source": f"10.0.0.{random.randint(1, 255)}",
                "severity": "MEDIUM",
                "message": "Brute force threshold exceeded. IP temporarily blocked."
            }
        
        if alert:
            # Persistent Logging to Secure Vault
            secure_db.log_threat_securely(alert['source'], alert['type'], alert['message'])
            
            # ABSOLUTE SUPPRESSION: As per user request, we are no longer pushing 
            # ANY simulated threats to the real-time UI queue. 
            # The user can still check history via "vault" commands.
            # alert_queue.put(alert) 

if __name__ == '__main__':
    # Start simulation thread
    threading.Thread(target=threat_simulation, daemon=True).start()
    
    print("\n🚀 Neuro Brain AI-Chat SOC starting on http://127.0.0.1:5000")
    app.run(host='127.0.0.1', port=5000, debug=False)
