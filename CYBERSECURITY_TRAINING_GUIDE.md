# 🛡️ SOC Intelligence: Neural Network Intrusion Detection System (IDS)

This document provides a highly technical, structured breakdown of the cybersecurity concepts utilized to train the `intrusion_detector.py` Neural Network model. 

---

## CONTEXT 1: Network Intrusion Detection Systems (NIDS)

### STEP 1 — Define the Concept
*   **Definition:** A Network Intrusion Detection System (NIDS) is a passive deployment that monitors inbound and outbound network traffic traversing switches and routers, analyzing protocols and flows for suspicious activity or known threat signatures.
*   **Purpose in Security:** To provide visibility into malicious lateral movement, data exfiltration, and active exploitation attempts before they compromise critical assets within the environment.
*   **Where it is commonly used:** Deployed strategically at chokepoints (e.g., inside the DMZ, behind edge firewalls, on core switches via SPAN/Mirror ports) in enterprise networks.

### Technical Breakdown
Modern NIDS relies on two primary methodologies:
1.  **Signature-Based (Legacy):** Relies on known Indicators of Compromise (IoCs) like specific byte sequences or known malicious IP addresses (e.g., Snort, Suricata). Ineffective against Zero-Day exploits.
2.  **Anomaly-Based (Neural/Heuristic):** Establishes a baseline of "normal" behavior. Any deviation (e.g., unexpected protocols, massive data transfers at 3 AM) triggers an alert. The Neuro Brain utilizes a Deep Neural Network (DNN) to learn these high-dimensional anomalies.

### Tools & Techniques
*   **Tools:** Zeek (Bro), Suricata, Snort, Corelight.
*   **Telemetry Generation:** In our model, we simulate features like `num_failed_logins`, `wrong_fragment`, and byte transfer ratios based on standard netflow metrics (e.g., PCAP analysis).

### Defending Cloud vs. On-Premise
*   **AWS:** VPC Flow Logs analyzed by Amazon GuardDuty (which uses ML, similar to our script).
*   **Azure:** NSG Flow Logs analyzed by Azure Sentinel and Defender for Cloud.

---

## CONTEXT 2: Distributed Denial of Service (DDoS) 

### STEP 1 — Define the Concept
*   **Definition:** A malicious attempt to disrupt the normal traffic of a targeted server, service, or network by overwhelming the target or its surrounding infrastructure with a flood of internet traffic.
*   **Purpose in Security:** Attackers use DDoS to cause financial damage, distract SOC teams while a secondary attack occurs (smokescreen), or for ideological reasons (hacktivism). 
*   **Where it is commonly used:** Targeted against internet-facing web applications, DNS infrastructure, and API endpoints.

### Technical Breakdown
Our model specifically looks for **Volumetric Attacks** (e.g., UDP floods, ICMP floods, DNS Amplification). 
In the training data, a DDoS is simulated when the network telemetry shows:
*   `duration = 0` (Connections are opened and abandoned instantly).
*   `src_bytes = small` (Small spoofed requests).
*   `count = massive` (Thousands of connections per second).

### Defensive Countermeasures
*   **Architecture:** Anycast routing to absorb traffic (e.g., Cloudflare, AWS Shield).
*   **Detection:** Recognizing abnormal spikes in connection velocity. 
*   **Mitigation:** BGP Blackholing, rate-limiting at the WAF level.

### Risk Assessment (Severity: HIGH)
While rarely leading to data breaches (Confidentiality), DDoS severely impacts Availability.
*   **MITRE ATT&CK:** T1498 (Network Denial of Service)

---

## CONTEXT 3: Brute Force & Credential Stuffing

### STEP 1 — Define the Concept
*   **Definition:** An automated, trial-and-error method used by attackers to guess credentials (usernames/passwords) or cryptographic keys. 
*   **Purpose in Security:** To gain unauthorized Initial Access to a system, bypass authentication, and establish persistence.
*   **Where it is commonly used:** Targeted at external portals (RDP, SSH, VPNs, OWA, Admin panels).

### Technical Breakdown
In our model, Brute Force is identified when telemetry exhibits:
*   High `num_failed_logins` originating from a single IP or a distributed botnet.
*   Repetitive, identical `dst_bytes` sizes (the server sending the exact same "Authentication Failed" response packet).

### Tools & Techniques (Offensive)
*   **Tools:** Hydra, Medusa, Hashcat (Offline), Burp Suite Intruder.
*   **Command Line Example (Hydra):**
    ```bash
    # Attacking an SSH service with a password list
    hydra -L users.txt -P rockyou.txt ssh://10.10.10.50
    ```

### Defensive Countermeasures
*   **Authentication:** Mandatory Multi-Factor Authentication (MFA), enforcing strict password complexity, and disabling legacy protocols.
*   **Network:** Implementing dynamic banning mechanisms like Fail2Ban.
*   **Zero Trust Architecture:** Never trust an internal unauthenticated connection; verify explicitly.

### Risk Assessment (Severity: CRITICAL)
*   **MITRE ATT&CK:** T1110 (Brute Force)
*   **CVSS Note:** Success leads directly to Account Takeover (ATO) and deep lateral movement.

---

## CONTEXT 4: Privilege Escalation (Exploitation)

### STEP 1 — Define the Concept
*   **Definition:** The act of exploiting a bug, design flaw, or configuration oversight in an operating system or software application to gain elevated access to resources that are normally protected (e.g., moving from a standard user to `root` or `SYSTEM`).
*   **Purpose in Security:** Lower unprivileged access rarely allows attackers to modify registries, dump credential hashes (like LSASS), or install persistent rootkits. Total system control requires escalation.
*   **Where it is commonly used:** Post-exploitation phases on compromised servers, workstations, and cloud instances.

### Technical Breakdown
In the generated training data, Privilege Escalation traces are identified by:
*   `root_shell = 1` or `su_attempted = 1` triggered unexpectedly.
*   Anomalous `num_file_creations` associated with dropping exploit payloads.
*   Large bidirectional data transfers indicative of a reverse shell execution (`src_bytes` and `dst_bytes`).

### Tools & Techniques
*   **Windows:** Exploiting misconfigured Service Permissions (e.g., Unquoted Service Paths), Token Impersonation (PrintSpoofer), or kernel exploits (e.g., CVE-2021-36934 "SeriousSAM").
*   **Linux:** SUID binary abuses, kernel exploits (e.g., "Dirty COW"), or exploiting generous `sudo` configurations.

### Real-World Scenarios
**Scenario:** An attacker gains unprivileged web shell access via a vulnerable WordPress plugin (RCE). They upload a compiled C binary exploiting a Local Privilege Escalation (LPE) vulnerability in an outdated Linux kernel to gain `root` access.

### Defensive Countermeasures
*   **Patch Management:** Strict adherence to updating kernels and core services. 
*   **Least Privilege (NIST PR.AC-4):** Web servers should run under isolated service accounts (`www-data`) with incredibly restricted privileges.
*   **Detection Engineering:** Monitoring for process spawning anomalies (e.g., `apache2` spawning `/bin/bash` or `whoami`).

### Risk Assessment (Severity: CRITICAL)
*   **MITRE ATT&CK:** T1068 (Exploitation for Privilege Escalation)
*   **Impact:** Complete loss of confidentiality, integrity, and availability of the host.

---

## CONTEXT 5: SQL Injection (SQLi)

### STEP 1 — Define the Concept
*   **Definition:** A web security vulnerability that allows an attacker to interfere with the queries that an application makes to its database by injecting malicious SQL payloads into input fields.
*   **Purpose in Security:** Used by attackers to view data they are not normally able to retrieve (passwords, PII), modify or delete data, or in severe cases, execute administrative operations on the backend database server.
*   **Where it is commonly used:** Exploited in web applications (login forms, search bars, URL parameters) that concatenate user input directly into database queries without sanitization.

### Technical Breakdown
Our WAF AI is trained on character-level TF-IDF vectors precisely to catch SQLi strings regardless of how they are nested or encoded, breaking away from standard regex limitations.
*   **Boolean/Inferential SQLi:** `username=' OR 1=1--` forces the `WHERE` clause to evaluate to TRUE, granting unauthorized access.
*   **UNION-Based SQLi:** `id=1 UNION SELECT username, password FROM users` appends unauthorized tables into the legitimate response.
*   **Stacked Queries:** `id=1; EXEC xp_cmdshell('whoami')` allows code execution directly from the Database Management System (DBMS).

### Tools & Techniques
*   **Tools:** SQLMap (Automated exploitation), Burp Suite (Manual payload fuzzer).
*   **Command Line Example (SQLMap):**
    ```bash
    sqlmap -u "http://10.10.10.50/view.php?id=1" --dbs --batch
    ```

### Defensive Countermeasures
*   **Architecture:** Mandatory use of **Parameterized Queries / Prepared Statements** (e.g., PDO in PHP) or Object-Relational Mappers (ORMs) which strictly separate code from data.
*   **Detection (WAF):** Identifying SQL syntax anomalies (`UNION`, `SELECT`, `--`, `#`) in HTTP parameter telemetry.

### Risk Assessment (Severity: CRITICAL)
*   **MITRE ATT&CK:** T1190 (Exploit Public-Facing Application)
*   **OWASP Top 10:** A03:2021 (Injection)

---

## CONTEXT 6: Cross-Site Scripting (XSS)

### STEP 1 — Define the Concept
*   **Definition:** A vulnerability where an attacker injects malicious executable scripts (usually JavaScript) into a trusted website, which is then executed by the browser of an unsuspecting victim visiting that site.
*   **Purpose in Security:** To hijack user sessions (stealing `document.cookie`), redirect users to malicious domains, or bypass CSRF protections to perform unauthorized actions on behalf of the victim.
*   **Where it is commonly used:** Found in comment sections, user profiles, or reflected error messages where unvalidated input is echoed back to the DOM.

### Technical Breakdown
XSS payloads in our WAF dataset represent a diverse array of obfuscated vectors designed to run scripts.
*   **Reflected XSS:** Payload delivered in a URL parameter (`?search=<script>alert(1)</script>`) and executed immediately.
*   **Stored XSS:** Payload saved to the database (e.g., a forum post `<img src=x onerror=alert(document.cookie)>`) executing every time the page is loaded.

### Defensive Countermeasures
*   **Input Validation:** Sanitize and strictly type-cast all incoming HTTP requests.
*   **Output Encoding:** Automatically HTML Entity encode user inputs rendering `<script>` as `&lt;script&gt;` so the browser reads it purely as text, not code.
*   **Content Security Policy (CSP):** Emitting strict HTTP headers preventing the execution of inline scripts and restricting external script sources.

### Risk Assessment (Severity: HIGH)
Unlike SQLi which attacks the server, XSS attacks the *user's client browser*.
*   **MITRE ATT&CK:** T1189 (Drive-by Compromise)
*   **OWASP Top 10:** A03:2021 (Injection - Cross-Site Scripting)

---

## CONTEXT 7: Fileless Malware & Memory Forensics (Python-Driven)

### ✅ STEP 1 — Define the Concept
*   **Definition:** Fileless Malware is a category of computer threats that resides entirely in a computer's volatile memory (RAM) rather than being stored as a file on the hard drive. 
*   **Purpose in Security:** It bypasses traditional file-based signature detection (disk-based AV) and leaves a minimal forensic footprint.
*   **Common Python Usage:** Used for automating memory dump extraction (Volatility API) and writing memory scraping scripts.

### ✅ STEP 2 — Technical Foundation
*   **Technical Workflow:** Leverages system tools (PowerShell, WMI) and APIs (Windows API: `VirtualAllocEx`, `WriteProcessMemory`) to inject shellcode into legitimate process memory spaces.
*   **Python Terminology:** Involves `ctypes` library to interact directly with OS memory addresses, `binary` data structures, and `bytearray` analysis for identifying NOP sleds (`0x90`).

### ✅ STEP 3 — Python Attack Simulation (Educational Only)
**Reconnaissance:** Identify a target process (e.g., `notepad.exe`).
**Payload Creation:** Create a dummy shellcode bytearray.
**Exploit Simulation:**
```python
import ctypes
# Simulation of the logic used in Process Injection (Simplified)
def simulate_injection():
    # Placeholder for shellcode bytes
    shellcode = b"\x90\x90\x90\xCC" # NOP Sled + Software Breakpoint
    # In a real scenario, Python would use OpenProcess and VirtualAllocEx
    print(f"[*] Simulating injection of {len(shellcode)} bytes into volatile memory...")
    # Mitigation: This is where Endpoint Detection (EDR) would intercept API calls
```

### ✅ STEP 4 — Python Tools & Libraries
*   **Volatility3:** The gold standard for memory forensics.
*   **PyMemoryEditor:** Scans and interacts with process memory.
*   **Command:** `vol -f mem.dmp windows.info`

### ✅ STEP 5 — Detection & Monitoring (Python-Based)
Defensive operations require a mix of signature-based (rules/hashes) and behavioral (AI/Entropy) detection.

#### 🔎 Example 1: Log Monitoring (Brute Force Detection)
Python's `re` module is essential for parsing unstructured log data.
```python
import re
failed_attempts = {}
def detect_brute_force(log_line):
    # Pattern: Look for 'Failed password' and extract IP
    match = re.search(r'Failed password.*from (\d+\.\d+\.\d+\.\d+)', log_line)
    if match:
        ip = match.group(1)
        failed_attempts[ip] = failed_attempts.get(ip, 0) + 1
        return ip if failed_attempts[ip] > 5 else None
```

#### 🔎 Example 2: File Integrity Monitor (Hash Check)
Detects unauthorized modifications (Defacement or Ransomware) by comparing hashes.
```python
import hashlib
def check_fim(file_path, original_hash):
    current_hash = hashlib.sha256(open(file_path, 'rb').read()).hexdigest()
    if current_hash != original_hash:
        print(f"⚠️ ALERT: File {file_path} has been tampered with!")
```

#### 🔎 Example 3: Behavioral Anomaly Detection
The **Neuro Brain SOC** uses AI models to detect patterns that rules miss, such as a process "hollowing" itself in memory without touching the disk.

### ✅ STEP 6 — Mitigation & Prevention
Secure coding includes disabling unnecessary debug privileges and using compiled languages with memory safety.
```python
# Python script to check for suspicious processes with high memory usage
import psutil
def check_rogue_processes():
    for proc in psutil.process_iter(['pid', 'name', 'memory_info']):
        if proc.info['memory_info'].rss > (500 * 1024 * 1024): # Over 500MB
            # Potential memory-resident threat or leak
            print(f"[*] Audit Triggered: {proc.info['name']} (PID: {proc.info['pid']})")
```

### ✅ STEP 7 — Risk & Severity Assessment
*   **Impact:** **CRITICAL** for Integrity and Confidentiality. Allows for persistent, "invisible" spying.
*   **CIA:** Direct compromise of Availability if a critical system process is hollowed out and crashes.

### ✅ STEP 8 — Real-World Scenario
**Scenario:** A Cobalt Strike beacon is injected into `svchost.exe`. 
**Detection:** Python scans the memory for unbacked `PAGE_EXECUTE_READWRITE` segments.
**Outcome:** The SOC team identifies the process and terminates the parent thread before data exfiltration occurs.

---

---

## CONTEXT 8: AI SOC Dashboard & Real-Time Visualization

### ✅ STEP 1 — Define the Concept
*   **Definition:** An AI-Driven SOC Dashboard is a centralized interface that aggregates telemetry from Network (NIDS), Web (WAF), and Endpoint (Forensics) layers to provide a unified "Single Pane of Glass" view of an organization's security posture.
*   **Purpose in Security:** It reduces "Alert Fatigue" by using AI to correlate independent events and prioritize high-confidence threats for human analysts.
*   **Python Applications:** Building custom security consoles with Tkinter/Dash, automating incident response playbooks, and real-time log streaming.

### ✅ STEP 2 — Technical Foundation
*   **Architecture:** Uses a **Producer-Consumer** pattern with **Multi-Threading**. The producer generates/sniffs telemetry, and the consumer (the AI model) performs inference without locking the UI main loop.
*   **Data Structures:** Uses `numpy` arrays for high-speed matrix operations (required for CNN inference) and `pandas` DataFrames for historical trend analysis.

### ✅ STEP 3 — Python Attack Simulation (Educational Only)
**Stage 1: Reconnaissance (Network Scanning)**
```python
import socket
def scan_network(target_ip):
    # Simulating a stealthy port scan logic
    print(f"[*] Reconnaissance: Scanning {target_ip} for open management ports...")
    # Logic: try to connect to 22, 80, 443
```
**Stage 2: Payload Creation (WAF Evasion)**
```python
def create_sqli_payload():
    # Creating an obfuscated SQLi string
    payload = "admin' AND (SELECT 1 FROM (SELECT(SLEEP(5)))a)--"
    print(f"[*] Payload Created: {payload}")
    return payload
```
**Stage 3: Testing & Detection (SOC Log Trigger)**
```python
# Pass payload to the WAF AI Model
detection = waf_ai.predict(create_sqli_payload())
if detection['label'] != "Benign":
    print(f"[!] SOC REACTION: WAF Indicator Triggered - {detection['label']} detected.")
```

### ✅ STEP 4 — Python Tools & Libraries
For enterprise-level detection, use specialized libraries for speed and scale:
*   **scapy:** Low-level packet crafting and sniffing.
*   **pandas:** High-speed log analysis and threat correlation.
*   **psutil:** Monitoring process memory and CPU signatures.
*   **requests:** Testing WAF resilience via simulated HTTP payloads.

#### 🔎 Enterprise Enhancement: SIEM Integration
A professional SOC stores detections in **JSON** format for SIEM (Splunk/ELK) ingestion:
```python
import json
alert = {"timestamp": "2026-03-04T03:00:00", "type": "SQLi", "severity": "CRITICAL"}
print(json.dumps(alert)) # SIEM-ready output
```

### ✅ STEP 5 — Python-Based Detection
The dashboard integrates `QuickPredictor` to run `predict_intrusion()`, `predict_waf()`, and `predict_forensics()` concurrently on incoming telemetry.

### ✅ STEP 6 — Mitigation & Protection
**Active Response Script:**
```python
import psutil
def auto_mitigate_threat(pid):
    # Terminate process detected as shellcode runner
    p = psutil.Process(pid)
    p.terminate()
    print(f"[!] MITIGATION: Terminated malicious PID {pid}")
```

### ✅ STEP 7 — Risk & Severity Assessment
*   **Impact:** High. Ensures 24/7 autonomous monitoring.
*   **Criticality:** Without a dashboard, sophisticated "low-and-slow" attacks often go unnoticed in individual log silos.

### ✅ STEP 8 — Real-World Scenario
**Scenario:** A company is hit by a credential stuffing attack (Brute Force).
**SOC Workflow:** The NIDS model detects a burst of failed logins. The Dashboard automatically flashes **CRITICAL**, triggers the `BLOCK_IP` response, and adds the attacker to the firewall blocklist before any account is compromised.

---


---

# 🛡️ MASTER PLAYBOOK: SQL INJECTION (SQLi) DEFENSE

A complete guide to understanding, detecting, and preventing SQL Injection using Python.

### ✅ STEP 1 — Define the Threat
*   **Identification:** SQL Injection (SQLi).
*   **How Attackers Breach:** Attackers insert malicious SQL code into input fields (like login forms or URL parameters). If the application doesn't sanitize this input, the code is executed directly by the database.
*   **Python Context:** Python web applications using `sqlite3` or `psycopg2` are vulnerable if they use string formatting (e.g., `f"SELECT * FROM users WHERE id = {user_id}"`) instead of parameterized queries.

### ✅ STEP 2 — Technical Foundation
*   **Target:** Database Layer (L7 Application Layer).
*   **Protocols:** HTTP/HTTPS.
*   **Methods:** Boolean-based, Error-based, UNION-based, and Out-of-band SQLi.
*   **Python Simulation Logic:**
```python
# VULNERABLE LOGIC (DO NOT USE)
query = "SELECT * FROM users WHERE username = '" + user_input + "';"
# If user_input is "' OR '1'='1", the query becomes:
# SELECT * FROM users WHERE username = '' OR '1'='1'; (LOGS IN AS FIRST USER)
```

### ✅ STEP 3 — Detection (Python-Based)
Python scripts can detect SQLi by scanning logs or live traffic for common keywords like `UNION`, `SELECT`, `OR 1=1`, and `--`.
```python
import re
def detect_sqli(payload):
    # Indicators of Compromise (Keywords)
    sqli_pattern = r"(\'|--|union|select|insert|delete|drop|or\s+1=1)"
    if re.search(sqli_pattern, payload, re.IGNORECASE):
        return True
    return False

test_input = "admin' OR 1=1--"
if detect_sqli(test_input):
    print("⚠️ SQL Injection Detected in Payload!")
```

### ✅ STEP 4 — Prevention & Hardening
The #1 defense is using **Parameterized Queries** (Prepared Statements).
```python
import sqlite3
def secure_login(username, password):
    conn = sqlite3.connect("users.db")
    cursor = conn.cursor()
    # ✅ SECURE: Using '?' as a placeholder
    cursor.execute("SELECT * FROM users WHERE user=? AND pass=?", (username, password))
    return cursor.fetchone()
```

### ✅ STEP 5 — Network Security Monitoring
Monitor L7 traffic for SQLi patterns using Scapy to inspect HTTP GET/POST requests.
```python
from scapy.all import sniff, TCP, Raw
def sniff_sql_traffic(pkt):
    if pkt.haslayer(Raw):
        payload = pkt[Raw].load.decode(errors='ignore')
        if "SELECT" in payload or "UNION" in payload:
            print(f"⚠️ Suspicious SQL-like traffic from {pkt[IP].src}")

# sniff(filter="tcp port 80", prn=sniff_sql_traffic)
```

### ✅ STEP 6 — Real-Time Alerts
Automatically notify the security team when a high-confidence SQLi attempt is blocked.
```python
import json
def send_siem_log(alert_type, source_ip):
    alert = {
        "timestamp": str(datetime.now()),
        "event": alert_type,
        "src": source_ip,
        "severity": "CRITICAL"
    }
    print(f"[📤] SIEM LOG: {json.dumps(alert)}")
```

### ✅ STEP 7 — Risk Assessment
*   **Confidentiality:** **CRITICAL**. Attackers can dump the entire user database.
*   **Integrity:** **HIGH**. Attackers can modify prices, delete tables, or change credentials.
*   **Availability:** **MEDIUM**. `DROP TABLE` or `SHUTDOWN` commands can kill the service.

### ✅ STEP 8 — Summary & Defense Strategy
1.  **Stop String Concatenation:** Always use parameterized queries.
2.  **Web Application Firewall (WAF):** Use AI models (like our `waf_ai.py`) to block obfuscated attacks.
3.  **Principle of Least Privilege:** Run the database user with minimal permissions (e.g., no `DROP` access).
4.  **Logging:** Monitor all database queries for anomalies using Python SIEM scripts.

---

# 🛡️ MASTER PLAYBOOK: DDoS DEFENSE (DISTRIBUTED DENIAL OF SERVICE)

A strategic guide to mitigating volumetric and protocol-based attacks using Python.

### ✅ STEP 1 — Define the Threat
*   **Identification:** Distributed Denial of Service (DDoS).
*   **How Attackers Breach:** Attackers use a network of compromised devices (Botnets) to flood a target system with overwhelming traffic. The goal is not theft, but **Availability** compromise—making the service unreachable for legitimate users.
*   **Python Context:** Python is often used to build "Stress Testers" (benchmarking tools) or for creating lightweight traffic analyzers that detect sudden pps (packets per second) spikes.

### ✅ STEP 2 — Technical Foundation
*   **Target:** Network (L3), Transport (L4), or Application (L7) Layers.
*   **Protocols:** UDP, ICMP (Ping floods), TCP (SYN floods), and HTTP (GET floods).
*   **Methods:** Amplication attacks (DNS/NTP) and low-and-slow HTTP depletion.
*   **Python Simulation Logic (L4 SYN Flood):**
```python
from scapy.all import IP, TCP, send
# WARNING: Educational simulation only
def simulate_syn_flood(target_ip, target_port):
    pkt = IP(dst=target_ip)/TCP(dport=target_port, flags="S")
    send(pkt, loop=1, verbose=0) 
```

### ✅ STEP 3 — Detection (Python-Based)
The most effective detection involves monitoring **Packet Velocity** and **IP Distribution**.
```python
import time
traffic_counter = {} # {ip: count}
WINDOW = 10 # 10-second window
THRESHOLD = 500 # pps threshold

def monitor_traffic(src_ip):
    traffic_counter[src_ip] = traffic_counter.get(src_ip, 0) + 1
    if traffic_counter[src_ip] > THRESHOLD:
        print(f"⚠️ DDoS ALERT: Anomalous packet rate from {src_ip}!")
        return True
    return False
```

### ✅ STEP 4 — Prevention & Hardening
Hardening involves rate-limiting at the kernel or application level.
1.  **SYN Cookies:** Enable in OS to handle incomplete handshakes.
2.  **Rate Limiting:** Python decorators can enforce per-IP request limits.
```python
from functools import wraps
from flask import request, abort

# Simple Flask Rate Limiter
def rate_limit(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        ip = request.remote_addr
        # Check against Redis/In-memory store...
        # if limit_exceeded: abort(429)
        return f(*args, **kwargs)
    return decorated_function
```

### ✅ STEP 5 — Network Security Monitoring
Use Scapy to analyze the entropy of incoming IP headers. Random/Spoofed source IPs are a hallmark of DDoS.
```python
from scapy.all import sniff, IP
def detect_spoofing(pkt):
    if pkt.haslayer(IP):
        # Logic: If 1,000 unique IPs arrive in 1sec, suspect Botnet
        pass
```

### ✅ STEP 6 — Real-Time Alerts
Integrating with Edge services like Cloudflare or AWS Shield APIs to trigger auto-mitigation.
```python
import requests
def trigger_ddos_shield(ip_to_block):
    # API call to firewall or CDN to block range
    print(f"[🔥] SHIELD: Auto-blocking malicious traffic source {ip_to_block}/24")
```

### ✅ STEP 7 — Risk Assessment
*   **Confidentiality:** **LOW**. Usually no data is stolen.
*   **Integrity:** **LOW**. Data remains unchanged.
*   **Availability:** **CRITICAL**. Total service blackouts leading to financial loss and reputational damage.

### ✅ STEP 8 — Summary & Defense Strategy
1.  **Over-Provisioning:** Ensure bandwidth headroom.
2.  **AI Filtering:** Use volumetric analysis (NIDS) to distinguish "Flash Crowds" from "Botnets."
3.  **Automatic Sinkholing:** Programmatically route malicious traffic to invalid addresses using Python orchestration scripts.

---

## Conclusion
The Neuro Brain project now features a complete defensive ecosystem: from raw data preparation to AI model training, and finally, a real-time **SOC Dashboard** for proactive cybersecurity operations.

