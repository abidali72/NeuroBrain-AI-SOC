"""
🛡️ Neuro Brain — Secure Database Connector
Hardened interface for database operations using Parameterized Queries.
FOLLOWS ✅ STEP 8 — Summary & Defense Strategy
"""

import sqlite3
import logging
from pathlib import Path

# Setup logging for security audits
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [SECURE_DB] - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SecureDatabase:
    def __init__(self, db_path="data/soc_vault.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = None
        self._initialize_db()

    def _initialize_db(self):
        """Creates the initial hardened structure."""
        try:
            self.connection = sqlite3.connect(self.db_path)
            cursor = self.connection.cursor()
            
            # Create a table for storing suspicious activity
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS threat_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT,
                    source_ip TEXT,
                    attack_type TEXT,
                    details TEXT
                )
            ''')
            self.connection.commit()
            logger.info("Hardened Database Initialized Securely.")
        except sqlite3.Error as e:
            logger.error(f"Database Initialization Failed: {e}")

    def log_threat_securely(self, ip, attack_type, details):
        """
        ✅ SECURE: Uses Parameterized Queries (?) to prevent SQL Injection.
        Never use f-strings or string concatenation for values in SQL.
        """
        try:
            cursor = self.connection.cursor()
            query = "INSERT INTO threat_logs (timestamp, source_ip, attack_type, details) VALUES (?, ?, ?, ?)"
            values = (datetime.now().isoformat(), ip, attack_type, details)
            
            # The library handles sanitization behind the scenes
            cursor.execute(query, values)
            self.connection.commit()
            logger.info(f"Threat logged securely from source: {ip}")
        except sqlite3.Error as e:
            logger.error(f"Failed to log threat: {e}")

    def query_threats_by_ip(self, ip):
        """
        ✅ SECURE: Prevents 'OR 1=1' attacks via parameter binding.
        """
        try:
            cursor = self.connection.cursor()
            # Even if the user sends "' OR 1=1 --", it is treated as a literal search string
            query = "SELECT * FROM threat_logs WHERE source_ip = ?"
            cursor.execute(query, (ip,))
            return cursor.fetchall()
        except sqlite3.Error as e:
            logger.error(f"Secure query failed: {e}")
            return []

    def close(self):
        if self.connection:
            self.connection.close()

if __name__ == "__main__":
    from datetime import datetime
    # Tactical demonstration of hardened logic
    db = SecureDatabase()
    
    # Simulating a safe log entry
    db.log_threat_securely("192.168.1.162", "SQLi Attempt", "Detected via WAF AI")
    
    # Simulating a "Hacker" trying to bypass via input
    malicious_input = "' OR 1=1 --"
    print(f"\n[*] Testing Hardening against Payload: {malicious_input}")
    results = db.query_threats_by_ip(malicious_input)
    
    if not results:
        print("[✅] HARDENING SUCCESS: SQL Injection payload was neutralized and treated as plain text.")
    else:
        print("[❌] HARDENING FAILURE: Payload bypassed logic.")
        
    db.close()
