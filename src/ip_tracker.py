"""
🌍 Neuro Brain — AI-Powered IP Tracker
Geolocation, DNS Analysis, and World Map Visualization for SOC Operations.
"""

import requests
import socket
import re
import folium
import pandas as pd
from datetime import datetime
from pathlib import Path

class IPTracker:
    def __init__(self, log_path="data/logs/ip_queries.csv"):
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.map_dir = Path("static/maps")
        self.map_dir.mkdir(parents=True, exist_ok=True)

    def validate_ip(self, ip):
        """Validates IPv4 format using regex."""
        pattern = r'^(\d{1,3}\.){3}\d{1,3}$'
        if re.match(pattern, ip):
            # Check if each octet is within 0-255
            octets = ip.split('.')
            return all(0 <= int(o) <= 255 for o in octets)
        return False

    def get_geolocation(self, ip):
        """Queries ip-api.com for geolocation data."""
        try:
            url = f"http://ip-api.com/json/{ip}"
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                data = response.json()
                if data.get("status") == "success":
                    return {
                        "IP": ip,
                        "City": data.get("city"),
                        "Region": data.get("regionName"),
                        "Country": data.get("country"),
                        "ISP": data.get("isp"),
                        "Timezone": data.get("timezone"),
                        "Latitude": data.get("lat"),
                        "Longitude": data.get("lon"),
                        "Type": "IPv4" if "." in ip else "IPv6"
                    }
            return {"error": f"API Error: {response.status_code}"}
        except Exception as e:
            return {"error": str(e)}

    def get_hostname(self, ip):
        """Performs a reverse DNS lookup."""
        try:
            return socket.gethostbyaddr(ip)[0]
        except (socket.herror, socket.gaierror, socket.timeout):
            return "Hostname not found"

    def log_query(self, data):
        """Logs the tracked IP details to CSV."""
        try:
            data["Timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            df = pd.DataFrame([data])
            # Write header only if file doesn't exist
            df.to_csv(self.log_path, mode='a', index=False, header=not self.log_path.exists())
            return True
        except Exception as e:
            print(f"Logging Error: {e}")
            return False

    def generate_map(self, ip_data_list, filename="latest_track.html"):
        """Generates an interactive Folium map for one or more IPs."""
        try:
            # Start at center of world or at first IP location
            start_coord = [0, 0]
            if ip_data_list and "Latitude" in ip_data_list[0]:
                start_coord = [ip_data_list[0]["Latitude"], ip_data_list[0]["Longitude"]]
            
            world_map = folium.Map(location=start_coord, zoom_start=3, tiles="CartoDB dark_matter")
            
            for data in ip_data_list:
                if "Latitude" in data and "Longitude" in data:
                    folium.Marker(
                        location=[data["Latitude"], data["Longitude"]],
                        popup=f"<b>IP:</b> {data['IP']}<br><b>Loc:</b> {data['City']}, {data['Country']}<br><b>ISP:</b> {data['ISP']}",
                        icon=folium.Icon(color='blue', icon='info-sign')
                    ).add_to(world_map)
            
            save_path = self.map_dir / filename
            world_map.save(str(save_path))
            return f"/static/maps/{filename}"
        except Exception as e:
            print(f"Map Generation Error: {e}")
            return None

    def check_threat_score(self, geo_data):
        """Educational simulation of a threat intelligence check."""
        score = 0
        reasons = []
        
        # Simulated suspicious regions/ISPs
        high_risk_countries = ["China", "Russia", "North Korea"]
        suspicious_isps = ["DigitalOcean", "Linode", "Hetzner", "OVH"] # Often used for VPNs/Bots
        
        if geo_data.get("Country") in high_risk_countries:
            score += 40
            reasons.append(f"High-risk geo-location: {geo_data['Country']}")
            
        if any(isp in (geo_data.get("ISP") or "") for isp in suspicious_isps):
            score += 30
            reasons.append(f"Infrastructure provider commonly used for anonymous traffic: {geo_data['ISP']}")
            
        # Random simulation factor
        if "VPN" in (geo_data.get("ISP") or "").upper():
            score += 50
            reasons.append("VPN/Proxy detected.")
            
        return {
            "Score": min(score, 100),
            "RiskLevel": "CRITICAL" if score >= 70 else "MEDIUM" if score >= 30 else "LOW",
            "Reasons": reasons
        }

    def track_batch(self, ips):
        """Processes a list of IPs and returns consolidated data."""
        results = []
        for ip in ips:
            if self.validate_ip(ip):
                geo = self.get_geolocation(ip)
                if "error" not in geo:
                    geo["Hostname"] = self.get_hostname(ip)
                    geo["Threat"] = self.check_threat_score(geo)
                    self.log_query(geo)
                    results.append(geo)
        return results

    def get_risk_summary(self, results):
        """Generates statistical summary for a batch of tracked IPs."""
        if not results:
            return None
        
        df = pd.DataFrame(results)
        return {
            "Total": len(results),
            "HighRisk": len([r for r in results if r["Threat"]["RiskLevel"] == "CRITICAL"]),
            "TopCountry": df['Country'].mode()[0] if not df['Country'].mode().empty else "Unknown",
            "TopISP": df['ISP'].mode()[0] if not df['ISP'].mode().empty else "Unknown"
        }

if __name__ == "__main__":
    # Tactical Self-Test: Batch Reconnaissance
    tracker = IPTracker()
    test_ips = ["8.8.8.8", "1.1.1.1", "203.0.113.152"]
    print(f"[*] Batch Tracking: {test_ips}...")
    
    batch_results = tracker.track_batch(test_ips)
    summary = tracker.get_risk_summary(batch_results)
    
    print(f"[+] Summary: {summary}")
    map_url = tracker.generate_map(batch_results, filename="batch_test.html")
    print(f"[+] Multi-Marker Map: {map_url}")
