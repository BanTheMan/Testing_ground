import requests
import json
import os
from datetime import datetime

# Configuration
# MU_Features_new contains detailed point features
BASE_FEATURE_SERVER = "https://muop-mapapp.umad.umsystem.edu:6443/arcgis/rest/services/MU_Features_new/MapServer"

# Layer IDs
LAYER_EMERGENCY_PHONES = 1
LAYER_ACCESSIBLE_ENTRANCES = 3

# Construct path relative to this script file
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "../../data/campus_boundary"))

def setup_directories():
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        # Using same dir as campus boundary for now

def fetch_features(layer_id, asset_name):
    """
    Fetches point features from ArcGIS MapServer as GeoJSON.
    """
    print(f"\n--- Fetching {asset_name} (Layer {layer_id}) ---")
    
    url = f"{BASE_FEATURE_SERVER}/{layer_id}/query"
    
    params = {
        'where': '1=1',
        'outFields': '*',
        'f': 'geojson',
        'returnGeometry': 'true',
        'outSR': '4326'
    }
    
    try:
        # Disable SSL verify as per previous success/observation with umsystem servers
        response = requests.get(url, params=params, verify=False)
        response.raise_for_status()
        data = response.json()
        
        features = data.get('features', [])
        print(f"Found {len(features)} {asset_name}.")
        
        if features:
            # Save Raw GeoJSON
            filename = os.path.join(DATA_DIR, f"safety_asset_{asset_name.replace(' ', '_').lower()}_{datetime.now().strftime('%Y%m%d')}.geojson")
            with open(filename, 'w') as f:
                json.dump(data, f, indent=2)
            print(f"Saved to {filename}")
            
            # Simple attribute inspection of first item
            if len(features) > 0:
                print("Sample properties:", features[0].get('properties'))
            
    except Exception as e:
        print(f"Error fetching {asset_name}: {e}")

if __name__ == "__main__":
    import urllib3
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    
    setup_directories()
    fetch_features(LAYER_EMERGENCY_PHONES, "Emergency Phones")
    fetch_features(LAYER_ACCESSIBLE_ENTRANCES, "Accessible Entrances")
