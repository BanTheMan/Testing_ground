import requests
import json
import os
from datetime import datetime

# Configuration
# Found via manual inspection of map.missouri.edu network traffic
BASE_MAP_SERVER = "https://muop-mapapp.umad.umsystem.edu:6443/arcgis/rest/services/MU_Base_new/MapServer"

# Layer IDs
LAYER_BOUNDARY = 66
LAYER_BUILDINGS = 58 
# Could also use Layer 13 in MU_Features_new for detailed buildings if needed

# Construct path relative to this script file
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "../../data/campus_boundary"))

def setup_directories():
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        print(f"Created directory: {DATA_DIR}")

def fetch_layer(layer_id, layer_name):
    """
    Fetches a specific layer from the ArcGIS MapServer as GeoJSON.
    """
    print(f"\n--- Fetching {layer_name} (Layer {layer_id}) ---")
    
    url = f"{BASE_MAP_SERVER}/{layer_id}/query"
    
    # Standard ArcGIS query parameters
    params = {
        'where': '1=1', # Select all
        'outFields': '*',
        'f': 'geojson', # Request GeoJSON format directly
        'returnGeometry': 'true',
        'outSR': '4326' # Request WGS84 coordinates (Lat/Lon)
    }
    
    try:
        response = requests.get(url, params=params, verify=False) # Skip SSL verify for internal-looking servers if needed, usually safer to verify but let's try.
        # Note: umsystem.edu certs might be tricky. If script fails on SSL, we'll disable verify.
        
        response.raise_for_status()
        data = response.json()
        
        features = data.get('features', [])
        print(f"Found {len(features)} features for {layer_name}.")
        
        if features:
            # Save Raw GeoJSON
            filename = os.path.join(DATA_DIR, f"{layer_name}_{datetime.now().strftime('%Y%m%d')}.geojson")
            with open(filename, 'w') as f:
                json.dump(data, f, indent=2)
            print(f"Saved {layer_name} to {filename}")
            
    except Exception as e:
        print(f"Error fetching {layer_name}: {e}")

if __name__ == "__main__":
    # Suppress InsecureRequestWarning if we need to verify=False
    import urllib3
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    
    setup_directories()
    fetch_layer(LAYER_BOUNDARY, "campus_boundary")
    fetch_layer(LAYER_BUILDINGS, "campus_buildings")
