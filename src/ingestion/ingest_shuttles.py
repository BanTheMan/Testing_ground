import requests
import pandas as pd
import polyline  # library for decoding encoded polylines
import os
from datetime import datetime

# Configuration
API_URL = "https://gocomotransit.etaspot.net/service.php"
TOKEN = "TESTING"

# Construct path relative to this script file
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "../../data/shuttle_data"))

def setup_directories():
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        print(f"Created directory: {DATA_DIR}")

def ingest_routes():
    """
    Fetches shuttle routes and decodes their geometry.
    """
    print("\n--- Fetching Shuttle Routes ---")
    
    payload = {
        'service': 'get_routes',
        'token': TOKEN
    }
    
    try:
        response = requests.get(API_URL, params=payload)
        response.raise_for_status()
        data = response.json()
        
        routes = data.get('get_routes', [])
        print(f"Found {len(routes)} routes.")
        
        parsed_routes = []
        for r in routes:
            # encLine contains the geometry in Google Encoded Polyline format
            encoded_line = r.get('encLine')
            decoded_points = []
            
            if encoded_line:
                try:
                    # Returns list of (lat, lon) tuples
                    decoded_points = polyline.decode(encoded_line)
                except Exception as e:
                    print(f"Error decoding line for route {r.get('name')}: {e}")

            parsed_routes.append({
                'route_id': r.get('id'),
                'name': r.get('name'),
                'abbr': r.get('abbr'),
                'color': r.get('color'),
                'agency_id': r.get('agencyID'),
                'encoded_polyline': encoded_line,
                'decoded_points_count': len(decoded_points)
                # In a real DB ingestion, we'd convert decoded_points to WKT or GeoJSON here
            })
            
        df = pd.DataFrame(parsed_routes)
        
        filename = os.path.join(DATA_DIR, f"shuttle_routes_{datetime.now().strftime('%Y%m%d')}.csv")
        df.to_csv(filename, index=False)
        print(f"Saved Routes to {filename}")
        print(df[['name', 'decoded_points_count']].head())
        return df

    except Exception as e:
        print(f"Error fetching routes: {e}")
        return None

def ingest_stops():
    """
    Fetches shuttle stop locations.
    """
    print("\n--- Fetching Shuttle Stops ---")
    
    payload = {
        'service': 'get_stops',
        'token': TOKEN
    }
    
    try:
        response = requests.get(API_URL, params=payload)
        response.raise_for_status()
        data = response.json()
        
        stops = data.get('get_stops', [])
        print(f"Found {len(stops)} stops.")
        
        parsed_stops = []
        for s in stops:
            parsed_stops.append({
                'stop_id': s.get('id'),
                'name': s.get('name'),
                'lat': s.get('lat'),
                'lng': s.get('lng'),
            })
            
        df = pd.DataFrame(parsed_stops)
        
        filename = os.path.join(DATA_DIR, f"shuttle_stops_{datetime.now().strftime('%Y%m%d')}.csv")
        df.to_csv(filename, index=False)
        print(f"Saved Stops to {filename}")
        print(df.head())
        return df

    except Exception as e:
        print(f"Error fetching stops: {e}")
        return None

if __name__ == "__main__":
    setup_directories()
    ingest_routes()
    ingest_stops()
