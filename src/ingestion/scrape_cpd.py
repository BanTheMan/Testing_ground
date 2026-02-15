import requests
import json
import pandas as pd
from datetime import datetime
import os

# Configuration
# FeatureServer endpoint for "Crimes_public"
BASE_URL = "https://services.arcgis.com/GHhNHT1xiCkCAXvo/arcgis/rest/services/Crimes_public_5065541538a64410b0dc75b9276284aa/FeatureServer/0/query"

# Construct path relative to this script file
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "../../data/crime_logs"))

def setup_directories():
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        print(f"Created directory: {DATA_DIR}")

def scrape_cpd_data():
    """
    Fetches public crime data from Columbia Police Department's ArcGIS FeatureServer.
    """
    print("\n--- Fetching CPD Crime Data (ArcGIS) ---")
    
    # Query parameters
    # fetching data from Jan 1, 2024 to present 
    # (ArcGIS queries are often limited by record count, may need pagination if > 2000)
    params = {
        'where': "reportdate > '2024-01-01'",
        'outFields': '*',  # Fetch all fields
        'f': 'json',       # Format as JSON
        'resultOffset': 0, # For pagination
        'resultRecordCount': 1000, # Request 1000 at a time
        'orderByFields': 'reportdate DESC'
    }
    
    all_features = []
    
    while True:
        print(f"Fetching records with offset {params['resultOffset']}...")
        try:
            response = requests.get(BASE_URL, params=params)
            response.raise_for_status()
            data = response.json()
            
            if 'error' in data:
                print(f"ArcGIS Error: {data['error']}")
                break
                
            features = data.get('features', [])
            if not features:
                print("No more features found.")
                break
                
            all_features.extend(features)
            print(f"  Retrieved {len(features)} records.")
            
            # Check if we've reached the end? 
            # If we got fewer records than requested, we are properly done.
            if len(features) < params['resultRecordCount']:
                break
            
            # Prepare next page
            params['resultOffset'] += len(features)
            
        except Exception as e:
            print(f"Error fetching CPD data: {e}")
            break
            
    if all_features:
        print(f"\nTotal CPD Records Found: {len(all_features)}")
        
        # Parse into a cleaner list of dicts
        parsed_data = []
        for feat in all_features:
            attrs = feat.get('attributes', {})
            # ArcGIS geometry is usually x/y (Web Mercator) or lat/lon depending on coord system
            # Here we just grab attributes first. If geometry is needed, we'd process 'geometry' key.
            # Usually the attributes contain enough info or we request geojson.
            
            # Convert timestamp (epoch ms) to readable date
            report_ts = attrs.get('reportdate')
            report_dt = datetime.fromtimestamp(report_ts/1000.0) if report_ts else None
            
            parsed_data.append({
                'offense_id': attrs.get('offenseid'),
                'report_date': report_dt,
                'nibrs_description': attrs.get('nibrsdesc'),
                'incident_description': attrs.get('incident_desc'), # Sometimes present
                'case_status': attrs.get('casestatus'),
                'full_address': attrs.get('fulladdr'),
                'city': attrs.get('city'),
                'zip': attrs.get('zip'),
                # Lat/Lon might be in attributes or geometry. 
                # Inspecting one record shows: attributes usually just has data. 
                # Geometry object is separate. For CSV, we'll stick to attributes explicitly requested.
            })

        df = pd.DataFrame(parsed_data)
        
        # Save Raw Data
        filename = os.path.join(DATA_DIR, f"cpd_crime_data_{datetime.now().strftime('%Y%m%d')}.csv")
        df.to_csv(filename, index=False)
        print(f"Saved CPD Data to {filename}")
        
        print("Sample Data:")
        print(df.head())
        return df
    else:
        print("No CPD data retrieved.")
        return None

if __name__ == "__main__":
    setup_directories()
    scrape_cpd_data()
