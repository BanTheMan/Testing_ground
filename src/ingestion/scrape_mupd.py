import requests
import pandas as pd
from datetime import datetime, timedelta
import os
import time

# Configuration
BASE_URL_CRIME = "https://muop-mupdreports.missouri.edu/dclog.php"
BASE_URL_INCIDENT = "https://muop-mupdreports.missouri.edu/dilog.php"

# Construct path relative to this script file
# Script is in src/ingestion/, data is in ../../data/crime_logs/
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "../../data/crime_logs"))

def setup_directories():
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        print(f"Created directory: {DATA_DIR}")

def scrape_crime_log():
    """
    Scrapes the MUPD Daily Crime Log (dclog.php) for the past year.
    Uses 'sfilter': '12' parameter.
    """
    print("\n--- Scraping Daily Crime Log (dclog.php) ---")
    try:
        # payload sfilter=12 fetches last 12 months
        response = requests.post(BASE_URL_CRIME, data={'sfilter': '12'})
        response.raise_for_status()

        dfs = pd.read_html(response.text)
        
        if not dfs:
            print("No tables found in Crime Log response.")
            return

        df = dfs[0]
        print(f"Found {len(df)} records in Crime Log.")

        # Save Raw Data
        filename = os.path.join(DATA_DIR, f"mupd_crime_log_{datetime.now().strftime('%Y%m%d')}.csv")
        df.to_csv(filename, index=False)
        print(f"Saved Crime Log to {filename}")
        
        # Display Sample
        print("Sample Data:")
        print(df.head())
        return df

    except Exception as e:
        print(f"Error scraping Crime Log: {e}")
        return None

def scrape_incident_log():
    """
    Scrapes the MUPD Daily Incident Log (dilog.php) for the past year.
    Loops through data month by month to avoid timeouts or large payload issues.
    """
    print("\n--- Scraping Daily Incident Log (dilog.php) ---")
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)
    
    all_incidents = []
    
    current_start = start_date
    session = requests.Session() # Use session to persist cookies
    
    # 1. Visit page first to get cookies/PHP session
    try:
        session.get(BASE_URL_INCIDENT)
    except Exception as e:
        print(f"Error initializing session: {e}")
        return

    while current_start < end_date:
        # Fetch in 30-day chunks
        current_end = min(current_start + timedelta(days=30), end_date)
        
        # FIX: The server expects YYYY-MM-DD format (based on browser inspection)
        str_start = current_start.strftime("%Y-%m-%d")
        str_end = current_end.strftime("%Y-%m-%d")
        
        print(f"Fetching Incidents: {str_start} - {str_end}...")
        
        payload = {
            'from_date': str_start, # Correct field name
            'to_date': str_end,     # Correct field name
            'type': 'View-All',     # Correct value
            'address': '',
            'search': 'Search',     # Required submit button
            'page_size': '500'       # Request larger page size to minimize pagination
        }
        
        try:
            response = session.post(BASE_URL_INCIDENT, data=payload)
            response.raise_for_status()
            
            # Checks if table exists
            if "No records found" in response.text:
                 print("  No records found for this period.")
            else:
                dfs = pd.read_html(response.text)
                if dfs:
                    chunk_df = dfs[0]
                    # Filter out empty rows or header repetitions if any
                    all_incidents.append(chunk_df)
                    print(f"  Retrieved {len(chunk_df)} records.")
        
        except Exception as e:
            print(f"  Error fetching chunk: {e}")

        # Move to next chunk
        current_start = current_end + timedelta(days=1)
        time.sleep(1) # Be polite to the server

    if all_incidents:
        final_df = pd.concat(all_incidents, ignore_index=True)
        print(f"\nTotal Incident Records Found: {len(final_df)}")
        
        # Save Raw Data
        filename = os.path.join(DATA_DIR, f"mupd_incident_log_{datetime.now().strftime('%Y%m%d')}.csv")
        final_df.to_csv(filename, index=False)
        print(f"Saved Incident Log to {filename}")
        
        print("Sample Data:")
        print(final_df.head())
        return final_df
    else:
        print("No incident data retrieved.")
        return None

if __name__ == "__main__":
    setup_directories()
    scrape_crime_log()
    scrape_incident_log()
