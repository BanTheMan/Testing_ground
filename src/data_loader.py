"""
Central data loading module for TigerSafe.

Loads and normalizes all raw data from the data/ directory:
- CPD crime data (CSV with NIBRS descriptions)
- MUPD crime log (CSV with campus incidents)
- MUPD incident log (CSV with police calls)
- Shuttle routes and stops (CSV with coordinates)
- Campus buildings, boundary, emergency phones (GeoJSON)
- Traffic stops (CSV for patrol pattern analysis)
"""

import json
import os
from datetime import datetime
from glob import glob
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Point

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# Known Columbia / Mizzou street coordinates for local geocoding
# Maps street name fragments to approximate (lat, lon) center points.
_STREET_COORDS = {
    "conley": (38.9400, -92.3260),
    "college ave": (38.9400, -92.3280),
    "college": (38.9400, -92.3280),
    "providence": (38.9410, -92.3310),
    "stadium": (38.9360, -92.3260),
    "rollins": (38.9380, -92.3270),
    "hitt": (38.9405, -92.3270),
    "university": (38.9410, -92.3280),
    "tiger": (38.9445, -92.3300),
    "broadway": (38.9470, -92.3290),
    "walnut": (38.9475, -92.3280),
    "elm": (38.9460, -92.3290),
    "cherry": (38.9460, -92.3310),
    "ash": (38.9465, -92.3305),
    "locust": (38.9465, -92.3295),
    "park de ville": (38.9435, -92.3240),
    "parkade": (38.9350, -92.3150),
    "vandiver": (38.9520, -92.2930),
    "grindstone": (38.9110, -92.3240),
    "nifong": (38.9100, -92.3290),
    "blue ridge": (38.9200, -92.3000),
    "chapel hill": (38.9150, -92.3700),
    "forum": (38.9080, -92.3290),
    "worley": (38.9480, -92.3310),
    "business loop": (38.9550, -92.3350),
    "range line": (38.9450, -92.3120),
    "east campus": (38.9380, -92.3250),
    "discovery": (38.9430, -92.3240),
    "hospital": (38.9355, -92.3295),
    "research park": (38.9450, -92.3250),
    "hearnes": (38.9365, -92.3255),
    "champions": (38.9360, -92.3320),
    "monk": (38.9385, -92.3300),
    "kentucky": (38.9405, -92.3235),
    "maryland": (38.9415, -92.3235),
    "stewart": (38.9393, -92.3290),
    "lowry": (38.9420, -92.3300),
    "turner": (38.9425, -92.3290),
}


def _approx_geocode(address: str) -> tuple[float, float] | None:
    """Approximate geocoding using known street names in Columbia, MO."""
    if not address or not isinstance(address, str):
        return None
    addr_lower = address.lower()
    for street, (lat, lon) in _STREET_COORDS.items():
        if street in addr_lower:
            # Add small jitter based on block number if present
            jitter_lat = np.random.default_rng(hash(address) % 2**31).normal(0, 0.001)
            jitter_lon = np.random.default_rng(hash(address) % 2**31 + 1).normal(0, 0.001)
            return (lat + jitter_lat, lon + jitter_lon)
    return None


def load_cpd_crimes() -> pd.DataFrame:
    """Load CPD crime data from CSV files in data/crime_logs/."""
    crime_dir = DATA_DIR / "crime_logs"
    files = sorted(crime_dir.glob("cpd_crime_data_*.csv"))
    if not files:
        return pd.DataFrame()

    frames = []
    for f in files:
        df = pd.read_csv(f, parse_dates=["report_date"])
        frames.append(df)
    df = pd.concat(frames, ignore_index=True).drop_duplicates(subset=["offense_id"])

    # Geocode addresses
    coords = df["full_address"].apply(_approx_geocode)
    df["lat"] = coords.apply(lambda c: c[0] if c else None)
    df["lon"] = coords.apply(lambda c: c[1] if c else None)
    df = df.dropna(subset=["lat", "lon"])

    # Parse hour from report_date
    if "report_date" in df.columns:
        df["hour"] = pd.to_datetime(df["report_date"]).dt.hour
        df["day_of_week"] = pd.to_datetime(df["report_date"]).dt.day_name()

    return df


def load_mupd_crimes() -> pd.DataFrame:
    """Load MUPD Daily Crime Log from CSV files in data/crime_logs/."""
    crime_dir = DATA_DIR / "crime_logs"
    files = sorted(crime_dir.glob("mupd_crime_log_*.csv"))
    if not files:
        return pd.DataFrame()

    frames = []
    for f in files:
        df = pd.read_csv(f)
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)

    # The MUPD CSV has a duplicate header row — filter it out
    if "Case Number" in df.columns:
        df = df[df["Case Number"] != "Case Number"]

    # Parse dates from the "Date/Time Reported" or similar column
    date_col = None
    for col in df.columns:
        if "date" in col.lower() and "reported" in col.lower():
            date_col = col
            break
        if "date" in col.lower() and "occured" in col.lower():
            date_col = col
            break

    if date_col:
        # Handle range formats like "02/14/2026 12:30:00 am-02/14/2026 12:30:00 am"
        df["_date_str"] = df[date_col].astype(str).str.split("-").str[0].str.strip()
        df["report_date"] = pd.to_datetime(df["_date_str"], errors="coerce")
        df["hour"] = df["report_date"].dt.hour
        df["day_of_week"] = df["report_date"].dt.day_name()
        df.drop(columns=["_date_str"], inplace=True)

    # Geocode locations
    loc_col = None
    for col in df.columns:
        if "location" in col.lower():
            loc_col = col
            break

    if loc_col:
        coords = df[loc_col].apply(_approx_geocode)
        df["lat"] = coords.apply(lambda c: c[0] if c else None)
        df["lon"] = coords.apply(lambda c: c[1] if c else None)
    df = df.dropna(subset=["lat", "lon"])

    # Normalize column names
    rename_map = {}
    for col in df.columns:
        if "incident type" in col.lower():
            rename_map[col] = "incident_type"
        if "criminal offense" in col.lower():
            rename_map[col] = "criminal_offense"
        if "case number" in col.lower():
            rename_map[col] = "case_number"
        if "location" in col.lower() and "occurr" in col.lower():
            rename_map[col] = "location_name"
        if "disposition" in col.lower():
            rename_map[col] = "disposition"
        if "domestic" in col.lower():
            rename_map[col] = "domestic_violence"
    df.rename(columns=rename_map, inplace=True)

    return df


def load_mupd_incidents() -> pd.DataFrame:
    """Load MUPD Daily Incident Log (police calls) from CSV."""
    crime_dir = DATA_DIR / "crime_logs"
    files = sorted(crime_dir.glob("mupd_incident_log_*.csv"))
    if not files:
        return pd.DataFrame()

    frames = []
    for f in files:
        df = pd.read_csv(f)
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)

    # Parse date and time
    if "Call Date" in df.columns and "Call Time" in df.columns:
        df["datetime_str"] = df["Call Date"].astype(str) + " " + df["Call Time"].astype(str)
        df["report_date"] = pd.to_datetime(df["datetime_str"], errors="coerce")
        df["hour"] = df["report_date"].dt.hour
        df.drop(columns=["datetime_str"], inplace=True)

    # Geocode addresses
    if "Address" in df.columns:
        coords = df["Address"].apply(_approx_geocode)
        df["lat"] = coords.apply(lambda c: c[0] if c else None)
        df["lon"] = coords.apply(lambda c: c[1] if c else None)
    df = df.dropna(subset=["lat", "lon"])

    return df


def load_shuttle_routes() -> pd.DataFrame:
    """Load shuttle route data from CSV."""
    shuttle_dir = DATA_DIR / "shuttle_data"
    files = sorted(shuttle_dir.glob("shuttle_routes_*.csv"))
    if not files:
        return pd.DataFrame()
    return pd.read_csv(files[-1])  # Use most recent


def load_shuttle_stops() -> pd.DataFrame:
    """Load shuttle stop data from CSV."""
    shuttle_dir = DATA_DIR / "shuttle_data"
    files = sorted(shuttle_dir.glob("shuttle_stops_*.csv"))
    if not files:
        return pd.DataFrame()
    return pd.read_csv(files[-1])


def load_campus_buildings() -> gpd.GeoDataFrame:
    """Load campus buildings from GeoJSON."""
    boundary_dir = DATA_DIR / "campus_boundary"
    files = sorted(boundary_dir.glob("campus_buildings_*.geojson"))
    if not files:
        return gpd.GeoDataFrame()
    return gpd.read_file(files[-1])


def load_campus_boundary() -> gpd.GeoDataFrame:
    """Load campus boundary polygon from GeoJSON."""
    boundary_dir = DATA_DIR / "campus_boundary"
    files = sorted(boundary_dir.glob("campus_boundary_*.geojson"))
    if not files:
        return gpd.GeoDataFrame()
    return gpd.read_file(files[-1])


def load_emergency_phones() -> gpd.GeoDataFrame:
    """Load emergency phone locations from GeoJSON."""
    boundary_dir = DATA_DIR / "campus_boundary"
    files = sorted(boundary_dir.glob("safety_asset_emergency_phones_*.geojson"))
    if not files:
        return gpd.GeoDataFrame()
    return gpd.read_file(files[-1])


def load_accessible_entrances() -> gpd.GeoDataFrame:
    """Load accessible entrance locations from GeoJSON."""
    boundary_dir = DATA_DIR / "campus_boundary"
    files = sorted(boundary_dir.glob("safety_asset_accessible_entrances_*.geojson"))
    if not files:
        return gpd.GeoDataFrame()
    return gpd.read_file(files[-1])


def load_traffic_stops(recent_years: int = 3) -> pd.DataFrame:
    """Load traffic stop data for patrol pattern analysis.

    Only loads the most recent N years to keep memory manageable.
    """
    traffic_dir = DATA_DIR / "traffic_stops"
    files = sorted(traffic_dir.glob("CPD*vehicle_stop_data_*.csv"))
    if not files:
        return pd.DataFrame()

    # Take most recent years
    files = files[-recent_years:] if len(files) > recent_years else files

    frames = []
    for f in files:
        try:
            df = pd.read_csv(f, low_memory=False)
            frames.append(df)
        except Exception:
            continue

    if not frames:
        return pd.DataFrame()

    df = pd.concat(frames, ignore_index=True)

    # Extract useful columns for patrol patterns
    cols_keep = []
    for col in ["street", "stoptime", "beat", "hour_stop", "DoW", "day_night", "Location"]:
        if col in df.columns:
            cols_keep.append(col)
    if cols_keep:
        df = df[cols_keep].copy()

    # Geocode street locations
    if "street" in df.columns:
        coords = df["street"].apply(_approx_geocode)
        df["lat"] = coords.apply(lambda c: c[0] if c else None)
        df["lon"] = coords.apply(lambda c: c[1] if c else None)

    return df


def load_all_crimes_unified() -> gpd.GeoDataFrame:
    """Load and unify all crime data into a single GeoDataFrame.

    Merges CPD and MUPD sources with normalized columns.
    """
    from src.crime_analyzer import normalize_crime_category

    all_records = []

    # CPD crimes
    cpd = load_cpd_crimes()
    if not cpd.empty:
        for _, row in cpd.iterrows():
            cat_info = normalize_crime_category(
                row.get("nibrs_description", ""), source="cpd"
            )
            all_records.append({
                "incident_id": str(row.get("offense_id", "")),
                "source": "cpd",
                "report_date": row.get("report_date"),
                "hour": row.get("hour"),
                "day_of_week": row.get("day_of_week"),
                "category": cat_info["category"],
                "severity": cat_info["severity"],
                "is_violent": cat_info["is_violent"],
                "original_description": row.get("nibrs_description", ""),
                "address": row.get("full_address", ""),
                "geometry": Point(row["lon"], row["lat"]),
            })

    # MUPD crimes
    mupd = load_mupd_crimes()
    if not mupd.empty:
        for _, row in mupd.iterrows():
            desc = row.get("incident_type", row.get("criminal_offense", ""))
            cat_info = normalize_crime_category(str(desc), source="mupd")
            all_records.append({
                "incident_id": str(row.get("case_number", "")),
                "source": "mupd",
                "report_date": row.get("report_date"),
                "hour": row.get("hour"),
                "day_of_week": row.get("day_of_week"),
                "category": cat_info["category"],
                "severity": cat_info["severity"],
                "is_violent": cat_info["is_violent"],
                "original_description": str(desc),
                "address": row.get("location_name", ""),
                "geometry": Point(row["lon"], row["lat"]),
            })

    if not all_records:
        return gpd.GeoDataFrame(
            columns=[
                "incident_id", "source", "report_date", "hour",
                "day_of_week", "category", "severity", "is_violent",
                "original_description", "address", "geometry",
            ],
            crs="EPSG:4326",
        )

    gdf = gpd.GeoDataFrame(all_records, crs="EPSG:4326")
    gdf["report_date"] = pd.to_datetime(gdf["report_date"], errors="coerce")
    return gdf
