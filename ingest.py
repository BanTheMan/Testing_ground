"""
Data ingestion module for Campus Dispatch Copilot.

Downloads and processes:
- OpenStreetMap walking network for Columbia, MO (via osmnx)
- Crime data from CPD Transparency Portal (ArcGIS REST API)
- Campus infrastructure from MU ArcGIS services (emergency phones, buildings)
"""

import json
import os
import pickle
from pathlib import Path

import geopandas as gpd
import numpy as np
import osmnx as ox
import pandas as pd
import requests
from shapely.geometry import Point

DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)

# Columbia, MO / Mizzou campus bounding box (approximate)
CAMPUS_CENTER = (38.9404, -92.3277)
CAMPUS_BBOX_NORTH = 38.955
CAMPUS_BBOX_SOUTH = 38.925
CAMPUS_BBOX_EAST = -92.305
CAMPUS_BBOX_WEST = -92.345

# CPD Transparency Portal endpoints (ArcGIS REST)
CPD_CRIME_URL = (
    "https://gis.como.gov/arcgis/rest/services/Public/PublicSafety/MapServer/0/query"
)

# MU Campus Map ArcGIS REST
MU_ARCGIS_BASE = "https://map.missouri.edu:6443/arcgis/rest/services"
MU_EMERGENCY_PHONES_URL = (
    f"{MU_ARCGIS_BASE}/MU_Features_new/FeatureServer/5/query"
)
MU_BUILDINGS_URL = (
    f"{MU_ARCGIS_BASE}/MU_Features_new/FeatureServer/1/query"
)


def download_osm_network(force: bool = False) -> None:
    """Download walking network graph for campus area via osmnx."""
    graph_path = DATA_DIR / "columbia_walk.graphml"
    if graph_path.exists() and not force:
        print("OSM network already downloaded.")
        return

    print("Downloading OSM walking network for Columbia, MO campus area...")
    G = ox.graph_from_bbox(
        bbox=(CAMPUS_BBOX_NORTH, CAMPUS_BBOX_SOUTH, CAMPUS_BBOX_EAST, CAMPUS_BBOX_WEST),
        network_type="walk",
    )
    ox.save_graphml(G, graph_path)
    print(f"Saved graph with {G.number_of_nodes()} nodes, {G.number_of_edges()} edges.")


def fetch_cpd_crimes(days_back: int = 180, force: bool = False) -> gpd.GeoDataFrame:
    """Fetch recent crime data from CPD ArcGIS REST API."""
    cache_path = DATA_DIR / "cpd_crimes.geojson"
    if cache_path.exists() and not force:
        print("Loading cached CPD crime data...")
        return gpd.read_file(cache_path)

    print(f"Fetching CPD crime data (last {days_back} days)...")
    since = pd.Timestamp.now() - pd.Timedelta(days=days_back)
    since_ms = int(since.timestamp() * 1000)

    params = {
        "where": f"Date_Occurred >= {since_ms}",
        "outFields": "*",
        "outSR": "4326",
        "f": "geojson",
        "resultRecordCount": 2000,
    }

    try:
        resp = requests.get(CPD_CRIME_URL, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        if "features" in data and len(data["features"]) > 0:
            gdf = gpd.GeoDataFrame.from_features(data["features"], crs="EPSG:4326")
            gdf.to_file(cache_path, driver="GeoJSON")
            print(f"Fetched {len(gdf)} crime records from CPD.")
            return gdf
        else:
            print("No features returned from CPD API. Using sample data.")
            return _generate_sample_crimes()
    except Exception as e:
        print(f"CPD API error: {e}. Using sample data.")
        return _generate_sample_crimes()


def fetch_mu_emergency_phones(force: bool = False) -> gpd.GeoDataFrame:
    """Fetch emergency phone locations from MU ArcGIS."""
    cache_path = DATA_DIR / "emergency_phones.geojson"
    if cache_path.exists() and not force:
        print("Loading cached emergency phone data...")
        return gpd.read_file(cache_path)

    print("Fetching MU emergency phone locations...")
    params = {
        "where": "1=1",
        "outFields": "*",
        "outSR": "4326",
        "f": "geojson",
    }

    try:
        resp = requests.get(MU_EMERGENCY_PHONES_URL, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        if "features" in data and len(data["features"]) > 0:
            gdf = gpd.GeoDataFrame.from_features(data["features"], crs="EPSG:4326")
            gdf.to_file(cache_path, driver="GeoJSON")
            print(f"Fetched {len(gdf)} emergency phone locations.")
            return gdf
        else:
            print("No emergency phone data returned. Using sample data.")
            return _generate_sample_phones()
    except Exception as e:
        print(f"MU ArcGIS error: {e}. Using sample data.")
        return _generate_sample_phones()


def fetch_mu_buildings(force: bool = False) -> gpd.GeoDataFrame:
    """Fetch campus building footprints from MU ArcGIS."""
    cache_path = DATA_DIR / "buildings.geojson"
    if cache_path.exists() and not force:
        print("Loading cached building data...")
        return gpd.read_file(cache_path)

    print("Fetching MU building data...")
    params = {
        "where": "1=1",
        "outFields": "BUILDING_NAME,BUILDING_NUMBER",
        "outSR": "4326",
        "f": "geojson",
        "resultRecordCount": 500,
    }

    try:
        resp = requests.get(MU_BUILDINGS_URL, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        if "features" in data and len(data["features"]) > 0:
            gdf = gpd.GeoDataFrame.from_features(data["features"], crs="EPSG:4326")
            gdf.to_file(cache_path, driver="GeoJSON")
            print(f"Fetched {len(gdf)} building records.")
            return gdf
        else:
            print("No building data returned. Using sample data.")
            return _generate_sample_buildings()
    except Exception as e:
        print(f"MU ArcGIS buildings error: {e}. Using sample data.")
        return _generate_sample_buildings()


# ---------------------------------------------------------------------------
# Sample / fallback data generators (for when APIs are unavailable)
# ---------------------------------------------------------------------------

def _generate_sample_crimes() -> gpd.GeoDataFrame:
    """Generate realistic sample crime data around Mizzou campus."""
    rng = np.random.default_rng(42)
    n = 200

    categories = [
        ("Theft/Larceny", 0.3, False, 0.40),
        ("Burglary", 0.4, False, 0.10),
        ("Simple Assault", 0.6, True, 0.10),
        ("Vandalism", 0.2, False, 0.15),
        ("Drug Offense", 0.3, False, 0.10),
        ("Robbery", 0.9, True, 0.03),
        ("Aggravated Assault", 0.85, True, 0.05),
        ("Weapons Violation", 0.5, False, 0.02),
        ("Sexual Assault", 0.95, True, 0.03),
        ("DUI", 0.25, False, 0.02),
    ]

    names, severities, violents, probs = zip(*categories)
    probs = np.array(probs)
    probs /= probs.sum()

    cat_indices = rng.choice(len(categories), size=n, p=probs)
    lats = rng.normal(CAMPUS_CENTER[0], 0.005, size=n)
    lons = rng.normal(CAMPUS_CENTER[1], 0.007, size=n)
    dates = pd.date_range(end=pd.Timestamp.now(), periods=n, freq="16h")
    hours = rng.integers(0, 24, size=n)

    records = []
    for i in range(n):
        ci = cat_indices[i]
        records.append({
            "incident_id": f"SAMPLE-{i:04d}",
            "category": names[ci],
            "severity": severities[ci],
            "is_violent": violents[ci],
            "date_occurred": dates[i],
            "hour": int(hours[i]),
            "day_of_week": dates[i].day_name(),
            "source": "sample",
            "geometry": Point(lons[i], lats[i]),
        })

    gdf = gpd.GeoDataFrame(records, crs="EPSG:4326")
    gdf.to_file(DATA_DIR / "cpd_crimes.geojson", driver="GeoJSON")
    print(f"Generated {n} sample crime records.")
    return gdf


def _generate_sample_phones() -> gpd.GeoDataFrame:
    """Generate sample emergency phone locations on campus."""
    phones = [
        ("Phone - Memorial Union", 38.9407, -92.3280),
        ("Phone - Jesse Hall", 38.9395, -92.3290),
        ("Phone - Engineering", 38.9420, -92.3260),
        ("Phone - Rec Center", 38.9380, -92.3310),
        ("Phone - Library", 38.9410, -92.3305),
        ("Phone - Student Center", 38.9390, -92.3275),
        ("Phone - Speakers Circle", 38.9402, -92.3295),
        ("Phone - Carnahan Quad", 38.9415, -92.3288),
        ("Phone - Virginia Ave Garage", 38.9430, -92.3245),
        ("Phone - Hitt Street", 38.9398, -92.3270),
        ("Phone - Rollins Street", 38.9375, -92.3260),
        ("Phone - Tiger Ave", 38.9445, -92.3300),
    ]
    records = [
        {"name": name, "geometry": Point(lon, lat)}
        for name, lat, lon in phones
    ]
    gdf = gpd.GeoDataFrame(records, crs="EPSG:4326")
    gdf.to_file(DATA_DIR / "emergency_phones.geojson", driver="GeoJSON")
    print(f"Generated {len(gdf)} sample emergency phone locations.")
    return gdf


def _generate_sample_buildings() -> gpd.GeoDataFrame:
    """Generate sample campus building locations."""
    buildings = [
        ("Memorial Union", 38.9407, -92.3280),
        ("Jesse Hall", 38.9395, -92.3290),
        ("Ellis Library", 38.9410, -92.3305),
        ("Lafferre Hall", 38.9422, -92.3258),
        ("Engineering Building East", 38.9418, -92.3250),
        ("Student Recreation Complex", 38.9380, -92.3310),
        ("Mizzou Student Center", 38.9390, -92.3275),
        ("Strickland Hall", 38.9412, -92.3278),
        ("Tate Hall", 38.9416, -92.3292),
        ("Middlebush Hall", 38.9405, -92.3298),
        ("Switzler Hall", 38.9400, -92.3302),
        ("Neff Hall", 38.9408, -92.3265),
        ("Townsend Hall", 38.9425, -92.3270),
        ("Hearnes Center", 38.9365, -92.3255),
        ("Faurot Field", 38.9360, -92.3320),
        ("Mizzou Arena", 38.9370, -92.3335),
        ("Discovery Hall", 38.9430, -92.3240),
        ("Chemistry Building", 38.9415, -92.3275),
    ]
    records = [
        {"BUILDING_NAME": name, "geometry": Point(lon, lat)}
        for name, lat, lon in buildings
    ]
    gdf = gpd.GeoDataFrame(records, crs="EPSG:4326")
    gdf.to_file(DATA_DIR / "buildings.geojson", driver="GeoJSON")
    print(f"Generated {len(gdf)} sample building locations.")
    return gdf


def ingest_all(force: bool = False) -> dict:
    """Run full data ingestion pipeline. Returns dict of loaded data."""
    download_osm_network(force=force)
    crimes = fetch_cpd_crimes(force=force)
    phones = fetch_mu_emergency_phones(force=force)
    buildings = fetch_mu_buildings(force=force)

    return {
        "crimes": crimes,
        "phones": phones,
        "buildings": buildings,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Ingest data for Campus Dispatch Copilot")
    parser.add_argument("--force", action="store_true", help="Re-download all data")
    args = parser.parse_args()

    results = ingest_all(force=args.force)
    for name, gdf in results.items():
        print(f"  {name}: {len(gdf)} records")
    print("Data ingestion complete.")
