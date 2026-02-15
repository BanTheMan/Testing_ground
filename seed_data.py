"""
Seed data generator for Campus Dispatch Copilot.

Generates all sample data locally so the app can run without network access
or API keys for the data layer. The AI chat still requires an Anthropic API key.

Usage:
    python seed_data.py
"""

from pathlib import Path

import geopandas as gpd
import numpy as np
import osmnx as ox
import pandas as pd
from shapely.geometry import Point

DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)

# Campus bounds
CAMPUS_CENTER = (38.9404, -92.3277)
BBOX_NORTH = 38.955
BBOX_SOUTH = 38.925
BBOX_EAST = -92.305
BBOX_WEST = -92.345


def seed_osm_network():
    """Download the OSM walking network (requires internet once)."""
    graph_path = DATA_DIR / "columbia_walk.graphml"
    if graph_path.exists():
        print("OSM network already exists, skipping.")
        return

    print("Downloading OSM walking network...")
    G = ox.graph_from_bbox(
        bbox=(BBOX_NORTH, BBOX_SOUTH, BBOX_EAST, BBOX_WEST),
        network_type="walk",
    )
    ox.save_graphml(G, graph_path)
    print(f"Saved: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")


def seed_crime_data():
    """Generate realistic sample crime data."""
    path = DATA_DIR / "cpd_crimes.geojson"
    if path.exists():
        print("Crime data already exists, skipping.")
        return

    print("Generating sample crime data...")
    rng = np.random.default_rng(42)
    n = 250

    categories = [
        ("Theft/Larceny", 0.3, False, 0.35),
        ("Burglary", 0.4, False, 0.08),
        ("Simple Assault", 0.6, True, 0.10),
        ("Vandalism", 0.2, False, 0.15),
        ("Drug Offense", 0.3, False, 0.10),
        ("Robbery", 0.9, True, 0.04),
        ("Aggravated Assault", 0.85, True, 0.05),
        ("Weapons Violation", 0.5, False, 0.02),
        ("Sexual Assault", 0.95, True, 0.03),
        ("DUI", 0.25, False, 0.03),
        ("Trespass", 0.15, False, 0.03),
        ("Fraud", 0.2, False, 0.02),
    ]

    names, severities, violents, probs = zip(*categories)
    probs = np.array(probs)
    probs /= probs.sum()

    cat_indices = rng.choice(len(categories), size=n, p=probs)

    # Cluster crimes in realistic hotspots
    hotspots = [
        (38.9470, -92.3290, 0.25),  # Downtown/Broadway
        (38.9407, -92.3280, 0.15),  # Memorial Union area
        (38.9380, -92.3260, 0.10),  # South campus
        (38.9440, -92.3310, 0.15),  # Providence/campus edge
        (38.9360, -92.3320, 0.10),  # Stadium area
        (38.9420, -92.3240, 0.10),  # East campus
        (38.9395, -92.3310, 0.15),  # Rec center area
    ]

    lats = []
    lons = []
    for _ in range(n):
        # Pick a hotspot
        h_probs = np.array([h[2] for h in hotspots])
        h_probs /= h_probs.sum()
        h_idx = rng.choice(len(hotspots), p=h_probs)
        h_lat, h_lon, _ = hotspots[h_idx]
        lats.append(rng.normal(h_lat, 0.003))
        lons.append(rng.normal(h_lon, 0.004))

    dates = pd.date_range(end=pd.Timestamp.now(), periods=n, freq="12h")
    hours = rng.integers(0, 24, size=n)
    # Bias hours toward night for violent crimes
    for i in range(n):
        if violents[cat_indices[i]]:
            hours[i] = rng.choice([21, 22, 23, 0, 1, 2, 3])

    records = []
    for i in range(n):
        ci = cat_indices[i]
        records.append({
            "incident_id": f"SEED-{i:04d}",
            "category": names[ci],
            "severity": severities[ci],
            "is_violent": violents[ci],
            "date_occurred": str(dates[i])[:19],
            "hour": int(hours[i]),
            "day_of_week": dates[i].day_name(),
            "source": "seed_data",
            "geometry": Point(lons[i], lats[i]),
        })

    gdf = gpd.GeoDataFrame(records, crs="EPSG:4326")
    gdf.to_file(path, driver="GeoJSON")
    print(f"Generated {n} crime records.")


def seed_emergency_phones():
    """Generate emergency phone locations."""
    path = DATA_DIR / "emergency_phones.geojson"
    if path.exists():
        print("Emergency phone data already exists, skipping.")
        return

    print("Generating emergency phone locations...")
    phones = [
        ("Blue Light - Memorial Union", 38.9407, -92.3280),
        ("Blue Light - Jesse Hall", 38.9395, -92.3290),
        ("Blue Light - Engineering East", 38.9420, -92.3260),
        ("Blue Light - Rec Center", 38.9380, -92.3310),
        ("Blue Light - Ellis Library", 38.9410, -92.3305),
        ("Blue Light - Student Center", 38.9390, -92.3275),
        ("Blue Light - Speakers Circle", 38.9402, -92.3295),
        ("Blue Light - Carnahan Quad", 38.9415, -92.3288),
        ("Blue Light - Virginia Ave Garage", 38.9430, -92.3245),
        ("Blue Light - Hitt Street", 38.9398, -92.3270),
        ("Blue Light - Rollins & College", 38.9375, -92.3260),
        ("Blue Light - Tiger Avenue", 38.9445, -92.3300),
        ("Blue Light - Research Park", 38.9450, -92.3250),
        ("Blue Light - Conley Ave", 38.9388, -92.3248),
        ("Blue Light - Hospital Drive", 38.9355, -92.3295),
        ("Blue Light - Stadium Blvd", 38.9360, -92.3335),
    ]

    records = [
        {"name": name, "geometry": Point(lon, lat)}
        for name, lat, lon in phones
    ]
    gdf = gpd.GeoDataFrame(records, crs="EPSG:4326")
    gdf.to_file(path, driver="GeoJSON")
    print(f"Generated {len(gdf)} emergency phone locations.")


def seed_buildings():
    """Generate campus building locations."""
    path = DATA_DIR / "buildings.geojson"
    if path.exists():
        print("Building data already exists, skipping.")
        return

    print("Generating building locations...")
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
        ("Geological Sciences", 38.9418, -92.3285),
        ("Tucker Hall", 38.9413, -92.3295),
        ("Schweitzer Hall", 38.9408, -92.3288),
        ("Stewart Hall", 38.9392, -92.3285),
        ("Gentry Hall", 38.9397, -92.3305),
        ("Waters Hall", 38.9435, -92.3265),
    ]

    records = [
        {"BUILDING_NAME": name, "geometry": Point(lon, lat)}
        for name, lat, lon in buildings
    ]
    gdf = gpd.GeoDataFrame(records, crs="EPSG:4326")
    gdf.to_file(path, driver="GeoJSON")
    print(f"Generated {len(gdf)} building locations.")


def seed_all():
    """Generate all seed data."""
    print("=" * 50)
    print("Campus Dispatch Copilot — Seed Data Generator")
    print("=" * 50)

    seed_osm_network()
    seed_crime_data()
    seed_emergency_phones()
    seed_buildings()

    print("=" * 50)
    print("All seed data generated successfully!")
    print(f"Data directory: {DATA_DIR.resolve()}")
    print("=" * 50)


if __name__ == "__main__":
    seed_all()
