"""
Safety scoring and routing engine for Campus Dispatch Copilot.

Implements:
- Crime density calculation per network edge
- Temporal risk multipliers
- Emergency infrastructure scoring
- Composite safety-weighted shortest paths
- Risk score breakdown for explainability
"""

from datetime import datetime
from pathlib import Path

import geopandas as gpd
import networkx as nx
import numpy as np
import osmnx as ox
import pandas as pd
from shapely.geometry import LineString, Point
from sklearn.neighbors import KernelDensity

DATA_DIR = Path(__file__).parent / "data"

# Routing weight parameters (from design doc)
ALPHA = 0.4  # distance weight
BETA = 0.4   # crime density weight
GAMMA = 0.2  # lighting/infrastructure weight

# Temporal multipliers (from data_analysis.md)
TEMPORAL_MULTIPLIERS = {
    "late_night": (0, 6, 1.8),     # midnight–6am
    "early_morning": (6, 9, 0.5),  # 6am–9am
    "daytime": (9, 17, 0.3),       # 9am–5pm
    "evening": (17, 20, 0.6),      # 5pm–8pm
    "night": (20, 22, 1.0),        # 8pm–10pm
    "late_evening": (22, 24, 1.5), # 10pm–midnight
}

# Risk score parameters (from design doc)
INCIDENT_BASE_SCORE = 10
RECENCY_WEIGHTS = {30: 5.0, 90: 2.0, 180: 1.0}  # days: multiplier
EMERGENCY_PHONE_REDUCTION = -5  # per phone, max -15
NIGHTTIME_MULTIPLIER = 2.0
HIGH_PATROL_REDUCTION = -10


def get_temporal_multiplier(hour: int) -> float:
    """Get risk multiplier based on time of day."""
    for _, (start, end, mult) in TEMPORAL_MULTIPLIERS.items():
        if start <= hour < end:
            return mult
    return 1.0


def load_graph():
    """Load the OSM walking network graph."""
    graph_path = DATA_DIR / "columbia_walk.graphml"
    if not graph_path.exists():
        raise FileNotFoundError(
            "OSM graph not found. Run `python ingest.py` first."
        )
    return ox.load_graphml(graph_path)


def load_crimes() -> gpd.GeoDataFrame:
    """Load crime data."""
    path = DATA_DIR / "cpd_crimes.geojson"
    if not path.exists():
        raise FileNotFoundError(
            "Crime data not found. Run `python ingest.py` first."
        )
    return gpd.read_file(path)


def load_emergency_phones() -> gpd.GeoDataFrame:
    """Load emergency phone locations."""
    path = DATA_DIR / "emergency_phones.geojson"
    if not path.exists():
        return gpd.GeoDataFrame(columns=["name", "geometry"], crs="EPSG:4326")
    return gpd.read_file(path)


def compute_edge_crime_density(
    G: nx.MultiDiGraph,
    crimes: gpd.GeoDataFrame,
    buffer_m: float = 100.0,
) -> nx.MultiDiGraph:
    """Assign crime density scores to each edge in the graph.

    For each edge, counts crimes within `buffer_m` meters and computes
    density = count / edge_length_m.
    """
    # Project to UTM for metric operations
    crimes_proj = crimes.to_crs("EPSG:32615")

    # Extract edge geometries
    edges = ox.graph_to_gdfs(G, nodes=False)
    edges_proj = edges.to_crs("EPSG:32615")

    for idx in edges_proj.index:
        edge_geom = edges_proj.loc[idx, "geometry"]
        buffer_zone = edge_geom.buffer(buffer_m)
        crime_count = crimes_proj.within(buffer_zone).sum()
        length_m = edge_geom.length
        density = crime_count / max(length_m, 1.0)

        # Count violent crimes specifically
        violent_mask = crimes_proj.within(buffer_zone)
        if "is_violent" in crimes_proj.columns:
            violent_count = (
                crimes_proj.loc[violent_mask, "is_violent"]
                .astype(bool)
                .sum()
            )
        else:
            violent_count = 0

        u, v, k = idx
        G[u][v][k]["crime_count"] = int(crime_count)
        G[u][v][k]["crime_density"] = float(density)
        G[u][v][k]["violent_crime_count"] = int(violent_count)

    return G


def compute_edge_phone_score(
    G: nx.MultiDiGraph,
    phones: gpd.GeoDataFrame,
    radius_m: float = 200.0,
) -> nx.MultiDiGraph:
    """Score edges by proximity to emergency phones."""
    if phones.empty:
        for u, v, k in G.edges(keys=True):
            G[u][v][k]["phone_score"] = 0.0
        return G

    phones_proj = phones.to_crs("EPSG:32615")
    edges = ox.graph_to_gdfs(G, nodes=False)
    edges_proj = edges.to_crs("EPSG:32615")

    for idx in edges_proj.index:
        edge_center = edges_proj.loc[idx, "geometry"].centroid
        # Count phones within radius
        distances = phones_proj.geometry.distance(edge_center)
        nearby_count = (distances <= radius_m).sum()
        # Score: more phones = safer (0 to 1)
        score = min(nearby_count / 3.0, 1.0)
        u, v, k = idx
        G[u][v][k]["phone_score"] = float(score)

    return G


def compute_safety_weights(
    G: nx.MultiDiGraph,
    hour: int = None,
    alpha: float = ALPHA,
    beta: float = BETA,
    gamma: float = GAMMA,
) -> nx.MultiDiGraph:
    """Compute composite safety weight for each edge.

    weight = alpha * length + beta * (crime_density * temporal_mult) + gamma * (1 - phone_score)
    """
    if hour is None:
        hour = datetime.now().hour

    temporal_mult = get_temporal_multiplier(hour)

    for u, v, k, data in G.edges(keys=True, data=True):
        length = data.get("length", 100.0)
        crime_density = data.get("crime_density", 0.0)
        phone_score = data.get("phone_score", 0.0)

        # Normalize length to 0-1 range (assume max edge ~500m)
        norm_length = min(length / 500.0, 1.0)

        safety_weight = (
            alpha * norm_length
            + beta * (crime_density * temporal_mult)
            + gamma * (1.0 - phone_score)
        )

        G[u][v][k]["safety_weight"] = max(safety_weight, 0.001)

    return G


def find_safest_route(
    G: nx.MultiDiGraph,
    origin: tuple[float, float],
    destination: tuple[float, float],
    hour: int = None,
) -> dict:
    """Find the safest walking route between two points.

    Args:
        G: The weighted graph (must have safety_weight on edges).
        origin: (lat, lon) tuple.
        destination: (lat, lon) tuple.

    Returns:
        dict with route geometry, distance, duration, safety score, etc.
    """
    orig_node = ox.nearest_nodes(G, origin[1], origin[0])
    dest_node = ox.nearest_nodes(G, destination[1], destination[0])

    try:
        route_nodes = nx.shortest_path(
            G, orig_node, dest_node, weight="safety_weight"
        )
    except nx.NetworkXNoPath:
        return {"error": "No path found between the given locations."}

    # Gather route stats
    total_length = 0
    total_crime_count = 0
    total_violent = 0
    total_phone_score = 0
    edge_count = 0
    coords = []

    nodes_gdf = ox.graph_to_gdfs(G, edges=False)

    for i in range(len(route_nodes) - 1):
        u, v = route_nodes[i], route_nodes[i + 1]
        # Get best edge (lowest safety weight)
        edge_data = min(
            G[u][v].values(), key=lambda d: d.get("safety_weight", float("inf"))
        )
        total_length += edge_data.get("length", 0)
        total_crime_count += edge_data.get("crime_count", 0)
        total_violent += edge_data.get("violent_crime_count", 0)
        total_phone_score += edge_data.get("phone_score", 0)
        edge_count += 1

    # Build coordinate list from node positions
    for node_id in route_nodes:
        node = nodes_gdf.loc[node_id]
        coords.append((node.geometry.y, node.geometry.x))  # (lat, lon)

    # Calculate average safety metrics
    avg_phone_score = total_phone_score / max(edge_count, 1)
    walking_speed_mps = 1.4  # ~5 km/h
    duration_min = (total_length / walking_speed_mps) / 60

    # Risk score calculation (from design doc)
    risk_score = _calculate_risk_score(
        crime_count=total_crime_count,
        violent_count=total_violent,
        phone_score=avg_phone_score,
        hour=hour or datetime.now().hour,
        route_length_m=total_length,
    )

    return {
        "coordinates": coords,
        "distance_m": round(total_length, 1),
        "duration_min": round(duration_min, 1),
        "risk_score": risk_score,
        "crime_count_nearby": total_crime_count,
        "violent_crimes_nearby": total_violent,
        "emergency_phones_score": round(avg_phone_score, 2),
        "num_edges": edge_count,
        "origin": origin,
        "destination": destination,
    }


def find_fastest_route(
    G: nx.MultiDiGraph,
    origin: tuple[float, float],
    destination: tuple[float, float],
    hour: int = None,
) -> dict:
    """Find the shortest (fastest) walking route by distance."""
    orig_node = ox.nearest_nodes(G, origin[1], origin[0])
    dest_node = ox.nearest_nodes(G, destination[1], destination[0])

    try:
        route_nodes = nx.shortest_path(G, orig_node, dest_node, weight="length")
    except nx.NetworkXNoPath:
        return {"error": "No path found between the given locations."}

    total_length = 0
    total_crime_count = 0
    total_violent = 0
    total_phone_score = 0
    edge_count = 0
    coords = []

    nodes_gdf = ox.graph_to_gdfs(G, edges=False)

    for i in range(len(route_nodes) - 1):
        u, v = route_nodes[i], route_nodes[i + 1]
        edge_data = min(
            G[u][v].values(), key=lambda d: d.get("length", float("inf"))
        )
        total_length += edge_data.get("length", 0)
        total_crime_count += edge_data.get("crime_count", 0)
        total_violent += edge_data.get("violent_crime_count", 0)
        total_phone_score += edge_data.get("phone_score", 0)
        edge_count += 1

    for node_id in route_nodes:
        node = nodes_gdf.loc[node_id]
        coords.append((node.geometry.y, node.geometry.x))

    avg_phone_score = total_phone_score / max(edge_count, 1)
    walking_speed_mps = 1.4
    duration_min = (total_length / walking_speed_mps) / 60

    risk_score = _calculate_risk_score(
        crime_count=total_crime_count,
        violent_count=total_violent,
        phone_score=avg_phone_score,
        hour=hour or datetime.now().hour,
        route_length_m=total_length,
    )

    return {
        "coordinates": coords,
        "distance_m": round(total_length, 1),
        "duration_min": round(duration_min, 1),
        "risk_score": risk_score,
        "crime_count_nearby": total_crime_count,
        "violent_crimes_nearby": total_violent,
        "emergency_phones_score": round(avg_phone_score, 2),
        "num_edges": edge_count,
        "origin": origin,
        "destination": destination,
    }


def _calculate_risk_score(
    crime_count: int,
    violent_count: int,
    phone_score: float,
    hour: int,
    route_length_m: float,
) -> dict:
    """Calculate risk score following the design doc algorithm.

    Returns dict with overall score, level, and breakdown.
    """
    # Base score: 10 per incident
    base = crime_count * INCIDENT_BASE_SCORE

    # Violent crime bonus
    violent_bonus = violent_count * 15

    # Time-of-day multiplier
    is_night = hour >= 22 or hour < 6
    time_mult = NIGHTTIME_MULTIPLIER if is_night else 1.0

    # Infrastructure adjustments
    phone_reduction = min(phone_score * 3, 1.0) * EMERGENCY_PHONE_REDUCTION * -1
    phone_adj = -phone_reduction  # more phones = lower score

    # Normalize by route length (per 100m)
    length_factor = max(route_length_m / 100.0, 1.0)

    raw_score = ((base + violent_bonus) * time_mult + phone_adj) / length_factor
    final_score = max(0, min(100, raw_score))

    # Classification
    if final_score <= 20:
        level = "Very Safe"
        color = "green"
    elif final_score <= 40:
        level = "Safe"
        color = "blue"
    elif final_score <= 60:
        level = "Moderate Risk"
        color = "orange"
    elif final_score <= 80:
        level = "Higher Risk"
        color = "red"
    else:
        level = "High Risk"
        color = "darkred"

    return {
        "score": round(final_score, 1),
        "level": level,
        "color": color,
        "breakdown": {
            "base_crime_score": base,
            "violent_crime_bonus": violent_bonus,
            "time_multiplier": time_mult,
            "phone_adjustment": round(phone_adj, 1),
            "length_normalization_factor": round(length_factor, 1),
            "is_nighttime": is_night,
        },
    }


def compute_crime_heatmap_data(
    crimes: gpd.GeoDataFrame,
    hour_filter: int = None,
) -> list[list[float]]:
    """Compute heatmap data points for Folium.

    Returns list of [lat, lon, weight] triples.
    """
    if crimes.empty:
        return []

    if hour_filter is not None and "hour" in crimes.columns:
        # Filter to +/- 2 hours of target
        mask = crimes["hour"].apply(
            lambda h: min(abs(h - hour_filter), 24 - abs(h - hour_filter)) <= 2
        )
        filtered = crimes[mask]
    else:
        filtered = crimes

    if filtered.empty:
        return []

    # Weight by severity if available
    points = []
    for _, row in filtered.iterrows():
        lat = row.geometry.y
        lon = row.geometry.x
        weight = row.get("severity", 0.5) if "severity" in filtered.columns else 0.5
        points.append([lat, lon, float(weight)])

    return points


def get_nearby_crimes(
    crimes: gpd.GeoDataFrame,
    lat: float,
    lon: float,
    radius_m: float = 500,
) -> gpd.GeoDataFrame:
    """Get crimes within radius_m of a point."""
    point = gpd.GeoSeries([Point(lon, lat)], crs="EPSG:4326").to_crs("EPSG:32615")
    crimes_proj = crimes.to_crs("EPSG:32615")
    distances = crimes_proj.geometry.distance(point.iloc[0])
    mask = distances <= radius_m
    result = crimes[mask].copy()
    result["distance_m"] = distances[mask].values
    return result.sort_values("distance_m")


def prepare_graph(hour: int = None) -> nx.MultiDiGraph:
    """Full pipeline: load graph, compute all weights, return ready graph."""
    G = load_graph()
    crimes = load_crimes()
    phones = load_emergency_phones()

    G = compute_edge_crime_density(G, crimes)
    G = compute_edge_phone_score(G, phones)
    G = compute_safety_weights(G, hour=hour)

    return G
