"""
Risk scoring engine for TigerSafe.

Computes comprehensive safety scores for routes by combining:
- Crime density along path (weighted by severity and recency)
- Time-of-day risk multipliers
- Emergency phone proximity
- Patrol frequency from traffic stop data
- Mode of travel adjustments
"""

from datetime import datetime

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import LineString, Point

from src.crime_analyzer import compute_crime_density_along_route

# Time-of-day risk periods and multipliers
TEMPORAL_PERIODS = {
    "late_night":    {"hours": range(0, 6),   "multiplier": 1.8, "label": "Late Night (12am-6am)"},
    "early_morning": {"hours": range(6, 9),   "multiplier": 0.5, "label": "Early Morning (6am-9am)"},
    "daytime":       {"hours": range(9, 17),  "multiplier": 0.3, "label": "Daytime (9am-5pm)"},
    "evening":       {"hours": range(17, 20), "multiplier": 0.6, "label": "Evening (5pm-8pm)"},
    "night":         {"hours": range(20, 22), "multiplier": 1.0, "label": "Night (8pm-10pm)"},
    "late_evening":  {"hours": range(22, 24), "multiplier": 1.5, "label": "Late Evening (10pm-12am)"},
}

# Risk adjustments for travel mode
MODE_MULTIPLIERS = {
    "walk": 1.3,   # Most vulnerable
    "bike": 1.0,   # Moderate
    "drive": 0.5,  # Most protected
}

# Weights for composite score
CRIME_WEIGHT = 0.45
TEMPORAL_WEIGHT = 0.25
INFRASTRUCTURE_WEIGHT = 0.15
RECENCY_WEIGHT = 0.15


def get_temporal_period(hour: int) -> dict:
    """Get the risk period info for a given hour."""
    for name, info in TEMPORAL_PERIODS.items():
        if hour in info["hours"]:
            return {"name": name, **info}
    return {"name": "night", "multiplier": 1.0, "label": "Night", "hours": range(20, 22)}


def get_temporal_multiplier(hour: int) -> float:
    """Get the risk multiplier for a given hour."""
    return get_temporal_period(hour)["multiplier"]


def count_emergency_phones_along_route(
    phones: gpd.GeoDataFrame,
    route_coords: list[tuple[float, float]],
    buffer_m: float = 200.0,
) -> int:
    """Count emergency phones within buffer distance of a route."""
    if phones is None or phones.empty or not route_coords:
        return 0

    line_coords = [(lon, lat) for lat, lon in route_coords]
    if len(line_coords) < 2:
        return 0

    route_line = LineString(line_coords)
    route_gdf = gpd.GeoDataFrame([{"geometry": route_line}], crs="EPSG:4326")
    route_proj = route_gdf.to_crs("EPSG:32615")
    phones_proj = phones.to_crs("EPSG:32615")

    buffer_zone = route_proj.geometry.iloc[0].buffer(buffer_m)
    return int(phones_proj.within(buffer_zone).sum())


def estimate_patrol_frequency(
    traffic_stops: pd.DataFrame,
    route_coords: list[tuple[float, float]],
    buffer_m: float = 500.0,
) -> dict:
    """Estimate police patrol frequency near a route from traffic stop data.

    Returns:
        Dict with patrol_level ("high"/"moderate"/"low"), stop_count,
        and time_distribution.
    """
    if traffic_stops is None or traffic_stops.empty or not route_coords:
        return {"patrol_level": "unknown", "stop_count": 0, "time_distribution": {}}

    if "lat" not in traffic_stops.columns or "lon" not in traffic_stops.columns:
        return {"patrol_level": "unknown", "stop_count": 0, "time_distribution": {}}

    stops_with_coords = traffic_stops.dropna(subset=["lat", "lon"])
    if stops_with_coords.empty:
        return {"patrol_level": "unknown", "stop_count": 0, "time_distribution": {}}

    # Build route centroid for approximate area match
    lats = [c[0] for c in route_coords]
    lons = [c[1] for c in route_coords]
    center_lat = np.mean(lats)
    center_lon = np.mean(lons)

    # Approximate distance filter using haversine
    R = 6371000
    lat_r = np.radians(center_lat)
    stop_lats = stops_with_coords["lat"].values
    stop_lons = stops_with_coords["lon"].values

    dlat = np.radians(stop_lats - center_lat)
    dlon = np.radians(stop_lons - center_lon)
    a = np.sin(dlat / 2) ** 2 + np.cos(lat_r) * np.cos(np.radians(stop_lats)) * np.sin(dlon / 2) ** 2
    distances = 2 * R * np.arcsin(np.sqrt(np.clip(a, 0, 1)))

    nearby = stops_with_coords[distances <= buffer_m]
    count = len(nearby)

    # Time distribution
    time_dist = {}
    if "hour_stop" in nearby.columns:
        time_dist = nearby["hour_stop"].dropna().astype(int).value_counts().sort_index().to_dict()

    # Classify patrol level (thresholds based on typical campus area)
    if count >= 50:
        level = "high"
    elif count >= 15:
        level = "moderate"
    else:
        level = "low"

    return {
        "patrol_level": level,
        "stop_count": count,
        "time_distribution": time_dist,
    }


def score_route(
    route: dict,
    crimes: gpd.GeoDataFrame,
    phones: gpd.GeoDataFrame,
    traffic_stops: pd.DataFrame = None,
    hour: int = None,
    mode: str = "walk",
) -> dict:
    """Compute a comprehensive risk score for a route.

    Args:
        route: Route dict with "coordinates" key (list of (lat, lon)).
        crimes: Unified crime GeoDataFrame.
        phones: Emergency phone GeoDataFrame.
        traffic_stops: Traffic stop DataFrame (optional).
        hour: Hour of day (0-23). Defaults to current hour.
        mode: Travel mode ("walk", "bike", "drive").

    Returns:
        Dict with overall score (0-100), level, color, and detailed breakdown.
    """
    if hour is None:
        hour = datetime.now().hour

    coords = route.get("coordinates", [])
    distance_m = route.get("distance_m", 0)

    # 1. Crime analysis along route
    crime_stats = compute_crime_density_along_route(crimes, coords, buffer_m=200)
    total_crimes = crime_stats["total_crimes"]
    violent_crimes = crime_stats["violent_crimes"]
    avg_severity = crime_stats["avg_severity"]
    recent_30d = crime_stats["recent_crimes_30d"]
    recent_7d = crime_stats["recent_crimes_7d"]

    # Crime score: base + severity + violent bonus
    # Normalized per 100m of route
    length_factor = max(distance_m / 100.0, 1.0)
    crime_base = (total_crimes * 8 + violent_crimes * 15) / length_factor
    crime_score = min(crime_base * (1 + avg_severity), 100)

    # 2. Temporal risk
    temporal = get_temporal_period(hour)
    temporal_mult = temporal["multiplier"]

    # 3. Recency weighting (more recent = higher risk)
    recency_score = 0
    if recent_7d > 0:
        recency_score = min(recent_7d * 5, 30)
    elif recent_30d > 0:
        recency_score = min(recent_30d * 2, 20)

    # 4. Infrastructure (emergency phones)
    phone_count = count_emergency_phones_along_route(phones, coords)
    phone_reduction = min(phone_count * 3, 15)

    # 5. Patrol frequency
    patrol_info = estimate_patrol_frequency(traffic_stops, coords)
    patrol_reduction = {"high": 8, "moderate": 3, "low": 0, "unknown": 0}.get(
        patrol_info["patrol_level"], 0
    )

    # 6. Mode adjustment
    mode_mult = MODE_MULTIPLIERS.get(mode, 1.0)

    # Composite score
    raw_score = (
        crime_score * CRIME_WEIGHT
        + (temporal_mult * 30) * TEMPORAL_WEIGHT
        + recency_score * RECENCY_WEIGHT
        - phone_reduction * INFRASTRUCTURE_WEIGHT
        - patrol_reduction * INFRASTRUCTURE_WEIGHT
    ) * mode_mult

    final_score = max(0, min(100, raw_score))

    # Classification
    if final_score <= 20:
        level, color = "Very Safe", "#22c55e"
    elif final_score <= 40:
        level, color = "Safe", "#3b82f6"
    elif final_score <= 60:
        level, color = "Moderate Risk", "#eab308"
    elif final_score <= 80:
        level, color = "Higher Risk", "#f97316"
    else:
        level, color = "High Risk", "#ef4444"

    return {
        "score": round(final_score, 1),
        "level": level,
        "color": color,
        "breakdown": {
            "crime_score": round(crime_score, 1),
            "temporal_period": temporal["label"],
            "temporal_multiplier": temporal_mult,
            "recency_score": recency_score,
            "emergency_phones_nearby": phone_count,
            "phone_reduction": phone_reduction,
            "patrol_level": patrol_info["patrol_level"],
            "patrol_reduction": patrol_reduction,
            "mode": mode,
            "mode_multiplier": mode_mult,
        },
        "crime_stats": crime_stats,
    }


def compare_routes(
    routes: list[dict],
    crimes: gpd.GeoDataFrame,
    phones: gpd.GeoDataFrame,
    traffic_stops: pd.DataFrame = None,
    hour: int = None,
    mode: str = "walk",
) -> list[dict]:
    """Score and compare multiple routes.

    Returns routes sorted by risk score (safest first), each
    augmented with risk_score and crime_stats.
    """
    scored = []
    for route in routes:
        risk = score_route(route, crimes, phones, traffic_stops, hour, mode)
        route_copy = dict(route)
        route_copy["risk_score"] = risk
        route_copy["crime_stats"] = risk["crime_stats"]
        scored.append(route_copy)

    # Sort by risk score (safest first)
    scored.sort(key=lambda r: r["risk_score"]["score"])

    # Add relative labels
    if scored:
        scored[0]["recommendation"] = "Safest Route"
    for i, r in enumerate(scored):
        if i > 0:
            fastest = min(scored, key=lambda x: x.get("estimated_time_min", float("inf")))
            if r is fastest:
                r["recommendation"] = "Fastest Route"
            else:
                r["recommendation"] = f"Alternative {i}"

    return scored
