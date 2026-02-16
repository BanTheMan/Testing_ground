"""
Crime analysis engine for TigerSafe.

Handles:
- NIBRS-based crime category normalization across CPD/MUPD sources
- Severity scoring calibrated for pedestrian safety
- Temporal analysis (time of day, day of week patterns)
- Spatial crime density along routes
- Recency weighting for risk calculations
- Crime type breakdown for specific areas
"""

from datetime import datetime, timedelta

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import LineString, Point

# Unified crime taxonomy with severity weights for pedestrian safety.
# Maps keyword fragments from NIBRS/MUPD descriptions to categories.
_CATEGORY_MAP = [
    # (keywords, unified_category, severity, is_violent)
    (["homicide", "murder", "manslaughter"], "Homicide", 1.0, True),
    (["sexual assault", "rape", "fondling", "sex offense", "sexual abuse"], "Sexual Assault", 0.95, True),
    (["robbery"], "Robbery", 0.9, True),
    (["aggravated assault", "assault 1st", "assault-agg"], "Aggravated Assault", 0.85, True),
    (["simple assault", "assault 3rd", "assault-simple", "assault-"], "Simple Assault", 0.6, True),
    (["kidnap", "abduction"], "Kidnapping", 0.9, True),
    (["arson"], "Arson", 0.7, True),
    (["burglary", "breaking and entering", "b&e"], "Burglary", 0.4, False),
    (["motor vehicle theft", "auto theft", "stolen vehicle"], "Motor Vehicle Theft", 0.35, False),
    (["theft", "larceny", "stealing", "shoplifting"], "Theft/Larceny", 0.3, False),
    (["vandalism", "destruction", "damage", "property damage", "criminal mischief"], "Vandalism", 0.2, False),
    (["drug", "narcotic", "marijuana", "controlled substance"], "Drug Offense", 0.3, False),
    (["weapon", "firearm", "gun"], "Weapons Violation", 0.5, False),
    (["dui", "dwi", "driving under", "driving while"], "DUI", 0.25, False),
    (["trespass"], "Trespass", 0.15, False),
    (["fraud", "forgery", "counterfeit", "identity theft"], "Fraud", 0.2, False),
    (["disorderly", "disturbance", "peace"], "Disorderly Conduct", 0.15, False),
    (["harassment", "stalking", "intimidation"], "Harassment", 0.5, True),
    (["crash", "accident", "traffic"], "Traffic Incident", 0.1, False),
]


def normalize_crime_category(
    description: str, source: str = "cpd"
) -> dict:
    """Map a crime description to a unified category with severity.

    Args:
        description: Raw crime description from CPD or MUPD.
        source: Data source identifier ("cpd" or "mupd").

    Returns:
        Dict with keys: category, severity, is_violent.
    """
    if not description or not isinstance(description, str):
        return {"category": "Other", "severity": 0.2, "is_violent": False}

    desc_lower = description.lower().strip()

    for keywords, category, severity, is_violent in _CATEGORY_MAP:
        for kw in keywords:
            if kw in desc_lower:
                return {
                    "category": category,
                    "severity": severity,
                    "is_violent": is_violent,
                }

    return {"category": "Other", "severity": 0.2, "is_violent": False}


def compute_crime_density_along_route(
    crimes: gpd.GeoDataFrame,
    route_coords: list[tuple[float, float]],
    buffer_m: float = 200.0,
) -> dict:
    """Compute crime statistics along a route.

    Args:
        crimes: GeoDataFrame with crime points.
        route_coords: List of (lat, lon) tuples forming the route.
        buffer_m: Buffer distance in meters around the route.

    Returns:
        Dict with crime counts, breakdown by category, severity stats.
    """
    if crimes.empty or not route_coords:
        return {
            "total_crimes": 0,
            "violent_crimes": 0,
            "by_category": {},
            "avg_severity": 0.0,
            "recent_crimes_30d": 0,
            "recent_crimes_7d": 0,
        }

    # Build route LineString (lon, lat for shapely)
    line_coords = [(lon, lat) for lat, lon in route_coords]
    if len(line_coords) < 2:
        return {
            "total_crimes": 0,
            "violent_crimes": 0,
            "by_category": {},
            "avg_severity": 0.0,
            "recent_crimes_30d": 0,
            "recent_crimes_7d": 0,
        }

    route_line = LineString(line_coords)
    route_gdf = gpd.GeoDataFrame(
        [{"geometry": route_line}], crs="EPSG:4326"
    )

    # Project to UTM for metric buffer
    route_proj = route_gdf.to_crs("EPSG:32615")
    crimes_proj = crimes.to_crs("EPSG:32615")
    buffer_zone = route_proj.geometry.iloc[0].buffer(buffer_m)

    # Find crimes within buffer
    mask = crimes_proj.within(buffer_zone)
    nearby = crimes[mask].copy()

    if nearby.empty:
        return {
            "total_crimes": 0,
            "violent_crimes": 0,
            "by_category": {},
            "avg_severity": 0.0,
            "recent_crimes_30d": 0,
            "recent_crimes_7d": 0,
        }

    # Category breakdown
    by_category = {}
    if "category" in nearby.columns:
        by_category = nearby["category"].value_counts().to_dict()

    # Violent count
    violent_count = 0
    if "is_violent" in nearby.columns:
        violent_count = int(nearby["is_violent"].astype(bool).sum())

    # Average severity
    avg_severity = 0.0
    if "severity" in nearby.columns:
        avg_severity = float(nearby["severity"].mean())

    # Recency counts
    now = pd.Timestamp.now()
    recent_30d = 0
    recent_7d = 0
    if "report_date" in nearby.columns:
        dates = pd.to_datetime(nearby["report_date"], errors="coerce")
        recent_30d = int((dates >= now - pd.Timedelta(days=30)).sum())
        recent_7d = int((dates >= now - pd.Timedelta(days=7)).sum())

    return {
        "total_crimes": len(nearby),
        "violent_crimes": violent_count,
        "by_category": by_category,
        "avg_severity": round(avg_severity, 3),
        "recent_crimes_30d": recent_30d,
        "recent_crimes_7d": recent_7d,
    }


def compute_temporal_crime_pattern(
    crimes: gpd.GeoDataFrame,
    lat: float,
    lon: float,
    radius_m: float = 500.0,
) -> dict:
    """Analyze crime patterns by time of day and day of week near a point.

    Returns hour-of-day and day-of-week distributions for nearby crimes.
    """
    if crimes.empty:
        return {"by_hour": {}, "by_day": {}, "peak_hours": [], "peak_days": []}

    point = gpd.GeoSeries([Point(lon, lat)], crs="EPSG:4326").to_crs("EPSG:32615")
    crimes_proj = crimes.to_crs("EPSG:32615")
    dists = crimes_proj.geometry.distance(point.iloc[0])
    nearby = crimes[dists <= radius_m]

    if nearby.empty:
        return {"by_hour": {}, "by_day": {}, "peak_hours": [], "peak_days": []}

    by_hour = {}
    if "hour" in nearby.columns:
        by_hour = nearby["hour"].dropna().astype(int).value_counts().sort_index().to_dict()

    by_day = {}
    if "day_of_week" in nearby.columns:
        by_day = nearby["day_of_week"].value_counts().to_dict()

    # Identify peak hours (above average)
    peak_hours = []
    if by_hour:
        avg_count = np.mean(list(by_hour.values()))
        peak_hours = [h for h, c in by_hour.items() if c > avg_count * 1.5]

    peak_days = []
    if by_day:
        avg_count = np.mean(list(by_day.values()))
        peak_days = [d for d, c in by_day.items() if c > avg_count * 1.3]

    return {
        "by_hour": by_hour,
        "by_day": by_day,
        "peak_hours": peak_hours,
        "peak_days": peak_days,
    }


def get_recent_incidents_near(
    crimes: gpd.GeoDataFrame,
    lat: float,
    lon: float,
    radius_m: float = 500.0,
    limit: int = 10,
) -> list[dict]:
    """Get the most recent crime incidents near a location.

    Returns list of incident dicts sorted by date (most recent first).
    """
    if crimes.empty:
        return []

    point = gpd.GeoSeries([Point(lon, lat)], crs="EPSG:4326").to_crs("EPSG:32615")
    crimes_proj = crimes.to_crs("EPSG:32615")
    dists = crimes_proj.geometry.distance(point.iloc[0])
    mask = dists <= radius_m
    nearby = crimes[mask].copy()
    nearby["distance_m"] = dists[mask].values

    if nearby.empty:
        return []

    if "report_date" in nearby.columns:
        nearby = nearby.sort_values("report_date", ascending=False)

    results = []
    for _, row in nearby.head(limit).iterrows():
        results.append({
            "category": row.get("category", "Unknown"),
            "severity": float(row.get("severity", 0.2)),
            "is_violent": bool(row.get("is_violent", False)),
            "date": str(row.get("report_date", ""))[:10],
            "description": row.get("original_description", ""),
            "distance_m": round(float(row.get("distance_m", 0)), 0),
            "source": row.get("source", ""),
        })

    return results
