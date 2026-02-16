"""
Shuttle service module for TigerSafe.

Loads real Tiger Line shuttle data from data/shuttle_data/ and provides:
- Route information (name, geometry, color)
- Stop locations with proximity search
- Schedule and availability checking
- Rider eligibility information
- Shuttle-based route suggestions
"""

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import polyline as polyline_lib
except ImportError:
    polyline_lib = None

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# Tiger Line shuttle schedule and eligibility.
# Based on actual MU Tiger Line operations.
SHUTTLE_INFO = {
    "Tiger Line 405 Campus Loop": {
        "abbr": "T5",
        "description": "General campus circulation connecting major buildings and parking lots",
        "schedule": {
            "weekday": {"start": "07:00", "end": "22:00"},
            "saturday": None,  # No service
            "sunday": None,
        },
        "frequency_min": 10,
        "eligibility": "All MU students, faculty, and staff with valid MU ID. Free to ride.",
        "key_stops": ["Memorial Union", "Student Center", "Rec Center", "Engineering"],
    },
    "Tiger Line Hearnes": {
        "abbr": "Hearnes",
        "description": "Serves Hearnes Center, south campus, residence halls, and parking areas",
        "schedule": {
            "weekday": {"start": "07:00", "end": "18:00"},
            "saturday": None,
            "sunday": None,
        },
        "frequency_min": 12,
        "eligibility": "All MU students, faculty, and staff with valid MU ID. Free to ride.",
        "key_stops": ["Hearnes Center", "Stadium", "South Campus"],
    },
    "Tiger Line Trowbridge": {
        "abbr": "Trowbridge",
        "description": "Connects residence halls on east side to central campus",
        "schedule": {
            "weekday": {"start": "07:00", "end": "18:00"},
            "saturday": None,
            "sunday": None,
        },
        "frequency_min": 12,
        "eligibility": "All MU students, faculty, and staff with valid MU ID. Free to ride.",
        "key_stops": ["Trowbridge", "Discovery Hall", "East Campus"],
    },
    "Tiger Line Reactor": {
        "abbr": "Reactor",
        "description": "Serves Research Park and north campus research facilities",
        "schedule": {
            "weekday": {"start": "07:00", "end": "18:00"},
            "saturday": None,
            "sunday": None,
        },
        "frequency_min": 18,
        "eligibility": "All MU students, faculty, and staff with valid MU ID. Free to ride.",
        "key_stops": ["Research Park", "Reactor", "North Campus"],
    },
    "Tiger Line MU Health Care": {
        "abbr": "Health",
        "description": "Connects campus to University Hospital and health care facilities",
        "schedule": {
            "weekday": {"start": "06:30", "end": "18:00"},
            "saturday": None,
            "sunday": None,
        },
        "frequency_min": 18,
        "eligibility": "All MU students, faculty, staff, patients, and visitors. Free to ride.",
        "key_stops": ["University Hospital", "Medical Center", "Campus"],
    },
}


def load_shuttle_stops() -> pd.DataFrame:
    """Load shuttle stop locations from CSV data."""
    shuttle_dir = DATA_DIR / "shuttle_data"
    files = sorted(shuttle_dir.glob("shuttle_stops_*.csv"))
    if not files:
        return pd.DataFrame(columns=["stop_id", "name", "lat", "lng"])
    df = pd.read_csv(files[-1])
    # Ensure numeric coordinates
    df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    df["lng"] = pd.to_numeric(df["lng"], errors="coerce")
    return df.dropna(subset=["lat", "lng"])


def load_shuttle_routes() -> pd.DataFrame:
    """Load shuttle route data from CSV."""
    shuttle_dir = DATA_DIR / "shuttle_data"
    files = sorted(shuttle_dir.glob("shuttle_routes_*.csv"))
    if not files:
        return pd.DataFrame()
    return pd.read_csv(files[-1])


def decode_route_polyline(encoded: str) -> list[tuple[float, float]]:
    """Decode a Google Encoded Polyline to a list of (lat, lon) points.

    Falls back to empty list if polyline library is unavailable.
    """
    if not encoded or not isinstance(encoded, str):
        return []
    if polyline_lib is None:
        return []
    try:
        return polyline_lib.decode(encoded)
    except Exception:
        return []


def get_route_geometries() -> dict[str, list[tuple[float, float]]]:
    """Get decoded geometries for all shuttle routes.

    Returns:
        Dict mapping route name to list of (lat, lon) points.
    """
    routes_df = load_shuttle_routes()
    if routes_df.empty:
        return {}

    result = {}
    for _, row in routes_df.iterrows():
        name = row.get("name", "")
        encoded = row.get("encoded_polyline", "")
        points = decode_route_polyline(str(encoded))
        if points:
            result[name] = points
    return result


def find_nearest_stops(
    lat: float, lon: float, radius_m: float = 500.0, limit: int = 5
) -> list[dict]:
    """Find shuttle stops within radius of a point.

    Args:
        lat: Latitude.
        lon: Longitude.
        radius_m: Search radius in meters.
        limit: Maximum stops to return.

    Returns:
        List of stop dicts with name, lat, lng, distance_m.
    """
    stops = load_shuttle_stops()
    if stops.empty:
        return []

    # Approximate distance using haversine
    R = 6371000  # Earth radius in meters
    lat_r = np.radians(lat)
    stop_lats_r = np.radians(stops["lat"].values)
    dlat = stop_lats_r - lat_r
    dlon = np.radians(stops["lng"].values - lon)

    a = np.sin(dlat / 2) ** 2 + np.cos(lat_r) * np.cos(stop_lats_r) * np.sin(dlon / 2) ** 2
    distances = 2 * R * np.arcsin(np.sqrt(a))

    stops = stops.copy()
    stops["distance_m"] = distances
    nearby = stops[stops["distance_m"] <= radius_m].sort_values("distance_m")

    results = []
    for _, row in nearby.head(limit).iterrows():
        results.append({
            "stop_id": int(row.get("stop_id", 0)),
            "name": row.get("name", "Unknown Stop"),
            "lat": float(row["lat"]),
            "lng": float(row["lng"]),
            "distance_m": round(float(row["distance_m"]), 0),
        })
    return results


def check_shuttle_availability(
    route_name: str = None, dt: datetime = None
) -> dict:
    """Check if shuttle service is currently available.

    Args:
        route_name: Specific route to check (or None for all).
        dt: DateTime to check (defaults to now).

    Returns:
        Dict with availability info per route.
    """
    if dt is None:
        dt = datetime.now()

    day_of_week = dt.weekday()  # 0=Monday
    current_time = dt.strftime("%H:%M")
    is_weekend = day_of_week >= 5

    results = {}
    for name, info in SHUTTLE_INFO.items():
        if route_name and route_name.lower() not in name.lower():
            continue

        if is_weekend:
            day_key = "saturday" if day_of_week == 5 else "sunday"
        else:
            day_key = "weekday"

        schedule = info["schedule"].get(day_key)
        if schedule is None:
            results[name] = {
                "available": False,
                "reason": f"No service on {'Saturday' if day_of_week == 5 else 'Sunday' if day_of_week == 6 else 'this day'}",
                "next_service": "Next weekday at " + info["schedule"]["weekday"]["start"] if info["schedule"]["weekday"] else "Unknown",
                **info,
            }
        elif current_time < schedule["start"]:
            results[name] = {
                "available": False,
                "reason": f"Service starts at {schedule['start']}",
                "next_service": f"Today at {schedule['start']}",
                **info,
            }
        elif current_time > schedule["end"]:
            results[name] = {
                "available": False,
                "reason": f"Service ended at {schedule['end']}",
                "next_service": "Tomorrow at " + (info["schedule"]["weekday"]["start"] if day_of_week < 4 else "Next Monday"),
                **info,
            }
        else:
            results[name] = {
                "available": True,
                "reason": "Currently running",
                "frequency": f"Every {info['frequency_min']} minutes",
                "ends_at": schedule["end"],
                **info,
            }

    return results


def get_shuttle_for_trip(
    origin: tuple[float, float],
    dest: tuple[float, float],
    dt: datetime = None,
) -> dict | None:
    """Check if a shuttle can help with a trip.

    Looks for shuttle stops near both origin and destination.

    Returns:
        Dict with shuttle route info if a shuttle covers part of the trip,
        or None if no useful shuttle route exists.
    """
    if dt is None:
        dt = datetime.now()

    origin_stops = find_nearest_stops(origin[0], origin[1], radius_m=400)
    dest_stops = find_nearest_stops(dest[0], dest[1], radius_m=400)

    if not origin_stops or not dest_stops:
        return None

    availability = check_shuttle_availability(dt=dt)
    available_routes = {
        name: info for name, info in availability.items() if info.get("available")
    }

    if not available_routes:
        return {
            "available": False,
            "nearest_origin_stop": origin_stops[0] if origin_stops else None,
            "nearest_dest_stop": dest_stops[0] if dest_stops else None,
            "reason": "No shuttle routes currently running",
            "next_service": next(
                (info.get("next_service", "") for info in availability.values()),
                "Unknown"
            ),
        }

    return {
        "available": True,
        "nearest_origin_stop": origin_stops[0],
        "nearest_dest_stop": dest_stops[0],
        "available_routes": list(available_routes.keys()),
        "route_details": available_routes,
        "walk_to_stop_m": origin_stops[0]["distance_m"],
        "walk_from_stop_m": dest_stops[0]["distance_m"],
        "eligibility": "All MU students, faculty, and staff with valid MU ID. Free to ride.",
        "real_time_tracking": "https://tiger.etaspot.net",
    }
