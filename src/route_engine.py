"""
Multi-modal routing engine for TigerSafe.

Supports walking, biking, and driving routes using OSMnx graphs.
Generates multiple route alternatives per mode and computes
travel time estimates.
"""

from datetime import datetime
from pathlib import Path

import networkx as nx
import osmnx as ox

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# Campus area bounding box for graph downloads
CAMPUS_BBOX = (38.960, 38.920, -92.300, -92.350)  # N, S, E, W

# Average travel speeds by mode (m/s)
SPEEDS = {
    "walk": 1.4,    # ~5 km/h
    "bike": 4.2,    # ~15 km/h
    "drive": 8.3,   # ~30 km/h in campus area
}

# OSMnx network type mapping
_NETWORK_TYPES = {
    "walk": "walk",
    "bike": "bike",
    "drive": "drive",
}


def _graph_path(mode: str) -> Path:
    return DATA_DIR / f"columbia_{mode}.graphml"


def download_graph(mode: str = "walk", force: bool = False) -> nx.MultiDiGraph:
    """Download and cache an OSM graph for the given travel mode.

    Args:
        mode: One of "walk", "bike", "drive".
        force: Re-download even if cached.

    Returns:
        NetworkX MultiDiGraph with edge lengths.
    """
    path = _graph_path(mode)
    if path.exists() and not force:
        return ox.load_graphml(path)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    network_type = _NETWORK_TYPES.get(mode, "walk")
    G = ox.graph_from_bbox(bbox=CAMPUS_BBOX, network_type=network_type)
    ox.save_graphml(G, path)
    return G


def load_graph(mode: str = "walk") -> nx.MultiDiGraph:
    """Load a cached graph or download it if not available."""
    path = _graph_path(mode)
    if path.exists():
        return ox.load_graphml(path)
    return download_graph(mode)


def find_route(
    G: nx.MultiDiGraph,
    origin: tuple[float, float],
    dest: tuple[float, float],
    weight: str = "length",
) -> dict | None:
    """Find a single route between origin and destination.

    Args:
        G: OSMnx graph.
        origin: (lat, lon).
        dest: (lat, lon).
        weight: Edge attribute to minimize ("length" or "safety_weight").

    Returns:
        Dict with route info or None if no path found.
    """
    orig_node = ox.nearest_nodes(G, origin[1], origin[0])
    dest_node = ox.nearest_nodes(G, dest[1], dest[0])

    try:
        route_nodes = nx.shortest_path(G, orig_node, dest_node, weight=weight)
    except nx.NetworkXNoPath:
        return None

    return _extract_route_info(G, route_nodes, origin, dest)


def find_alternative_routes(
    G: nx.MultiDiGraph,
    origin: tuple[float, float],
    dest: tuple[float, float],
    num_alternatives: int = 3,
    weight: str = "length",
) -> list[dict]:
    """Find multiple alternative routes using k-shortest-paths.

    Returns up to `num_alternatives` distinct routes, ordered by
    increasing distance.
    """
    orig_node = ox.nearest_nodes(G, origin[1], origin[0])
    dest_node = ox.nearest_nodes(G, dest[1], dest[0])

    routes = []
    seen_lengths = set()

    try:
        path_gen = nx.shortest_simple_paths(G, orig_node, dest_node, weight=weight)
        count = 0
        for route_nodes in path_gen:
            if count >= num_alternatives * 5:
                break

            info = _extract_route_info(G, route_nodes, origin, dest)
            if info is None:
                continue

            # Skip near-duplicate routes (within 5% length)
            length_key = round(info["distance_m"] / 50) * 50
            if length_key in seen_lengths:
                continue
            seen_lengths.add(length_key)

            routes.append(info)
            if len(routes) >= num_alternatives:
                break
            count += 1
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        pass

    # If no alternatives found, try the basic shortest path
    if not routes:
        single = find_route(G, origin, dest, weight=weight)
        if single:
            routes.append(single)

    return routes


def _extract_route_info(
    G: nx.MultiDiGraph,
    route_nodes: list,
    origin: tuple[float, float],
    dest: tuple[float, float],
) -> dict | None:
    """Extract route information from a list of graph nodes."""
    if len(route_nodes) < 2:
        return None

    nodes_gdf = ox.graph_to_gdfs(G, edges=False)
    coords = []
    total_length = 0.0

    for node_id in route_nodes:
        if node_id in nodes_gdf.index:
            node = nodes_gdf.loc[node_id]
            coords.append((node.geometry.y, node.geometry.x))

    for i in range(len(route_nodes) - 1):
        u, v = route_nodes[i], route_nodes[i + 1]
        if v in G[u]:
            edge_data = min(
                G[u][v].values(),
                key=lambda d: d.get("length", float("inf")),
            )
            total_length += edge_data.get("length", 0)

    if not coords:
        return None

    return {
        "coordinates": coords,
        "distance_m": round(total_length, 1),
        "num_nodes": len(route_nodes),
        "origin": origin,
        "destination": dest,
    }


def estimate_travel_time(distance_m: float, mode: str) -> float:
    """Estimate travel time in minutes given distance and mode."""
    speed = SPEEDS.get(mode, SPEEDS["walk"])
    return round((distance_m / speed) / 60, 1)


def compute_routes_for_mode(
    origin: tuple[float, float],
    dest: tuple[float, float],
    mode: str = "walk",
    num_alternatives: int = 3,
) -> list[dict]:
    """Full pipeline: load graph, find alternatives, add time estimates.

    Args:
        origin: (lat, lon).
        dest: (lat, lon).
        mode: "walk", "bike", or "drive".
        num_alternatives: Number of route alternatives to generate.

    Returns:
        List of route dicts with coordinates, distance, estimated time.
    """
    G = load_graph(mode)
    routes = find_alternative_routes(G, origin, dest, num_alternatives)

    for route in routes:
        route["mode"] = mode
        route["estimated_time_min"] = estimate_travel_time(
            route["distance_m"], mode
        )

    return routes
