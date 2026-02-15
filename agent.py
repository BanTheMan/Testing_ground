"""
TigerSafe AI Agent — Claude-powered safety routing assistant.

Implements the agent described in the design doc using Claude's tool-use API.
Five tools:
1. query_crimes_near_location - spatial crime queries
2. get_route_safety_score - Dijkstra with safety weights
3. check_shuttle_schedule - GTFS schedule info
4. get_recent_incidents - recent area crime lookup
5. explain_safety_factors - lighting, density, temporal risk
"""

import json
import os
from datetime import datetime

import anthropic

# Tool definitions for Claude
TOOLS = [
    {
        "name": "query_crimes_near_location",
        "description": (
            "Query crime incidents near a specific location on or around the "
            "Mizzou campus. Returns crime counts, categories, and severity "
            "within the specified radius."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "latitude": {
                    "type": "number",
                    "description": "Latitude of the location to query",
                },
                "longitude": {
                    "type": "number",
                    "description": "Longitude of the location to query",
                },
                "radius_meters": {
                    "type": "number",
                    "description": "Search radius in meters (default 500)",
                    "default": 500,
                },
            },
            "required": ["latitude", "longitude"],
        },
    },
    {
        "name": "get_route_safety_score",
        "description": (
            "Calculate and compare routes between two locations. Returns both "
            "the safest route and the fastest route with safety scores, distance, "
            "duration, and risk breakdowns."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "origin_lat": {
                    "type": "number",
                    "description": "Origin latitude",
                },
                "origin_lon": {
                    "type": "number",
                    "description": "Origin longitude",
                },
                "dest_lat": {
                    "type": "number",
                    "description": "Destination latitude",
                },
                "dest_lon": {
                    "type": "number",
                    "description": "Destination longitude",
                },
                "hour": {
                    "type": "integer",
                    "description": "Hour of day (0-23) for temporal risk. Defaults to current hour.",
                },
            },
            "required": ["origin_lat", "origin_lon", "dest_lat", "dest_lon"],
        },
    },
    {
        "name": "check_shuttle_schedule",
        "description": (
            "Check Tiger Line shuttle availability and schedule. Returns "
            "information about shuttle routes near a location."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "latitude": {
                    "type": "number",
                    "description": "Latitude to check shuttle routes near",
                },
                "longitude": {
                    "type": "number",
                    "description": "Longitude to check shuttle routes near",
                },
            },
            "required": ["latitude", "longitude"],
        },
    },
    {
        "name": "get_recent_incidents",
        "description": (
            "Get the most recent crime incidents in a specific area. Returns "
            "details about recent crimes including category, date, and location."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "latitude": {
                    "type": "number",
                    "description": "Center latitude",
                },
                "longitude": {
                    "type": "number",
                    "description": "Center longitude",
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of incidents to return (default 10)",
                    "default": 10,
                },
            },
            "required": ["latitude", "longitude"],
        },
    },
    {
        "name": "explain_safety_factors",
        "description": (
            "Explain the safety factors for a specific location including "
            "crime density, lighting conditions, emergency phone proximity, "
            "patrol frequency, and time-of-day risk."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "latitude": {
                    "type": "number",
                    "description": "Latitude of the location",
                },
                "longitude": {
                    "type": "number",
                    "description": "Longitude of the location",
                },
                "hour": {
                    "type": "integer",
                    "description": "Hour of day (0-23) for temporal analysis",
                },
            },
            "required": ["latitude", "longitude"],
        },
    },
]

# Well-known campus locations for geocoding
CAMPUS_LOCATIONS = {
    "memorial union": (38.9407, -92.3280),
    "mem union": (38.9407, -92.3280),
    "jesse hall": (38.9395, -92.3290),
    "jesse": (38.9395, -92.3290),
    "ellis library": (38.9410, -92.3305),
    "library": (38.9410, -92.3305),
    "ellis": (38.9410, -92.3305),
    "lafferre hall": (38.9422, -92.3258),
    "lafferre": (38.9422, -92.3258),
    "engineering": (38.9418, -92.3250),
    "engineering building": (38.9418, -92.3250),
    "rec center": (38.9380, -92.3310),
    "student recreation": (38.9380, -92.3310),
    "rec complex": (38.9380, -92.3310),
    "student center": (38.9390, -92.3275),
    "mizzou student center": (38.9390, -92.3275),
    "strickland hall": (38.9412, -92.3278),
    "strickland": (38.9412, -92.3278),
    "tate hall": (38.9416, -92.3292),
    "tate": (38.9416, -92.3292),
    "middlebush hall": (38.9405, -92.3298),
    "middlebush": (38.9405, -92.3298),
    "speakers circle": (38.9402, -92.3295),
    "neff hall": (38.9408, -92.3265),
    "neff": (38.9408, -92.3265),
    "townsend hall": (38.9425, -92.3270),
    "townsend": (38.9425, -92.3270),
    "hearnes center": (38.9365, -92.3255),
    "hearnes": (38.9365, -92.3255),
    "faurot field": (38.9360, -92.3320),
    "faurot": (38.9360, -92.3320),
    "mizzou arena": (38.9370, -92.3335),
    "arena": (38.9370, -92.3335),
    "discovery hall": (38.9430, -92.3240),
    "discovery": (38.9430, -92.3240),
    "chemistry building": (38.9415, -92.3275),
    "chemistry": (38.9415, -92.3275),
    "columns": (38.9400, -92.3290),
    "the columns": (38.9400, -92.3290),
    "francis quadrangle": (38.9400, -92.3295),
    "the quad": (38.9400, -92.3295),
    "carnahan quad": (38.9415, -92.3288),
    "switzler hall": (38.9400, -92.3302),
    "switzler": (38.9400, -92.3302),
    "a&s": (38.9398, -92.3285),
    "arts and science": (38.9398, -92.3285),
    "tiger hotel": (38.9462, -92.3295),
    "downtown": (38.9470, -92.3290),
    "broadway": (38.9460, -92.3290),
    "flat branch park": (38.9490, -92.3320),
}

SYSTEM_PROMPT = """You are TigerSafe, an AI safety assistant for Mizzou (University of Missouri) students. Your role is to help students navigate campus safely by providing:

1. **Safe route recommendations** between campus locations
2. **Crime awareness** — recent incidents and patterns near locations
3. **Safety factor explanations** — what makes an area safer or riskier
4. **Shuttle information** — Tiger Line routes and availability
5. **General safety advice** tailored to time of day and location

## Important Guidelines:
- Always consider the time of day when assessing safety
- Provide balanced information — don't unnecessarily alarm users, but be honest about risks
- When recommending routes, explain the trade-offs between speed and safety
- Cite specific data (crime counts, risk scores) to support your recommendations
- If asked about a campus location by name, use the known coordinates to look it up
- Be concise but thorough — students need actionable information quickly

## Known Campus Locations:
You know the coordinates for major campus buildings. When a user mentions a building name, you can use those coordinates directly with the tools.

## Risk Score Interpretation:
- 0-20: Very Safe (green) — well-lit, low crime, near emergency phones
- 21-40: Safe (blue) — generally safe with minor concerns
- 41-60: Moderate Risk (orange) — exercise normal caution
- 61-80: Higher Risk (red) — be alert, consider alternatives
- 81-100: High Risk (dark red) — avoid if possible, especially at night
"""


def execute_tool(tool_name: str, tool_input: dict, app_state: dict) -> str:
    """Execute a tool call and return the result as a JSON string."""
    from safety import (
        find_fastest_route,
        find_safest_route,
        get_nearby_crimes,
        get_temporal_multiplier,
        load_crimes,
        load_emergency_phones,
    )

    try:
        if tool_name == "query_crimes_near_location":
            crimes = app_state.get("crimes")
            if crimes is None:
                crimes = load_crimes()
            lat = tool_input["latitude"]
            lon = tool_input["longitude"]
            radius = tool_input.get("radius_meters", 500)
            nearby = get_nearby_crimes(crimes, lat, lon, radius)

            if nearby.empty:
                return json.dumps({
                    "total_incidents": 0,
                    "message": f"No crime incidents found within {radius}m of this location.",
                })

            # Summarize by category
            summary = {}
            if "category" in nearby.columns:
                for cat, group in nearby.groupby("category"):
                    summary[cat] = len(group)

            violent_count = 0
            if "is_violent" in nearby.columns:
                violent_count = int(nearby["is_violent"].astype(bool).sum())

            return json.dumps({
                "total_incidents": len(nearby),
                "violent_incidents": violent_count,
                "by_category": summary,
                "radius_meters": radius,
                "location": {"lat": lat, "lon": lon},
            })

        elif tool_name == "get_route_safety_score":
            G = app_state.get("graph")
            if G is None:
                return json.dumps({"error": "Graph not loaded. Please run data ingestion first."})

            origin = (tool_input["origin_lat"], tool_input["origin_lon"])
            dest = (tool_input["dest_lat"], tool_input["dest_lon"])
            hour = tool_input.get("hour", datetime.now().hour)

            safest = find_safest_route(G, origin, dest, hour=hour)
            fastest = find_fastest_route(G, origin, dest, hour=hour)

            # Store routes for map display
            app_state["last_safest_route"] = safest
            app_state["last_fastest_route"] = fastest

            result = {
                "safest_route": {
                    "distance_m": safest.get("distance_m"),
                    "duration_min": safest.get("duration_min"),
                    "risk_score": safest.get("risk_score"),
                    "crime_count_nearby": safest.get("crime_count_nearby"),
                    "violent_crimes_nearby": safest.get("violent_crimes_nearby"),
                },
                "fastest_route": {
                    "distance_m": fastest.get("distance_m"),
                    "duration_min": fastest.get("duration_min"),
                    "risk_score": fastest.get("risk_score"),
                    "crime_count_nearby": fastest.get("crime_count_nearby"),
                    "violent_crimes_nearby": fastest.get("violent_crimes_nearby"),
                },
                "hour_analyzed": hour,
                "temporal_multiplier": get_temporal_multiplier(hour),
            }
            return json.dumps(result)

        elif tool_name == "check_shuttle_schedule":
            lat = tool_input["latitude"]
            lon = tool_input["longitude"]

            # Static shuttle route info (from GTFS data in data_analysis.md)
            shuttle_routes = [
                {
                    "name": "Hearnes Route",
                    "description": "Serves Hearnes Center, south campus, and residential areas",
                    "hours": "7:00 AM - 6:00 PM (Mon-Fri)",
                    "frequency": "Every 10-15 minutes",
                },
                {
                    "name": "Trowbridge Route",
                    "description": "Connects residence halls to central campus",
                    "hours": "7:00 AM - 6:00 PM (Mon-Fri)",
                    "frequency": "Every 10-15 minutes",
                },
                {
                    "name": "Reactor Route",
                    "description": "Serves Research Park and north campus",
                    "hours": "7:00 AM - 6:00 PM (Mon-Fri)",
                    "frequency": "Every 15-20 minutes",
                },
                {
                    "name": "Campus Route",
                    "description": "General campus circulation route",
                    "hours": "7:00 AM - 10:00 PM (Mon-Fri)",
                    "frequency": "Every 10 minutes",
                },
                {
                    "name": "MU Health Care Route",
                    "description": "Connects campus to University Hospital",
                    "hours": "6:30 AM - 6:00 PM (Mon-Fri)",
                    "frequency": "Every 15-20 minutes",
                },
            ]

            current_hour = datetime.now().hour
            is_weekend = datetime.now().weekday() >= 5

            return json.dumps({
                "location": {"lat": lat, "lon": lon},
                "shuttle_routes": shuttle_routes,
                "service_note": (
                    "Weekend — limited or no shuttle service"
                    if is_weekend
                    else (
                        "Shuttles are currently running"
                        if 7 <= current_hour <= 22
                        else "Shuttle service has ended for the day"
                    )
                ),
                "real_time_tracking": "https://tiger.etaspot.net",
            })

        elif tool_name == "get_recent_incidents":
            crimes = app_state.get("crimes")
            if crimes is None:
                crimes = load_crimes()
            lat = tool_input["latitude"]
            lon = tool_input["longitude"]
            limit = tool_input.get("limit", 10)

            nearby = get_nearby_crimes(crimes, lat, lon, radius_m=800)
            if nearby.empty:
                return json.dumps({
                    "incidents": [],
                    "message": "No recent incidents found near this location.",
                })

            # Sort by date if available
            if "date_occurred" in nearby.columns:
                nearby = nearby.sort_values("date_occurred", ascending=False)

            incidents = []
            for _, row in nearby.head(limit).iterrows():
                incident = {
                    "category": row.get("category", "Unknown"),
                    "severity": float(row.get("severity", 0.5)),
                    "distance_m": round(row.get("distance_m", 0), 0),
                }
                if "date_occurred" in nearby.columns:
                    incident["date"] = str(row["date_occurred"])[:10]
                if "is_violent" in nearby.columns:
                    incident["is_violent"] = bool(row["is_violent"])
                incidents.append(incident)

            return json.dumps({
                "incidents": incidents,
                "total_nearby": len(nearby),
            })

        elif tool_name == "explain_safety_factors":
            lat = tool_input["latitude"]
            lon = tool_input["longitude"]
            hour = tool_input.get("hour", datetime.now().hour)

            crimes = app_state.get("crimes")
            if crimes is None:
                crimes = load_crimes()
            phones = app_state.get("phones")
            if phones is None:
                phones = load_emergency_phones()

            # Crime density analysis
            nearby_crimes = get_nearby_crimes(crimes, lat, lon, radius_m=500)
            violent_count = 0
            if not nearby_crimes.empty and "is_violent" in nearby_crimes.columns:
                violent_count = int(nearby_crimes["is_violent"].astype(bool).sum())

            # Emergency phone proximity
            if not phones.empty:
                from shapely.geometry import Point as ShapelyPoint
                import geopandas as _gpd
                point_proj = _gpd.GeoSeries(
                    [ShapelyPoint(lon, lat)], crs="EPSG:4326"
                ).to_crs("EPSG:32615")
                phones_proj = phones.to_crs("EPSG:32615")
                phone_dists = phones_proj.geometry.distance(point_proj.iloc[0])
                nearest_phone_m = round(float(phone_dists.min()), 0)
                phones_within_200m = int((phone_dists <= 200).sum())
            else:
                nearest_phone_m = None
                phones_within_200m = 0

            temporal_mult = get_temporal_multiplier(hour)
            time_period = _get_time_period_name(hour)

            return json.dumps({
                "location": {"lat": lat, "lon": lon},
                "crime_analysis": {
                    "incidents_within_500m": len(nearby_crimes),
                    "violent_incidents": violent_count,
                    "non_violent_incidents": len(nearby_crimes) - violent_count,
                },
                "infrastructure": {
                    "nearest_emergency_phone_m": nearest_phone_m,
                    "phones_within_200m": phones_within_200m,
                },
                "temporal_risk": {
                    "hour": hour,
                    "period": time_period,
                    "risk_multiplier": temporal_mult,
                    "assessment": _temporal_assessment(temporal_mult),
                },
            })

        else:
            return json.dumps({"error": f"Unknown tool: {tool_name}"})

    except Exception as e:
        return json.dumps({"error": str(e)})


def _get_time_period_name(hour: int) -> str:
    """Human-readable time period name."""
    if 0 <= hour < 6:
        return "Late Night"
    elif 6 <= hour < 9:
        return "Early Morning"
    elif 9 <= hour < 17:
        return "Daytime"
    elif 17 <= hour < 20:
        return "Evening"
    elif 20 <= hour < 22:
        return "Night"
    else:
        return "Late Evening"


def _temporal_assessment(multiplier: float) -> str:
    """Assessment text for temporal risk."""
    if multiplier <= 0.5:
        return "Low risk period — good visibility, high foot traffic"
    elif multiplier <= 0.8:
        return "Moderate risk — normal caution advised"
    elif multiplier <= 1.2:
        return "Elevated risk — be aware of surroundings"
    elif multiplier <= 1.5:
        return "Higher risk — stick to well-lit, populated paths"
    else:
        return "Highest risk period — avoid walking alone, use shuttles or buddy system"


def chat(
    user_message: str,
    conversation_history: list[dict],
    app_state: dict,
    api_key: str = None,
) -> tuple[str, list[dict]]:
    """Process a user message through the TigerSafe agent.

    Args:
        user_message: The user's message.
        conversation_history: List of previous messages.
        app_state: Shared state dict (graph, crimes, etc.).
        api_key: Anthropic API key.

    Returns:
        Tuple of (assistant_response_text, updated_conversation_history).
    """
    if not api_key:
        api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        return (
            "Please set your Anthropic API key in the sidebar to use TigerSafe.",
            conversation_history,
        )

    client = anthropic.Anthropic(api_key=api_key)

    # Add user message to history
    conversation_history.append({"role": "user", "content": user_message})

    # Agentic loop — keep calling until no more tool_use
    max_iterations = 10
    for _ in range(max_iterations):
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=4096,
            system=SYSTEM_PROMPT,
            tools=TOOLS,
            messages=conversation_history,
        )

        # Check if we need to handle tool calls
        if response.stop_reason == "tool_use":
            # Add assistant response to history
            conversation_history.append({
                "role": "assistant",
                "content": response.content,
            })

            # Process all tool calls
            tool_results = []
            for block in response.content:
                if block.type == "tool_use":
                    result = execute_tool(block.name, block.input, app_state)
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result,
                    })

            # Add tool results to history
            conversation_history.append({
                "role": "user",
                "content": tool_results,
            })

        else:
            # Final response — extract text
            conversation_history.append({
                "role": "assistant",
                "content": response.content,
            })

            text_parts = []
            for block in response.content:
                if hasattr(block, "text"):
                    text_parts.append(block.text)

            return "\n".join(text_parts), conversation_history

    return (
        "I apologize, but I encountered too many processing steps. "
        "Please try rephrasing your question.",
        conversation_history,
    )


def resolve_location(name: str) -> tuple[float, float] | None:
    """Try to resolve a campus location name to coordinates."""
    key = name.strip().lower()
    if key in CAMPUS_LOCATIONS:
        return CAMPUS_LOCATIONS[key]

    # Fuzzy match — check if any key is contained in the input
    for loc_key, coords in CAMPUS_LOCATIONS.items():
        if loc_key in key or key in loc_key:
            return coords

    return None
