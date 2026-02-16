"""
TigerSafe — Campus Dispatch Copilot

Streamlit application providing safe campus navigation with:
- Current location access and destination selection
- Multi-modal routing (walk, bike, drive) with alternatives
- Shuttle integration (availability, eligibility, routes, schedule)
- Crime-aware risk scoring per path
- AI-powered safety advisor with RAG
- Interactive map with crime heatmap and infrastructure

Run with: streamlit run app.py
"""

import json
import os
from datetime import datetime
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import folium
import geopandas as gpd
import pandas as pd
import streamlit as st
from folium.plugins import HeatMap
from streamlit_folium import st_folium

from src.ai_advisor import (
    CAMPUS_LOCATIONS,
    build_route_context,
    chat_with_advisor,
    get_route_analysis,
    resolve_location,
)
from src.crime_analyzer import (
    compute_crime_density_along_route,
    get_recent_incidents_near,
)
from src.data_loader import (
    load_all_crimes_unified,
    load_campus_buildings,
    load_emergency_phones,
    load_shuttle_stops,
    load_traffic_stops,
)
from src.risk_scorer import (
    compare_routes,
    get_temporal_multiplier,
    get_temporal_period,
)
from src.route_engine import compute_routes_for_mode, download_graph
from src.shuttle_service import (
    check_shuttle_availability,
    find_nearest_stops,
    get_shuttle_for_trip,
    get_route_geometries,
)

DATA_DIR = Path(__file__).parent / "data"

# --- Page Config ---
st.set_page_config(
    page_title="TigerSafe — Campus Dispatch Copilot",
    page_icon="🐯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Custom CSS ---
st.markdown("""
<style>
    .risk-very-safe { color: #22c55e; font-weight: bold; }
    .risk-safe { color: #3b82f6; font-weight: bold; }
    .risk-moderate { color: #eab308; font-weight: bold; }
    .risk-higher { color: #f97316; font-weight: bold; }
    .risk-high { color: #ef4444; font-weight: bold; }
    .route-card {
        border: 1px solid #ddd; border-radius: 8px;
        padding: 12px; margin: 4px 0;
    }
    .stChatMessage { max-height: 500px; overflow-y: auto; }
    div[data-testid="stMetric"] { background: #f8f9fa; border-radius: 8px; padding: 8px; }
</style>
""", unsafe_allow_html=True)


# --- Session State ---
def init_state():
    defaults = {
        "messages": [],
        "conversation_history": [],
        "crimes": None,
        "phones": None,
        "buildings": None,
        "traffic_stops": None,
        "data_loaded": False,
        "routes": [],
        "scored_routes": [],
        "shuttle_info": None,
        "ai_analysis": "",
        "route_context": "",
        "origin_name": "",
        "dest_name": "",
        "origin_coords": None,
        "dest_coords": None,
        "travel_mode": "walk",
        "current_hour": datetime.now().hour,
        "show_heatmap": True,
        "show_phones": True,
        "show_buildings": False,
        "show_shuttle_stops": True,
        "graphs_ready": set(),
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


init_state()


# --- Data Loading ---
@st.cache_data(show_spinner=False)
def cached_load_crimes():
    return load_all_crimes_unified()


@st.cache_data(show_spinner=False)
def cached_load_phones():
    return load_emergency_phones()


@st.cache_data(show_spinner=False)
def cached_load_buildings():
    return load_campus_buildings()


@st.cache_data(show_spinner=False)
def cached_load_shuttle_stops():
    return load_shuttle_stops()


@st.cache_data(show_spinner=False)
def cached_load_traffic_stops():
    return load_traffic_stops(recent_years=3)


def ensure_data_loaded():
    """Load all data sources (cached after first load)."""
    if not st.session_state.data_loaded:
        st.session_state.crimes = cached_load_crimes()
        st.session_state.phones = cached_load_phones()
        st.session_state.buildings = cached_load_buildings()
        st.session_state.traffic_stops = cached_load_traffic_stops()
        st.session_state.data_loaded = True


# --- Sidebar ---
with st.sidebar:
    st.title("TigerSafe")
    st.caption("Campus Dispatch Copilot — Mizzou")
    st.divider()

    # API Key
    api_key = st.text_input(
        "Anthropic API Key",
        type="password",
        value=os.environ.get("ANTHROPIC_API_KEY", ""),
        help="Required for AI advisor. Set in .env file or enter here.",
    )

    st.divider()

    # Time display
    now = datetime.now()
    st.session_state.current_hour = now.hour
    period = get_temporal_period(now.hour)
    st.markdown(f"**Current Time**: {now.strftime('%I:%M %p, %A')}")
    st.markdown(f"**Risk Period**: {period['label']}")
    st.markdown(f"**Risk Multiplier**: {period['multiplier']}x")

    st.divider()

    # Map layers
    st.subheader("Map Layers")
    st.session_state.show_heatmap = st.checkbox("Crime Heatmap", value=st.session_state.show_heatmap)
    st.session_state.show_phones = st.checkbox("Emergency Phones", value=st.session_state.show_phones)
    st.session_state.show_buildings = st.checkbox("Buildings", value=st.session_state.show_buildings)
    st.session_state.show_shuttle_stops = st.checkbox("Shuttle Stops", value=st.session_state.show_shuttle_stops)

    st.divider()

    # Data status
    ensure_data_loaded()
    crime_count = len(st.session_state.crimes) if st.session_state.crimes is not None else 0
    phone_count = len(st.session_state.phones) if st.session_state.phones is not None and not st.session_state.phones.empty else 0
    st.caption(f"Data: {crime_count} crime records | {phone_count} emergency phones")

    st.divider()
    if st.button("Clear Session", use_container_width=True):
        for key in ["messages", "conversation_history", "routes", "scored_routes",
                     "shuttle_info", "ai_analysis", "route_context"]:
            st.session_state[key] = [] if "routes" in key or "messages" in key or "history" in key else ""
        st.rerun()


# --- Main Layout ---
st.header("TigerSafe — Campus Safety Navigation")

# === LOCATION & MODE SELECTION ===
loc_col1, loc_col2, mode_col = st.columns([2, 2, 1])

# Build clean location list (deduplicated, showing most specific names)
_seen_coords = {}
clean_locations = ["(Type or select a location)"]
for k in sorted(CAMPUS_LOCATIONS.keys(), key=len, reverse=True):
    coords = CAMPUS_LOCATIONS[k]
    if coords not in _seen_coords:
        _seen_coords[coords] = k.title()
        clean_locations.append(k.title())
clean_locations.sort()

with loc_col1:
    st.subheader("Origin")
    origin_method = st.radio(
        "How to set origin",
        ["Select from list", "Enter coordinates"],
        horizontal=True,
        key="origin_method",
        label_visibility="collapsed",
    )
    if origin_method == "Select from list":
        origin_pick = st.selectbox(
            "Starting location",
            options=clean_locations,
            key="origin_select",
        )
        if origin_pick and origin_pick != "(Type or select a location)":
            coords = resolve_location(origin_pick)
            if coords:
                st.session_state.origin_coords = coords
                st.session_state.origin_name = origin_pick
    else:
        lat_in = st.number_input("Latitude", value=38.9404, format="%.6f", key="olat")
        lon_in = st.number_input("Longitude", value=-92.3277, format="%.6f", key="olon")
        st.session_state.origin_coords = (lat_in, lon_in)
        st.session_state.origin_name = f"({lat_in:.4f}, {lon_in:.4f})"

with loc_col2:
    st.subheader("Destination")
    dest_method = st.radio(
        "How to set destination",
        ["Select from list", "Enter coordinates"],
        horizontal=True,
        key="dest_method",
        label_visibility="collapsed",
    )
    if dest_method == "Select from list":
        dest_pick = st.selectbox(
            "Destination",
            options=clean_locations,
            key="dest_select",
        )
        if dest_pick and dest_pick != "(Type or select a location)":
            coords = resolve_location(dest_pick)
            if coords:
                st.session_state.dest_coords = coords
                st.session_state.dest_name = dest_pick
    else:
        dlat_in = st.number_input("Latitude", value=38.9410, format="%.6f", key="dlat")
        dlon_in = st.number_input("Longitude", value=-92.3305, format="%.6f", key="dlon")
        st.session_state.dest_coords = (dlat_in, dlon_in)
        st.session_state.dest_name = f"({dlat_in:.4f}, {dlon_in:.4f})"

with mode_col:
    st.subheader("Mode")
    st.session_state.travel_mode = st.radio(
        "Travel mode",
        ["walk", "bike", "drive"],
        format_func=lambda x: {"walk": "Walking", "bike": "Biking", "drive": "Driving"}[x],
        key="mode_radio",
        label_visibility="collapsed",
    )

# === CALCULATE ROUTES BUTTON ===
calc_col1, calc_col2, calc_col3 = st.columns([2, 1, 2])
with calc_col1:
    calculate_pressed = st.button(
        "Calculate Routes",
        use_container_width=True,
        type="primary",
        disabled=(st.session_state.origin_coords is None or st.session_state.dest_coords is None),
    )

if calculate_pressed and st.session_state.origin_coords and st.session_state.dest_coords:
    origin = st.session_state.origin_coords
    dest = st.session_state.dest_coords
    mode = st.session_state.travel_mode
    hour = st.session_state.current_hour

    with st.spinner(f"Finding {mode} routes and analyzing safety..."):
        # 1. Ensure graph is downloaded
        if mode not in st.session_state.graphs_ready:
            try:
                download_graph(mode)
                st.session_state.graphs_ready.add(mode)
            except Exception as e:
                st.error(f"Failed to download routing graph for {mode}: {e}")
                st.stop()

        # 2. Find routes
        try:
            routes = compute_routes_for_mode(origin, dest, mode=mode, num_alternatives=3)
        except Exception as e:
            st.error(f"Routing error: {e}")
            routes = []

        if not routes:
            st.warning("No routes found between the selected locations. Try different locations or a different travel mode.")
        else:
            # 3. Score routes
            crimes = st.session_state.crimes
            phones = st.session_state.phones
            traffic = st.session_state.traffic_stops

            scored = compare_routes(routes, crimes, phones, traffic, hour=hour, mode=mode)
            st.session_state.scored_routes = scored

            # 4. Check shuttle
            shuttle = get_shuttle_for_trip(origin, dest)
            st.session_state.shuttle_info = shuttle

            # 5. Build context and get AI analysis
            context = build_route_context(scored, shuttle, hour, mode)
            st.session_state.route_context = context

            analysis = get_route_analysis(scored, shuttle, hour, mode, api_key=api_key)
            st.session_state.ai_analysis = analysis

    st.rerun()


# === RESULTS DISPLAY ===
if st.session_state.scored_routes:
    st.divider()

    # Map and details columns
    map_col, detail_col = st.columns([3, 2])

    # --- MAP ---
    with map_col:
        st.subheader("Route Map")

        # Center on route midpoint
        all_coords = []
        for r in st.session_state.scored_routes:
            all_coords.extend(r.get("coordinates", []))
        if all_coords:
            center_lat = sum(c[0] for c in all_coords) / len(all_coords)
            center_lon = sum(c[1] for c in all_coords) / len(all_coords)
        else:
            center_lat, center_lon = 38.9404, -92.3277

        m = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=16,
            tiles="CartoDB positron",
        )

        # Crime heatmap
        if st.session_state.show_heatmap and st.session_state.crimes is not None and not st.session_state.crimes.empty:
            crimes = st.session_state.crimes
            heatmap_data = []
            for _, row in crimes.iterrows():
                lat = row.geometry.y
                lon = row.geometry.x
                weight = float(row.get("severity", 0.5))
                heatmap_data.append([lat, lon, weight])
            if heatmap_data:
                HeatMap(
                    heatmap_data,
                    radius=25, blur=15,
                    gradient={0.2: "blue", 0.4: "lime", 0.6: "yellow", 0.8: "orange", 1.0: "red"},
                    name="Crime Heatmap",
                ).add_to(m)

        # Emergency phones
        if st.session_state.show_phones and st.session_state.phones is not None and not st.session_state.phones.empty:
            phone_group = folium.FeatureGroup(name="Emergency Phones")
            for _, row in st.session_state.phones.iterrows():
                folium.Marker(
                    location=[row.geometry.y, row.geometry.x],
                    popup=str(row.get("DESCRIPTIO", row.get("name", "Emergency Phone"))),
                    icon=folium.Icon(color="blue", icon="phone", prefix="fa"),
                ).add_to(phone_group)
            phone_group.add_to(m)

        # Buildings
        if st.session_state.show_buildings and st.session_state.buildings is not None and not st.session_state.buildings.empty:
            bldg_group = folium.FeatureGroup(name="Buildings")
            for _, row in st.session_state.buildings.iterrows():
                geom = row.geometry
                if geom.geom_type == "Point":
                    loc = [geom.y, geom.x]
                else:
                    centroid = geom.centroid
                    loc = [centroid.y, centroid.x]
                name = row.get("BUILDING_N", row.get("BUILDING_NAME", row.get("name", "Building")))
                folium.Marker(
                    location=loc,
                    popup=str(name),
                    icon=folium.Icon(color="gray", icon="building", prefix="fa"),
                ).add_to(bldg_group)
            bldg_group.add_to(m)

        # Shuttle stops
        if st.session_state.show_shuttle_stops:
            shuttle_stops_df = cached_load_shuttle_stops()
            if not shuttle_stops_df.empty:
                shuttle_group = folium.FeatureGroup(name="Shuttle Stops")
                for _, row in shuttle_stops_df.iterrows():
                    if pd.notna(row.get("lat")) and pd.notna(row.get("lng")):
                        folium.CircleMarker(
                            location=[row["lat"], row["lng"]],
                            radius=5,
                            color="#127AD1",
                            fill=True,
                            fill_opacity=0.7,
                            popup=str(row.get("name", "Shuttle Stop")),
                        ).add_to(shuttle_group)
                shuttle_group.add_to(m)

        # Routes
        for i, route in enumerate(st.session_state.scored_routes):
            coords = route.get("coordinates", [])
            risk = route.get("risk_score", {})
            color = risk.get("color", "#3b82f6")
            rec = route.get("recommendation", f"Route {i+1}")
            score = risk.get("score", 0)
            level = risk.get("level", "?")
            time_min = route.get("estimated_time_min", 0)
            dist = route.get("distance_m", 0)

            weight = 7 if i == 0 else 4
            opacity = 0.9 if i == 0 else 0.6
            dash = None if i == 0 else "8"

            folium.PolyLine(
                locations=coords,
                color=color,
                weight=weight,
                opacity=opacity,
                dash_array=dash,
                popup=f"{rec}: {dist:.0f}m, ~{time_min:.0f}min, Risk {score}/100 ({level})",
                tooltip=rec,
            ).add_to(m)

        # Origin and destination markers
        if st.session_state.origin_coords:
            folium.Marker(
                location=st.session_state.origin_coords,
                popup=f"Origin: {st.session_state.origin_name}",
                icon=folium.Icon(color="green", icon="play", prefix="fa"),
            ).add_to(m)
        if st.session_state.dest_coords:
            folium.Marker(
                location=st.session_state.dest_coords,
                popup=f"Destination: {st.session_state.dest_name}",
                icon=folium.Icon(color="red", icon="flag", prefix="fa"),
            ).add_to(m)

        folium.LayerControl().add_to(m)
        st_folium(m, width=None, height=500)

    # --- ROUTE DETAILS ---
    with detail_col:
        st.subheader("Route Comparison")

        mode_label = {"walk": "Walking", "bike": "Biking", "drive": "Driving"}[st.session_state.travel_mode]

        for i, route in enumerate(st.session_state.scored_routes):
            risk = route.get("risk_score", {})
            crime = route.get("crime_stats", {})
            breakdown = risk.get("breakdown", {})
            rec = route.get("recommendation", f"Route {i+1}")

            with st.expander(f"{'>>> ' if i == 0 else ''}{rec} — {risk.get('level', '?')}", expanded=(i == 0)):
                # Key metrics
                c1, c2, c3 = st.columns(3)
                c1.metric("Distance", f"{route.get('distance_m', 0):.0f}m")
                c2.metric("Time", f"{route.get('estimated_time_min', 0):.0f} min")
                c3.metric("Risk", f"{risk.get('score', 0)}/100")

                st.markdown(f"**Mode**: {mode_label}")
                st.markdown(f"**Risk Level**: {risk.get('level', 'Unknown')}")

                # Crime breakdown
                st.markdown("**Crime Along Route**:")
                total_c = crime.get("total_crimes", 0)
                violent_c = crime.get("violent_crimes", 0)
                recent_c = crime.get("recent_crimes_30d", 0)
                st.markdown(f"- Total: {total_c} | Violent: {violent_c} | Last 30 days: {recent_c}")

                by_cat = crime.get("by_category", {})
                if by_cat:
                    top_cats = sorted(by_cat.items(), key=lambda x: -x[1])[:5]
                    for cat_name, count in top_cats:
                        st.markdown(f"  - {cat_name}: {count}")

                # Risk factors
                st.markdown("**Risk Factors**:")
                st.markdown(f"- Time period: {breakdown.get('temporal_period', '?')} ({breakdown.get('temporal_multiplier', 1.0)}x)")
                st.markdown(f"- Emergency phones: {breakdown.get('emergency_phones_nearby', 0)}")
                st.markdown(f"- Patrol level: {breakdown.get('patrol_level', 'unknown')}")
                st.markdown(f"- Mode factor: {breakdown.get('mode_multiplier', 1.0)}x")

        # Shuttle info
        shuttle = st.session_state.shuttle_info
        if shuttle:
            st.divider()
            st.subheader("Shuttle Option")
            if shuttle.get("available"):
                st.success("Tiger Line shuttle is currently running!")
                origin_stop = shuttle.get("nearest_origin_stop", {})
                dest_stop = shuttle.get("nearest_dest_stop", {})
                st.markdown(f"**Walk to stop**: {origin_stop.get('name', 'N/A')} ({origin_stop.get('distance_m', 0):.0f}m)")
                st.markdown(f"**Walk from stop**: {dest_stop.get('name', 'N/A')} ({dest_stop.get('distance_m', 0):.0f}m)")
                st.markdown(f"**Routes**: {', '.join(shuttle.get('available_routes', []))}")
                st.markdown(f"**Eligibility**: {shuttle.get('eligibility', 'MU ID holders')}")
                st.markdown(f"**Track live**: [tiger.etaspot.net](https://tiger.etaspot.net)")
            else:
                st.info(f"Shuttle not currently available: {shuttle.get('reason', '')}")
                if shuttle.get("next_service"):
                    st.markdown(f"Next service: {shuttle['next_service']}")
                if shuttle.get("nearest_origin_stop"):
                    st.markdown(f"Nearest stop to origin: {shuttle['nearest_origin_stop'].get('name', 'N/A')}")

    # === AI ANALYSIS ===
    st.divider()
    st.subheader("AI Safety Advisor")

    if st.session_state.ai_analysis:
        st.markdown(st.session_state.ai_analysis)

    st.divider()

    # Chat interface
    st.subheader("Ask TigerSafe")
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("Ask about safety, routes, or campus..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response, updated_history = chat_with_advisor(
                    user_message=prompt,
                    conversation_history=st.session_state.conversation_history,
                    route_context=st.session_state.route_context,
                    api_key=api_key,
                )
                st.markdown(response)

        st.session_state.messages.append({"role": "assistant", "content": response})
        st.session_state.conversation_history = updated_history

else:
    # No routes calculated yet — show intro
    st.info("Select an origin and destination above, choose your travel mode, and click **Calculate Routes** to get started.")

    # Show map with campus overview
    m = folium.Map(location=[38.9404, -92.3277], zoom_start=15, tiles="CartoDB positron")

    # Crime heatmap
    if st.session_state.show_heatmap and st.session_state.crimes is not None and not st.session_state.crimes.empty:
        heatmap_data = []
        for _, row in st.session_state.crimes.iterrows():
            heatmap_data.append([row.geometry.y, row.geometry.x, float(row.get("severity", 0.5))])
        if heatmap_data:
            HeatMap(heatmap_data, radius=25, blur=15, name="Crime Heatmap").add_to(m)

    # Emergency phones
    if st.session_state.show_phones and st.session_state.phones is not None and not st.session_state.phones.empty:
        for _, row in st.session_state.phones.iterrows():
            folium.Marker(
                location=[row.geometry.y, row.geometry.x],
                popup=str(row.get("DESCRIPTIO", row.get("name", "Emergency Phone"))),
                icon=folium.Icon(color="blue", icon="phone", prefix="fa"),
            ).add_to(m)

    # Shuttle stops
    if st.session_state.show_shuttle_stops:
        shuttle_stops_df = cached_load_shuttle_stops()
        if not shuttle_stops_df.empty:
            for _, row in shuttle_stops_df.iterrows():
                if pd.notna(row.get("lat")) and pd.notna(row.get("lng")):
                    folium.CircleMarker(
                        location=[row["lat"], row["lng"]],
                        radius=4, color="#127AD1", fill=True,
                        fill_opacity=0.6, popup=str(row.get("name", "Stop")),
                    ).add_to(m)

    folium.LayerControl().add_to(m)
    st_folium(m, width=None, height=500)

    # Quick actions
    st.divider()
    st.subheader("Shuttle Status")
    availability = check_shuttle_availability()
    for name, info in availability.items():
        status = "Running" if info.get("available") else "Not running"
        reason = info.get("reason", "")
        freq = info.get("frequency", "")
        elig = info.get("eligibility", "")
        st.markdown(f"**{name}**: {status} — {reason} {freq}")
        if elig:
            st.caption(f"  Eligibility: {elig}")

    # Chat even without routes
    st.divider()
    st.subheader("Ask TigerSafe")
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("Ask about campus safety..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response, updated_history = chat_with_advisor(
                    user_message=prompt,
                    conversation_history=st.session_state.conversation_history,
                    route_context="",
                    api_key=api_key,
                )
                st.markdown(response)

        st.session_state.messages.append({"role": "assistant", "content": response})
        st.session_state.conversation_history = updated_history
