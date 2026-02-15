"""
Campus Dispatch Copilot — TigerSafe

Streamlit application with Folium map visualization and Claude-powered
AI chat for safe campus routing at Mizzou.

Run with: streamlit run app.py
"""

import json
import os
from datetime import datetime
from pathlib import Path

import folium
import geopandas as gpd
import pandas as pd
import streamlit as st
from folium.plugins import HeatMap
from streamlit_folium import st_folium

from agent import CAMPUS_LOCATIONS, chat, resolve_location
from ingest import ingest_all
from safety import (
    compute_crime_heatmap_data,
    find_fastest_route,
    find_safest_route,
    get_nearby_crimes,
    get_temporal_multiplier,
    load_crimes,
    load_emergency_phones,
    prepare_graph,
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
    .risk-very-safe { color: #28a745; font-weight: bold; }
    .risk-safe { color: #17a2b8; font-weight: bold; }
    .risk-moderate { color: #ffc107; font-weight: bold; }
    .risk-higher { color: #dc3545; font-weight: bold; }
    .risk-high { color: #721c24; font-weight: bold; }
    .stChatMessage { max-height: 400px; overflow-y: auto; }
</style>
""", unsafe_allow_html=True)


# --- Session State Initialization ---
def init_session_state():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "conversation_history" not in st.session_state:
        st.session_state.conversation_history = []
    if "data_loaded" not in st.session_state:
        st.session_state.data_loaded = False
    if "graph" not in st.session_state:
        st.session_state.graph = None
    if "crimes" not in st.session_state:
        st.session_state.crimes = None
    if "phones" not in st.session_state:
        st.session_state.phones = None
    if "buildings" not in st.session_state:
        st.session_state.buildings = None
    if "last_safest_route" not in st.session_state:
        st.session_state.last_safest_route = None
    if "last_fastest_route" not in st.session_state:
        st.session_state.last_fastest_route = None
    if "show_heatmap" not in st.session_state:
        st.session_state.show_heatmap = True
    if "show_phones" not in st.session_state:
        st.session_state.show_phones = True
    if "show_buildings" not in st.session_state:
        st.session_state.show_buildings = False
    if "hour_filter" not in st.session_state:
        st.session_state.hour_filter = datetime.now().hour
    if "map_click" not in st.session_state:
        st.session_state.map_click = None


init_session_state()


# --- Sidebar ---
with st.sidebar:
    st.title("TigerSafe")
    st.caption("Campus Dispatch Copilot for Mizzou")
    st.divider()

    # API Key
    api_key = st.text_input(
        "Anthropic API Key",
        type="password",
        value=os.environ.get("ANTHROPIC_API_KEY", ""),
        help="Required for AI chat. Set ANTHROPIC_API_KEY env var or enter here.",
    )

    st.divider()

    # Data loading
    st.subheader("Data")
    if st.button("Load / Refresh Data", use_container_width=True):
        with st.spinner("Ingesting data..."):
            try:
                ingest_all(force=False)
                st.session_state.crimes = load_crimes()
                st.session_state.phones = load_emergency_phones()
                st.session_state.data_loaded = True
                st.success(f"Loaded {len(st.session_state.crimes)} crime records")
            except Exception as e:
                st.error(f"Data load error: {e}")

    if st.button("Prepare Routing Graph", use_container_width=True):
        with st.spinner("Building safety-weighted graph (this may take a moment)..."):
            try:
                hour = st.session_state.hour_filter
                st.session_state.graph = prepare_graph(hour=hour)
                st.success("Routing graph ready!")
            except Exception as e:
                st.error(f"Graph error: {e}")

    data_status = "Ready" if st.session_state.data_loaded else "Not loaded"
    graph_status = "Ready" if st.session_state.graph is not None else "Not loaded"
    st.caption(f"Data: {data_status} | Graph: {graph_status}")

    st.divider()

    # Map controls
    st.subheader("Map Layers")
    st.session_state.show_heatmap = st.checkbox("Crime Heatmap", value=st.session_state.show_heatmap)
    st.session_state.show_phones = st.checkbox("Emergency Phones", value=st.session_state.show_phones)
    st.session_state.show_buildings = st.checkbox("Buildings", value=st.session_state.show_buildings)

    st.divider()

    # Time control
    st.subheader("Time of Day")
    st.session_state.hour_filter = st.slider(
        "Hour (0-23)",
        min_value=0,
        max_value=23,
        value=st.session_state.hour_filter,
        help="Adjust to see risk at different times",
    )
    mult = get_temporal_multiplier(st.session_state.hour_filter)
    period_names = {0.3: "Daytime", 0.5: "Early Morning", 0.6: "Evening", 1.0: "Night", 1.5: "Late Evening", 1.8: "Late Night"}
    period = period_names.get(mult, "")
    st.caption(f"{period} — Risk multiplier: {mult}x")

    st.divider()

    # Quick locations
    st.subheader("Quick Locations")
    quick_loc = st.selectbox(
        "Jump to location",
        options=[""] + sorted(CAMPUS_LOCATIONS.keys()),
        format_func=lambda x: x.title() if x else "Select a location...",
    )

    st.divider()
    if st.button("Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.conversation_history = []
        st.session_state.last_safest_route = None
        st.session_state.last_fastest_route = None
        st.rerun()


# --- Main Layout ---
map_col, chat_col = st.columns([3, 2])


# --- Map ---
with map_col:
    st.subheader("Campus Safety Map")

    # Determine map center
    center_lat, center_lon = 38.9404, -92.3277
    zoom = 15
    if quick_loc and quick_loc in CAMPUS_LOCATIONS:
        center_lat, center_lon = CAMPUS_LOCATIONS[quick_loc]
        zoom = 17

    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=zoom,
        tiles="CartoDB positron",
    )

    # Crime heatmap layer
    if st.session_state.show_heatmap and st.session_state.crimes is not None:
        heatmap_data = compute_crime_heatmap_data(
            st.session_state.crimes,
            hour_filter=st.session_state.hour_filter,
        )
        if heatmap_data:
            HeatMap(
                heatmap_data,
                radius=25,
                blur=15,
                gradient={0.2: "blue", 0.4: "lime", 0.6: "yellow", 0.8: "orange", 1.0: "red"},
                name="Crime Heatmap",
            ).add_to(m)

    # Emergency phones
    if st.session_state.show_phones and st.session_state.phones is not None:
        phone_group = folium.FeatureGroup(name="Emergency Phones")
        for _, row in st.session_state.phones.iterrows():
            folium.Marker(
                location=[row.geometry.y, row.geometry.x],
                popup=row.get("name", "Emergency Phone"),
                icon=folium.Icon(color="blue", icon="phone", prefix="fa"),
            ).add_to(phone_group)
        phone_group.add_to(m)

    # Buildings
    if st.session_state.show_buildings and st.session_state.buildings is not None:
        bldg_group = folium.FeatureGroup(name="Buildings")
        for _, row in st.session_state.buildings.iterrows():
            name = row.get("BUILDING_NAME", "Building")
            folium.Marker(
                location=[row.geometry.y, row.geometry.x],
                popup=name,
                icon=folium.Icon(color="gray", icon="building", prefix="fa"),
            ).add_to(bldg_group)
        bldg_group.add_to(m)

    # Routes
    if st.session_state.last_safest_route and "coordinates" in st.session_state.last_safest_route:
        route = st.session_state.last_safest_route
        risk = route.get("risk_score", {})
        color = risk.get("color", "green") if isinstance(risk, dict) else "green"
        score = risk.get("score", "?") if isinstance(risk, dict) else "?"

        folium.PolyLine(
            locations=route["coordinates"],
            color=color,
            weight=6,
            opacity=0.8,
            popup=f"Safest Route — Risk: {score}",
            tooltip="Safest Route",
        ).add_to(m)

        # Origin marker
        origin = route["coordinates"][0]
        folium.Marker(
            location=origin,
            popup="Origin",
            icon=folium.Icon(color="green", icon="play", prefix="fa"),
        ).add_to(m)

        # Destination marker
        dest = route["coordinates"][-1]
        folium.Marker(
            location=dest,
            popup="Destination",
            icon=folium.Icon(color="red", icon="flag", prefix="fa"),
        ).add_to(m)

    if st.session_state.last_fastest_route and "coordinates" in st.session_state.last_fastest_route:
        route = st.session_state.last_fastest_route
        folium.PolyLine(
            locations=route["coordinates"],
            color="blue",
            weight=4,
            opacity=0.5,
            dash_array="10",
            popup="Fastest Route",
            tooltip="Fastest Route (dashed)",
        ).add_to(m)

    # Layer control
    folium.LayerControl().add_to(m)

    # Render map
    map_data = st_folium(m, width=None, height=500, returned_objects=["last_clicked"])

    # Handle map clicks
    if map_data and map_data.get("last_clicked"):
        clicked = map_data["last_clicked"]
        st.session_state.map_click = (clicked["lat"], clicked["lng"])
        st.caption(f"Clicked: ({clicked['lat']:.5f}, {clicked['lng']:.5f})")

    # Route info cards below map
    if st.session_state.last_safest_route and "risk_score" in st.session_state.last_safest_route:
        sr = st.session_state.last_safest_route
        fr = st.session_state.last_fastest_route

        col1, col2 = st.columns(2)
        with col1:
            risk = sr.get("risk_score", {})
            st.markdown(f"""
            **Safest Route**
            - Distance: {sr.get('distance_m', '?')}m
            - Walking time: ~{sr.get('duration_min', '?')} min
            - Risk: {risk.get('score', '?')} ({risk.get('level', '?')})
            - Crimes nearby: {sr.get('crime_count_nearby', 0)}
            """)

        with col2:
            if fr:
                risk_f = fr.get("risk_score", {})
                st.markdown(f"""
                **Fastest Route**
                - Distance: {fr.get('distance_m', '?')}m
                - Walking time: ~{fr.get('duration_min', '?')} min
                - Risk: {risk_f.get('score', '?')} ({risk_f.get('level', '?')})
                - Crimes nearby: {fr.get('crime_count_nearby', 0)}
                """)


# --- Chat ---
with chat_col:
    st.subheader("Chat with TigerSafe")

    # Display chat messages
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Chat input
    if prompt := st.chat_input("Ask about campus safety, routes, or incidents..."):
        # Display user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Build app state for agent
        app_state = {
            "graph": st.session_state.graph,
            "crimes": st.session_state.crimes,
            "phones": st.session_state.phones,
            "last_safest_route": st.session_state.last_safest_route,
            "last_fastest_route": st.session_state.last_fastest_route,
        }

        # Get AI response
        with st.chat_message("assistant"):
            with st.spinner("Analyzing..."):
                response_text, updated_history = chat(
                    user_message=prompt,
                    conversation_history=st.session_state.conversation_history,
                    app_state=app_state,
                    api_key=api_key,
                )
                st.markdown(response_text)

        # Update state
        st.session_state.messages.append({"role": "assistant", "content": response_text})
        st.session_state.conversation_history = updated_history

        # Check if agent produced route data
        if app_state.get("last_safest_route") != st.session_state.last_safest_route:
            st.session_state.last_safest_route = app_state["last_safest_route"]
            st.session_state.last_fastest_route = app_state.get("last_fastest_route")
            st.rerun()

    # Quick action buttons
    st.divider()
    st.caption("Quick Actions")

    qcol1, qcol2 = st.columns(2)
    with qcol1:
        if st.button("What's safe right now?", use_container_width=True):
            hour = datetime.now().hour
            auto_msg = f"What areas of campus are safest to walk through right now (it's {hour}:00)?"
            st.session_state.messages.append({"role": "user", "content": auto_msg})
            st.rerun()

    with qcol2:
        if st.button("Recent incidents", use_container_width=True):
            auto_msg = "What are the most recent crime incidents on or near campus?"
            st.session_state.messages.append({"role": "user", "content": auto_msg})
            st.rerun()

    qcol3, qcol4 = st.columns(2)
    with qcol3:
        if st.button("Shuttle info", use_container_width=True):
            auto_msg = "What shuttle routes are available right now?"
            st.session_state.messages.append({"role": "user", "content": auto_msg})
            st.rerun()

    with qcol4:
        if st.button("Safety tips", use_container_width=True):
            auto_msg = "What are some general safety tips for walking on campus at night?"
            st.session_state.messages.append({"role": "user", "content": auto_msg})
            st.rerun()
