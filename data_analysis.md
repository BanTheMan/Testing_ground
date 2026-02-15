# Building a safety-aware routing app for Mizzou students

**The most efficient path to a working prototype combines osmnx for network routing, Claude's direct tool-use API for the AI layer, and Streamlit with Folium for visualization—all backed by a SQLite/GeoPandas crime database pre-loaded before the hackathon.** Of the nine data sources identified, six offer structured or semi-structured access (CSV exports, GTFS feeds, ArcGIS REST APIs), two require HTML scraping, and one demands PDF extraction. The critical architectural decision is building **pre-computed safety-weighted edge attributes** on the OSM road graph, updated by time-of-day multipliers at query time, rather than computing safety scores on the fly.

---

## A. Data ingestion playbook for all nine sources

### Source 1: OpenStreetMap road and walkway network via osmnx

The Geofabrik Missouri extract (176 MB PBF, 341 MB shapefile, updated daily) covers the entire state, but **osmnx is the clear winner** for a Columbia-scale project. A single function call downloads, cleans, and returns a routable NetworkX MultiDiGraph:

```python
import osmnx as ox
G_walk = ox.graph_from_place("Columbia, Missouri, USA", network_type="walk")
G_bike = ox.graph_from_place("Columbia, Missouri, USA", network_type="bike")
G_drive = ox.graph_from_place("Columbia, Missouri, USA", network_type="drive")
```

osmnx queries the Overpass API by place name, bounding box, or point-with-radius, automatically simplifies topology, and caches results locally. For safety-relevant infrastructure, fetch street lighting and footpath data separately:

```python
lighting = ox.features_from_place("Columbia, Missouri, USA", tags={"highway": "street_lamp"})
lit_roads = ox.features_from_place("Columbia, Missouri, USA", tags={"lit": True})
```

**Important caveat**: OSM's `lit=yes/no` tag coverage in Columbia is incomplete—community-contributed data varies. Supplement with MU campus ArcGIS infrastructure layers (Source 8) where possible. The `graph_to_gdfs()` and `graph_from_gdfs()` conversion functions are essential: modify edge attributes in the GeoDataFrame using vectorized pandas operations, then rebuild the graph for routing.

### Source 2: Columbia Police dispatch data with CSV export

The City of Columbia Police provides a **near-real-time dispatch feed** at `como.gov/CMS/911dispatch/police.php` with a CSV export endpoint at `police_csvexport.php`. Data arrives with a **6-hour delay** and includes date/time, incident number, location (block-level addresses like "1200 BLOCK ELLETA BLVD"), and incident type across **60+ categories** (assault, shots fired, robbery, burglary, suspicious person, etc.).

```python
import pandas as pd
dispatch_df = pd.read_csv("https://www.como.gov/CMS/911dispatch/police_csvexport.php")
```

Additionally, **yearly vehicle stop CSVs** (2014–2024) are available at predictable URLs: `como.gov/wp-content/uploads/YYYY/MM/CPD_vehicle_stop_data_YYYY.csv`. The city also operates an ArcGIS-powered 911 Calls Map at `gis.gocolumbiamo.com/911Calls/` which likely has queryable REST endpoints, and an open data portal at `datahub-gocolumbiamo.opendata.arcgis.com/`. Block-level addresses will need geocoding—use a batch geocoder like the Census Bureau's or Nominatim before the hackathon to avoid API rate limits during the event.

### Source 3: MUPD crime data hub and daily logs

The MUPD news-crime-data page links to four key resources: the Daily Crime Log (Source 5), a Daily Incident Log at `dilog.php`, Missouri State Highway Patrol crime statistics filtered to the MU jurisdiction, and the annual Clery report (Source 6). **No APIs exist anywhere in the MUPD ecosystem**—all automated access requires scraping. The MSHP statistics at `showmecrime.mo.gov` provide UCR/NIBRS-compliant data filterable to "University of Missouri—Columbia," offering a structured complement to the scraped logs.

### Source 4: CPD Transparency Portal is an ArcGIS Hub with REST access

This is a **confirmed ArcGIS Hub site** providing NIBRS-compliant crime statistics, use-of-force data, vehicle stop breakdowns, and officer diversity data. The underlying ArcGIS server at `gis.gocolumbiamo.com/arcgis/rest/services/` supports REST queries returning JSON and GeoJSON. ArcGIS Hub datasets typically offer **CSV, GeoJSON, Shapefile, and KML** downloads directly from dataset pages.

```python
# Once you discover the FeatureServer URL via browser inspection:
import requests, geopandas as gpd
url = "https://gis.gocolumbiamo.com/arcgis/rest/services/.../FeatureServer/0/query"
params = {"where": "1=1", "outFields": "*", "f": "geojson"}
gdf = gpd.read_file(requests.get(url, params=params).text)
```

The portal pages render via JavaScript, so the exact Feature Service URLs need discovery through browser DevTools (Network tab) or the ArcGIS Hub search API. Reports update **daily** with a ~3-day review lag and only include incidents with completed police reports. **This is likely the highest-quality structured crime dataset available** for the project, since it uses standardized NIBRS codes and includes geolocation.

### Source 5: MUPD Daily Crime Log is a scrapable HTML table

The PHP page at `muop-mupdreports.missouri.edu/dclog.php` serves a **server-rendered HTML table** (not JavaScript-loaded) with seven confirmed columns: Case Number, Date/Time Reported, Location of Occurrence, Domestic DV Relationship, Incident Type, Criminal Offense, and Disposition. A sub-row carries Date/Time Occurred. The page accepts a time-range parameter (1 month through "All"), and the default 1-month view returned **206 records** as of mid-February 2026.

```python
import pandas as pd
tables = pd.read_html("https://muop-mupdreports.missouri.edu/dclog.php")
crime_log = tables[0]  # May need cleanup for merged cells
```

**Key advantage**: addresses are street-level (e.g., "820 CONLEY AVE, COLUMBIA MO, 65201") rather than block-level, making geocoding more precise. Some entries use intersection format ("799 S COLLEGE AVE/ROLLINS ST"), requiring special parsing. Multiple offense types per case produce multiple rows—deduplicate by case number when counting incidents.

### Source 6: Clery report provides context, not routing-grade data

The 2025 Annual Security and Fire Safety Report is a **201-page PDF** with crime statistics tables starting around page 141. It covers the three most recent calendar years across four broad geographic categories: On-Campus, On-Campus Student Housing, Non-Campus, and Public Property. **Location granularity is far too coarse for route-level analysis**—this data is useful only for aggregate risk context and trend narratives in the AI advisor.

Extract tables with **pdfplumber** (preferred for Clery-style lattice tables):

```python
import pdfplumber
with pdfplumber.open("2025-Clery-Report.pdf") as pdf:
    for page in pdf.pages[140:178]:
        tables = page.extract_tables()
```

An alternative structured source is the **Department of Education Campus Safety database** at `ope.ed.gov/campussafety/`, which provides the same Clery statistics in a queryable web interface—potentially easier than PDF extraction.

### Source 7: Tiger Line Shuttle has a confirmed GTFS feed

The Tiger Line operates **five routes** (Hearnes Loop, Trowbridge Loop, MU Reactor Loop, Campus Loop, MU Health Care Loop) in partnership with GoCOMO transit. The GTFS static feed is hosted at:

```
https://s3.amazonaws.com/etatransit.gtfs/gocomotransit.etaspot.net/gtfs.zip
```

This ZIP contains standard GTFS text files (stops.txt, routes.txt, trips.txt, stop_times.txt, shapes.txt). Filter Tiger Line routes by name pattern matching. Real-time vehicle tracking uses the **ETA SPOT platform** (GoCOMO switched from DoubleMap in December 2024) at `gocomotransit.etaspot.net`. GTFS-Realtime feeds may exist at predictable S3 URLs following the ETA SPOT pattern (`position_updates.pb`, `trip_updates.pb`, `alerts.pb`) but need verification.

```python
import partridge as ptg
feed = ptg.load_feed("https://s3.amazonaws.com/etatransit.gtfs/gocomotransit.etaspot.net/gtfs.zip")
tiger_routes = feed.routes[feed.routes.route_long_name.str.contains(
    'Tiger|Hearnes|Trowbridge|Reactor|Campus', case=False)]
```

### Sources 8 and 9: MU Campus Map and Dashboard expose ArcGIS REST APIs

Both applications run on MU's ArcGIS infrastructure with the services directory at `map.missouri.edu:6443/arcgis/rest/services/`. The Campus Map (Source 8) draws from **MU_Base_new/MapServer** (basemap with buildings, roads, annotations) and **MU_Features_new/FeatureServer** (queryable feature layers including construction projects at Layer 11, chair lifts at Layer 7, and likely parking, paths, emergency phones, and bike racks on other layers). The Dashboard (Source 9) is specifically the **Construction Map** feeding from FeatureServer/11.

```python
import requests
# Discover all layers
resp = requests.get(
    "https://map.missouri.edu:6443/arcgis/rest/services/MU_Features_new/FeatureServer?f=json",
    verify=False  # SSL cert may be self-signed
)
layers = resp.json()["layers"]  # Lists all available layers with IDs and names

# Query construction zones (avoidance areas for routing)
params = {"where": "1=1", "outFields": "*", "outSR": "4326", "f": "geojson"}
construction = requests.get(
    "https://map.missouri.edu:6443/arcgis/rest/services/MU_Features_new/FeatureServer/11/query",
    params=params, verify=False
)
```

**Note**: The native spatial reference is NAD 1983 StatePlane Missouri Central (WKID: 102697)—always request `outSR=4326` for WGS84 compatibility with GeoPandas. SSL certificate issues may require `verify=False` in requests.

---

## B. Unified schema and crime taxonomy normalization

### A multi-layer GeoDataFrame architecture

Store each data type as a separate GeoDataFrame with a consistent CRS (EPSG:4326 for storage, EPSG:32615/UTM Zone 15N for spatial operations). All layers link to the road network through `(u, v, key)` edge identifiers or nearest-edge references.

The **crime incidents layer** needs: `incident_id`, `source` (CPD/MUPD/Clery), `datetime`, `hour`, `day_of_week`, `crime_category` (unified), `severity` (0–1 float), `is_violent` (boolean), `original_category`, `geometry` (Point), and `nearest_edge_u/v/key` for graph linkage. The **network edges layer** extends osmnx's default attributes with `crime_count`, `crime_density` (per 100m), `violent_crime_count`, `safety_score`, `lighting_score`, and precomputed temporal weights (`weight_day`, `weight_night`). Transit and infrastructure layers carry their own schemas for stops, routes, streetlights, and emergency phones.

Use **GeoPackage** (`.gpkg`) for persistent multi-layer storage in a single file—it supports multiple geometry types and attribute tables natively.

### Normalizing CPD, MUPD, and Clery crime categories

**NIBRS (National Incident-Based Reporting System) is the natural common backbone.** CPD already reports NIBRS-compliant data through the transparency portal. MUPD categories map closely to NIBRS offense codes. Clery categories are a strict subset of NIBRS. Build a mapping dictionary from each source's original categories to a unified taxonomy with **severity weights calibrated for pedestrian safety**:

| Unified Category | Severity | Violent? | Maps From |
|---|---|---|---|
| Homicide | 1.0 | Yes | Murder/Non-Negligent Manslaughter (Clery), Homicide (NIBRS 09A/09B) |
| Sexual assault | 0.95 | Yes | Rape, Fondling (Clery), Sex Offenses (NIBRS 11A-11D) |
| Robbery | 0.9 | Yes | ROBBERY 1ST/2ND (CPD), Robbery (MUPD/Clery) |
| Aggravated assault | 0.85 | Yes | ASSAULT 1ST (CPD), Assault-Aggravated (MUPD) |
| Simple assault | 0.6 | Yes | ASSAULT 3RD (CPD), Assault-Simple (MUPD) |
| Burglary | 0.4 | No | BURGLARY (CPD/MUPD/Clery) |
| Theft/larceny | 0.3 | No | STEALING (CPD), Larceny (MUPD) |
| Vandalism | 0.2 | No | PROPERTY DAMAGE (CPD/MUPD) |
| Drug offense | 0.3 | No | Drug violations across all sources |
| Weapons violation | 0.5 | No | Weapons Law (MUPD), Weapons (CPD) |

These severity weights drive the routing engine: a road segment near recent aggravated assaults gets a much higher cost penalty than one near property thefts.

### Spatial joining crimes to the road network

The pipeline for attributing crime points to road segments uses GeoPandas `sjoin_nearest` with a **50-meter maximum snap distance** in a projected CRS:

```python
crimes_proj = crimes_gdf.to_crs(epsg=32615)
edges_proj = gdf_edges.to_crs(epsg=32615)
crime_to_edge = gpd.sjoin_nearest(crimes_proj, edges_proj, how="inner",
                                   max_distance=50, distance_col="snap_dist")
crime_counts = crime_to_edge.groupby(["u", "v", "key"]).agg(
    crime_count=("index_left", "count"),
    violent_count=("is_violent", "sum")
).reset_index()
gdf_edges = gdf_edges.merge(crime_counts, on=["u", "v", "key"], how="left").fillna(0)
gdf_edges["crime_density"] = gdf_edges["crime_count"] / (gdf_edges["length"] / 100)
```

Crimes beyond 50 meters from any road (in buildings, parks) are still valuable—assign them to the nearest edge but flag `snap_dist > 30` for potential down-weighting.

---

## C. How to build the safety-weighted routing engine

### Composite edge weights combine distance, crime, and temporal risk

The routing engine assigns each edge a **composite weight** that Dijkstra's algorithm minimizes. The formula is a weighted linear combination:

```
composite_weight = α × length + β × (crime_density × temporal_multiplier) + γ × (1 - lighting_score)
```

where α, β, γ are tunable parameters (start with **0.4, 0.4, 0.2**). Convert edge attributes to a GeoDataFrame, compute composite weights as a new column, rebuild the graph, then route:

```python
route = ox.routing.shortest_path(G, orig_node, dest_node, weight="composite_weight")
```

For the hackathon, precompute two weight columns (`weight_day` and `weight_night`) and swap the active weight based on query time. A more granular approach uses **six time periods** with empirically calibrated multipliers: early morning (0.3×), morning (0.5×), afternoon (0.7×), evening (1.0×), night (1.5×), late night (1.8×). Use the `astral` library to compute actual sunset/sunrise for Columbia, MO and apply linear interpolation during twilight transitions.

### The SafePaths paper provides the theoretical foundation

Galbrun, Pelechrinis, and Terzi's 2016 paper "Urban Navigation Beyond Shortest Route: The Case of Safe Paths" formalized **safety-aware routing as a biobjective shortest path problem**—minimizing both distance and cumulative crime risk simultaneously. Their approach generates a **Pareto-optimal set** of routes showing explicit distance-vs-safety tradeoffs. This is the gold standard for presenting users with meaningful choices ("5 min longer but avoids high-crime area"). Other notable implementations include the "There and Back Again" project (San Francisco crime-aware routing using OSRM with custom Lua profiles) and the "Navigating Crime" capstone (LAPD data + XGBoost prediction + ArcGIS routing).

### Crime heatmaps via kernel density estimation

Generate continuous crime probability surfaces using scikit-learn's `KernelDensity` (preferred for larger datasets over scipy's `gaussian_kde`):

```python
from sklearn.neighbors import KernelDensity
kde = KernelDensity(kernel="gaussian", bandwidth=0.001)  # ~100m in degrees
kde.fit(crime_points)
```

Sample the KDE along route segments by interpolating points at regular intervals along each LineString geometry, scoring each sample, then averaging. Visualize with `folium.plugins.HeatMap` for the interactive map overlay, using a gradient from blue (safe) through yellow to red (dangerous). For temporal animation, `HeatMapWithTime` shows crime density shifting across hours of the day—a powerful demo feature.

---

## D. The agentic AI layer should use Claude's tool-use API directly

### Skip the frameworks—direct tool use is fastest for a hackathon

After evaluating LangChain, LangGraph, CrewAI, AutoGen, and Claude's native tool-use API, **direct Claude tool use with the Anthropic Python SDK wins decisively for hackathon speed**. The complete agent loop is ~50 lines of Python. No framework learning curve, no dependency conflicts, no abstraction layers to fight. LangGraph is designed for complex multi-step stateful workflows with conditional branching—overkill for a single-purpose safety agent. LangChain adds value mainly for RAG pipelines or multi-provider setups, neither of which this project needs.

The agent needs **five tools**: `query_crimes_near_location` (spatial query against GeoPandas), `get_route_safety_score` (runs Dijkstra with safety weights and returns the score breakdown), `check_shuttle_schedule` (queries GTFS data for next departures), `get_recent_incidents` (area-based crime lookup), and `explain_safety_factors` (returns lighting, crime density, and temporal risk for a location). Each tool is a Python function that encapsulates geospatial logic—**never let the LLM generate spatial queries directly**, as research confirms LLMs struggle with complex spatial reasoning.

### System prompt and tool schemas

Name the agent "TigerSafe." The system prompt should instruct it to always use tools before making safety claims, never fabricate statistics, suggest safer alternatives when risk is elevated, and mention shuttle options when relevant. Tool schemas use JSON Schema with `strict: true` for guaranteed parameter validation. Format tool results as concise JSON—drop geometry columns, limit records, and include summary statistics rather than raw data.

```python
import anthropic
client = anthropic.Anthropic()

def run_safety_agent(user_message, history=[]):
    messages = history + [{"role": "user", "content": user_message}]
    while True:
        response = client.messages.create(
            model="claude-sonnet-4-5-20250929",
            max_tokens=4096, system=system_prompt,
            tools=tools, messages=messages)
        messages.append({"role": "assistant", "content": response.content})
        if response.stop_reason == "end_turn":
            return response, messages
        if response.stop_reason == "tool_use":
            results = []
            for block in response.content:
                if block.type == "tool_use":
                    result = execute_tool(block.name, block.input)
                    results.append({"type": "tool_result",
                                    "tool_use_id": block.id,
                                    "content": json.dumps(result)})
            messages.append({"role": "user", "content": results})
```

---

## E. Hackathon execution strategy and tech stack

### The 24-hour build order

The MVP is one sentence: **a user asks "Is it safe to walk from the Rec Center to downtown at 11pm?" and gets a data-backed answer with a color-coded route on a map.** Everything else is polish.

| Hours | Task | Output |
|-------|------|--------|
| 0–2 | Environment setup, stack alignment | Repo, venv, API keys |
| 2–4 | Data prep: load CPD dispatch CSV + MUPD crime log into SQLite, geocode addresses | `safety_data.sqlite` with lat/lon |
| 4–8 | Core backend: osmnx graph + crime spatial join + 2 Claude tools (`query_crimes`, `get_safety_score`) | Working agent that answers safety questions |
| 8–12 | Streamlit app: Folium map (left) + chat interface (right) | Visual prototype |
| 12–16 | Integration: route visualization, heatmap overlay, map click for origin/dest | Connected frontend-backend |
| 16–20 | Polish: add shuttle data from GTFS, time-of-day slider, improve prompts | Feature-complete demo |
| 20–24 | Demo prep, bug fixes, script walkthrough scenarios | Presentation-ready |

**Pre-hackathon** (if rules allow): geocode all CPD and MUPD addresses, download and cache the osmnx graph, parse the GTFS feed, and pre-compute crime density per edge. This alone saves 3–4 hours.

### Streamlit + Folium is the optimal frontend stack

For a Python team at a hackathon, **Streamlit with streamlit-folium** delivers the best ratio of visual polish to development time. The two-column layout puts the interactive map on the left and the AI chat on the right:

```python
import streamlit as st
from streamlit_folium import st_folium
col_map, col_chat = st.columns([3, 2])
```

Folium handles the crime heatmap layer (`folium.plugins.HeatMap`), route polylines color-coded by safety score (green → yellow → red), shuttle stop markers, and construction zone polygons. The `st_folium` component returns click coordinates, enabling map-click origin/destination selection. Streamlit's `st.chat_input` and `st.chat_message` create the conversational interface. Use `st.session_state` for chat history persistence and `@st.cache_data` for expensive data loads.

Total frontend dependencies: `pip install streamlit streamlit-folium folium anthropic geopandas shapely pandas osmnx`.

### Data pipeline: pre-fetch everything for the demo

**Do not build a real-time pipeline during the hackathon.** Pre-fetch all crime data into a SQLite database with simple lat/lon columns (avoid SpatiaLite dependency headaches). Load into GeoPandas at app startup with `@st.cache_data`. For the "live" demo effect, add a sidebar showing 3–5 simulated recent alerts that rotate on refresh. Cache Claude API responses during development using `lru_cache` with coordinates rounded to 3 decimal places (~111m precision) to increase cache hit rates. Use Claude Haiku for testing iterations and switch to Sonnet for the final demo.

### Five pitfalls that kill hackathon geo projects

- **CRS mismatches**: Standardize everything to EPSG:4326 immediately on ingest. Project to EPSG:32615 only for distance calculations, then convert back.
- **Geocoding during the event**: Batch-geocode all addresses before the hackathon. Census Bureau's geocoder handles "820 CONLEY AVE, COLUMBIA MO" format well.
- **Map rendering performance**: Limit displayed crime points to the most recent 1,000. Use marker clustering for larger datasets.
- **SpatiaLite installation failures**: Use plain SQLite with lat/lon columns plus haversine distance in Python—avoids native library compilation issues entirely.
- **Overscoping**: The team does not need real-time scraping, production databases, or custom ML models. Pre-loaded data + osmnx routing + Claude tool use + Folium visualization is a complete, impressive demo.

---

## Conclusion

The nine data sources break cleanly into three tiers of accessibility. **Tier 1** (ready to use): OSM via osmnx, CPD dispatch CSV export, GoCOMO GTFS feed, and ArcGIS REST APIs for campus features—these require minimal engineering to ingest. **Tier 2** (light scraping): MUPD Daily Crime Log via `pandas.read_html` and CPD Transparency Portal via ArcGIS Hub dataset downloads—functional with a few hours of work. **Tier 3** (reference only): the Clery PDF provides trend context but lacks routing-grade location data.

The architecture that maximizes hackathon impact is: **precomputed safety-weighted osmnx graph → Claude tool-use agent with 5 geospatial tools → Streamlit + Folium frontend**. The biobjective routing approach from the SafePaths literature (presenting Pareto-optimal distance-vs-safety tradeoffs) is the most principled method, but a simpler weighted composite score gets 80% of the value in 20% of the implementation time. The single most valuable pre-hackathon task is geocoding all CPD and MUPD crime records and computing per-edge crime densities—this transforms the demo from "we have data" to "watch the route change in real time when you switch from day to night."