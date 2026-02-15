# TigerSafe — Campus Dispatch Copilot

A safety-aware routing system for Mizzou students. TigerSafe combines real crime data, OpenStreetMap walking networks, and an AI-powered chat assistant to help students find safe routes across campus.

## Features

- **Safety-weighted routing** — Finds routes that balance distance with crime density, emergency phone proximity, and time-of-day risk
- **Interactive crime heatmap** — Folium-based map with crime density visualization, emergency phone markers, and building overlays
- **AI chat assistant** — Claude-powered "TigerSafe" agent that answers safety questions, compares routes, and explains risk factors
- **Temporal risk analysis** — Risk scores adjust based on time of day (late night vs. daytime)
- **Explainable risk scores** — Every recommendation comes with a breakdown of contributing factors

## Architecture

```
┌──────────────┐     ┌──────────────────┐     ┌──────────────┐
│  Streamlit   │────▶│  Claude Agent    │────▶│   Safety     │
│  Frontend    │     │  (tool-use API)  │     │   Engine     │
│  + Folium    │     │  5 tools         │     │  + OSMnx     │
└──────────────┘     └──────────────────┘     └──────────────┘
       │                                            │
       └──────────── Shared Data Layer ─────────────┘
                    (GeoJSON + GraphML)
```

**Modules:**
| File | Purpose |
|------|---------|
| `app.py` | Streamlit UI with Folium map and chat interface |
| `agent.py` | Claude-powered AI agent with 5 safety tools |
| `safety.py` | Risk scoring engine, safety-weighted routing |
| `ingest.py` | Data ingestion from CPD, MU ArcGIS, and OSM |
| `seed_data.py` | Sample data generator for demo/offline use |

## Quick Start

### 1. Install dependencies

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Generate seed data

This downloads the OSM walking network (requires internet) and generates sample crime/infrastructure data:

```bash
python seed_data.py
```

### 3. Set your API key

The AI chat requires an Anthropic API key:

```bash
export ANTHROPIC_API_KEY="sk-ant-..."
```

Or enter it in the sidebar when the app launches.

### 4. Run the app

```bash
streamlit run app.py
```

### 5. In the app

1. Click **"Load / Refresh Data"** in the sidebar to load crime and infrastructure data
2. Click **"Prepare Routing Graph"** to build the safety-weighted network (takes a moment)
3. Start chatting! Try: *"What's the safest route from Ellis Library to the Rec Center?"*

## Example Queries

- "What's the safest route from Jesse Hall to Engineering?"
- "Are there any recent incidents near Memorial Union?"
- "What safety factors should I be aware of near the Columns at night?"
- "Is the shuttle running right now?"
- "Compare the safest and fastest routes from the Student Center to Hearnes Center"

## Data Sources

| Source | Type | Access |
|--------|------|--------|
| OpenStreetMap (osmnx) | Walking network | API (free) |
| CPD Transparency Portal | Crime statistics | ArcGIS REST API |
| MUPD Daily Crime Log | Campus incidents | HTML scraping |
| MU Campus Map | Buildings, emergency phones | ArcGIS REST API |
| Tiger Line GTFS | Shuttle schedules | Static feed |

The app falls back to realistic sample data if live APIs are unavailable.

## Risk Score Algorithm

Each route receives a risk score (0-100) based on:

- **Base crime score**: 10 points per nearby incident
- **Violent crime bonus**: +15 for violent offenses
- **Time-of-day multiplier**: 2x for 10pm-6am, 0.3x for daytime
- **Emergency phone proximity**: Reduces score near blue-light phones
- **Length normalization**: Score is per-100m of route

| Score | Level | Color |
|-------|-------|-------|
| 0-20 | Very Safe | Green |
| 21-40 | Safe | Blue |
| 41-60 | Moderate Risk | Orange |
| 61-80 | Higher Risk | Red |
| 81-100 | High Risk | Dark Red |

## Project Structure

```
.
├── app.py              # Streamlit application
├── agent.py            # Claude AI agent with tools
├── safety.py           # Safety scoring & routing engine
├── ingest.py           # Data ingestion pipelines
├── seed_data.py        # Sample data generator
├── requirements.txt    # Python dependencies
├── data/               # Generated data files (gitignored)
│   ├── columbia_walk.graphml
│   ├── cpd_crimes.geojson
│   ├── emergency_phones.geojson
│   └── buildings.geojson
├── design.md           # Architectural design document
└── data_analysis.md    # Data source analysis
```
