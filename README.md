# TigerSafe — Campus Dispatch Copilot

A safety-aware routing system for Mizzou students. TigerSafe combines real CPD/MUPD crime data, OpenStreetMap multi-modal networks, Tiger Line shuttle integration, and a Claude-powered AI safety advisor to help students find safe routes across campus.

## Features

- **Multi-modal routing** — Walk, bike, or drive routes via OSMnx with k-shortest-path alternatives
- **Tiger Line shuttle integration** — Real schedule/stop data, availability checks, eligibility info, and proximity search
- **Crime-aware risk scoring** — Composite score from crime density, severity, recency, time-of-day, emergency phone proximity, patrol frequency, and travel mode
- **Interactive map** — Folium-based with crime heatmap, emergency phones, buildings, shuttle stops, and color-coded route overlays
- **AI safety advisor** — Claude-powered RAG assistant using Clery Report context, route analysis, and a structured 4-beat response framework
- **Path comparison** — Side-by-side route metrics with pros/cons, labeled Safest / Fastest / Alternative

## Architecture

```
┌──────────────┐     ┌──────────────────┐     ┌──────────────┐
│  Streamlit   │────▶│  AI Advisor      │────▶│  Route       │
│  Frontend    │     │  (Claude + RAG)  │     │  Engine      │
│  + Folium    │     │  Clery Report    │     │  (OSMnx)     │
└──────────────┘     └──────────────────┘     └──────────────┘
       │                                            │
       └──────────── src/ Module Layer ─────────────┘
                    data_loader · crime_analyzer
                    risk_scorer · shuttle_service
```

**Modules:**
| File | Purpose |
|------|---------|
| `app.py` | Streamlit UI — map, route display, shuttle panel, AI chat |
| `src/data_loader.py` | Loads all CSV/GeoJSON data from `data/`, local geocoding |
| `src/crime_analyzer.py` | NIBRS-based crime normalization, spatial density analysis |
| `src/route_engine.py` | Multi-modal graph routing via OSMnx + NetworkX |
| `src/risk_scorer.py` | Composite risk scoring with temporal/mode/infrastructure factors |
| `src/shuttle_service.py` | Tiger Line schedule, stop proximity, availability checks |
| `src/ai_advisor.py` | Claude RAG advisor with Clery Report context and fallback analysis |

## Quick Start

### 1. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate  # macOS/Linux
# venv\Scripts\activate   # Windows
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Set your API key

Copy the example env file and add your Anthropic API key:

```bash
cp .env.example .env
# Edit .env and replace the placeholder with your key
```

Or enter the key in the sidebar when the app launches. The app works without an API key (fallback analysis mode), but the AI advisor requires one.

### 4. Run the app

```bash
streamlit run app.py
```

### 5. In the app

1. Select your **origin** and **destination** from the campus location dropdown (or enter coordinates manually)
2. Choose a **travel mode** (walk, bike, or drive)
3. Click **"Calculate Routes"** — the app downloads the OSMnx graph (first run only), finds alternative routes, scores them for safety, checks shuttle availability, and generates AI analysis
4. Explore the map, compare routes, check shuttle details, and chat with the AI advisor

## Example Queries (AI Chat)

- "What's the safest route from Jesse Hall to Engineering?"
- "Are there any recent incidents near Memorial Union?"
- "What safety factors should I be aware of near the Columns at night?"
- "Is the shuttle running right now?"
- "Compare the safest and fastest routes from the Student Center to Hearnes Center"

## Data Sources

| Source | Directory | Contents |
|--------|-----------|----------|
| CPD Crime Data | `data/crime_logs/` | Columbia Police Department crime CSV |
| MUPD Logs | `data/crime_logs/` | MU Police crime log + incident log CSVs |
| Campus GeoJSON | `data/campus_boundary/` | Boundary, buildings, emergency phones, accessible entrances |
| Tiger Line Shuttle | `data/shuttle_data/` | Routes (encoded polylines) and stop locations |
| Traffic Stops | `data/traffic_stops/` | CPD vehicle stop data (2014–2024) for patrol frequency |
| Clery Report | `data/rag_sources/` | Annual Security Report PDF for RAG context |
| OSMnx | Downloaded at runtime | Walk/bike/drive street networks, cached as GraphML |

## Risk Score Algorithm

Each route receives a composite score (0–100) from weighted factors:

| Factor | Weight | Description |
|--------|--------|-------------|
| Crime density | 45% | Incidents within 200m buffer, weighted by severity and violence |
| Temporal risk | 25% | Time-of-day multiplier (late night 1.8×, daytime 0.3×) |
| Recency | 15% | Recent crimes (7d/30d) weighted more heavily |
| Infrastructure | 15% | Emergency phone proximity (reduces score) and patrol frequency |

Mode multipliers adjust the final score: walk 1.3×, bike 1.0×, drive 0.5×.

| Score | Level | Color |
|-------|-------|-------|
| 0–20 | Very Safe | Green |
| 21–40 | Safe | Blue |
| 41–60 | Moderate Risk | Yellow |
| 61–80 | Higher Risk | Orange |
| 81–100 | High Risk | Red |

## Project Structure

```
.
├── app.py                  # Streamlit application
├── src/
│   ├── __init__.py
│   ├── data_loader.py      # Central data loading and geocoding
│   ├── crime_analyzer.py   # Crime normalization and spatial analysis
│   ├── route_engine.py     # Multi-modal OSMnx routing
│   ├── risk_scorer.py      # Composite risk scoring engine
│   ├── shuttle_service.py  # Tiger Line shuttle integration
│   └── ai_advisor.py       # Claude RAG safety advisor
├── data/
│   ├── crime_logs/         # CPD + MUPD crime data CSVs
│   ├── campus_boundary/    # GeoJSON: boundary, buildings, phones
│   ├── shuttle_data/       # Shuttle routes and stops CSVs
│   ├── traffic_stops/      # CPD vehicle stop CSVs (2014–2024)
│   └── rag_sources/        # Clery Report PDF
├── requirements.txt        # Python dependencies
├── .env.example            # API key template
├── design.md               # Architectural design document
└── data_analysis.md        # Data source analysis
```
