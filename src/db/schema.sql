-- Enable PostGIS extension for spatial data
CREATE EXTENSION IF NOT EXISTS postgis;

-- 1. Crime Incidents Table (from Daily Crime Log / dclog.php)
-- Stores processed crime reports with categories and dispositions.
CREATE TABLE IF NOT EXISTS crime_incidents (
    id SERIAL PRIMARY KEY,
    case_number VARCHAR(50) UNIQUE NOT NULL,
    incident_type VARCHAR(100) NOT NULL,
    location_name VARCHAR(255),
    location_geo GEOGRAPHY(POINT, 4326), -- Populated via geocoding later
    date_occurred TIMESTAMP,
    date_reported TIMESTAMP,
    disposition VARCHAR(100),
    domestic_violence BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Index for spatial queries coverage (e.g., "find crimes within 500m")
CREATE INDEX IF NOT EXISTS idx_crime_location ON crime_incidents USING GIST(location_geo);
CREATE INDEX IF NOT EXISTS idx_crime_date ON crime_incidents(date_occurred);


-- 2. Police Calls Table (from Daily Incident Log / dilog.php)
-- Stores raw calls for service, often less severe or preliminary compared to crime logs.
CREATE TABLE IF NOT EXISTS police_calls (
    id SERIAL PRIMARY KEY,
    incident_number VARCHAR(50) UNIQUE NOT NULL,
    call_time TIMESTAMP NOT NULL, -- Combined Date + Time
    incident_type VARCHAR(100),
    location_address VARCHAR(255),
    location_geo GEOGRAPHY(POINT, 4326),
    description TEXT,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_call_location ON police_calls USING GIST(location_geo);


-- 3. CPD Incidents Table (from Columbia Police Dept / ArcGIS)
-- Stores public crime reports from the city transparency portal.
CREATE TABLE IF NOT EXISTS cpd_incidents (
    id SERIAL PRIMARY KEY,
    offense_id VARCHAR(50) UNIQUE NOT NULL,
    report_date TIMESTAMP,
    nibrs_description VARCHAR(100), -- Standardized offense description
    full_address VARCHAR(255),
    location_geo GEOGRAPHY(POINT, 4326),
    case_status VARCHAR(50),
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_cpd_location ON cpd_incidents USING GIST(location_geo);
CREATE INDEX IF NOT EXISTS idx_cpd_date ON cpd_incidents(report_date);


-- 4. Traffic Stops Table (from CPD Data)
-- Stores vehicle stop data for patrol pattern analysis.
CREATE TABLE IF NOT EXISTS traffic_stops (
    id SERIAL PRIMARY KEY,
    agency VARCHAR(50) DEFAULT 'CPD',
    stop_date TIMESTAMP,
    location_name VARCHAR(255),
    location_geo GEOGRAPHY(POINT, 4326),
    violation_type VARCHAR(100),
    driver_race VARCHAR(50),
    driver_gender VARCHAR(20),
    search_conducted BOOLEAN,
    result VARCHAR(100), -- Citation, Warning, Arrest
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_traffic_location ON traffic_stops USING GIST(location_geo);


-- 5. Shuttle Routes Table (Tiger Line / Go COMO)
-- Stores static route definitions and geometries.
CREATE TABLE IF NOT EXISTS shuttle_routes (
    id SERIAL PRIMARY KEY,
    route_id INT UNIQUE NOT NULL, -- e.g. 405
    route_name VARCHAR(100),
    agency_id INT, -- To distinguish Tiger Line vs City
    color VARCHAR(10),
    geometry GEOGRAPHY(LINESTRING, 4326), -- Decoded from encLine
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT NOW()
);

-- 6. Shuttle Stops Table
-- Stores stop locations for the shuttles.
CREATE TABLE IF NOT EXISTS shuttle_stops (
    id SERIAL PRIMARY KEY,
    stop_id INT UNIQUE NOT NULL,
    stop_name VARCHAR(100),
    location_geo GEOGRAPHY(POINT, 4326),
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_shuttle_stop_loc ON shuttle_stops USING GIST(location_geo);


-- 7. Campus Boundary Table (from map.missouri.edu)
-- Used for geofencing: "Is this user on campus?"
CREATE TABLE IF NOT EXISTS campus_boundary (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100),
    geometry GEOGRAPHY(POLYGON, 4326),
    created_at TIMESTAMP DEFAULT NOW()
);

-- 8. Campus Buildings Table (from map.missouri.edu)
-- Used for: "Which building is this?" and routing to entrances.
CREATE TABLE IF NOT EXISTS campus_buildings (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100),
    building_number VARCHAR(50),
    geometry GEOGRAPHY(POLYGON, 4326), 
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_campus_boundary ON campus_boundary USING GIST(geometry);
CREATE INDEX IF NOT EXISTS idx_campus_buildings ON campus_buildings USING GIST(geometry);


-- 9. Safety Assets Table (from MU_Features_new)
-- Stores locations of Emergency Phones, Accessible Entrances, etc.
CREATE TABLE IF NOT EXISTS safety_assets (
    id SERIAL PRIMARY KEY,
    asset_id INT NOT NULL, -- Original Object ID
    asset_type VARCHAR(50) NOT NULL, -- 'Emergency Phone', 'Accessible Entrance'
    description VARCHAR(255),
    location_geo GEOGRAPHY(POINT, 4326),
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_safety_assets ON safety_assets USING GIST(location_geo);
CREATE INDEX IF NOT EXISTS idx_safety_type ON safety_assets(asset_type);
