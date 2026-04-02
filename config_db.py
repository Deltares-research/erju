"""
Configuration file for NetCDF database creation.
All user-configurable parameters are defined here.

Author: Fabian Campos
Project: Rail4Earth - Subtask 3.3.3
Date: February 2026
"""

import json
from pathlib import Path

# ==============================================================================
# SITE + TIME REFERENCE
# ==============================================================================

# Site from where the data is recorded. Used for metadata and file organization.
# To switch sites: change this to match a JSON file in the sites/ folder.
# Available sites: Check sites/ directory for .json files (e.g., holten.json)
# Example: SITE_NAME = "Amsterdam" will load sites/amsterdam.json
SITE_NAME = "Holten"

# Load site-specific configuration from JSON file
# This automatically loads measurement points, sensor mappings, distances, etc.
_config_dir = Path(__file__).parent
_site_config_file = _config_dir / "sites" / f"{SITE_NAME.lower()}.json"

if not _site_config_file.exists():
    raise FileNotFoundError(
        f"Site configuration file not found: {_site_config_file}\n"
        f"Please create a JSON file for site '{SITE_NAME}' in the sites/ folder.\n"
        f"See sites/template.json for an example structure."
    )

with open(_site_config_file, "r") as f:
    _site_config = json.load(f)

# IMPORTANT:
# Define the time basis of ACCEL_START_DATE / ACCEL_END_DATE.
# Use "UTC" if your database timestamps are UTC; use "Europe/Amsterdam" if local.
TIMEZONE = "UTC"

# ==============================================================================
# INPUT/OUTPUT PATHS - MAIN CONFIGURATION
# ==============================================================================

# Path to folder containing fiber optic H5 files (.h5 format)
# (Not used yet if you are building accelerometer-only NetCDF files first)
# FO_DATA_PATH = (
#     r"F:\recording_2024-08-26T12_59_54Z_5kHzping_1kHzlog_1mCS_2mGL_3000channels"
# )
FO_DATA_PATH = r"C:\fo_holten_sample"

# Enable/disable FO extraction per event.
FO_ENABLE = True

# FO reader type used by BaseFOdata factory.
FO_READER = "optasense"

# FO channel window definition around center channel.
FO_CENTER_CHANNEL = 1190
FO_CHANNEL_HALF_WINDOW = 6

# Save per-event FO availability report (CSV) next to NetCDF outputs.
FO_SAVE_AVAILABILITY_REPORT = True

# Path to accelerometer SQLite database
ACCEL_DB_PATH = r"P:\archivedprojects\11207352-stem\database\Wielrondheid_132887.db"

# Output folder where database files will be saved
OUTPUT_FOLDER = r"P:\11210978-erju-ai\holten_db"

# ==============================================================================
# ACCELEROMETER EVENT EXTRACTION CONFIGURATION
# ==============================================================================

# Time range for extracting accelerometer events
ACCEL_START_DATE = "2024-08-26 13:00:00"
ACCEL_END_DATE = "2024-08-29 07:00:00"

# Query chunk size in days for large date ranges.
# Use 1 for day-by-day processing. Set to None or <=0 to disable chunking.
ACCEL_QUERY_CHUNK_DAYS = 1

# Campaign name (set to None to include all campaigns)
ACCEL_CAMPAIGNS = None

# Train type filter (e.g., "VIRM", "ICM", or None for all)
ACCEL_TRAINTYPE = "ICM"

# Track filter (recommend integer for consistent metadata typing)
# (Set to None for all tracks)
ACCEL_TRACK = None

# ==============================================================================
# ACCELEROMETER SENSOR SELECTION + STANDARDIZED IDS
# ==============================================================================
# Site-specific configuration loaded from sites/{SITE_NAME}.json

# Raw measurement point names as they appear in the accelerometer database.
ACCEL_MEASUREMENT_POINTS = _site_config["measurement_points"]

# Stable short sensor IDs used in the NetCDF structure:
# - /geometry/acc_sensor_id
# - /acc/<SENSOR_ID>/...
ACCEL_SENSOR_ID_MAP = _site_config["sensor_id_map"]

# Distance from each accelerometer to the track centerline (meters).
# These values must match the IDs used in ACCEL_SENSOR_ID_MAP.
ACCEL_DISTANCE_TO_TRACK_M = _site_config["distance_to_track_m"]

# Optional: side of track convention for each sensor:
# -1 = left, +1 = right, 0 = unknown.
# If you don't trust this information, set all to 0 and fill later.
ACCEL_SIDE_OF_TRACK = _site_config["side_of_track"]

# Optional: axis availability mask per sensor (x,y,z).
# Use this if some sensors are Z-only.
# If not specified, your writer can infer it from the data.
ACCEL_AXIS_MASK = _site_config["axis_mask"]

# ==============================================================================
# DATABASE OUTPUT CONFIGURATION
# ==============================================================================

# Output file format: "netcdf", "pickle", or "both"
DATABASE_FORMAT = "netcdf"

# Enable compression to reduce file size (recommended)
DATABASE_COMPRESSION = True

# Compression level (1-9, higher = smaller files but slower)
DATABASE_COMPRESSION_LEVEL = 9

# File naming format for NetCDF files (e.g., EVENT_0001, EVENT_0002, etc.)
NAME_FORMAT = "EVENT_{:04d}"

# Include rich metadata in output files (timestamps, source filters, etc.)
DATABASE_INCLUDE_METADATA = True

# Pipeline version (for reproducibility tracking)
PIPELINE_VERSION = "1.0.0"

# ==============================================================================
# NETCDF VARIABLE NAMING + METADATA
# ==============================================================================

# Accelerometer axis labels (order must match acceleration_mps2 columns)
ACCEL_AXIS_LABELS = ["x", "y", "z"]

# Variable names and metadata (recommend using NetCDF/CF-style "long_name")
VAR_TIME = {
    "name": "time_s",
    "units": "s",
    "long_name": "Time relative to event_t0_utc",
}

VAR_FREQUENCY = {
    "name": "fs_hz",
    "units": "Hz",
    "long_name": "Sampling frequency",
}

VAR_ACCELERATION = {
    "name": "acceleration_mps2",
    "units": "m/s^2",
    "long_name": "Acceleration time series (x, y, z)",
}

VAR_DISTANCE = {
    "name": "acc_distance_to_track_m",
    "units": "m",
    "long_name": "Distance from accelerometer to track centerline",
}

VAR_AXIS_MASK = {
    "name": "axis_mask",
    "units": "-",
    "long_name": "Axis availability (1=present, 0=missing)",
}
