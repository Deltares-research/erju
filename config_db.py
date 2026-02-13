"""
Configuration file for NetCDF database creation.
All user-configurable parameters are defined here.

Author: Fabian Campos
Project: Rail4Earth - Subtask 3.3.3
Date: February 2026
"""

import os

# ==============================================================================
# INPUT/OUTPUT PATHS - MAIN CONFIGURATION
# ==============================================================================

# Site from where the data is recorded. Used for metadata and file organization.
SITE_NAME = "Holten"

# Path to folder containing fiber optic H5 files (.h5 format)
FO_DATA_PATH = (
    r"F:\recording_2024-08-26T12_59_54Z_5kHzping_1kHzlog_1mCS_2mGL_3000channels"
)

# Output folder where database files will be saved
OUTPUT_FOLDER = r"P:\11210978-erju-ai\holten_db"


# ==============================================================================
# ACCELEROMETER DATABASE CONFIGURATION
# ==============================================================================

# Path to accelerometer SQLite database
ACCEL_DB_PATH = r"P:\archivedprojects\11207352-stem\database\Wielrondheid_132887.db"

# Time range for extracting accelerometer events
ACCEL_START_DATE = "2024-08-26 13:00:00"
ACCEL_END_DATE = "2024-08-29 07:00:00"

# Location name(s) for accelerometer data
# List of measurement points to extract. Each MP will be stored in separate groups.
ACCEL_MEASUREMENT_POINTS = [
    "Meetjournal_MP8_Holten_zuid_4m_C",  # Centre, 4m from track
    "Meetjournal_MP9_Holten_zuid_4m_D",  # Right, 4m from track
    "Meetjournal_MP10_Holten_zuid_8m_C",  # Centre, 8m from track
]

# Campaign name (set to None to include all campaigns)
ACCEL_CAMPAIGNS = None

# Train type filter (e.g., 'VIRM', 'ICM', or None for all)
ACCEL_TRAINTYPE = "VIRM"

# Track filter (e.g., '1', '2', or None for all)
ACCEL_TRACK = "1"


# ==============================================================================
# DATABASE OUTPUT CONFIGURATION
# ==============================================================================

# Output file format: "netcdf", "pickle", or "both"
# NetCDF is recommended for long-term storage and sharing
DATABASE_FORMAT = "netcdf"

# Enable compression to reduce file size (recommended)
DATABASE_COMPRESSION = True

# File naming format for NetCDF files (e.g., EVENT_0001, EVENT_0002, etc.)
NAME_FORMAT = "EVENT_{:04d}"

# Compression level (1-9, higher = smaller files but slower)
# 9 is maximum compression (recommended for archival storage)
DATABASE_COMPRESSION_LEVEL = 9

# Include rich metadata in output files (timestamps, source files, etc.)
DATABASE_INCLUDE_METADATA = True
