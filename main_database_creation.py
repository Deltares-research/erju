"""
Main script for creating NetCDF databases from accelerometer events.

Author: Fabian Campos
Project: Rail4Earth - Subtask 3.3.3
Date: February 2026

Usage:
    python main_database_creation.py
"""

import sys
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

import config_db as config
from src.utils.db_utils import fetch_multi_mp_accel_data
from src.db.netcdf_creator import create_netcdf_database


def main():
    """Main entry point for database creation."""

    print("\n" + "=" * 80)
    print("ACCELEROMETER DATABASE CREATION - STARTING")
    print("=" * 80)
    print(f"\nSite:         {config.SITE_NAME}")
    print(f"Database:     {config.ACCEL_DB_PATH}")
    print(f"Date Range:   {config.ACCEL_START_DATE} → {config.ACCEL_END_DATE}")
    print(f"\nMeasurement Points ({len(config.ACCEL_MEASUREMENT_POINTS)}):")
    for mp in config.ACCEL_MEASUREMENT_POINTS:
        print(f"  - {mp}")
    print(f"\nFilters:")
    print(f"  Train Type: {config.ACCEL_TRAINTYPE}")
    print(f"  Track:      {config.ACCEL_TRACK}")
    print(f"\nOutput:       {config.OUTPUT_FOLDER}")
    print("\n" + "=" * 80 + "\n")

    # 1. Fetch accelerometer data from multiple measurement points
    print("Step 1: Fetching accelerometer events from database...")
    events_dict = fetch_multi_mp_accel_data(
        path_db=config.ACCEL_DB_PATH,
        start_date=config.ACCEL_START_DATE,
        end_date=config.ACCEL_END_DATE,
        measurement_points=config.ACCEL_MEASUREMENT_POINTS,
        campaigns=config.ACCEL_CAMPAIGNS,
        traintype=config.ACCEL_TRAINTYPE,
        track=config.ACCEL_TRACK,
    )

    if len(events_dict) == 0:
        print("\nNo events found! Check your date range and filters.")
        return

    # 2. Create timestamped subfolder
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_subfolder = Path(config.OUTPUT_FOLDER) / f"netcdf_{timestamp}"
    output_subfolder.mkdir(parents=True, exist_ok=True)
    print(f"Step 2: Creating subfolder: {output_subfolder.name}\n")

    # 3. Create NetCDF database (one file per event, all MPs included)
    print(f"\nStep 3: Creating {len(events_dict)} NetCDF files (one per event)...")
    output_files = create_netcdf_database(
        events_dict=events_dict,
        output_folder=output_subfolder,
        site_name=config.SITE_NAME,
        name_format=config.NAME_FORMAT,
        compression_level=9 if config.DATABASE_COMPRESSION else 0,
    )

    print(f"\nCreated {len(output_files)} NetCDF files")
    print(f"Location: {output_subfolder}")

    print("\n" + "=" * 80)
    print("DATABASE CREATION - COMPLETE")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
