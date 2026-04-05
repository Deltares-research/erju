# Database Creation System - Current Guide

## Overview

This guide reflects the current production approach for building the event NetCDF database.

Pipeline characteristics:
- Config-driven (single source of truth in config_db.py and site JSON files)
- Event-centric (one NetCDF file per train event)
- Multi-modality (accelerometer always, fibre optics optional per event)
- Shared time reference (event_t0_utc)

## Current Entry Point

Run the pipeline with:

```bash
python main_database_creation.py
```

There is no CLI argument parser in the current production entrypoint.
All runtime settings are defined in config_db.py.

## Current Architecture

Main modules used by the production pipeline:

- main_database_creation.py
    - Orchestrates chunked DB querying, optional FO enrichment, NetCDF writing, and reporting.
- config_db.py
    - Loads site metadata from sites/{site}.json and defines extraction/writer settings.
- src/utils/db_utils.py
    - Fetches and groups accelerometer event data from the SQLite database.
- src/utils/fo_utils.py
    - Selects FO files by event window and extracts/crops FO strain to match ACC event windows.
- src/erju/process_FO_base.py
    - Reader factory for FO formats (optasense/silixa/nptdms).
- src/erju/process_FO_optasense.py
    - OptaSense H5 reader and metadata extraction.
- src/db/netcdf_creator.py
    - Writes the current NetCDF schema (meta_acc, geometry_acc, acc/*, optional FO groups).

## Configuration Workflow

1. Select site in config_db.py:

```python
SITE_NAME = "Holten"
```

2. Confirm paths and extraction window in config_db.py:
- ACCEL_DB_PATH
- FO_DATA_PATH
- OUTPUT_FOLDER
- ACCEL_START_DATE / ACCEL_END_DATE

3. Configure FO extraction in config_db.py:
- FO_ENABLE
- FO_READER
- FO_CENTER_CHANNEL
- FO_CHANNEL_HALF_WINDOW

4. Configure site metadata in sites/{site}.json:
- accelerometer section (measurement points, mapping, geometry, axis masks)
- fibre_optics section (side, approximate distance, manual metadata fallback)

## FO Metadata Strategy (Current)

FO metadata uses robust precedence:

1. Manual defaults from sites/{site}.json via FO_METADATA_MANUAL in config_db.py.
2. Overwrite with H5 properties when available (preferred source).

Persisted FO metadata includes:
- acquisition_id
- gauge_length
- gauge_length_unit
- spatial_sampling_interval
- spatial_sampling_interval_unit
- raw_data_unit
- raw_description
- fibre_refractive_index
- number_of_measurements

## NetCDF Output

Per event output:
- EVENT_XXXX.nc files in a timestamped subfolder under OUTPUT_FOLDER

Optional report:
- fo_event_availability.csv

Authoritative schema references:
- docs/NETCDF_SCHEMA_CURRENT.md
- docs/NETCDF_SCHEMA_CURRENT.csv

## Processing Steps

1. Build output subfolder with timestamp.
2. Query ACC events (chunked by ACCEL_QUERY_CHUNK_DAYS).
3. Group ACC traces by event_id across measurement points.
4. If FO_ENABLE is true:
     - Select matching FO files for each event window.
     - Extract, concatenate, and crop FO strain to event time span.
     - Merge FO metadata with file-first precedence.
5. Write one NetCDF file per event.
6. Save FO availability CSV report.

## Legacy Note

Historical modules still exist for reference but are not part of the current production pipeline.
Use main_database_creation.py + src/db/netcdf_creator.py as the canonical path.
