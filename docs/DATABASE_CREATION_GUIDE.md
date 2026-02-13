# Database Creation System - Quick Start Guide

## Overview

This system provides a modern, config-driven approach to creating NetCDF databases from fiber optic (FO) sensor data and accelerometer measurements for railway vibration analysis.

## New Files Structure

```
erju/
├── config_db.py                           # Configuration system
├── main_database_creation.py              # Main entry point
└── src/
    └── erju/
        ├── create_database_legacy.py      # Old code (archived)
        └── netcdf_writer.py               # NetCDF export module
```

## Quick Start

### 1. Basic Usage with Preset Configuration

```bash
# Use Culemborg preset configuration
python main_database_creation.py --config culemborg

# Test configuration without processing
python main_database_creation.py --config culemborg --test

# Enable verbose output
python main_database_creation.py --config culemborg --verbose
```

### 2. Custom Configuration

```bash
# Override paths
python main_database_creation.py --config default \
    --fo-path "D:/data/fo" \
    --output "D:/output/database"

# Override channels and threshold
python main_database_creation.py --config culemborg \
    --channels 4270 4280 4290 \
    --threshold 600
```

### 3. Different Processing Modes

The system supports three processing modes:

#### FO-Only Mode (Default)
Detects events directly from FO data:
```python
# In config_db.py
processing = ProcessingMode(
    mode='fo_only',
    extend_signal=True,
    generate_plots=True
)
```

#### Accelerometer-Driven Mode
Uses accelerometer to detect events, then extracts corresponding FO data:
```python
processing = ProcessingMode(
    mode='accel_driven',
    use_logbook_filter=True
)
```

#### Both Mode
Processes FO and accelerometer independently, then merges:
```python
processing = ProcessingMode(
    mode='both'
)
```

## Configuration System

### Creating a Custom Configuration

Edit `config_db.py` to create your own preset:

```python
def get_my_custom_config() -> DatabaseCreationConfig:
    """My custom configuration."""
    paths = PathConfig(
        fo_data_path=r"D:\my_data\fo",
        acc_data_path=r"D:\my_data\accel",
        output_path=r"D:\output"
    )
    
    fo_config = FOProcessingConfig(
        reader_type='silixa',  # or 'optasense', 'nptdms'
        selected_channels=[4270, 4280],
        threshold=500
    )
    
    stalta = STALTAConfig(
        nsta=1.0,      # Short window (seconds)
        nlta=8.0,      # Long window (seconds)
        trigger_on=5.0,
        trigger_off=0.5
    )
    
    database = DatabaseConfig(
        format='netcdf',  # or 'pickle', 'both'
        compression=True,
        compression_level=9
    )
    
    return DatabaseCreationConfig(
        paths=paths,
        fo_processing=fo_config,
        database=database
    )
```

### Configuration Classes

#### PathConfig
- `fo_data_path`: FO TDMS files location
- `acc_data_path`: Accelerometer .asc files location (optional)
- `logbook_path`: Excel logbook for validation (optional)
- `output_path`: Where to save database files
- `plots_path`: Where to save diagnostic plots

#### FOProcessingConfig
- `reader_type`: 'silixa', 'optasense', or 'nptdms'
- `selected_channels`: List of FO channels to process
- `file_time_coverage`: Time per file (seconds)
- `buffer_seconds`: Buffer around events
- `threshold`: Activity detection threshold

#### STALTAConfig
- `nsta`: Short-term average window (seconds)
- `nlta`: Long-term average window (seconds)
- `trigger_on`: Start event threshold ratio
- `trigger_off`: End event threshold ratio
- `window_buffer`: Extra time around events (seconds)

#### DatabaseConfig
- `format`: 'netcdf', 'pickle', or 'both'
- `compression`: Enable compression
- `compression_level`: 1-9 (higher = more compression)
- `include_metadata`: Add metadata to files

## Output Format

### NetCDF File Structure

Each event is saved as a separate NetCDF file with this structure:

```
event_20241120_103045123456.nc
│
├── Dimensions
│   ├── time_fo: 10000
│   └── time_accel: 10000 (if accelerometer data included)
│
├── Variables
│   ├── time_fo(time_fo): datetime array
│   ├── fo_signal(time_fo): FO strain rate values
│   ├── time_accel(time_accel): datetime array (optional)
│   ├── accel_x(time_accel): X-axis acceleration (optional)
│   ├── accel_y(time_accel): Y-axis acceleration (optional)
│   └── accel_z(time_accel): Z-axis acceleration (optional)
│
└── Global Attributes
    ├── event_start_time: ISO datetime
    ├── event_end_time: ISO datetime
    ├── event_duration_seconds: float
    ├── source_files: comma-separated list
    ├── channel: FO channel number
    └── ... (CF convention attributes)
```

### Reading NetCDF Files

```python
from src.erju.netcdf_writer import NetCDFReader
from pathlib import Path

# Read event data
reader = NetCDFReader()
event_data = reader.read_event_file(Path("event_20241120_103045.nc"))

# Access FO signal
fo_time = event_data['fo_signal']['time']
fo_values = event_data['fo_signal']['values']
channel = event_data['fo_signal']['channel']

# Access metadata
metadata = event_data['metadata']
```

## Pipeline Steps

The system executes these steps automatically:

1. **File Discovery**: Scans input directories for FO/accelerometer files
2. **Pre-screening** (FO mode only): Identifies files with significant activity
3. **Event Detection**: Applies STA/LTA to detect vibration events
4. **Signal Extraction**: Extracts signals for detected time windows
5. **Database Export**: Writes NetCDF files with metadata

## Development Status

### ✅ Implemented (Skeleton)
- Configuration system with dataclasses
- Main orchestration framework
- NetCDF writer/reader
- Command-line interface
- Logging system
- Preset configurations

### 🚧 To Be Implemented (Step-by-Step)
- [ ] File discovery methods
- [ ] FO signal pre-screening
- [ ] STA/LTA event detection
- [ ] Signal extraction for time windows
- [ ] Multi-file signal joining
- [ ] Logbook filtering
- [ ] Plot generation

## Next Steps

The skeleton is ready! We'll now implement each component step by step:

1. **File Discovery**: Implement methods to scan and list input files
2. **FO Pre-screening**: Average signals to find active files
3. **Event Detection**: Implement STA/LTA algorithm
4. **Signal Extraction**: Extract data for time windows
5. **Export**: Write to NetCDF format

## Testing

Test the configuration system:
```bash
python config_db.py
```

Test the main script setup:
```bash
python main_database_creation.py --config culemborg --test
```

Test NetCDF writer:
```bash
python src/erju/netcdf_writer.py
```

## Contact

For questions or issues:
- Author: Fabian Campos
- Email: fabian.campos@deltares.nl
- Project: Rail4Earth - Subtask 3.3.3
