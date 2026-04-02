"""
NetCDF database creator for accelerometer events.

Creates hierarchical NetCDF files with support for multiple measurement points.

Author: Fabian Campos
Date: February 2026
"""

import netCDF4 as nc
import numpy as np
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo


def _apply_config_attributes(nc_var, var_config: dict, long_name_override: str = None):
    """Apply NetCDF variable attributes from config, mapping description -> long_name."""
    if not var_config:
        return

    long_name = (
        long_name_override
        if long_name_override is not None
        else var_config.get("long_name", var_config.get("description"))
    )
    if long_name is not None:
        nc_var.long_name = long_name

    for attr_name, attr_value in var_config.items():
        if attr_name in {"name", "long_name", "description"}:
            continue
        setattr(nc_var, attr_name, attr_value)


def _safe_int_or_missing(value, missing_value: int = -1):
    """Convert value to int, returning missing_value when missing/invalid."""
    if value is None:
        return missing_value
    if isinstance(value, str) and value.strip() == "":
        return missing_value
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return missing_value


def validate_config(config):
    """
    Validate configuration before writing NetCDF files.

    Raises:
        ValueError: If configuration is invalid or inconsistent.
    """
    # Check axis labels length
    if len(config.ACCEL_AXIS_LABELS) != 3:
        raise ValueError(
            f"ACCEL_AXIS_LABELS must have length 3, got {len(config.ACCEL_AXIS_LABELS)}"
        )

    # Check all measurement points have sensor ID mappings
    for mp_name in config.ACCEL_MEASUREMENT_POINTS:
        if mp_name not in config.ACCEL_SENSOR_ID_MAP:
            raise ValueError(
                f"Measurement point '{mp_name}' in ACCEL_MEASUREMENT_POINTS "
                f"not found in ACCEL_SENSOR_ID_MAP"
            )

    # Get all mapped sensor IDs
    mapped_sensor_ids = [
        config.ACCEL_SENSOR_ID_MAP[mp] for mp in config.ACCEL_MEASUREMENT_POINTS
    ]

    # Check all mapped sensor IDs have distance values
    for sensor_id in mapped_sensor_ids:
        if sensor_id not in config.ACCEL_DISTANCE_TO_TRACK_M:
            raise ValueError(
                f"Sensor ID '{sensor_id}' not found in ACCEL_DISTANCE_TO_TRACK_M"
            )

    # Check all mapped sensor IDs have axis masks
    for sensor_id in mapped_sensor_ids:
        if sensor_id not in config.ACCEL_AXIS_MASK:
            raise ValueError(f"Sensor ID '{sensor_id}' not found in ACCEL_AXIS_MASK")
        # Check axis mask length
        if len(config.ACCEL_AXIS_MASK[sensor_id]) != 3:
            raise ValueError(
                f"Axis mask for sensor '{sensor_id}' must have length 3, "
                f"got {len(config.ACCEL_AXIS_MASK[sensor_id])}"
            )

    if getattr(config, "FO_ENABLE", False):
        if getattr(config, "FO_CHANNEL_HALF_WINDOW", -1) < 0:
            raise ValueError("FO_CHANNEL_HALF_WINDOW must be >= 0")
        if getattr(config, "FO_CENTER_CHANNEL", -1) < 0:
            raise ValueError("FO_CENTER_CHANNEL must be >= 0")


def create_netcdf_database(
    events_dict: dict,
    output_folder: str,
    site_name: str,
    config,
    name_format: str = "EVENT_{:04d}",
    compression_level: int = 9,
    start_index: int = 1,
):
    """
    Create hierarchical NetCDF database from accelerometer events.

    Handles both single and multiple measurement points per event.
    Each NetCDF file represents one train passing event and contains:
    - Root attributes: Global metadata (event_id, site_id, timestamps)
    - meta_acc/: Accelerometer event metadata (train info, timing)
    - geometry_acc/: Accelerometer sensor geometry information
    - acc/<SENSOR_ID>/: Accelerometer data for each measurement point

    Args:
        events_dict: Dictionary from fetch_multi_mp_accel_data() organized by event_id.
        output_folder: Path to output folder.
        site_name: Name of the site (e.g., "Holten").
        config: Configuration module with metadata settings.
        name_format: Format string for file naming.
        compression_level: Compression level (1-9).
        start_index: Starting index for file numbering (1-based).

    Returns:
        list: List of created NetCDF file paths.
    """
    # Validate configuration
    validate_config(config)

    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    output_files = []
    total_events = len(events_dict)

    for local_idx, (event_id, event_data) in enumerate(events_dict.items(), start=1):
        # Generate file ID and path
        file_id = name_format.format(start_index + local_idx - 1)
        netcdf_filename = f"{file_id}.nc"
        netcdf_path = output_folder / netcdf_filename

        # Extract metadata
        metadata = event_data["event_metadata"]
        mps = event_data["measurement_points"]

        # Reference time (t0) - convert to UTC if needed
        t0_naive = datetime.strptime(metadata["start_time"], "%Y-%m-%d %H:%M:%S")
        if config.TIMEZONE == "UTC":
            t0_utc = t0_naive.replace(tzinfo=ZoneInfo("UTC"))
        else:
            t0_local = t0_naive.replace(tzinfo=ZoneInfo(config.TIMEZONE))
            t0_utc = t0_local.astimezone(ZoneInfo("UTC"))

        # Create NetCDF file with hierarchical groups
        with nc.Dataset(netcdf_path, "w", format="NETCDF4") as dataset:

            # ===================================================================
            # ROOT ATTRIBUTES - Global metadata
            # ===================================================================
            dataset.event_id = str(event_id)
            dataset.site_id = site_name
            event_t0_utc = t0_utc.isoformat().replace("+00:00", "Z")
            dataset.event_t0_utc = event_t0_utc
            dataset.created_utc = (
                datetime.now(ZoneInfo("UTC")).isoformat().replace("+00:00", "Z")
            )
            dataset.pipeline_version = config.PIPELINE_VERSION
            dataset.software_git_hash = "N/A"  # TODO: Get from git

            # Add database generation metadata if enabled
            if config.DATABASE_INCLUDE_METADATA:
                dataset.query_start_date = config.ACCEL_START_DATE
                dataset.query_end_date = config.ACCEL_END_DATE
                dataset.query_train_type_filter = (
                    str(config.ACCEL_TRAINTYPE) if config.ACCEL_TRAINTYPE else "None"
                )
                dataset.query_track_filter = (
                    str(config.ACCEL_TRACK)
                    if config.ACCEL_TRACK is not None
                    else "None"
                )
                dataset.query_timezone = config.TIMEZONE

            # ===================================================================
            # GROUP: meta_acc - Accelerometer event metadata
            # ===================================================================
            meta_grp = dataset.createGroup("meta_acc")

            # Train type as string variable (scalar)
            train_type_var = meta_grp.createVariable("train_type", str, ())
            train_type_var[0] = metadata.get("traintype", "unknown")
            train_type_var.long_name = "Train type/classification during passage"

            # Calculate event timing relative to t0
            time_window = metadata["time_window"]
            event_start_offset = (
                time_window[0] - t0_utc.replace(tzinfo=None)
            ).total_seconds()
            event_end_offset = (
                time_window[1] - t0_utc.replace(tzinfo=None)
            ).total_seconds()

            meta_grp.createVariable("event_start_offset_s", "f4")
            meta_grp.createVariable("event_end_offset_s", "f4")
            train_speed_var = meta_grp.createVariable("train_speed_kmh", "f4")

            meta_grp["event_start_offset_s"][:] = event_start_offset
            meta_grp["event_end_offset_s"][:] = event_end_offset

            speed_kmh = metadata.get("speed", np.nan)
            if speed_kmh is None:
                speed_kmh = np.nan
            train_speed_var[:] = speed_kmh
            train_speed_var.units = "km/h"
            train_speed_var.long_name = (
                "Train speed during passage from source database"
            )

            track_number_var = meta_grp.createVariable("track_number", "i4")
            track_number = _safe_int_or_missing(metadata.get("track"), missing_value=-1)
            track_number_var[:] = track_number
            track_number_var.long_name = (
                "Track number during passage from source database"
            )
            track_number_var.missing_value = np.int32(-1)
            track_number_var.comment = "-1 indicates unknown or missing track number"

            # ===================================================================
            # ROOT DIMENSIONS - Shared across all groups
            # ===================================================================
            # Create global acc_axis dimension (used by geometry and all acc groups)
            dataset.createDimension("acc_axis", 3)

            # ===================================================================
            # GROUP: geometry_acc - Accelerometer sensor geometry
            # ===================================================================
            geom_grp = dataset.createGroup("geometry_acc")

            # Map measurement point names to sensor IDs
            # Note: mps already has sensor IDs as keys (from fetch_multi_mp_accel_data)
            sensor_ids = list(mps.keys())
            n_sensors = len(sensor_ids)

            # Create dimensions
            geom_grp.createDimension("acc_sensor", n_sensors)

            # Sensor IDs variable
            sensor_id_var = geom_grp.createVariable(
                "acc_sensor_id", str, ("acc_sensor",)
            )
            for idx, sensor_id in enumerate(sensor_ids):
                sensor_id_var[idx] = sensor_id

            # Axis labels variable
            axis_labels_var = geom_grp.createVariable("axis_labels", str, ("acc_axis",))
            for idx, label in enumerate(config.ACCEL_AXIS_LABELS):
                axis_labels_var[idx] = label

            # Distance to track variable
            dist_var = geom_grp.createVariable(
                config.VAR_DISTANCE["name"], "f4", ("acc_sensor",)
            )
            for idx, sensor_id in enumerate(sensor_ids):
                dist_var[idx] = config.ACCEL_DISTANCE_TO_TRACK_M[sensor_id]
            _apply_config_attributes(dist_var, config.VAR_DISTANCE)

            # Side of track variable (int8 for -1, 0, +1)
            side_var = geom_grp.createVariable(
                "acc_side_of_track", "i1", ("acc_sensor",)
            )
            for idx, sensor_id in enumerate(sensor_ids):
                side_var[idx] = config.ACCEL_SIDE_OF_TRACK[sensor_id]
            side_var.long_name = "Side of track (-1=left, 0=unknown, +1=right)"

            # Axis mask variable (2D: sensor x axis)
            mask_var = geom_grp.createVariable(
                config.VAR_AXIS_MASK["name"], "i1", ("acc_sensor", "acc_axis")
            )
            for idx, sensor_id in enumerate(sensor_ids):
                mask_var[idx, :] = config.ACCEL_AXIS_MASK[sensor_id]
            _apply_config_attributes(mask_var, config.VAR_AXIS_MASK)

            # ===================================================================
            # GROUP: acc/<SENSOR_ID> - Accelerometer data for each measurement point
            # ===================================================================
            for sensor_id, mp_data in mps.items():
                # Create group using sensor ID (e.g., acc/MP8)
                acc_grp = dataset.createGroup(f"acc/{sensor_id}")

                # Extract data
                absolute_time = mp_data["absolute_time"]
                trace_x = mp_data["trace_x"]
                trace_y = mp_data["trace_y"]
                trace_z = mp_data["trace_z"]
                fs_hz = mp_data["fs_hz"]
                n_samples = mp_data["n_samples"]

                # Convert to relative time (seconds from t0_utc)
                time_s = np.array(
                    [
                        (t - t0_utc.replace(tzinfo=None)).total_seconds()
                        for t in absolute_time
                    ]
                )

                # Create dimensions (reuse global acc_axis from root)
                acc_grp.createDimension("acc_time", n_samples)
                # acc_axis dimension inherited from root

                # Time variable
                time_var = acc_grp.createVariable(
                    config.VAR_TIME["name"],
                    "f8",
                    ("acc_time",),
                    compression="zlib" if compression_level > 0 else None,
                    complevel=compression_level if compression_level > 0 else 0,
                )
                time_var[:] = time_s
                time_long_name = config.VAR_TIME.get(
                    "long_name", config.VAR_TIME.get("description")
                )
                if time_long_name:
                    time_long_name = f"{time_long_name} for {sensor_id}"
                _apply_config_attributes(
                    time_var,
                    config.VAR_TIME,
                    long_name_override=time_long_name,
                )

                # Sampling frequency
                fs_var = acc_grp.createVariable(config.VAR_FREQUENCY["name"], "f4")
                fs_var[:] = fs_hz
                _apply_config_attributes(fs_var, config.VAR_FREQUENCY)

                # Per-sensor axis availability mask [axis]
                sensor_axis_mask_var = acc_grp.createVariable(
                    config.VAR_AXIS_MASK["name"],
                    "i1",
                    ("acc_axis",),
                )
                sensor_axis_mask_var[:] = np.asarray(
                    config.ACCEL_AXIS_MASK[sensor_id], dtype=np.int8
                )
                axis_mask_long_name = config.VAR_AXIS_MASK.get(
                    "long_name", config.VAR_AXIS_MASK.get("description")
                )
                if axis_mask_long_name:
                    axis_mask_long_name = f"{axis_mask_long_name} for {sensor_id}"
                _apply_config_attributes(
                    sensor_axis_mask_var,
                    config.VAR_AXIS_MASK,
                    long_name_override=axis_mask_long_name,
                )

                # Acceleration matrix [time, axis] - uses inherited acc_axis dimension
                accel_var = acc_grp.createVariable(
                    config.VAR_ACCELERATION["name"],
                    "f4",
                    ("acc_time", "acc_axis"),
                    compression="zlib" if compression_level > 0 else None,
                    complevel=compression_level if compression_level > 0 else 0,
                )
                accel_var[:, 0] = trace_x
                accel_var[:, 1] = trace_y
                accel_var[:, 2] = trace_z
                _apply_config_attributes(accel_var, config.VAR_ACCELERATION)

            # ===================================================================
            # FO modality (optional): meta_fo, geometry_fo, fo
            # ===================================================================
            fo_data = event_data.get("fo_data")
            if fo_data and fo_data.get("found", False):
                fo_timestamps = fo_data["timestamps"]
                fo_strain = np.asarray(fo_data["strain"], dtype=np.float32)
                fo_channel_ids = np.asarray(fo_data["channel_ids"], dtype=np.int32)

                if fo_strain.ndim != 2:
                    raise ValueError(
                        f"FO strain must be 2D [time, channel], got shape {fo_strain.shape}"
                    )

                n_fo_time, n_fo_channel = fo_strain.shape
                if n_fo_channel != len(fo_channel_ids):
                    raise ValueError(
                        "FO channel dimension mismatch between strain and channel_ids"
                    )

                # Root dimensions shared by geometry_fo and /fo
                dataset.createDimension("fo_channel", n_fo_channel)

                # GROUP: meta_fo
                meta_fo_grp = dataset.createGroup("meta_fo")
                fo_reader_var = meta_fo_grp.createVariable("fo_reader", str, ())
                fo_reader_var[0] = str(fo_data.get("fo_reader", "unknown"))
                fo_reader_var.long_name = "FO data reader type"

                fo_file_count_var = meta_fo_grp.createVariable("fo_file_count", "i4")
                fo_file_count_var[:] = len(fo_data.get("file_paths", []))
                fo_file_count_var.long_name = (
                    "Number of FO files concatenated for this event"
                )

                # GROUP: geometry_fo
                geometry_fo_grp = dataset.createGroup("geometry_fo")

                centre_fo_channel_var = geometry_fo_grp.createVariable(
                    "centre_fo_channel", "i4"
                )
                centre_fo_channel_var[:] = int(fo_data.get("center_channel"))

                fo_channel_half_window_var = geometry_fo_grp.createVariable(
                    "fo_channel_half_window", "i4"
                )
                fo_channel_half_window_var[:] = int(fo_data.get("channel_half_window"))

                fo_channel_id_var = geometry_fo_grp.createVariable(
                    "fo_channel_id", "i4", ("fo_channel",)
                )
                fo_channel_id_var[:] = fo_channel_ids
                fo_channel_id_var.long_name = "Absolute FO channel numbers"

                # Keep requested variable name while storing literal channel values
                fo_channel_position_var = geometry_fo_grp.createVariable(
                    "fo_channel_position_m", "f4", ("fo_channel",)
                )
                fo_channel_position_var[:] = fo_channel_ids.astype(np.float32)
                fo_channel_position_var.units = "-"
                fo_channel_position_var.long_name = "Literal FO channel values"
                fo_channel_position_var.comment = (
                    "No distance conversion applied; values equal fo_channel_id"
                )

                channel_spacing = fo_data.get("channel_spacing_m")
                if channel_spacing is not None:
                    channel_spacing_var = geometry_fo_grp.createVariable(
                        "fo_channel_spacing_m", "f4"
                    )
                    channel_spacing_var[:] = np.float32(channel_spacing)

                gauge_length = fo_data.get("gauge_length_m")
                if gauge_length is not None:
                    gauge_length_var = geometry_fo_grp.createVariable(
                        "fo_gauge_length_m", "f4"
                    )
                    gauge_length_var[:] = np.float32(gauge_length)

                # GROUP: fo
                fo_grp = dataset.createGroup("fo")
                fo_grp.createDimension("fo_time", n_fo_time)

                fo_time_s = np.array(
                    [
                        (t - t0_utc.replace(tzinfo=None)).total_seconds()
                        for t in fo_timestamps
                    ],
                    dtype=np.float64,
                )

                fo_time_var = fo_grp.createVariable(
                    "time_s",
                    "f8",
                    ("fo_time",),
                    compression="zlib" if compression_level > 0 else None,
                    complevel=compression_level if compression_level > 0 else 0,
                )
                fo_time_var[:] = fo_time_s
                fo_time_var.units = "s"
                fo_time_var.long_name = "FO time relative to event_t0_utc"

                fo_fs_var = fo_grp.createVariable("fs_hz", "f4")
                fo_fs_var[:] = np.float32(fo_data["fs_hz"])
                fo_fs_var.units = "Hz"
                fo_fs_var.long_name = "FO sampling frequency"

                fo_strain_var = fo_grp.createVariable(
                    "strain",
                    "f4",
                    ("fo_time", "fo_channel"),
                    compression="zlib" if compression_level > 0 else None,
                    complevel=compression_level if compression_level > 0 else 0,
                )
                fo_strain_var[:, :] = fo_strain
                fo_strain_var.long_name = "FO strain time series"

        output_files.append(str(netcdf_path))

        # Print progress
        if local_idx % 10 == 0 or local_idx == total_events:
            print(f"  Created {local_idx}/{total_events} files...")

    return output_files
