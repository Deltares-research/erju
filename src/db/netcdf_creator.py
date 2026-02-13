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


def create_netcdf_database(
    events_dict: dict,
    output_folder: str,
    site_name: str,
    config,
    name_format: str = "EVENT_{:04d}",
    compression_level: int = 9,
):
    """
    Create hierarchical NetCDF database from accelerometer events.

    Handles both single and multiple measurement points per event.
    Each NetCDF file represents one train passing event and contains:
    - Root attributes: Global metadata (event_id, site_id, timestamps)
    - meta/: Event metadata (train info, timing)
    - geometry/: Sensor geometry information
    - acc/<SENSOR_ID>/: Accelerometer data for each measurement point

    Args:
        events_dict: Dictionary from fetch_multi_mp_accel_data() organized by event_id.
        output_folder: Path to output folder.
        site_name: Name of the site (e.g., "Holten").
        config: Configuration module with metadata settings.
        name_format: Format string for file naming.
        compression_level: Compression level (1-9).

    Returns:
        list: List of created NetCDF file paths.
    """
    # Validate configuration
    validate_config(config)

    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    output_files = []
    total_events = len(events_dict)

    for i, (event_id, event_data) in enumerate(events_dict.items(), start=1):
        # Generate file ID and path
        file_id = name_format.format(i)
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
            dataset.t0_utc = t0_utc.isoformat().replace("+00:00", "Z")
            dataset.created_utc = (
                datetime.now(ZoneInfo("UTC")).isoformat().replace("+00:00", "Z")
            )
            dataset.pipeline_version = config.PIPELINE_VERSION
            dataset.software_git_hash = "N/A"  # TODO: Get from git

            # Add database generation metadata if enabled
            if config.DATABASE_INCLUDE_METADATA:
                dataset.accel_start_date = config.ACCEL_START_DATE
                dataset.accel_end_date = config.ACCEL_END_DATE
                dataset.accel_train_type_filter = (
                    str(config.ACCEL_TRAINTYPE) if config.ACCEL_TRAINTYPE else "None"
                )
                dataset.accel_track_filter = (
                    str(config.ACCEL_TRACK)
                    if config.ACCEL_TRACK is not None
                    else "None"
                )
                dataset.timezone = config.TIMEZONE

            # ===================================================================
            # GROUP: meta - Event metadata
            # ===================================================================
            meta_grp = dataset.createGroup("meta")

            # Train type as string variable (scalar)
            train_type_var = meta_grp.createVariable("train_type", str, ())
            train_type_var[0] = metadata.get("traintype", "unknown")
            train_type_var.long_name = "Train type/classification during passage"

            # Track as int32 scalar if available
            if config.ACCEL_TRACK is not None:
                track_var = meta_grp.createVariable("track", "i4")
                track_var[:] = config.ACCEL_TRACK

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
            meta_grp.createVariable(config.VAR_SPEED["name"], "f4")

            meta_grp["event_start_offset_s"][:] = event_start_offset
            meta_grp["event_end_offset_s"][:] = event_end_offset
            meta_grp[config.VAR_SPEED["name"]][:] = np.nan  # Unknown for now
            if "units" in config.VAR_SPEED:
                meta_grp[config.VAR_SPEED["name"]].units = config.VAR_SPEED["units"]
            if "long_name" in config.VAR_SPEED:
                meta_grp[config.VAR_SPEED["name"]].long_name = config.VAR_SPEED[
                    "long_name"
                ]

            # ===================================================================
            # ROOT DIMENSIONS - Shared across all groups
            # ===================================================================
            # Create global acc_axis dimension (used by geometry and all acc groups)
            dataset.createDimension("acc_axis", 3)

            # ===================================================================
            # GROUP: geometry - Sensor geometry
            # ===================================================================
            geom_grp = dataset.createGroup("geometry")

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
            if "units" in config.VAR_DISTANCE:
                dist_var.units = config.VAR_DISTANCE["units"]
            if "long_name" in config.VAR_DISTANCE:
                dist_var.long_name = config.VAR_DISTANCE["long_name"]

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
            if "long_name" in config.VAR_AXIS_MASK:
                mask_var.long_name = config.VAR_AXIS_MASK["long_name"]

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
                if "units" in config.VAR_TIME:
                    time_var.units = config.VAR_TIME["units"]
                if "long_name" in config.VAR_TIME:
                    time_var.long_name = (
                        f"{config.VAR_TIME['long_name']} for {sensor_id}"
                    )

                # Sampling frequency
                fs_var = acc_grp.createVariable(config.VAR_FREQUENCY["name"], "f4")
                fs_var[:] = fs_hz
                if "units" in config.VAR_FREQUENCY:
                    fs_var.units = config.VAR_FREQUENCY["units"]
                if "long_name" in config.VAR_FREQUENCY:
                    fs_var.long_name = config.VAR_FREQUENCY["long_name"]

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
                if "units" in config.VAR_ACCELERATION:
                    accel_var.units = config.VAR_ACCELERATION["units"]
                if "long_name" in config.VAR_ACCELERATION:
                    accel_var.long_name = config.VAR_ACCELERATION["long_name"]

        output_files.append(str(netcdf_path))

        # Print progress
        if i % 10 == 0 or i == total_events:
            print(f"  Created {i}/{total_events} files...")

    return output_files
