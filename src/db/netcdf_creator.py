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


def create_netcdf_database(
    events_dict: dict,
    output_folder: str,
    site_name: str,
    name_format: str = "EVENT_{:04d}",
    compression_level: int = 9,
):
    """
    Create hierarchical NetCDF database from accelerometer events.

    Handles both single and multiple measurement points per event.
    Each NetCDF file represents one train passing event and contains:
    - general/: Global metadata (event_id, site_id, timestamps)
    - meta/: Event metadata (train info, timing)
    - geometry/: Sensor geometry information
    - acc/<MP_ID>/: Accelerometer data for each measurement point
      - time_s: relative time vector
      - fs_hz: sampling frequency
      - acceleration_mps2: [time, 3] array (x, y, z)
      - axis_mask: [3] array indicating available axes

    Args:
        events_dict: Dictionary from fetch_multi_mp_accel_data() organized by event_id.
        output_folder: Path to output folder.
        site_name: Name of the site (e.g., "Holten").
        name_format: Format string for file naming.
        compression_level: Compression level (1-9).

    Returns:
        list: List of created NetCDF file paths.
    """
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

        # Reference time (t0) - use earliest time from all MPs
        t0_utc = datetime.strptime(metadata["start_time"], "%Y-%m-%d %H:%M:%S")

        # Create NetCDF file with hierarchical groups
        with nc.Dataset(netcdf_path, "w", format="NETCDF4") as dataset:

            # ===================================================================
            # GROUP: general - Global metadata
            # ===================================================================
            general_grp = dataset.createGroup("general")
            general_grp.event_id = str(event_id)
            general_grp.site_id = site_name
            general_grp.t0_utc = t0_utc.isoformat()
            general_grp.created_utc = datetime.now().isoformat()
            general_grp.pipeline_version = "1.0.0"
            general_grp.software_git_hash = "N/A"  # TODO: Get from git

            # ===================================================================
            # GROUP: meta - Event metadata
            # ===================================================================
            meta_grp = dataset.createGroup("meta")
            meta_grp.train_type = metadata.get("traintype", "unknown")
            meta_grp.track = metadata.get("track", "unknown")

            # Calculate event timing relative to t0
            time_window = metadata["time_window"]
            event_start_offset = (time_window[0] - t0_utc).total_seconds()
            event_end_offset = (time_window[1] - t0_utc).total_seconds()

            meta_grp.createVariable("event_start_offset_s", "f4")
            meta_grp.createVariable("event_end_offset_s", "f4")
            meta_grp.createVariable("train_speed_mps", "f4")

            meta_grp["event_start_offset_s"][:] = event_start_offset
            meta_grp["event_end_offset_s"][:] = event_end_offset
            meta_grp["train_speed_mps"][:] = np.nan  # Unknown for now
            meta_grp["train_speed_mps"].units = "m/s"

            # ===================================================================
            # GROUP: geometry - Sensor geometry
            # ===================================================================
            geom_grp = dataset.createGroup("geometry")
            n_sensors = len(mps)

            # Create dimension for sensors
            geom_grp.createDimension("acc_sensor", n_sensors)
            geom_grp.createDimension("acc_axis", 3)  # x, y, z

            # Sensor IDs
            mp_ids = list(mps.keys())
            sensor_id_var = geom_grp.createVariable(
                "acc_sensor_id", str, ("acc_sensor",)
            )
            for idx, mp_id in enumerate(mp_ids):
                sensor_id_var[idx] = mp_id

            # Axis labels
            axis_labels_var = geom_grp.createVariable("axis_labels", str, ("acc_axis",))
            axis_labels_var[0] = "x"
            axis_labels_var[1] = "y"
            axis_labels_var[2] = "z"

            # TODO: Add sensor positions, distances to track, etc. when available
            # For now, fill with placeholders
            dist_var = geom_grp.createVariable(
                "acc_distance_to_track_m", "f4", ("acc_sensor",)
            )
            dist_var[:] = np.nan  # Unknown
            dist_var.units = "m"

            # ===================================================================
            # GROUP: acc/<MP_ID> - Accelerometer data for each measurement point
            # ===================================================================
            for mp_id, mp_data in mps.items():
                acc_grp = dataset.createGroup(f"acc/{mp_id}")

                # Extract data
                absolute_time = mp_data["absolute_time"]
                trace_x = mp_data["trace_x"]
                trace_y = mp_data["trace_y"]
                trace_z = mp_data["trace_z"]
                fs_hz = mp_data["fs_hz"]
                n_samples = mp_data["n_samples"]

                # Convert to relative time (seconds from t0_utc)
                time_s = np.array([(t - t0_utc).total_seconds() for t in absolute_time])

                # Create dimension
                acc_grp.createDimension("acc_time", n_samples)
                acc_grp.createDimension("acc_axis", 3)

                # Time variable
                time_var = acc_grp.createVariable(
                    "time_s",
                    "f8",
                    ("acc_time",),
                    compression="zlib" if compression_level > 0 else None,
                    complevel=compression_level if compression_level > 0 else 0,
                )
                time_var[:] = time_s
                time_var.units = "s"
                time_var.long_name = f"Time for {mp_id} relative to t0_utc"

                # Sampling frequency
                fs_var = acc_grp.createVariable("fs_hz", "f4")
                fs_var[:] = fs_hz
                fs_var.units = "Hz"
                fs_var.long_name = "Sampling frequency"

                # Acceleration matrix [time, axis]
                accel_var = acc_grp.createVariable(
                    "acceleration_mps2",
                    "f4",
                    ("acc_time", "acc_axis"),
                    compression="zlib" if compression_level > 0 else None,
                    complevel=compression_level if compression_level > 0 else 0,
                )
                accel_var[:, 0] = trace_x
                accel_var[:, 1] = trace_y
                accel_var[:, 2] = trace_z
                accel_var.units = "m/s^2"
                accel_var.long_name = "Acceleration time series (x, y, z)"

                # Axis mask (all axes present in this case)
                mask_var = acc_grp.createVariable("axis_mask", "i1", ("acc_axis",))
                mask_var[:] = [1, 1, 1]  # All axes present
                mask_var.long_name = "Axis availability (1=present, 0=missing)"

        output_files.append(str(netcdf_path))

        # Print progress
        if i % 10 == 0 or i == total_events:
            print(f"  Created {i}/{total_events} files...")

    return output_files
