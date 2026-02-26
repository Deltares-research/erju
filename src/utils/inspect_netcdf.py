"""
Utility to inspect hierarchical NetCDF files with groups.

Author: Fabian Campos
Date: February 2026
"""

import netCDF4 as nc
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime, timedelta, timezone


def _to_python_scalar(value):
    """Convert NetCDF scalar values to friendly Python scalars for printing."""
    if hasattr(value, "item"):
        try:
            value = value.item()
        except Exception:
            pass
    if isinstance(value, bytes):
        try:
            value = value.decode("utf-8")
        except Exception:
            pass
    return value


def print_raw_structure(group, indent=0, group_path="ROOT"):
    """Print raw NetCDF structure (what's actually stored in the file)."""
    prefix = "  " * indent

    # Group attributes (STORED)
    if len(group.ncattrs()) > 0:
        for attr in group.ncattrs():
            value = getattr(group, attr)
            print(f"{prefix}[ATTR] {group_path}.{attr} = {value}")

    # Dimensions (STORED)
    if len(group.dimensions) > 0:
        for dim_name, dim in group.dimensions.items():
            size = len(dim) if not dim.isunlimited() else "UNLIMITED"
            print(f"{prefix}[DIM]  {group_path}.{dim_name} = {size}")

    # Variables (STORED)
    if len(group.variables) > 0:
        for var_name, var in group.variables.items():
            shape_str = str(var.shape)
            dtype_str = str(var.dtype)
            print(f"{prefix}[VAR]  {group_path}.{var_name} [{dtype_str}, {shape_str}]")
            if var.shape == ():
                try:
                    value = _to_python_scalar(var[()])
                    print(f"{prefix}       value = {value}")
                except Exception as exc:
                    print(f"{prefix}       value = <error reading scalar: {exc}>")
            # Variable attributes (STORED)
            for attr in var.ncattrs():
                val = getattr(var, attr)
                print(f"{prefix}       .{attr} = {val}")

    # Recurse into subgroups (STORED)
    if len(group.groups) > 0:
        for group_name, subgroup in group.groups.items():
            subgroup_path = f"{group_path}/{group_name}"
            print(f"{prefix}[GRP]  {subgroup_path}")
            print_raw_structure(subgroup, indent + 1, subgroup_path)


def print_formatted_view(group, indent=0):
    """Print formatted view with sample data for understanding."""
    prefix = "  " * indent

    # Group attributes
    if len(group.ncattrs()) > 0:
        print(f"{prefix}Attributes:")
        for attr in group.ncattrs():
            value = getattr(group, attr)
            print(f"{prefix}  {attr:30s} = {value}")
        print()

    # Dimensions
    if len(group.dimensions) > 0:
        print(f"{prefix}Dimensions:")
        for dim_name, dim in group.dimensions.items():
            size = len(dim) if not dim.isunlimited() else "UNLIMITED"
            print(f"{prefix}  {dim_name:30s} = {size}")
        print()

    # Variables
    if len(group.variables) > 0:
        print(f"{prefix}Variables:")
        for var_name, var in group.variables.items():
            shape_str = str(var.shape)
            dtype_str = str(var.dtype)
            print(f"{prefix}  {var_name:30s} {dtype_str:10s} {shape_str}")
            if var.shape == ():
                try:
                    value = _to_python_scalar(var[()])
                    print(f"{prefix}    value = {value}")
                except Exception as exc:
                    print(f"{prefix}    value = <error reading scalar: {exc}>")
            # Show attributes
            for attr in var.ncattrs():
                val = getattr(var, attr)
                print(f"{prefix}    {attr}: {val}")
            # Show first/last values (COMPUTED FOR DISPLAY)
            if var.size > 0:
                if var.ndim == 1:
                    print(f"{prefix}    First: {var[0]}")
                    print(f"{prefix}    Last:  {var[-1]}")
                elif var.ndim == 2:
                    print(f"{prefix}    First row: {var[0, :]}")
        print()

    # Recurse into subgroups
    if len(group.groups) > 0:
        for group_name, subgroup in group.groups.items():
            print(f"{prefix}Group: {group_name}")
            print(f"{prefix}{'-' * 60}")
            print_formatted_view(subgroup, indent + 1)


def _get_event_t0_utc(dataset):
    """Read event reference time from root attributes (event_t0_utc or legacy t0_utc)."""
    t0_str = getattr(dataset, "event_t0_utc", None) or getattr(dataset, "t0_utc", None)
    if t0_str is None:
        return None
    t0_str = _to_python_scalar(t0_str)
    try:
        return datetime.fromisoformat(str(t0_str).replace("Z", "+00:00")).astimezone(
            timezone.utc
        )
    except Exception:
        return None


def plot_sample_accel_timesignals(dataset, max_sensors=3, max_points=None):
    """Plot accelerometer time signals from the NetCDF file (no saving)."""
    if "acc" not in dataset.groups:
        print("\nNo 'acc' group found. Skipping accelerometer plotting.")
        return

    acc_root = dataset.groups["acc"]
    sensor_ids = list(acc_root.groups.keys())
    if len(sensor_ids) == 0:
        print("\nNo accelerometer sensor groups found under '/acc'.")
        return

    event_t0_utc = _get_event_t0_utc(dataset)

    sensors_to_plot = sensor_ids[:max_sensors]
    points_label = "full timeseries" if max_points is None else f"max {max_points} points"
    print(f"\nPLOTTING ACCELEROMETER SIGNALS ({len(sensors_to_plot)} sensor(s), {points_label})")
    print("=" * 80)

    for sensor_id in sensors_to_plot:
        sensor_group = acc_root.groups[sensor_id]

        if "time_s" in sensor_group.variables:
            time = np.asarray(sensor_group.variables["time_s"][:])
            if event_t0_utc is not None:
                time_plot = np.array(
                    [event_t0_utc + timedelta(seconds=float(t)) for t in time],
                    dtype=object,
                )
                time_label = "UTC time (from event_t0_utc)"
            else:
                time_plot = time
                time_label = "time_s"
        elif "time_accel" in sensor_group.variables:
            time = np.asarray(sensor_group.variables["time_accel"][:])
            time_plot = time
            time_label = "time_accel"
        else:
            print(f"  - {sensor_id}: no recognized time variable, skipping")
            continue

        traces = {}
        if "acceleration_mps2" in sensor_group.variables:
            accel = np.asarray(sensor_group.variables["acceleration_mps2"][:])
            if accel.ndim == 2 and accel.shape[1] >= 3:
                traces = {
                    "x": accel[:, 0],
                    "y": accel[:, 1],
                    "z": accel[:, 2],
                }
        else:
            for axis in ["x", "y", "z"]:
                var_name = f"accel_{axis}"
                if var_name in sensor_group.variables:
                    traces[axis] = np.asarray(sensor_group.variables[var_name][:])

        if len(traces) == 0:
            print(f"  - {sensor_id}: no recognized accelerometer traces, skipping")
            continue

        n = len(time_plot) if max_points is None else min(len(time_plot), max_points)
        if n == 0:
            print(f"  - {sensor_id}: empty signal, skipping")
            continue

        fig, axes = plt.subplots(3, 1, figsize=(11, 7), sharex=True)
        fig.suptitle(f"Accelerometer sample traces - {sensor_id}")

        for i, axis in enumerate(["x", "y", "z"]):
            if axis in traces:
                axes[i].plot(time_plot[:n], traces[axis][:n], linewidth=0.9)
                axes[i].set_ylabel(f"{axis.upper()} [m/s²]")
            else:
                axes[i].text(
                    0.5, 0.5, f"{axis.upper()} not available", ha="center", va="center"
                )
                axes[i].set_yticks([])
            axes[i].grid(True, alpha=0.3)

        axes[-1].set_xlabel(f"{time_label}")
        plt.tight_layout()
        plt.show()


def inspect_netcdf(
    file_path: str,
    plot_accel_samples: bool = True,
    max_plot_sensors: int = 3,
    max_plot_points=None,
):
    """Inspect NetCDF file (handles both flat and hierarchical with groups)."""
    file_path = Path(file_path)

    if not file_path.exists():
        print(f"File not found: {file_path}")
        return

    print("\n" + "=" * 80)
    print(f"NetCDF File: {file_path.name}")
    print("=" * 80 + "\n")

    with nc.Dataset(file_path, "r") as dataset:
        # PART 1: RAW NETCDF STRUCTURE (what's actually in the file)
        print("RAW NETCDF STRUCTURE (what's stored in file)")
        print("=" * 80)
        print("Legend: [ATTR]=Attribute, [DIM]=Dimension, [VAR]=Variable, [GRP]=Group")
        print("-" * 80)
        print_raw_structure(dataset)
        print()

        # PART 2: FORMATTED VIEW (human-readable with sample values)
        print("\nFORMATTED VIEW (for understanding)")
        print("=" * 80)
        print_formatted_view(dataset)

        if plot_accel_samples:
            plot_sample_accel_timesignals(
                dataset,
                max_sensors=max_plot_sensors,
                max_points=max_plot_points,
            )

    # File info
    print("\nFILE INFORMATION")
    print("=" * 80)
    print(f"  Path:       {file_path}")
    print(f"  Size:       {file_path.stat().st_size / 1024:.2f} KB")
    print("\n" + "=" * 80 + "\n")


if __name__ == "__main__":
    # Hardcoded path for easy testing (change as needed)
    file_path = r"P:\11210978-erju-ai\holten_db\netcdf_20260226_163053\EVENT_0090.nc"

    # Plotting switch: set True to show sample accelerometer plots, False to disable
    PLOT_ACCEL_SAMPLES = True
    MAX_PLOT_POINTS = None  # None = full timeseries

    inspect_netcdf(
        file_path,
        plot_accel_samples=PLOT_ACCEL_SAMPLES,
        max_plot_points=MAX_PLOT_POINTS,
    )
