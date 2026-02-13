"""
NetCDF writer module for database creation.

This module handles writing event data to NetCDF format with proper
compression, metadata, and CF conventions compliance.

Author: Fabian Campos
Project: Rail4Earth - Subtask 3.3.3
Date: February 2026
"""

import netCDF4 as nc
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List
from loguru import logger


class NetCDFWriter:
    """
    Writer class for creating CF-compliant NetCDF files for sensor event data.

    Supports:
    - Compression (zlib, complevel 1-9)
    - Multiple data variables (FO signal, accelerometer traces)
    - Rich metadata
    - Time series with proper datetime handling
    """

    def __init__(self, compression: bool = True, compression_level: int = 9):
        """
        Initialize NetCDF writer.

        Args:
            compression: Whether to compress variables.
            compression_level: Compression level (1-9, higher = more compression).
        """
        self.compression = compression
        self.compression_level = compression_level

    def write_event_file(
        self,
        output_path: Path,
        event_data: Dict[str, Any],
        file_metadata: Optional[Dict[str, Any]] = None,
    ) -> Path:
        """
        Write a single event to a NetCDF file.

        Args:
            output_path: Path where NetCDF file will be saved.
            event_data: Dictionary containing event data (signals, times, etc.).
            file_metadata: Optional metadata to include in file.

        Returns:
            Path: Path to created NetCDF file.

        Expected event_data structure:
            {
                'start_time': datetime,
                'end_time': datetime,
                'fo_signal': {
                    'time': array of datetime,
                    'values': array of float,
                    'channel': int
                },
                'accel_signal': {  # optional
                    'time': array of datetime,
                    'x': array of float,
                    'y': array of float,
                    'z': array of float
                },
                'source_files': list of str,
                'window_index': int
            }
        """
        output_path = Path(output_path)

        logger.debug(f"Writing NetCDF file: {output_path.name}")

        with nc.Dataset(output_path, "w", format="NETCDF4") as dataset:
            # Global attributes
            self._write_global_attributes(dataset, event_data, file_metadata)

            # Write FO data
            if "fo_signal" in event_data:
                self._write_fo_data(dataset, event_data["fo_signal"])

            # Write accelerometer data
            if "accel_signal" in event_data:
                self._write_accel_data(dataset, event_data["accel_signal"])

        logger.debug(f"NetCDF file created: {output_path}")
        return output_path

    def _write_global_attributes(
        self,
        dataset: nc.Dataset,
        event_data: Dict[str, Any],
        file_metadata: Optional[Dict[str, Any]],
    ):
        """Write global attributes to NetCDF file."""
        # CF conventions
        dataset.Conventions = "CF-1.8"
        dataset.title = "Railway Vibration Event Data"
        dataset.institution = "Deltares"
        dataset.source = "Fiber Optic Distributed Acoustic Sensing (DAS)"
        dataset.history = f"Created on {datetime.now().isoformat()}"

        # Event timing
        dataset.event_start_time = event_data["start_time"].isoformat()
        dataset.event_end_time = event_data["end_time"].isoformat()
        dataset.event_duration_seconds = (
            event_data["end_time"] - event_data["start_time"]
        ).total_seconds()

        # Source information
        if "source_files" in event_data:
            dataset.source_files = ", ".join(event_data["source_files"])

        if "window_index" in event_data:
            dataset.window_index = event_data["window_index"]

        # Additional metadata
        if file_metadata:
            for key, value in file_metadata.items():
                setattr(dataset, key, str(value))

    def _write_fo_data(self, dataset: nc.Dataset, fo_data: Dict[str, Any]):
        """
        Write fiber optic signal data to NetCDF file.

        Args:
            dataset: NetCDF dataset object.
            fo_data: Dictionary with 'time', 'values', and 'channel' keys.
        """
        # Create time dimension
        time_dim = dataset.createDimension("time_fo", len(fo_data["time"]))

        # Create time variable
        time_var = dataset.createVariable(
            "time_fo",
            np.float64,
            ("time_fo",),
            zlib=self.compression,
            complevel=self.compression_level,
        )
        time_var.units = "seconds since 1970-01-01 00:00:00"
        time_var.long_name = "time"
        time_var.standard_name = "time"
        time_var.calendar = "gregorian"

        # Convert datetime to seconds since epoch
        epoch = datetime(1970, 1, 1)
        time_values = np.array([(t - epoch).total_seconds() for t in fo_data["time"]])
        time_var[:] = time_values

        # Create FO signal variable
        signal_var = dataset.createVariable(
            "fo_signal",
            np.float32,
            ("time_fo",),
            zlib=self.compression,
            complevel=self.compression_level,
        )
        signal_var.long_name = "Fiber Optic Strain Rate Signal"
        signal_var.units = "strain_rate"
        signal_var.channel = fo_data["channel"]
        signal_var[:] = fo_data["values"]

        logger.debug(
            f"FO data written: {len(fo_data['values'])} samples, channel {fo_data['channel']}"
        )

    def _write_accel_data(self, dataset: nc.Dataset, accel_data: Dict[str, Any]):
        """
        Write accelerometer data to NetCDF file.

        Args:
            dataset: NetCDF dataset object.
            accel_data: Dictionary with 'time', 'x', 'y', 'z' keys.
        """
        # Create time dimension (separate from FO if sampling rates differ)
        time_dim = dataset.createDimension("time_accel", len(accel_data["time"]))

        # Create time variable
        time_var = dataset.createVariable(
            "time_accel",
            np.float64,
            ("time_accel",),
            zlib=self.compression,
            complevel=self.compression_level,
        )
        time_var.units = "seconds since 1970-01-01 00:00:00"
        time_var.long_name = "time"
        time_var.standard_name = "time"
        time_var.calendar = "gregorian"

        # Convert datetime to seconds since epoch
        epoch = datetime(1970, 1, 1)
        time_values = np.array(
            [(t - epoch).total_seconds() for t in accel_data["time"]]
        )
        time_var[:] = time_values

        # Create accelerometer variables for each axis
        for axis in ["x", "y", "z"]:
            if axis in accel_data:
                var = dataset.createVariable(
                    f"accel_{axis}",
                    np.float32,
                    ("time_accel",),
                    zlib=self.compression,
                    complevel=self.compression_level,
                )
                var.long_name = f"Acceleration {axis.upper()}-axis"
                var.units = "m/s^2"
                var.axis = axis.upper()
                var[:] = accel_data[axis]

        logger.debug(f"Accelerometer data written: {len(accel_data['time'])} samples")

    def write_batch(
        self,
        output_dir: Path,
        events: List[Dict[str, Any]],
        file_metadata: Optional[Dict[str, Any]] = None,
    ) -> List[Path]:
        """
        Write multiple events to separate NetCDF files.

        Args:
            output_dir: Directory where files will be saved.
            events: List of event dictionaries.
            file_metadata: Optional metadata for all files.

        Returns:
            List[Path]: List of paths to created files.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        output_files = []

        for i, event in enumerate(events, 1):
            # Generate filename from start time
            timestamp = event["start_time"].strftime("%Y%m%d_%H%M%S%f")
            filename = f"event_{timestamp}.nc"
            output_path = output_dir / filename

            try:
                self.write_event_file(output_path, event, file_metadata)
                output_files.append(output_path)
            except Exception as e:
                logger.error(f"Failed to write event {i}: {e}")

        logger.info(
            f"Batch write complete: {len(output_files)}/{len(events)} files written"
        )
        return output_files


class NetCDFReader:
    """
    Reader class for loading event data from NetCDF files.

    Companion to NetCDFWriter for reading back data.
    """

    @staticmethod
    def read_event_file(file_path: Path) -> Dict[str, Any]:
        """
        Read event data from a NetCDF file.

        Args:
            file_path: Path to NetCDF file.

        Returns:
            Dict containing event data and metadata.
        """
        file_path = Path(file_path)

        with nc.Dataset(file_path, "r") as dataset:
            event_data = {}

            # Read global attributes
            event_data["metadata"] = {
                attr: getattr(dataset, attr) for attr in dataset.ncattrs()
            }

            # Read FO data if present
            if "fo_signal" in dataset.variables:
                time_fo = dataset.variables["time_fo"][:]
                fo_signal = dataset.variables["fo_signal"][:]

                # Convert time back to datetime
                epoch = datetime(1970, 1, 1)
                time_fo_dt = [epoch + timedelta(seconds=float(t)) for t in time_fo]

                event_data["fo_signal"] = {
                    "time": time_fo_dt,
                    "values": fo_signal,
                    "channel": dataset.variables["fo_signal"].channel,
                }

            # Read accelerometer data if present
            if "time_accel" in dataset.dimensions:
                time_accel = dataset.variables["time_accel"][:]

                epoch = datetime(1970, 1, 1)
                time_accel_dt = [
                    epoch + timedelta(seconds=float(t)) for t in time_accel
                ]

                event_data["accel_signal"] = {"time": time_accel_dt}

                for axis in ["x", "y", "z"]:
                    var_name = f"accel_{axis}"
                    if var_name in dataset.variables:
                        event_data["accel_signal"][axis] = dataset.variables[var_name][
                            :
                        ]

            return event_data


# ======================================================================================
# UTILITY FUNCTIONS
# ======================================================================================


def get_netcdf_info(file_path: Path) -> Dict[str, Any]:
    """
    Get summary information about a NetCDF file.

    Args:
        file_path: Path to NetCDF file.

    Returns:
        Dict with file information.
    """
    with nc.Dataset(file_path, "r") as dataset:
        info = {
            "dimensions": {
                dim: len(dataset.dimensions[dim]) for dim in dataset.dimensions
            },
            "variables": list(dataset.variables.keys()),
            "attributes": {attr: getattr(dataset, attr) for attr in dataset.ncattrs()},
            "file_size_mb": file_path.stat().st_size / (1024 * 1024),
        }
    return info


def print_netcdf_summary(file_path: Path):
    """Print a formatted summary of a NetCDF file."""
    info = get_netcdf_info(file_path)

    print(f"\n{'='*80}")
    print(f"NetCDF File: {file_path.name}")
    print(f"{'='*80}")
    print(f"\nDimensions:")
    for dim, size in info["dimensions"].items():
        print(f"  {dim}: {size}")
    print(f"\nVariables: {', '.join(info['variables'])}")
    print(f"\nFile size: {info['file_size_mb']:.2f} MB")
    print(f"\nGlobal Attributes:")
    for attr, value in info["attributes"].items():
        print(f"  {attr}: {value}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    # Example usage
    from datetime import timedelta

    print("Testing NetCDF writer...")

    # Create sample event data
    start_time = datetime(2024, 11, 20, 10, 30, 0)
    end_time = start_time + timedelta(seconds=10)

    # Generate sample FO signal
    n_samples = 10000
    time_fo = [start_time + timedelta(seconds=i / 1000) for i in range(n_samples)]
    signal = np.sin(2 * np.pi * 5 * np.arange(n_samples) / 1000)  # 5 Hz sine wave

    event_data = {
        "start_time": start_time,
        "end_time": end_time,
        "fo_signal": {"time": time_fo, "values": signal, "channel": 4270},
        "source_files": ["test_file.tdms"],
        "window_index": 0,
    }

    # Write to file
    writer = NetCDFWriter(compression=True, compression_level=9)
    output_path = Path("test_event.nc")
    writer.write_event_file(output_path, event_data)

    print(f"✅ Test file created: {output_path}")

    # Read it back
    print_netcdf_summary(output_path)

    # Clean up
    output_path.unlink()
    print("✅ Test complete!")
