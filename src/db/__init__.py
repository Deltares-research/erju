"""
Database creation module.

Contains utilities for creating NetCDF databases from sensor data.
"""

try:
    from .netcdf import create_netcdf_database, inspect_netcdf
    __all__ = [
        "create_netcdf_database",
        "inspect_netcdf",
    ]
except ModuleNotFoundError:
    # netCDF4 is optional; not required for parquet/ML workflows
    __all__ = []
