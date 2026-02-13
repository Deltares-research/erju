"""
Utility to inspect hierarchical NetCDF files with groups.

Author: Fabian Campos
Date: February 2026
"""

import netCDF4 as nc
from pathlib import Path


def print_group(group, indent=0):
    """Recursively print group contents."""
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
            # Show attributes
            for attr in var.ncattrs():
                val = getattr(var, attr)
                print(f"{prefix}    {attr}: {val}")
            # Show first/last values
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
            print(f"{prefix}{'=' * 60}")
            print_group(subgroup, indent + 1)


def inspect_netcdf(file_path: str):
    """Inspect NetCDF file (handles both flat and hierarchical with groups)."""
    file_path = Path(file_path)

    if not file_path.exists():
        print(f"File not found: {file_path}")
        return

    print("\n" + "=" * 80)
    print(f"NetCDF File: {file_path.name}")
    print("=" * 80 + "\n")

    with nc.Dataset(file_path, "r") as dataset:
        print("ROOT GROUP")
        print("=" * 80)
        print_group(dataset)

    # File info
    print("FILE INFO")
    print("=" * 80)
    print(f"  Path:       {file_path}")
    print(f"  Size:       {file_path.stat().st_size / 1024:.2f} KB")
    print("\n" + "=" * 80 + "\n")


if __name__ == "__main__":
    # Hardcoded path for easy testing (change as needed)
    file_path = r"P:\11210978-erju-ai\holten_db\netcdf_20260213_155741\EVENT_0001.nc"
    inspect_netcdf(file_path)
