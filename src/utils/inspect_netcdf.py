"""
Utility to inspect hierarchical NetCDF files with groups.

Author: Fabian Campos
Date: February 2026
"""

import netCDF4 as nc
from pathlib import Path


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

    # File info
    print("\nFILE INFORMATION")
    print("=" * 80)
    print(f"  Path:       {file_path}")
    print(f"  Size:       {file_path.stat().st_size / 1024:.2f} KB")
    print("\n" + "=" * 80 + "\n")


if __name__ == "__main__":
    # Hardcoded path for easy testing (change as needed)
    file_path = r"P:\11210978-erju-ai\holten_db\netcdf_20260213_170828\EVENT_0002.nc"
    inspect_netcdf(file_path)
