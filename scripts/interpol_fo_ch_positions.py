"""
This script interpolates geographic coordinates along a known set of channel points.

It reads a GeoPackage containing channel coordinates (from ProRail), generates new evenly spaced channel points
at a user-defined interval, and interpolates their latitude and longitude based on the known data.
The interpolated points are reprojected to EPSG:28992 (RD New) and saved to a new GeoPackage layer.

Main steps:
1. Load known channel coordinates from a GeoPackage layer.
2. Define a range of new channel values at the desired spacing (e.g., 1 m).
3. Interpolate geographic positions for the new channels.
4. Reproject the results and save them to a new GeoPackage layer.

Useful for creating intermediate channel coordinates for fiber optic or linear sensor datasets.
"""
import geopandas as gpd
import pandas as pd
import numpy as np
from shapely.geometry import Point


def interpolate_along_known_coords(
        known_coords: gpd.GeoDataFrame, new_points: gpd.GeoDataFrame, channel_col: str = "channel"
) -> gpd.GeoDataFrame:
    """
    Interpolates geographic coordinates for new_points based on known channel coordinates.

    Args:
        known_coords: GeoDataFrame with known 'channel' values and Point geometries (in EPSG:4326).
        new_points: GeoDataFrame with the same 'channel' column, geometry will be overwritten.
        channel_col: Column name to interpolate along (default is 'channel').

    Returns:
        GeoDataFrame with interpolated Point geometries.
    """
    known_coords = known_coords.to_crs("epsg:4326").sort_values(by=channel_col).reset_index(drop=True)
    new_points_latitudes = np.interp(new_points[channel_col], known_coords[channel_col], known_coords.geometry.y)
    new_points_longitudes = np.interp(new_points[channel_col], known_coords[channel_col], known_coords.geometry.x)
    new_points["geometry"] = [Point(lon, lat) for lon, lat in zip(new_points_longitudes, new_points_latitudes)]
    return gpd.GeoDataFrame(new_points, geometry="geometry", crs="epsg:4326")


def main():
    # === USER SETTINGS ===
    input_gpkg = "holten_calibrations.gpkg"  # Input GeoPackage filename
    input_layer = "holten_calibrations"  # Layer name in the GeoPackage
    output_gpkg = "interpolated_channel_coordinates_rdnew.gpkg"  # Output file
    output_layer = "interpolated_channels"  # Output layer name
    channel_spacing = 1.0  # <-- Set your desired spacing in meters here

    # Load known coordinates
    known_coords = gpd.read_file(input_gpkg, layer=input_layer)

    # Define range of new channels at user-defined spacing
    min_channel = np.ceil(known_coords["channel"].min())
    max_channel = np.floor(known_coords["channel"].max())
    new_channels = np.arange(min_channel, max_channel + channel_spacing, channel_spacing)

    # Create new GeoDataFrame with those channel values
    new_points = gpd.GeoDataFrame(pd.DataFrame({"channel": new_channels}))

    # Interpolate
    interpolated = interpolate_along_known_coords(known_coords, new_points)

    # Reproject to EPSG:28992
    interpolated_rdnew = interpolated.to_crs(epsg=28992)

    # Save to GeoPackage
    interpolated_rdnew.to_file(output_gpkg, layer=output_layer, driver="GPKG")
    print(f"Saved interpolated points with spacing {channel_spacing} m to:")
    print(f"   {output_gpkg} (layer: {output_layer})")


if __name__ == "__main__":
    main()
