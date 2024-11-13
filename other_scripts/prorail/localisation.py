import geopandas as gpd
from numpy import interp
from shapely.geometry import Point


def interpolate_along_known_coords(
    known_coords: gpd.GeoDataFrame, new_points: gpd.GeoDataFrame, channel_col: str = "channel"
) -> gpd.GeoDataFrame:
    """Interpolates the coordinates of new_points based on the known_coords.

    Args:
        known_coords (gpd.GeoDataFrame): The known coordinates.
        new_points (gpd.GeoDataFrame): The new points.
        channel_col (str): The column name to use for interpolation

    Returns:
        gpd.GeoDataFrame: The new points with interpolated coordinates.
    """
    known_coords = known_coords.to_crs("epsg:4326").sort_values(by=channel_col).reset_index(drop=True)
    new_points_latitudes = interp(new_points[channel_col].values, known_coords[channel_col], known_coords.geometry.y)
    new_points_longitudes = interp(new_points[channel_col].values, known_coords[channel_col], known_coords.geometry.x)
    new_points["geometry"] = [Point(lon, lat) for lon, lat in zip(new_points_longitudes, new_points_latitudes)]
    return gpd.GeoDataFrame(new_points, geometry="geometry", crs="epsg:4326")
