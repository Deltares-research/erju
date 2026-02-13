# Site Configuration Files

This folder contains site-specific configuration files for different measurement locations.

## File Format

Each site should have its own JSON file named `{site_name}.json` (lowercase).

Example: `holten.json`, `amsterdam.json`, `utrecht.json`

## JSON Structure

```json
{
  "site_name": "Site Name",
  "description": "Brief description of the measurement site",
  "measurement_points": [
    "Raw_Measurement_Point_Name_1",
    "Raw_Measurement_Point_Name_2"
  ],
  "sensor_id_map": {
    "Raw_Measurement_Point_Name_1": "MP1",
    "Raw_Measurement_Point_Name_2": "MP2"
  },
  "distance_to_track_m": {
    "MP1": 4.0,
    "MP2": 8.0
  },
  "side_of_track": {
    "MP1": -1,
    "MP2": 1
  },
  "axis_mask": {
    "MP1": [1, 1, 1],
    "MP2": [1, 1, 1]
  }
}
```

## Field Descriptions

- **site_name**: Human-readable name of the site
- **description**: Brief description of the measurement campaign or location
- **measurement_points**: List of raw measurement point names as they appear in your database
- **sensor_id_map**: Mapping from raw names to short sensor IDs (MP1, MP2, etc.)
- **distance_to_track_m**: Distance from each sensor to track centerline in meters
- **side_of_track**: Side of track convention (-1=left, 0=unknown, +1=right)
- **axis_mask**: Which axes are available for each sensor [x, y, z] (1=present, 0=missing)

## Adding a New Site

1. Create a new JSON file with the site name in lowercase (e.g., `utrecht.json`)
2. Fill in all required fields following the structure above
3. In `config_db.py`, change `SITE_NAME = "Utrecht"` to use the new site
4. The configuration will be automatically loaded from `sites/utrecht.json`

## Notes

- Sensor IDs should be consistent and short (e.g., MP1, MP2, MP3...)
- All sensors listed in `measurement_points` must have entries in all other dictionaries
- Distance values can be approximate if exact measurements are not available (use 0.0 for unknown)
- Set side_of_track to 0 if uncertain about sensor positioning
