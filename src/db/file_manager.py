"""
File manager module for database creation.

Handles file discovery, unique ID assignment, and file registry management.

Author: Fabian Campos
Date: February 2026
"""

from pathlib import Path
from typing import Dict


def assign_file_ids(data_folder: str, file_extension: str = ".h5") -> Dict[str, str]:
    """
    Assign unique sequential IDs to all files in a folder.

    This creates a registry mapping file_id -> file_path for all files
    in the specified folder. Files are sorted by name before ID assignment
    to ensure consistent ordering across runs.

    Args:
        data_folder: Path to folder containing data files.
        file_extension: File extension to filter by (default: '.h5').

    Returns:
        Dict[str, str]: Dictionary mapping file IDs to file paths.
            Example: {'FILE_0001': 'path/to/file1.h5', 'FILE_0002': 'path/to/file2.h5'}

    Example:
        >>> registry = assign_file_ids(r"D:\data\fo_files")
        >>> print(f"Found {len(registry)} files")
        >>> print(registry['FILE_0001'])
    """
    data_folder = Path(data_folder)

    # Check if folder exists
    if not data_folder.exists():
        raise ValueError(f"Data folder does not exist: {data_folder}")

    # Find all files with the specified extension
    files = sorted(data_folder.glob(f"*{file_extension}"))

    if len(files) == 0:
        raise ValueError(f"No {file_extension} files found in {data_folder}")

    # Create registry with zero-padded IDs
    file_registry = {}
    id_width = len(str(len(files)))  # Calculate padding width based on total files

    for i, file_path in enumerate(files, start=1):
        file_id = f"FILE_{i:0{id_width}d}"
        file_registry[file_id] = str(file_path)

    return file_registry


def save_registry(registry: Dict[str, str], output_path: str):
    """
    Save file registry to a CSV file for reference.

    Args:
        registry: Dictionary mapping file IDs to file paths.
        output_path: Path where CSV file will be saved.
    """
    import csv

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["file_id", "file_path", "file_name"])

        for file_id, file_path in registry.items():
            file_name = Path(file_path).name
            writer.writerow([file_id, file_path, file_name])

    print(f"Registry saved to: {output_path}")


def load_registry(registry_path: str) -> Dict[str, str]:
    """
    Load file registry from a CSV file.

    Args:
        registry_path: Path to CSV file with registry data.

    Returns:
        Dict[str, str]: Dictionary mapping file IDs to file paths.
    """
    import csv

    registry = {}

    with open(registry_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            registry[row["file_id"]] = row["file_path"]

    return registry
