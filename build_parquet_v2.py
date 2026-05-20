"""Build Parquet v2 dataset from event-level NetCDF files.

v2 vs v1: FO spectral features use 1/3-octave bands (ISO 18405:2017, base-2)
starting from 1 Hz instead of the uniform 5-Hz linear bins used in v1.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, List

import netCDF4 as nc
import numpy as np
import pandas as pd

from src.db.parquet.config_parquet_v2 import CONFIG
from src.utils.geometry_utils import apply_corrected_distances
from src.db.parquet.parquet_v2_utils import (
    SkipRecord,
    assemble_output_row,
    compute_event_fo_features,
    compute_pgv_z_mms,
    create_build_folder,
    get_sensor_ids,
    load_event_metadata,
    load_sensor_geometry,
    open_netcdf_event,
    process_fo_event_signal,
    save_json,
    validate_row_eligibility,
    write_build_log,
    write_parquet,
)


def _build_train_type_code(df: pd.DataFrame) -> Dict[str, int]:
    values = sorted(str(v) for v in df["train_type"].fillna("unknown").unique())
    mapping = {name: idx for idx, name in enumerate(values)}
    df["train_type_code"] = (
        df["train_type"].map(lambda x: mapping.get(str(x), -1)).astype(int)
    )
    return mapping


def main() -> None:
    cfg = CONFIG

    input_folders = cfg.input_folder_paths()
    output_root = cfg.output_root_path()

    missing = [p for p in input_folders if not p.exists()]
    if missing:
        raise FileNotFoundError(
            f"Input NetCDF folder(s) do not exist: {[str(p) for p in missing]}"
        )

    netcdf_files: List[Path] = []
    for folder in input_folders:
        netcdf_files.extend(sorted(folder.glob(cfg.netcdf_glob)))
    netcdf_files = sorted(netcdf_files)

    if cfg.max_files > 0:
        netcdf_files = netcdf_files[: cfg.max_files]
        print(
            f"[DEBUG] max_files={cfg.max_files}: processing first {len(netcdf_files)} file(s) only."
        )

    total_files = len(netcdf_files)
    for folder in input_folders:
        print(f"Input : {folder}")
    print(f"Files : {total_files} NetCDF events found ({len(input_folders)} folders)")
    print(f"Output: {output_root}")
    print(f"Bands : {len(cfg.octave_bands)} 1/3-octave bands (1 Hz – 100 Hz)")
    if total_files == 0:
        print(
            "WARNING: no NetCDF files found — check input_netcdf_folders and netcdf_glob in config."
        )

    build_dir = create_build_folder(
        output_root=output_root, version_name=cfg.output.version_name
    )
    print(f"Build : {build_dir}\n")

    log_lines: List[str] = [
        f"Build folder: {build_dir}",
        *(f"Input folder: {p}" for p in input_folders),
        f"NetCDF files found: {total_files}",
        f"1/3-octave bands: {len(cfg.octave_bands)}",
    ]

    rows: List[Dict[str, Any]] = []
    skipped: List[SkipRecord] = []

    summary = {
        "netcdf_files_scanned": len(netcdf_files),
        "rows_created": 0,
        "skipped_events": 0,
        "skipped_sensors": 0,
        "missing_z_axis_count": 0,
        "warnings_count": 0,
        "skipped_event_records": [],
        "skipped_sensor_records": [],
        "warnings": [],
        "target_name": cfg.target.name,
        "row_unit": cfg.row_unit,
    }

    for file_idx, nc_path in enumerate(netcdf_files, start=1):
        print(f"[{file_idx:>4}/{total_files}] {nc_path.name} ...", end=" ", flush=True)
        rows_before_event = len(rows)

        try:
            with open_netcdf_event(nc_path) as dataset:
                event_meta = load_event_metadata(dataset)
                event_id = event_meta.get("event_id", "")

                fo_features = compute_event_fo_features(
                    dataset=dataset,
                    fo_config=cfg.fo_processing,
                    octave_bands=cfg.octave_bands,
                    reductions=cfg.channel_reductions,
                    include_time_domain=cfg.feature_families.fo_time_domain,
                    include_spectral=cfg.feature_families.fo_spectral_octave_bands,
                )

                if cfg.inclusion_rules.require_fo_data and fo_features is None:
                    skipped.append(
                        SkipRecord(
                            level="event",
                            event_id=event_id,
                            sensor_id=None,
                            reason="missing_or_invalid_fo",
                            file_path=str(nc_path),
                        )
                    )
                else:
                    sensor_ids = get_sensor_ids(dataset)
                    if len(sensor_ids) == 0:
                        skipped.append(
                            SkipRecord(
                                level="event",
                                event_id=event_id,
                                sensor_id=None,
                                reason="no_acc_sensors",
                                file_path=str(nc_path),
                            )
                        )
                    else:
                        for sensor_id in sensor_ids:
                            if sensor_id in cfg.exclude_sensor_ids:
                                skipped.append(
                                    SkipRecord(
                                        level="sensor",
                                        event_id=event_id,
                                        sensor_id=sensor_id,
                                        reason="excluded_sensor_id_in_config",
                                        file_path=str(nc_path),
                                    )
                                )
                                continue

                            eligible, reason = validate_row_eligibility(
                                dataset=dataset,
                                sensor_id=sensor_id,
                                require_z_axis=cfg.inclusion_rules.require_z_axis,
                            )
                            if not eligible:
                                skipped.append(
                                    SkipRecord(
                                        level="sensor",
                                        event_id=event_id,
                                        sensor_id=sensor_id,
                                        reason=reason,
                                        file_path=str(nc_path),
                                    )
                                )
                                if reason == "missing_z_axis":
                                    summary["missing_z_axis_count"] += 1
                                continue

                            sensor_group = dataset.groups["acc"].groups[sensor_id]
                            _vel_var = next(
                                (
                                    n
                                    for n in ("velocity_mms", "acceleration_mps2")
                                    if n in sensor_group.variables
                                ),
                                "velocity_mms",
                            )
                            vel_z_mms = np.asarray(
                                sensor_group.variables[_vel_var][:],
                                dtype=np.float64,
                            )[:, 2]

                            try:
                                target_pgv_z_mms = compute_pgv_z_mms(
                                    velocity_z_mms=vel_z_mms
                                )
                            except Exception as exc:
                                skipped.append(
                                    SkipRecord(
                                        level="sensor",
                                        event_id=event_id,
                                        sensor_id=sensor_id,
                                        reason=f"pgv_computation_failed: {exc}",
                                        file_path=str(nc_path),
                                    )
                                )
                                continue

                            if (
                                cfg.inclusion_rules.require_finite_target
                                and not np.isfinite(target_pgv_z_mms)
                            ):
                                skipped.append(
                                    SkipRecord(
                                        level="sensor",
                                        event_id=event_id,
                                        sensor_id=sensor_id,
                                        reason="non_finite_target_pgv_z_mms",
                                        file_path=str(nc_path),
                                    )
                                )
                                continue

                            geometry = load_sensor_geometry(
                                dataset, sensor_id=sensor_id
                            )

                            row = assemble_output_row(
                                event_meta=event_meta,
                                sensor_id=sensor_id,
                                target_pgv_z_mms=target_pgv_z_mms,
                                geometry=geometry,
                                fo_features=fo_features or {},
                            )
                            rows.append(row)

        except Exception as exc:
            skipped.append(
                SkipRecord(
                    level="event",
                    event_id="",
                    sensor_id=None,
                    reason=f"event_read_failed: {exc}",
                    file_path=str(nc_path),
                )
            )

        rows_after_event = len(rows)
        new_rows = rows_after_event - rows_before_event
        if new_rows == 0:
            summary["skipped_events"] += 1
            print("SKIPPED")
        else:
            print(f"OK  (+{new_rows} rows)")

    if len(rows) == 0:
        summary["warnings"].append(
            "No rows were created. Dataset.parquet was not written."
        )
        summary["warnings_count"] = len(summary["warnings"])
    else:
        df = pd.DataFrame(rows)

        # Fix distances: override acc_distance_to_track_m with correct geometry
        # from holten.json and add effective_distance_to_active_track_m.
        # The raw NetCDF values use field-label distances (always track 1, some
        # values rounded / wrong). This must be applied before any downstream
        # attenuation fitting or feature engineering.
        df = apply_corrected_distances(df)
        print(
            f"\nDistance correction applied: "
            f"acc_distance_to_track_m overridden, "
            f"effective_distance_to_active_track_m added."
        )

        if cfg.feature_families.include_train_type_code:
            train_type_mapping = _build_train_type_code(df)
        else:
            train_type_mapping = {}

        if cfg.sort_rows_by:
            valid_sort_cols = [c for c in cfg.sort_rows_by if c in df.columns]
            if valid_sort_cols:
                df = df.sort_values(valid_sort_cols).reset_index(drop=True)

        parquet_path = build_dir / cfg.output.parquet_filename
        write_parquet(df=df, output_path=parquet_path, engine=cfg.output.parquet_engine)

        summary["rows_created"] = int(len(df))
        summary["parquet_path"] = str(parquet_path)
        summary["columns"] = list(df.columns)
        summary["train_type_mapping"] = train_type_mapping

    skipped_event_records = []
    skipped_sensor_records = []
    for rec in skipped:
        payload = {
            "event_id": rec.event_id,
            "sensor_id": rec.sensor_id,
            "reason": rec.reason,
            "file_path": rec.file_path,
        }
        if rec.level == "event":
            skipped_event_records.append(payload)
        elif rec.level == "sensor":
            skipped_sensor_records.append(payload)

    summary["skipped_event_records"] = skipped_event_records
    summary["skipped_sensor_records"] = skipped_sensor_records
    summary["skipped_sensors"] = len(skipped_sensor_records)
    summary["warnings_count"] = len(summary["warnings"])

    config_snapshot_path = build_dir / cfg.output.config_snapshot_filename
    summary_path = build_dir / cfg.output.summary_filename
    log_path = build_dir / cfg.output.log_filename

    save_json(config_snapshot_path, cfg.as_dict())
    save_json(summary_path, summary)

    log_lines.append(f"Rows created: {summary['rows_created']}")
    log_lines.append(f"Skipped events: {summary['skipped_events']}")
    log_lines.append(f"Skipped sensors: {summary['skipped_sensors']}")
    log_lines.append(f"Missing z-axis count: {summary['missing_z_axis_count']}")
    log_lines.append(f"Warnings: {summary['warnings_count']}")
    if "parquet_path" in summary:
        log_lines.append(f"Parquet: {summary['parquet_path']}")

    write_build_log(log_path=log_path, lines=log_lines)

    print("=" * 80)
    print("PARQUET V2 BUILD COMPLETE")
    print("=" * 80)
    print(f"Build folder: {build_dir}")
    print(f"Rows created: {summary['rows_created']}")
    print(f"Skipped events: {summary['skipped_events']}")
    print(f"Skipped sensors: {summary['skipped_sensors']}")
    print(f"Config snapshot: {config_snapshot_path}")
    print(f"Summary: {summary_path}")
    print(f"Build log: {log_path}")


if __name__ == "__main__":
    main()
