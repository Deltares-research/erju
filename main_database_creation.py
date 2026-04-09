"""
Main script for creating NetCDF databases from accelerometer events.

Author: Fabian Campos
Project: Rail4Earth - Subtask 3.3.3
Date: February 2026

Usage:
    python main_database_creation.py
"""

import sys
import csv
from pathlib import Path
from datetime import datetime, timedelta
from time import perf_counter

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

import src.db.netcdf.config_db as config
from src.utils.db_utils import fetch_multi_mp_accel_data
from src.utils.fo_utils import extract_fo_event_data
from src.db.netcdf import create_netcdf_database


DATE_FMT = "%Y-%m-%d %H:%M:%S"


def iter_time_chunks(start_date: str, end_date: str, chunk_days: int):
    """Yield non-overlapping [start, end] datetime chunks for DB querying."""
    start_dt = datetime.strptime(start_date, DATE_FMT)
    end_dt = datetime.strptime(end_date, DATE_FMT)

    if chunk_days is None or chunk_days <= 0:
        yield start_dt, end_dt
        return

    chunk_start = start_dt
    while chunk_start <= end_dt:
        next_chunk_start = chunk_start + timedelta(days=chunk_days)
        chunk_end = min(end_dt, next_chunk_start - timedelta(seconds=1))
        yield chunk_start, chunk_end
        chunk_start = next_chunk_start


def main():
    """Main entry point for database creation."""

    print("\n" + "=" * 80)
    print("ACCELEROMETER DATABASE CREATION - STARTING")
    print("=" * 80)
    print(f"\nSite:         {config.SITE_NAME}")
    print(f"Database:     {config.ACCEL_DB_PATH}")
    print(f"Date Range:   {config.ACCEL_START_DATE} → {config.ACCEL_END_DATE}")
    print(f"\nMeasurement Points ({len(config.ACCEL_MEASUREMENT_POINTS)}):")
    for mp in config.ACCEL_MEASUREMENT_POINTS:
        print(f"  - {mp}")
    print(f"\nFilters:")
    print(f"  Train Type: {config.ACCEL_TRAINTYPE}")
    print(f"  Track:      {config.ACCEL_TRACK}")
    chunk_days = getattr(config, "ACCEL_QUERY_CHUNK_DAYS", 1)
    print(f"  Chunk Days: {chunk_days}")
    print(f"\nOutput:       {config.OUTPUT_FOLDER}")
    print("\n" + "=" * 80 + "\n")

    # 1. Create timestamped subfolder
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_subfolder = Path(config.OUTPUT_FOLDER) / f"netcdf_{timestamp}"
    output_subfolder.mkdir(parents=True, exist_ok=True)
    print(f"Step 1: Creating subfolder: {output_subfolder.name}\n")

    # 2. Fetch and process accelerometer data in date chunks
    chunk_ranges = list(
        iter_time_chunks(
            config.ACCEL_START_DATE,
            config.ACCEL_END_DATE,
            chunk_days,
        )
    )
    print(f"Step 2: Processing {len(chunk_ranges)} date chunk(s)...")

    output_files = []
    total_events_found = 0
    fo_report_rows = []

    run_t0 = perf_counter()

    for chunk_idx, (chunk_start_dt, chunk_end_dt) in enumerate(chunk_ranges, start=1):
        chunk_t0 = perf_counter()
        chunk_start = chunk_start_dt.strftime(DATE_FMT)
        chunk_end = chunk_end_dt.strftime(DATE_FMT)

        print(
            f"\nChunk {chunk_idx}/{len(chunk_ranges)}: " f"{chunk_start} → {chunk_end}"
        )
        print("  Fetching accelerometer events from database...")

        events_dict = fetch_multi_mp_accel_data(
            path_db=config.ACCEL_DB_PATH,
            start_date=chunk_start,
            end_date=chunk_end,
            measurement_points=config.ACCEL_MEASUREMENT_POINTS,
            sensor_id_map=config.ACCEL_SENSOR_ID_MAP,
            campaigns=config.ACCEL_CAMPAIGNS,
            traintype=config.ACCEL_TRAINTYPE,
            track=config.ACCEL_TRACK,
        )

        n_events = len(events_dict)
        total_events_found += n_events

        if n_events > 0:
            sensor_presence_counts = {}
            per_event_sensor_counts = []
            for event_data in events_dict.values():
                sensor_ids = list(event_data["measurement_points"].keys())
                per_event_sensor_counts.append(len(sensor_ids))
                for sid in sensor_ids:
                    sensor_presence_counts[sid] = sensor_presence_counts.get(sid, 0) + 1

            expected_sensors = sorted(set(config.ACCEL_SENSOR_ID_MAP.values()))
            missing_sensor_global = [
                sid for sid in expected_sensors if sensor_presence_counts.get(sid, 0) == 0
            ]

            print(
                f"  Events fetched: {n_events} | "
                f"Sensors/event min-avg-max: "
                f"{min(per_event_sensor_counts)}-"
                f"{(sum(per_event_sensor_counts) / len(per_event_sensor_counts)):.1f}-"
                f"{max(per_event_sensor_counts)}"
            )
            if missing_sensor_global:
                print(
                    "  WARNING: Sensors with zero presence in this chunk: "
                    + ", ".join(missing_sensor_global)
                )
            else:
                print("  All configured sensor IDs appeared at least once in this chunk.")

        if n_events == 0:
            print("  No events in this chunk.")
            continue

        # 2b. Enrich events with FO data using exact accelerometer time window
        if getattr(config, "FO_ENABLE", False):
            print("  Extracting FO data for event windows...")
            for event_id, event_data in events_dict.items():
                time_window = event_data["event_metadata"]["time_window"]

                try:
                    fo_result = extract_fo_event_data(
                        fo_data_path=config.FO_DATA_PATH,
                        time_window=time_window,
                        center_channel=config.FO_CENTER_CHANNEL,
                        channel_half_window=config.FO_CHANNEL_HALF_WINDOW,
                        reader=config.FO_READER,
                        manual_metadata=getattr(config, "FO_METADATA_MANUAL", None),
                    )
                except Exception as exc:
                    fo_result = {
                        "found": False,
                        "reason": f"fo_exception: {exc}",
                        "file_paths": [],
                    }

                if fo_result.get("found", False):
                    event_data["fo_data"] = fo_result

                fo_report_rows.append(
                    {
                        "event_id": event_id,
                        "chunk_start": chunk_start,
                        "chunk_end": chunk_end,
                        "has_fo": int(bool(fo_result.get("found", False))),
                        "reason": fo_result.get("reason", "unknown"),
                        "fo_file_count": len(fo_result.get("file_paths", [])),
                        "fo_n_samples": (
                            int(fo_result["strain"].shape[0])
                            if fo_result.get("found", False)
                            else 0
                        ),
                    }
                )

            chunk_fo_rows = fo_report_rows[-n_events:]
            fo_found_count = sum(int(r["has_fo"]) for r in chunk_fo_rows)
            fo_missing_count = n_events - fo_found_count
            print(
                f"  FO summary: found={fo_found_count} | missing={fo_missing_count} "
                f"({(100.0 * fo_found_count / n_events):.1f}% coverage)"
            )

            # Show top FO miss reasons for quick debugging.
            miss_reason_counts = {}
            for row in chunk_fo_rows:
                if int(row["has_fo"]) == 0:
                    reason = str(row.get("reason", "unknown"))
                    miss_reason_counts[reason] = miss_reason_counts.get(reason, 0) + 1
            if miss_reason_counts:
                sorted_reasons = sorted(
                    miss_reason_counts.items(), key=lambda kv: kv[1], reverse=True
                )
                top_reasons = ", ".join(
                    f"{reason}={count}" for reason, count in sorted_reasons[:5]
                )
                print(f"  FO missing reasons (top): {top_reasons}")

        print(f"  Writing {n_events} NetCDF file(s)...")
        write_t0 = perf_counter()
        chunk_output_files = create_netcdf_database(
            events_dict=events_dict,
            output_folder=output_subfolder,
            site_name=config.SITE_NAME,
            config=config,
            name_format=config.NAME_FORMAT,
            compression_level=9 if config.DATABASE_COMPRESSION else 0,
            start_index=len(output_files) + 1,
        )
        write_dt = perf_counter() - write_t0
        output_files.extend(chunk_output_files)

        chunk_dt = perf_counter() - chunk_t0
        print(
            f"  Chunk complete: wrote {len(chunk_output_files)} file(s) | "
            f"write_time={write_dt:.1f}s | total_chunk_time={chunk_dt:.1f}s"
        )

    if len(output_files) == 0:
        print("\nNo events found! Check your date range and filters.")
        return

    print(f"\nStep 3: Completed {len(chunk_ranges)} chunk(s)")
    print(f"Total events found: {total_events_found}")
    print(f"\nCreated {len(output_files)} NetCDF files")
    print(f"Location: {output_subfolder}")
    print(f"Total runtime: {perf_counter() - run_t0:.1f}s")

    if getattr(config, "FO_ENABLE", False) and getattr(
        config, "FO_SAVE_AVAILABILITY_REPORT", True
    ):
        report_path = output_subfolder / "fo_event_availability.csv"
        with open(report_path, "w", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(
                csv_file,
                fieldnames=[
                    "event_id",
                    "chunk_start",
                    "chunk_end",
                    "has_fo",
                    "reason",
                    "fo_file_count",
                    "fo_n_samples",
                ],
            )
            writer.writeheader()
            writer.writerows(fo_report_rows)
        print(f"FO availability report: {report_path}")

    print("\n" + "=" * 80)
    print("DATABASE CREATION - COMPLETE")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
