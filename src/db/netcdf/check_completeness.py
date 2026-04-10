"""
check_completeness.py  –  Scan a NetCDF database folder and report completeness.

Usage
-----
    python src/db/netcdf/check_completeness.py  P:\\path\\to\\netcdf_folder
    python src/db/netcdf/check_completeness.py  P:\\path\\to\\netcdf_folder --expected-sensors MP1 MP6 MP8
    python src/db/netcdf/check_completeness.py  P:\\path\\to\\netcdf_folder --verbose

Each EVENT_*.nc file is checked for:
  * Required groups and root attributes
  * Presence and non-empty data per accelerometer sensor
  * Missing sensors compared to the expected set
  * FO data presence and basic validity

One line is printed per file while scanning, and a full summary table is printed at
the end.
"""

import argparse
import sys
from pathlib import Path

import netCDF4 as nc
import numpy as np


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────


def _read_str_array(variable) -> list[str]:
    """Robustly read a netCDF4 string/char variable into a plain Python list."""
    raw = variable[:]
    if raw is None:
        return []
    out = []
    for item in raw:
        if isinstance(item, (bytes, bytearray)):
            out.append(item.decode("utf-8", errors="replace").strip("\x00"))
        elif hasattr(item, "tobytes"):
            try:
                out.append(
                    item.tobytes().decode("utf-8", errors="replace").strip("\x00")
                )
            except Exception:
                out.append(str(item))
        else:
            out.append(str(item).strip("\x00"))
    return out


def _check_file(path: Path, expected_sensors: list[str]) -> dict:
    """
    Check a single EVENT_*.nc file for completeness.

    Returns a dict with:
        ok           – True if no issues found
        issues       – list of string descriptions of problems
        sensors_found – list of sensor IDs found in geometry_acc
        has_fo        – True if FO group is present and non-empty
        fo_issues     – list of FO-specific issue strings
    """
    issues = []
    fo_issues = []
    sensors_found = []
    has_fo = False

    try:
        with nc.Dataset(path, "r") as ds:
            group_names = set(ds.groups.keys())

            # ── Root attributes ──────────────────────────────────────────────
            for attr in ["event_id", "site_id", "event_t0_utc"]:
                if not hasattr(ds, attr):
                    issues.append(f"missing_root_attr:{attr}")

            # ── meta_acc group ───────────────────────────────────────────────
            if "meta_acc" not in group_names:
                issues.append("missing_group:meta_acc")
            else:
                meta = ds.groups["meta_acc"]
                for var in [
                    "train_type",
                    "event_start_offset_s",
                    "event_end_offset_s",
                    "train_speed_kmh",
                    "track_number",
                ]:
                    if var not in meta.variables:
                        issues.append(f"meta_acc:missing_var:{var}")

            # ── geometry_acc group ───────────────────────────────────────────
            if "geometry_acc" not in group_names:
                issues.append("missing_group:geometry_acc")
            else:
                geom = ds.groups["geometry_acc"]
                for var in [
                    "acc_sensor_id",
                    "axis_labels",
                    "acc_distance_to_track_m",
                    "acc_side_of_track",
                    "axis_mask",
                ]:
                    if var not in geom.variables:
                        issues.append(f"geometry_acc:missing_var:{var}")

                if "acc_sensor_id" in geom.variables:
                    sensors_found = _read_str_array(geom.variables["acc_sensor_id"])

            # ── acc group and per-sensor data ────────────────────────────────
            if "acc" not in group_names:
                issues.append("missing_group:acc")
            else:
                acc_root = ds.groups["acc"]
                for sid in sensors_found:
                    if sid not in acc_root.groups:
                        issues.append(f"acc:missing_subgroup:{sid}")
                        continue
                    sg = acc_root.groups[sid]
                    # time_s
                    if "time_s" not in sg.variables:
                        issues.append(f"acc/{sid}:missing_var:time_s")
                    else:
                        n = (
                            sg.variables["time_s"].shape[0]
                            if sg.variables["time_s"].shape
                            else 0
                        )
                        if n == 0:
                            issues.append(f"acc/{sid}:empty:time_s")
                    # fs_hz
                    if "fs_hz" not in sg.variables:
                        issues.append(f"acc/{sid}:missing_var:fs_hz")
                    # data variable (velocity_mms or acceleration_g)
                    data_var = None
                    for candidate in ["velocity_mms", "acceleration_g"]:
                        if candidate in sg.variables:
                            data_var = candidate
                            break
                    if data_var is None:
                        issues.append(
                            f"acc/{sid}:missing_data_var:(velocity_mms|acceleration_g)"
                        )
                    else:
                        shape = sg.variables[data_var].shape
                        if len(shape) == 0 or shape[0] == 0:
                            issues.append(f"acc/{sid}:empty:{data_var}")

            # ── Expected-sensor coverage check ──────────────────────────────
            missing = [s for s in expected_sensors if s not in sensors_found]
            if missing:
                issues.append(f"missing_sensors:[{','.join(missing)}]")

            # ── FO groups ────────────────────────────────────────────────────
            if "meta_fo" in group_names or "fo" in group_names:
                # FO was intended – check all three groups exist
                for grp in ["meta_fo", "geometry_fo", "fo"]:
                    if grp not in group_names:
                        fo_issues.append(f"missing_group:{grp}")

                if "fo" in group_names:
                    fo_grp = ds.groups["fo"]
                    if "strain" not in fo_grp.variables:
                        fo_issues.append("fo:missing_var:strain")
                    else:
                        strain_shape = fo_grp.variables["strain"].shape
                        if len(strain_shape) == 0 or strain_shape[0] == 0:
                            fo_issues.append("fo:empty:strain")
                        else:
                            # Quick NaN-check on a small sample to avoid reading full array
                            sample = fo_grp.variables["strain"][:5, :]
                            if np.all(np.isnan(sample)):
                                fo_issues.append("fo:allnan:strain(sample)")
                    if "time_s" not in fo_grp.variables:
                        fo_issues.append("fo:missing_var:time_s")
                    if "fs_hz" not in fo_grp.variables:
                        fo_issues.append("fo:missing_var:fs_hz")

                has_fo = len(fo_issues) == 0 and "fo" in group_names

    except Exception as exc:
        issues.append(f"unreadable_file:{exc}")

    return {
        "ok": len(issues) == 0 and len(fo_issues) == 0,
        "issues": issues,
        "sensors_found": sensors_found,
        "has_fo": has_fo,
        "fo_issues": fo_issues,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Sensor reference detection from first file
# ──────────────────────────────────────────────────────────────────────────────


def _infer_expected_sensors(first_file: Path) -> list[str]:
    """Return the sensor IDs listed in geometry_acc/acc_sensor_id of the first file."""
    try:
        with nc.Dataset(first_file, "r") as ds:
            if "geometry_acc" in ds.groups:
                geom = ds.groups["geometry_acc"]
                if "acc_sensor_id" in geom.variables:
                    return _read_str_array(geom.variables["acc_sensor_id"])
    except Exception:
        pass
    return []


# ──────────────────────────────────────────────────────────────────────────────
# Main scan
# ──────────────────────────────────────────────────────────────────────────────


def scan_database(db_folder: Path, expected_sensors: list[str], verbose: bool) -> None:
    files = sorted(db_folder.glob("EVENT_*.nc"))
    if not files:
        print(f"No EVENT_*.nc files found in: {db_folder}")
        sys.exit(1)

    total = len(files)

    # Infer expected sensors if not provided
    if not expected_sensors:
        expected_sensors = _infer_expected_sensors(files[0])
        source = f"inferred from {files[0].name}"
    else:
        source = "provided via --expected-sensors"

    print(f"\nDB folder : {db_folder}")
    print(f"Files     : {total}")
    print(
        f"Expected sensors ({source}): {' '.join(expected_sensors) if expected_sensors else '(none)'}"
    )
    print()

    # ── Per-file scan ──────────────────────────────────────────────────────
    n_ok = 0
    n_issues = 0
    n_fo_present = 0
    n_fo_issues = 0
    n_unreadable = 0

    # For summary: per-sensor missing count + per-issue-type count
    sensor_missing_count: dict[str, int] = {s: 0 for s in expected_sensors}
    issue_type_count: dict[str, int] = {}

    for i, f in enumerate(files, start=1):
        result = _check_file(f, expected_sensors)

        if result["has_fo"]:
            n_fo_present += 1
        if result["fo_issues"]:
            n_fo_issues += 1

        all_issues = result["issues"] + result["fo_issues"]
        file_ok = len(all_issues) == 0

        if file_ok:
            n_ok += 1
            status = "OK"
            detail = f"{len(result['sensors_found'])} sensors"
            if result["has_fo"]:
                detail += ", FO ok"
            elif result["fo_issues"]:
                detail += ", FO issues"
            else:
                detail += ", no FO"
        else:
            n_issues += 1
            status = "ISSUES"
            detail = "  |  ".join(all_issues[:4])
            if len(all_issues) > 4:
                detail += f"  (+ {len(all_issues) - 4} more)"
            if "unreadable_file" in " ".join(all_issues):
                n_unreadable += 1

        # Accumulate sensor missing counts
        for s in expected_sensors:
            if s not in result["sensors_found"]:
                sensor_missing_count[s] = sensor_missing_count.get(s, 0) + 1

        # Accumulate issue type counts (coarse prefix only)
        for issue in result["issues"]:
            key = issue.split(":")[0]
            issue_type_count[key] = issue_type_count.get(key, 0) + 1

        # Print per-file line
        if verbose or not file_ok:
            print(f"[{i:>5}/{total}] {f.name:<20}  {status:<8}  {detail}")
        else:
            # Compact progress: print every 50 files or first/last
            if i == 1 or i == total or i % 50 == 0:
                pct = 100.0 * i / total
                bar = "#" * int(pct / 5) + "." * (20 - int(pct / 5))
                print(
                    f"  [{bar}] {i}/{total} ({pct:.0f}%)  ok={n_ok}  issues={n_issues}",
                    end="\r",
                )

    # Clear progress bar line
    if not verbose:
        print(" " * 80, end="\r")

    # ── Summary ───────────────────────────────────────────────────────────
    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'Total files':<30}: {total}")
    print(f"{'Complete (no issues)':<30}: {n_ok}  ({100.0*n_ok/total:.1f}%)")
    print(f"{'Files with issues':<30}: {n_issues}  ({100.0*n_issues/total:.1f}%)")
    if n_unreadable:
        print(f"{'  Unreadable':<30}: {n_unreadable}")
    print()

    print(
        f"{'FO present + valid':<30}: {n_fo_present}  ({100.0*n_fo_present/total:.1f}%)"
    )
    print(
        f"{'FO absent':<30}: {total - n_fo_present - n_fo_issues}  ({100.0*(total-n_fo_present-n_fo_issues)/total:.1f}%)"
    )
    if n_fo_issues:
        print(
            f"{'FO present but with issues':<30}: {n_fo_issues}  ({100.0*n_fo_issues/total:.1f}%)"
        )
    print()

    # Per-sensor missing count (only print sensors that are actually missing)
    if expected_sensors:
        any_missing = {s: c for s, c in sensor_missing_count.items() if c > 0}
        if any_missing:
            print("Sensors missing from at least one file:")
            for sid, count in sorted(any_missing.items(), key=lambda kv: -kv[1]):
                pct = 100.0 * count / total
                bar = "#" * int(pct / 5)
                print(f"  {sid:<8}: {count:>5}/{total}  ({pct:5.1f}%)  {bar}")
        else:
            print("All expected sensors present in every file.")
        print()

    if issue_type_count:
        print("Issue type breakdown (acc-level):")
        for key, cnt in sorted(issue_type_count.items(), key=lambda kv: -kv[1]):
            print(f"  {key:<35}: {cnt}")
        print()

    print("=" * 60)


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Check completeness of all EVENT_*.nc files in a NetCDF DB folder.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python src/db/netcdf/check_completeness.py P:\\my_db\\netcdf_20260409_163947
  python src/db/netcdf/check_completeness.py P:\\my_db\\netcdf_20260409_163947 --verbose
  python src/db/netcdf/check_completeness.py P:\\my_db\\netcdf_20260409_163947 --expected-sensors MP1 MP2 MP6
        """,
    )
    parser.add_argument(
        "db_folder",
        type=Path,
        help="Path to the folder containing EVENT_*.nc files.",
    )
    parser.add_argument(
        "--expected-sensors",
        nargs="+",
        metavar="SENSOR",
        default=None,
        help=(
            "Sensor IDs that must be present in every file "
            "(e.g. MP1 MP2 MP6). Default: inferred from the first file."
        ),
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print one line per file whether OK or not (default: only print files with issues).",
    )

    args = parser.parse_args()

    if not args.db_folder.exists():
        print(f"ERROR: folder not found: {args.db_folder}")
        sys.exit(1)
    if not args.db_folder.is_dir():
        print(f"ERROR: not a directory: {args.db_folder}")
        sys.exit(1)

    scan_database(
        db_folder=args.db_folder,
        expected_sensors=args.expected_sensors or [],
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
