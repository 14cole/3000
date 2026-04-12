#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import traceback
from typing import Any

from geometry_io import parse_geometry, build_geometry_snapshot
import rcs_solver


def _load_snapshot(geo_path: str):
    with open(geo_path, 'r') as f:
        text = f.read()
    title, segments, ibcs_entries, dielectric_entries = parse_geometry(text)
    snapshot = build_geometry_snapshot(title, segments, ibcs_entries, dielectric_entries)
    return snapshot, os.path.abspath(geo_path), os.path.dirname(os.path.abspath(geo_path))


def _panel_count(snapshot: dict[str, Any], units: str, mesh_ref_ghz: float) -> int:
    meters_scale = rcs_solver._unit_scale_to_meters(units)
    lambda_min = rcs_solver.C0 / (float(mesh_ref_ghz) * 1.0e9)
    panels = rcs_solver._build_panels(snapshot, meters_scale, lambda_min)
    return len(panels)


def _solve_one(snapshot: dict[str, Any], base_dir: str, freq_ghz: float, elev_deg: float,
               pol: str, units: str, basis_family: str, testing_family: str,
               mesh_reference_ghz: float | None):
    return rcs_solver.solve_monostatic_rcs_2d(
        geometry_snapshot=snapshot,
        frequencies_ghz=[float(freq_ghz)],
        elevations_deg=[float(elev_deg)],
        polarization=pol,
        geometry_units=units,
        material_base_dir=base_dir,
        compute_condition_number=False,
        parallel_elevations=False,
        reuse_angle_invariant_matrix=True,
        mesh_reference_ghz=mesh_reference_ghz,
        basis_family=basis_family,
        testing_family=testing_family,
    )


def main() -> int:
    ap = argparse.ArgumentParser(description='Check whether a crash above some frequency is caused by frequency-dependent remeshing.')
    ap.add_argument('geo', help='Path to .geo file')
    ap.add_argument('--freq-start', type=float, default=1.0)
    ap.add_argument('--freq-stop', type=float, default=15.0)
    ap.add_argument('--freq-step', type=float, default=1.0)
    ap.add_argument('--elev', type=float, default=0.0)
    ap.add_argument('--pol', default='TE')
    ap.add_argument('--units', default='inches')
    ap.add_argument('--method', choices=['pulse', 'galerkin'], default='pulse')
    ap.add_argument('--fixed-mesh-ghz', type=float, default=10.0,
                    help='Reference frequency for the fixed-mesh comparison run')
    ap.add_argument('--json-out', default='mesh_crash_compare.json')
    args = ap.parse_args()

    basis_family, testing_family = ('pulse', 'collocation') if args.method == 'pulse' else ('linear', 'galerkin')
    snapshot, source_path, base_dir = _load_snapshot(args.geo)

    freqs = []
    f = float(args.freq_start)
    while f <= float(args.freq_stop) + 1e-12:
        freqs.append(round(f, 12))
        f += float(args.freq_step)

    report: dict[str, Any] = {
        'geometry': source_path,
        'units': args.units,
        'polarization': args.pol,
        'method': args.method,
        'fixed_mesh_ghz': args.fixed_mesh_ghz,
        'results': [],
    }

    print(f'Geometry: {source_path}')
    print(f'Method: {basis_family}/{testing_family}')
    print(f'Fixed-mesh comparison reference: {args.fixed_mesh_ghz} GHz')
    print('')

    for freq in freqs:
        row: dict[str, Any] = {'freq_ghz': freq}
        try:
            row['adaptive_panel_count'] = _panel_count(snapshot, args.units, freq)
        except Exception as exc:
            row['adaptive_panel_count_error'] = f'{type(exc).__name__}: {exc}'
        try:
            row['fixed_panel_count'] = _panel_count(snapshot, args.units, args.fixed_mesh_ghz)
        except Exception as exc:
            row['fixed_panel_count_error'] = f'{type(exc).__name__}: {exc}'

        # Adaptive mesh run
        try:
            result_ad = _solve_one(snapshot, base_dir, freq, args.elev, args.pol, args.units,
                                   basis_family, testing_family, None)
            row['adaptive_status'] = 'PASS'
            row['adaptive_warning_count'] = int((result_ad.get('metadata', {}) or {}).get('warning_count', 0))
        except Exception as exc:
            row['adaptive_status'] = 'FAIL'
            row['adaptive_error'] = f'{type(exc).__name__}: {exc}'
            row['adaptive_traceback'] = traceback.format_exc()

        # Fixed mesh run
        try:
            result_fx = _solve_one(snapshot, base_dir, freq, args.elev, args.pol, args.units,
                                   basis_family, testing_family, float(args.fixed_mesh_ghz))
            row['fixed_status'] = 'PASS'
            row['fixed_warning_count'] = int((result_fx.get('metadata', {}) or {}).get('warning_count', 0))
        except Exception as exc:
            row['fixed_status'] = 'FAIL'
            row['fixed_error'] = f'{type(exc).__name__}: {exc}'
            row['fixed_traceback'] = traceback.format_exc()

        report['results'].append(row)

        adaptive_panels = row.get('adaptive_panel_count', '?')
        fixed_panels = row.get('fixed_panel_count', '?')
        print(
            f"{freq:>6g} GHz | adaptive panels={adaptive_panels!s:>6} | fixed panels={fixed_panels!s:>6} | "
            f"adaptive={row.get('adaptive_status')} | fixed={row.get('fixed_status')}"
        )
        if row.get('adaptive_status') == 'FAIL':
            print(f"         adaptive error: {row.get('adaptive_error')}")
        if row.get('fixed_status') == 'FAIL':
            print(f"         fixed error:    {row.get('fixed_error')}")

    with open(args.json_out, 'w') as f:
        json.dump(report, f, indent=2)

    print('')
    print(f'Wrote {args.json_out}')
    print('Interpretation:')
    print('- If adaptive starts failing at 11+ GHz but fixed-mesh keeps passing, the crash is almost certainly caused by frequency-dependent remeshing exposing a geometry/junction issue.')
    print('- If both adaptive and fixed fail at 11+ GHz, the issue is more likely in the frequency-dependent operator evaluation itself for this geometry/material setup.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
