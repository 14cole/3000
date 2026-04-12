#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np


def load_module(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module {name} from {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def first_bad_index(arr: np.ndarray):
    bad = np.argwhere(~np.isfinite(arr))
    if bad.size == 0:
        return None
    return tuple(int(v) for v in bad[0])


def row_stats(vec: np.ndarray):
    bad = np.argwhere(~np.isfinite(vec))
    return {
        "finite": bool(np.all(np.isfinite(vec))),
        "first_bad_col": None if bad.size == 0 else int(bad[0][0]),
        "bad_count": int(bad.shape[0]) if bad.size else 0,
        "max_abs_finite": float(np.max(np.abs(vec[np.isfinite(vec)]))) if np.any(np.isfinite(vec)) else None,
    }


def main():
    ap = argparse.ArgumentParser(description="Trace first bad row inside coupled pulse/collocation a_core assembly")
    ap.add_argument("geometry")
    ap.add_argument("--solver", default="rcs_solver.py")
    ap.add_argument("--geometry-io", default="geometry_io.py")
    ap.add_argument("--freq", type=float, required=True)
    ap.add_argument("--elev", type=float, default=0.0)
    ap.add_argument("--pol", default="TE")
    ap.add_argument("--units", default="inches")
    ap.add_argument("--mesh-ref", type=float, default=None)
    ap.add_argument("--out", default="coupled_row_trace_summary.json")
    args = ap.parse_args()

    solver = load_module("solver_mod", args.solver)
    gio = load_module("geometry_io_mod", args.geometry_io)

    with open(args.geometry, "r") as f:
        text = f.read()
    title, segments, ibcs_entries, dielectric_entries = gio.parse_geometry(text)
    snapshot = gio.build_geometry_snapshot(title, segments, ibcs_entries, dielectric_entries)
    base_dir = os.path.dirname(os.path.abspath(args.geometry))

    k0 = 2.0 * math.pi * (args.freq * 1e9) / solver.C0
    pol_internal = solver._normalize_polarization(args.pol)
    meters_scale = solver._unit_scale_to_meters(args.units)
    mesh_freq = float(args.mesh_ref) if args.mesh_ref is not None else float(args.freq)
    lambda_min = solver.C0 / (mesh_freq * 1e9)

    panels, preflight = solver._build_panels_from_snapshot(
        snapshot,
        meters_scale=meters_scale,
        min_wavelength=lambda_min,
        max_panels=solver.MAX_PANELS_DEFAULT,
    )
    materials = solver.MaterialLibrary.from_entries(
        ibcs_entries=snapshot.get("ibcs", []),
        dielectric_entries=snapshot.get("dielectrics", []),
        base_dir=base_dir,
    )
    infos = solver._build_coupled_panel_info(panels, materials, args.freq, pol_internal, k0)
    region_ops = solver._build_coupled_region_operators(panels, infos)

    n = len(panels)
    a = np.zeros((2 * n, 2 * n), dtype=np.complex128)
    rows = []
    row = 0

    for i, info in enumerate(infos):
        active_region = info.minus_region if info.minus_region >= 0 else info.plus_region
        s_active, k_active = region_ops[active_region]
        row_u, row_q = solver._assemble_coupled_region_row(i, active_region, s_active, k_active, infos)
        a[row, :n] = row_u
        a[row, n:] = row_q
        rows.append({
            "row": row,
            "panel_index": i,
            "panel_name": panels[i].name,
            "kind": "active_region",
            "active_region": int(active_region),
            "bc_kind": str(info.bc_kind),
            "stats": row_stats(a[row]),
        })
        row += 1

        if info.bc_kind == "transmission":
            passive_region = info.plus_region if active_region == info.minus_region else info.minus_region
            if passive_region >= 0:
                s_passive, k_passive = region_ops[passive_region]
                row_u, row_q = solver._assemble_coupled_region_row(i, passive_region, s_passive, k_passive, infos)
                a[row, :n] = row_u
                a[row, n:] = row_q
                rows.append({
                    "row": row,
                    "panel_index": i,
                    "panel_name": panels[i].name,
                    "kind": "passive_region",
                    "active_region": int(active_region),
                    "passive_region": int(passive_region),
                    "bc_kind": str(info.bc_kind),
                    "stats": row_stats(a[row]),
                })
                row += 1
                continue

        z = info.robin_impedance
        physical_region = active_region
        coeff_u, coeff_q = solver._region_side_trace_coefficients(info, physical_region)
        alpha = None
        if abs(z) <= solver.EPS:
            if pol_internal == "TM":
                a[row, i] = 1.0 + 0.0j
            else:
                a[row, i] = coeff_u
                a[row, n + i] = coeff_q
            kind = "pec_or_zero_impedance_bc"
        else:
            eps_phys = info.eps_minus if physical_region == info.minus_region else info.eps_plus
            mu_phys = info.mu_minus if physical_region == info.minus_region else info.mu_plus
            k_phys = info.k_minus if physical_region == info.minus_region else info.k_plus
            alpha = solver._surface_robin_alpha(pol_internal, eps_phys, mu_phys, k_phys, z)
            a[row, i] = coeff_u + alpha
            a[row, n + i] = coeff_q
            kind = "robin_bc"
        rows.append({
            "row": row,
            "panel_index": i,
            "panel_name": panels[i].name,
            "kind": kind,
            "active_region": int(active_region),
            "bc_kind": str(info.bc_kind),
            "coeff_u": [float(np.real(coeff_u)), float(np.imag(coeff_u))],
            "coeff_q": [float(np.real(coeff_q)), float(np.imag(coeff_q))],
            "alpha": None if alpha is None else [float(np.real(alpha)), float(np.imag(alpha))],
            "robin_impedance": [float(np.real(z)), float(np.imag(z))],
            "stats": row_stats(a[row]),
        })
        row += 1

    summary: dict[str, Any] = {
        "freq_ghz": float(args.freq),
        "mesh_ref_ghz": None if args.mesh_ref is None else float(args.mesh_ref),
        "pol_user": args.pol,
        "pol_internal": pol_internal,
        "panel_count": n,
        "preflight": preflight,
        "a_core_finite": bool(np.all(np.isfinite(a))),
        "a_core_first_bad": first_bad_index(a),
        "first_bad_row_record": None,
        "rows": rows,
    }

    if not summary["a_core_finite"]:
        bad_row = summary["a_core_first_bad"][0]
        summary["first_bad_row_record"] = rows[bad_row]

    with open(args.out, "w") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps({
        "a_core_finite": summary["a_core_finite"],
        "a_core_first_bad": summary["a_core_first_bad"],
        "first_bad_row_record": summary["first_bad_row_record"],
    }, indent=2))


if __name__ == "__main__":
    main()
