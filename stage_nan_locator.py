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


def load_module(path: str, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module {name!r} from {path!r}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def arr_report(arr: Any, label: str) -> dict[str, Any]:
    a = np.asarray(arr)
    finite = bool(np.all(np.isfinite(a)))
    out: dict[str, Any] = {
        "label": label,
        "shape": list(a.shape),
        "finite": finite,
    }
    if a.size:
        bad = np.argwhere(~np.isfinite(a))
        if bad.size:
            idx = tuple(int(v) for v in bad[0])
            out["first_bad_index"] = list(idx)
            try:
                val = a[idx]
                out["first_bad_value"] = {
                    "real": float(np.real(val)),
                    "imag": float(np.imag(val)),
                }
            except Exception:
                out["first_bad_value"] = str(a[idx])
        else:
            try:
                out["abs_max"] = float(np.nanmax(np.abs(a)))
                out["abs_min_nonzero"] = float(np.nanmin(np.abs(a[np.nonzero(a)]))) if np.any(a != 0) else 0.0
            except Exception:
                pass
    return out


def info_report(infos: list[Any]) -> dict[str, Any]:
    keys = [
        "eps_plus", "mu_plus", "eps_minus", "mu_minus",
        "k_plus", "k_minus", "q_plus_beta", "q_plus_gamma", "robin_impedance",
    ]
    rep: dict[str, Any] = {"panel_count": int(len(infos)), "finite": True}
    bad_entries: list[dict[str, Any]] = []
    for i, info in enumerate(infos):
        for key in keys:
            val = getattr(info, key)
            z = complex(val)
            if not (math.isfinite(z.real) and math.isfinite(z.imag)):
                rep["finite"] = False
                bad_entries.append({
                    "panel_index": int(i),
                    "field": key,
                    "real": float(np.real(z)),
                    "imag": float(np.imag(z)),
                    "seg_type": int(getattr(info, "seg_type", -999)),
                    "plus_region": int(getattr(info, "plus_region", -999)),
                    "minus_region": int(getattr(info, "minus_region", -999)),
                })
                if len(bad_entries) >= 10:
                    break
        if len(bad_entries) >= 10:
            break
    rep["bad_entries"] = bad_entries
    return rep


def main() -> int:
    ap = argparse.ArgumentParser(description="Locate the first stage that introduces NaN/Inf in the coupled pulse/collocation path.")
    ap.add_argument("geometry", help="Path to .geo file")
    ap.add_argument("--solver", default="rcs_solver.py", help="Path to rcs_solver.py")
    ap.add_argument("--geometry-io", default="geometry_io.py", help="Path to geometry_io.py")
    ap.add_argument("--freqs", default="10,11", help="Comma-separated GHz list")
    ap.add_argument("--elev", type=float, default=0.0, help="Elevation/incidence angle in deg")
    ap.add_argument("--pol", default="TE", help="TE/TM or VV/HH")
    ap.add_argument("--units", default="inches", help="inches or meters")
    ap.add_argument("--mesh-ref", type=float, default=None, help="Lock mesh reference GHz")
    ap.add_argument("--out", default="stage_nan_locator_summary.json", help="Output JSON path")
    args = ap.parse_args()

    solver_path = os.path.abspath(args.solver)
    geometry_io_path = os.path.abspath(args.geometry_io)
    geo_path = os.path.abspath(args.geometry)

    geomod = load_module(geometry_io_path, "geometry_io_stage_nan")
    solver = load_module(solver_path, "rcs_solver_stage_nan")

    with open(geo_path, "r", encoding="utf-8") as f:
        text = f.read()
    title, segments, ibcs_entries, dielectric_entries = geomod.parse_geometry(text)
    snapshot = geomod.build_geometry_snapshot(title, segments, ibcs_entries, dielectric_entries)

    pol = solver._normalize_polarization(args.pol)
    unit_scale = solver._unit_scale_to_meters(args.units)
    base_dir = os.path.dirname(geo_path)
    materials = solver.MaterialLibrary.from_entries(snapshot.get("ibcs", []) or [], snapshot.get("dielectrics", []) or [], base_dir=base_dir)
    preflight = solver.validate_geometry_snapshot_for_solver(snapshot, base_dir=base_dir)

    freqs = [float(tok) for tok in str(args.freqs).replace(" ", "").split(",") if tok]
    out: dict[str, Any] = {
        "geometry": geo_path,
        "solver": solver_path,
        "geometry_io": geometry_io_path,
        "freqs_ghz": freqs,
        "elev_deg": float(args.elev),
        "pol_user": args.pol,
        "pol_internal": pol,
        "units": args.units,
        "mesh_ref_ghz": args.mesh_ref,
        "preflight": preflight,
        "material_warnings": list(getattr(materials, "warnings", [])),
        "results": [],
    }

    for freq_ghz in freqs:
        freq_hz = float(freq_ghz) * 1.0e9
        k0 = 2.0 * math.pi * freq_hz / float(solver.C0)
        mesh_freq_ghz = float(args.mesh_ref) if args.mesh_ref is not None else float(freq_ghz)
        lambda_min = float(solver.C0) / (mesh_freq_ghz * 1.0e9)
        rec: dict[str, Any] = {
            "freq_ghz": float(freq_ghz),
            "mesh_freq_ghz": float(mesh_freq_ghz),
            "k0": {"real": float(np.real(k0)), "imag": float(np.imag(k0))},
        }
        try:
            panels = solver._build_panels(snapshot, unit_scale, lambda_min, max_panels=getattr(solver, "MAX_PANELS_DEFAULT", 20000))
            rec["panel_count"] = int(len(panels))
            rec["panel_length_min"] = float(min((p.length for p in panels), default=0.0))
            rec["panel_length_max"] = float(max((p.length for p in panels), default=0.0))

            infos = solver._build_coupled_panel_info(panels, materials, float(freq_ghz), pol, k0)
            rec["coupled_infos"] = info_report(infos)

            jc, js = solver._build_junction_trace_constraints(panels, infos=infos)
            rec["junction_stats"] = {k: (int(v) if isinstance(v, (int, np.integer)) else v) for k, v in dict(js).items()}
            rec["junction_constraints"] = arr_report(jc, "junction_constraints")

            region_ops = solver._build_coupled_region_operators(panels, infos)
            ro_rep: dict[str, Any] = {}
            for region_flag, pair in region_ops.items():
                s_mat, k_mat = pair
                ro_rep[str(region_flag)] = {
                    "S": arr_report(s_mat, f"region_{region_flag}_S"),
                    "K": arr_report(k_mat, f"region_{region_flag}_K"),
                }
            rec["region_ops"] = ro_rep

            a_core = solver._build_coupled_matrix(panels=panels, infos=infos, region_ops=region_ops, pol=pol)
            rec["a_core"] = arr_report(a_core, "a_core")

            n_panels = len(panels)
            rhs_pad_count = 0
            if np.asarray(jc).size > 0:
                a_mat, rhs_seed = solver._augment_system_with_constraints(
                    a_core,
                    np.zeros(2 * n_panels, dtype=np.complex128),
                    jc,
                )
                rhs_pad_count = int(max(0, rhs_seed.shape[0] - (2 * n_panels)))
            else:
                a_mat = a_core
            rec["a_mat"] = arr_report(a_mat, "a_mat")

            centers = np.array([p.center for p in panels], dtype=float)
            u_inc_air = solver._incident_plane_wave_many(centers, k0, np.asarray([float(args.elev)], dtype=float))
            rhs_mat = solver._build_coupled_rhs_many(infos=infos, u_inc_air=u_inc_air)
            if rhs_pad_count > 0:
                rhs_mat = np.vstack([rhs_mat, np.zeros((rhs_pad_count, rhs_mat.shape[1]), dtype=np.complex128)])
            rec["rhs_mat"] = arr_report(rhs_mat, "rhs_mat")

            try:
                solver._ensure_finite_linear_system(a_mat, rhs_mat, label="stage_nan_locator")
                rec["ensure_finite_linear_system"] = "pass"
            except Exception as exc:
                rec["ensure_finite_linear_system"] = f"FAIL: {exc}"

            # Add panel mapping for first bad A entry.
            for mat_name in ("a_core", "a_mat"):
                first = rec.get(mat_name, {}).get("first_bad_index")
                if first and len(first) >= 2:
                    row, col = int(first[0]), int(first[1])
                    mapping: dict[str, Any] = {"row": row, "col": col}
                    if row < 2 * n_panels:
                        mapping["row_panel"] = int(row % n_panels)
                        mapping["row_block"] = "u_trace" if row < n_panels else "q_minus"
                    else:
                        mapping["row_block"] = "constraint"
                    if col < 2 * n_panels:
                        mapping["col_panel"] = int(col % n_panels)
                        mapping["col_block"] = "u_trace" if col < n_panels else "q_minus"
                    else:
                        mapping["col_block"] = "constraint"
                    rec[mat_name]["panel_mapping"] = mapping

        except Exception as exc:
            rec["exception"] = f"{type(exc).__name__}: {exc}"
        out["results"].append(rec)

    out_path = os.path.abspath(args.out)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))
    print(f"\nWrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
