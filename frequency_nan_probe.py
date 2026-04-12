#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np


def _load_module(module_name: str, path: str):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module '{module_name}' from {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


def _first_bad_index(arr: np.ndarray):
    bad = np.argwhere(~np.isfinite(arr))
    if bad.size == 0:
        return None
    return tuple(int(v) for v in bad[0])


def _panel_summary(panel: Any) -> Dict[str, Any]:
    return {
        "name": str(getattr(panel, "name", "")),
        "seg_type": int(getattr(panel, "seg_type", -1)),
        "ibc_flag": int(getattr(panel, "ibc_flag", 0)),
        "ipn1": int(getattr(panel, "ipn1", 0)),
        "ipn2": int(getattr(panel, "ipn2", 0)),
        "length_m": float(getattr(panel, "length", float("nan"))),
        "center_m": [float(v) for v in np.asarray(getattr(panel, "center", [np.nan, np.nan]), dtype=float)],
        "normal": [float(v) for v in np.asarray(getattr(panel, "normal", [np.nan, np.nan]), dtype=float)],
    }


def _complex_dict(z: complex) -> Dict[str, float]:
    return {"real": float(np.real(z)), "imag": float(np.imag(z)), "abs": float(abs(z))}


def main() -> int:
    ap = argparse.ArgumentParser(description="Probe frequency-dependent NaN/Inf in coupled 2D RCS assembly.")
    ap.add_argument("geometry", help="Path to .geo file")
    ap.add_argument("--solver", default="rcs_solver.py", help="Path to rcs_solver.py")
    ap.add_argument("--geometry-io", default="geometry_io.py", help="Path to geometry_io.py")
    ap.add_argument("--freqs", default="10,11,12", help="Comma-separated GHz values to probe")
    ap.add_argument("--elev", type=float, default=0.0, help="Elevation angle in deg (for RHS sanity)")
    ap.add_argument("--pol", default="TE", help="User polarization label (TE/TM or VV/HH)")
    ap.add_argument("--units", default="inches", help="Geometry units")
    ap.add_argument("--mesh-ref", type=float, default=None, help="Fixed mesh reference GHz")
    ap.add_argument("--max-panels", type=int, default=20000)
    ap.add_argument("--out", default="freq_nan_probe_summary.json")
    args = ap.parse_args()

    geo_path = os.path.abspath(args.geometry)
    solver_path = os.path.abspath(args.solver)
    gio_path = os.path.abspath(args.geometry_io)
    if not os.path.isfile(geo_path):
        raise FileNotFoundError(geo_path)
    if not os.path.isfile(solver_path):
        raise FileNotFoundError(solver_path)
    if not os.path.isfile(gio_path):
        raise FileNotFoundError(gio_path)

    gio = _load_module("geometry_io_probe", gio_path)
    solver = _load_module("rcs_solver_probe", solver_path)

    with open(geo_path, "r", encoding="utf-8") as f:
        text = f.read()
    title, segments, ibcs_entries, dielectric_entries = gio.parse_geometry(text)
    snapshot = gio.build_geometry_snapshot(title, segments, ibcs_entries, dielectric_entries)
    base_dir = str(Path(geo_path).resolve().parent)

    freqs = [float(tok.strip()) for tok in str(args.freqs).split(",") if tok.strip()]
    if not freqs:
        raise ValueError("No frequencies provided.")

    summary: Dict[str, Any] = {
        "geometry": geo_path,
        "solver": solver_path,
        "geometry_io": gio_path,
        "title": title,
        "dielectrics": dielectric_entries,
        "ibcs": ibcs_entries,
        "complex_hankel_backend": getattr(solver, "_complex_hankel_backend_name")(),
        "real_bessel_backend": getattr(getattr(solver, "_BESSEL", None), "backend_name", "unknown"),
        "scipy_special_available": bool(getattr(solver, "_SCIPY_SPECIAL", None) is not None),
        "mpmath_available": bool(getattr(solver, "_MPMATH", None) is not None),
        "frequencies": [],
    }

    try:
        preflight = solver.validate_geometry_snapshot_for_solver(snapshot, base_dir=base_dir)
    except Exception as exc:
        summary["preflight_error"] = str(exc)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(f"Preflight failed: {exc}")
        print(f"Wrote {args.out}")
        return 1
    summary["preflight"] = preflight

    unit_scale = solver._unit_scale_to_meters(args.units)
    pol_internal = solver._normalize_polarization(args.pol)

    for freq_ghz in freqs:
        entry: Dict[str, Any] = {"frequency_ghz": freq_ghz}
        summary["frequencies"].append(entry)
        try:
            mesh_freq = float(args.mesh_ref) if args.mesh_ref is not None else float(freq_ghz)
            lambda_min = solver.C0 / (mesh_freq * 1.0e9)
            panels = solver._build_panels(
                geometry_snapshot=snapshot,
                meters_scale=unit_scale,
                min_wavelength=lambda_min,
                max_panels=int(args.max_panels),
            )
            materials = solver.MaterialLibrary.from_entries(
                snapshot.get("ibcs", []) or [],
                snapshot.get("dielectrics", []) or [],
                base_dir=base_dir,
            )
            k0 = 2.0 * math.pi * (freq_ghz * 1.0e9) / solver.C0
            infos = solver._build_coupled_panel_info(panels, materials, freq_ghz, pol_internal, k0)
            junc, jstats = solver._build_junction_trace_constraints(panels, infos)
            entry["panel_count"] = len(panels)
            entry["junction_stats"] = jstats
            entry["warnings"] = list(materials.warnings)
            entry["mesh_reference_ghz"] = mesh_freq
            entry["k0"] = float(k0)

            region_to_k: Dict[int, complex] = {}
            for info in infos:
                if int(info.minus_region) >= 0:
                    region_to_k[int(info.minus_region)] = complex(info.k_minus)
                if int(info.plus_region) >= 0:
                    region_to_k[int(info.plus_region)] = complex(info.k_plus)
            entry["region_k"] = {str(region): _complex_dict(kv) for region, kv in sorted(region_to_k.items())}

            pair_switch_count = 0
            pair_total = 0
            region_reports: Dict[str, Any] = {}
            for region, k_region in sorted(region_to_k.items()):
                # estimate whether this region is switching from real-path to complex-path at this frequency
                switch_examples: List[Dict[str, Any]] = []
                for i, p_i in enumerate(panels[: min(200, len(panels))]):
                    for j, p_j in enumerate(panels[: min(200, len(panels))]):
                        if i == j:
                            continue
                        r = float(np.linalg.norm(np.asarray(p_i.center) - np.asarray(p_j.center)))
                        z = complex(k_region) * max(r, solver.EPS)
                        pair_total += 1
                        if abs(z.imag) > 1e-14:
                            pair_switch_count += 1
                            if len(switch_examples) < 3:
                                switch_examples.append({
                                    "i": i,
                                    "j": j,
                                    "r_m": r,
                                    "z": _complex_dict(z),
                                })
                reg_rep: Dict[str, Any] = {
                    "k": _complex_dict(k_region),
                    "pairs_checked": pair_total,
                    "pairs_with_abs_imag_kr_gt_1e14": pair_switch_count,
                    "switch_examples": switch_examples,
                }
                try:
                    s_mat, k_mat = solver._build_operator_matrices_coupled(panels, k_region if abs(k_region) > solver.EPS else (solver.EPS + 0.0j))
                    reg_rep["s_finite"] = bool(np.all(np.isfinite(s_mat)))
                    reg_rep["k_finite"] = bool(np.all(np.isfinite(k_mat)))
                    if not reg_rep["s_finite"]:
                        idx = _first_bad_index(s_mat)
                        reg_rep["s_first_bad"] = idx
                        if idx is not None:
                            ii, jj = idx
                            reg_rep["s_bad_panels"] = {"obs": _panel_summary(panels[ii]), "src": _panel_summary(panels[jj])}
                    if not reg_rep["k_finite"]:
                        idx = _first_bad_index(k_mat)
                        reg_rep["k_first_bad"] = idx
                        if idx is not None:
                            ii, jj = idx
                            reg_rep["k_bad_panels"] = {"obs": _panel_summary(panels[ii]), "src": _panel_summary(panels[jj])}
                    # Diagnostic: strip tiny imaginary part from k and retry.
                    k_real = complex(float(np.real(k_region)), 0.0)
                    s_real, k_real_mat = solver._build_operator_matrices_coupled(panels, k_real if abs(k_real) > solver.EPS else (solver.EPS + 0.0j))
                    reg_rep["real_k_retry"] = {
                        "k_used": _complex_dict(k_real),
                        "s_finite": bool(np.all(np.isfinite(s_real))),
                        "k_finite": bool(np.all(np.isfinite(k_real_mat))),
                    }
                except Exception as exc:
                    reg_rep["operator_error"] = str(exc)
                region_reports[str(region)] = reg_rep
            entry["region_reports"] = region_reports

            # Full coupled matrix
            try:
                region_ops = solver._build_coupled_region_operators(panels, infos)
                a_core = solver._build_coupled_matrix(panels=panels, infos=infos, region_ops=region_ops, pol=pol_internal)
                rhs = solver._build_coupled_rhs_many(
                    infos=infos,
                    u_inc_air=solver._incident_plane_wave_many(
                        np.array([p.center for p in panels], dtype=float),
                        k0,
                        np.asarray([float(args.elev)], dtype=float),
                    ),
                )
                a_mat = a_core
                rhs_pad_count = 0
                if junc.size > 0:
                    a_mat, rhs_seed = solver._augment_system_with_constraints(
                        a_core,
                        np.zeros(2 * len(panels), dtype=np.complex128),
                        junc,
                    )
                    rhs_pad_count = int(max(0, rhs_seed.shape[0] - (2 * len(panels))))
                if rhs_pad_count > 0:
                    rhs = np.vstack([rhs, np.zeros((rhs_pad_count, rhs.shape[1]), dtype=np.complex128)])
                entry["a_core_finite"] = bool(np.all(np.isfinite(a_core)))
                entry["a_mat_finite"] = bool(np.all(np.isfinite(a_mat)))
                entry["rhs_finite"] = bool(np.all(np.isfinite(rhs)))
                if not entry["a_core_finite"]:
                    idx = _first_bad_index(a_core)
                    entry["a_core_first_bad"] = idx
                    if idx is not None:
                        row, col = idx
                        n = len(panels)
                        prow = row % n
                        pcol = col % n
                        entry["a_core_bad_location"] = {
                            "row_unknown_block": "u_trace" if row < n else "q_minus",
                            "col_unknown_block": "u_trace" if col < n else "q_minus",
                            "row_panel": _panel_summary(panels[prow]),
                            "col_panel": _panel_summary(panels[pcol]),
                        }
                if not entry["a_mat_finite"]:
                    idx = _first_bad_index(a_mat)
                    entry["a_mat_first_bad"] = idx
                if not entry["rhs_finite"]:
                    entry["rhs_first_bad"] = _first_bad_index(rhs)
            except Exception as exc:
                entry["matrix_error"] = str(exc)
        except Exception as exc:
            entry["setup_error"] = str(exc)

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))
    print(f"\nWrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
