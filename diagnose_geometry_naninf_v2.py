#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import sys
import traceback
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Tuple

import numpy as np


REPO_CANDIDATE_SOLVER = [
    "rcs_solver.py",
    "rcs_solver(32).py",
    "rcs_solver(31).py",
]
REPO_CANDIDATE_GEOM = [
    "geometry_io.py",
    "geometry_io(11).py",
    "geometry_io(10).py",
]


def _load_module(module_name: str, path: str):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module '{module_name}' from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _find_candidate(explicit: str | None, candidates: list[str], search_dir: str) -> str:
    if explicit:
        p = os.path.abspath(explicit)
        if not os.path.isfile(p):
            raise FileNotFoundError(f"File not found: {p}")
        return p
    for name in candidates:
        p = os.path.join(search_dir, name)
        if os.path.isfile(p):
            return os.path.abspath(p)
    raise FileNotFoundError(
        f"Could not find any of {candidates} in {search_dir}. Provide an explicit path."
    )


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    if isinstance(obj, complex):
        return {"real": float(obj.real), "imag": float(obj.imag)}
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_jsonable(v) for v in obj]
    return repr(obj)


def _finite_scalar(value: Any) -> bool:
    if isinstance(value, (bool, str)) or value is None:
        return True
    if isinstance(value, complex):
        return bool(np.isfinite(value.real) and np.isfinite(value.imag))
    if isinstance(value, (int, float, np.generic)):
        try:
            return bool(np.isfinite(value))
        except Exception:
            return True
    return True


def _array_nonfinite_report(
    arr: Any,
    label: str,
    index_mapper: Callable[[int], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    out: dict[str, Any] = {
        "label": label,
        "ok": True,
    }
    try:
        a = np.asarray(arr)
    except Exception as exc:
        out.update({"ok": False, "error": f"could not convert to ndarray: {exc}"})
        return out

    out["shape"] = list(a.shape)
    out["dtype"] = str(a.dtype)
    finite_mask = np.isfinite(a)
    if np.all(finite_mask):
        return out

    bad = np.argwhere(~finite_mask)
    first = tuple(int(v) for v in bad[0]) if bad.size else None
    out["ok"] = False
    out["nonfinite_count"] = int(np.size(finite_mask) - np.count_nonzero(finite_mask))
    out["first_index"] = list(first) if first is not None else None
    try:
        first_val = a[first] if first is not None else None
        out["first_value"] = _jsonable(first_val)
    except Exception:
        out["first_value"] = None

    if first is not None and index_mapper is not None:
        if a.ndim >= 1:
            try:
                out["row_detail"] = _jsonable(index_mapper(int(first[0])))
            except Exception as exc:
                out["row_detail_error"] = repr(exc)
        if a.ndim >= 2:
            try:
                out["col_detail"] = _jsonable(index_mapper(int(first[1])))
            except Exception as exc:
                out["col_detail_error"] = repr(exc)
    return out


def _excerpt_around(a: np.ndarray, index: tuple[int, ...], radius: int = 2) -> dict[str, Any]:
    if a.ndim != 2:
        return {}
    i, j = (int(index[0]), int(index[1]))
    i0 = max(0, i - radius)
    i1 = min(a.shape[0], i + radius + 1)
    j0 = max(0, j - radius)
    j1 = min(a.shape[1], j + radius + 1)
    sub = a[i0:i1, j0:j1]
    return {
        "row_range": [i0, i1],
        "col_range": [j0, j1],
        "values": _jsonable(sub),
    }


def _panel_unknown_mapper(panels: list[Any]) -> Callable[[int], dict[str, Any]]:
    n_panels = len(panels)

    def mapper(idx: int) -> dict[str, Any]:
        if idx < 0:
            return {"index": idx, "kind": "invalid"}
        if idx < n_panels:
            p = panels[idx]
            return {
                "index": idx,
                "unknown": "u_trace",
                "panel_index": idx,
                "panel_name": getattr(p, "name", None),
                "seg_type": int(getattr(p, "seg_type", 0)),
                "ibc_flag": int(getattr(p, "ibc_flag", 0)),
                "ipn1": int(getattr(p, "ipn1", 0)),
                "ipn2": int(getattr(p, "ipn2", 0)),
                "center_m": _jsonable(np.asarray(getattr(p, "center", [np.nan, np.nan]), dtype=float)),
            }
        q_idx = idx - n_panels
        if q_idx < n_panels:
            p = panels[q_idx]
            return {
                "index": idx,
                "unknown": "q_minus",
                "panel_index": q_idx,
                "panel_name": getattr(p, "name", None),
                "seg_type": int(getattr(p, "seg_type", 0)),
                "ibc_flag": int(getattr(p, "ibc_flag", 0)),
                "ipn1": int(getattr(p, "ipn1", 0)),
                "ipn2": int(getattr(p, "ipn2", 0)),
                "center_m": _jsonable(np.asarray(getattr(p, "center", [np.nan, np.nan]), dtype=float)),
            }
        return {
            "index": idx,
            "unknown": "constraint_row_or_col",
            "constraint_offset": idx - (2 * n_panels),
        }

    return mapper


def _linear_unknown_mapper(mesh: Any) -> Callable[[int], dict[str, Any]]:
    nnodes = len(mesh.nodes)

    def mapper(idx: int) -> dict[str, Any]:
        if idx < 0:
            return {"index": idx, "kind": "invalid"}
        if idx < nnodes:
            node = mesh.nodes[idx]
            return {
                "index": idx,
                "unknown": "u_trace_node",
                "node_index": idx,
                "xy_m": _jsonable(np.asarray(getattr(node, "xy", [np.nan, np.nan]), dtype=float)),
                "snap_key": _jsonable(getattr(node, "key", None)),
            }
        q_idx = idx - nnodes
        if q_idx < nnodes:
            node = mesh.nodes[q_idx]
            return {
                "index": idx,
                "unknown": "q_minus_node",
                "node_index": q_idx,
                "xy_m": _jsonable(np.asarray(getattr(node, "xy", [np.nan, np.nan]), dtype=float)),
                "snap_key": _jsonable(getattr(node, "key", None)),
            }
        return {
            "index": idx,
            "unknown": "constraint_row_or_col",
            "constraint_offset": idx - (2 * nnodes),
        }

    return mapper


def _inspect_coupled_infos(infos: list[Any], label: str) -> dict[str, Any]:
    bad: list[dict[str, Any]] = []
    for idx, info in enumerate(infos):
        record = vars(info)
        for key, value in record.items():
            if not _finite_scalar(value):
                bad.append(
                    {
                        "info_index": idx,
                        "field": key,
                        "value": _jsonable(value),
                        "seg_type": _jsonable(record.get("seg_type")),
                        "plus_region": _jsonable(record.get("plus_region")),
                        "minus_region": _jsonable(record.get("minus_region")),
                    }
                )
    return {
        "label": label,
        "ok": len(bad) == 0,
        "count": len(infos),
        "bad_fields": bad[:50],
        "bad_field_count": len(bad),
    }


def _summarize_panels(panels: list[Any]) -> dict[str, Any]:
    seg_types: dict[int, int] = {}
    by_name: dict[str, int] = {}
    min_len = math.inf
    max_len = 0.0
    for p in panels:
        st = int(getattr(p, "seg_type", 0))
        seg_types[st] = seg_types.get(st, 0) + 1
        nm = str(getattr(p, "name", ""))
        by_name[nm] = by_name.get(nm, 0) + 1
        length = float(getattr(p, "length", 0.0))
        min_len = min(min_len, length)
        max_len = max(max_len, length)
    return {
        "panel_count": len(panels),
        "seg_type_counts": seg_types,
        "segment_name_panel_counts": by_name,
        "panel_length_min_m": 0.0 if not panels else float(min_len),
        "panel_length_max_m": float(max_len),
    }


def _diagnose_pulse(
    solver: Any,
    panels: list[Any],
    materials: Any,
    freq_ghz: float,
    pol_internal: str,
    k0: float,
    elev_deg: float,
) -> dict[str, Any]:
    report: dict[str, Any] = {"method": "pulse/collocation"}
    panel_mapper = _panel_unknown_mapper(panels)
    panel_centers = np.asarray([p.center for p in panels], dtype=float)
    elevations_arr = np.asarray([float(elev_deg)], dtype=float)
    n_panels = len(panels)

    infos = solver._build_coupled_panel_info(panels, materials, freq_ghz, pol_internal, k0)
    report["coupled_infos"] = _inspect_coupled_infos(infos, "pulse_infos")

    junction_constraints, junction_stats = solver._build_junction_trace_constraints(panels, infos=infos)
    report["junction_stats"] = _jsonable(junction_stats)
    report["junction_constraints"] = _array_nonfinite_report(
        junction_constraints, "junction_constraints"
    )

    region_ops = solver._build_coupled_region_operators(panels, infos)
    reg_summary: dict[str, Any] = {}
    for region, ops in region_ops.items():
        s_mat, kp_mat = ops
        reg_summary[str(region)] = {
            "s_mat": _array_nonfinite_report(s_mat, f"region_{region}_S", panel_mapper),
            "kp_mat": _array_nonfinite_report(kp_mat, f"region_{region}_Kp", panel_mapper),
        }
    report["region_ops"] = reg_summary

    a_core = solver._build_coupled_matrix(panels=panels, infos=infos, region_ops=region_ops, pol=pol_internal)
    report["a_core"] = _array_nonfinite_report(a_core, "pulse_a_core", panel_mapper)
    if not report["a_core"]["ok"] and report["a_core"].get("first_index"):
        idx = tuple(report["a_core"]["first_index"])
        report["a_core_excerpt"] = _excerpt_around(np.asarray(a_core), idx)

    rhs_pad_count = 0
    if np.asarray(junction_constraints).size > 0:
        a_mat, rhs_seed = solver._augment_system_with_constraints(
            a_core,
            np.zeros(2 * n_panels, dtype=np.complex128),
            junction_constraints,
        )
        rhs_pad_count = int(max(0, rhs_seed.shape[0] - (2 * n_panels)))
    else:
        a_mat = a_core

    rhs_mat = solver._build_coupled_rhs_many(
        infos=infos,
        u_inc_air=solver._incident_plane_wave_many(panel_centers, k0, elevations_arr),
    )
    if rhs_pad_count > 0:
        rhs_mat = np.vstack([rhs_mat, np.zeros((rhs_pad_count, rhs_mat.shape[1]), dtype=np.complex128)])

    report["a_mat"] = _array_nonfinite_report(a_mat, "pulse_a_mat", panel_mapper)
    if not report["a_mat"]["ok"] and report["a_mat"].get("first_index"):
        idx = tuple(report["a_mat"]["first_index"])
        report["a_mat_excerpt"] = _excerpt_around(np.asarray(a_mat), idx)
    report["rhs_mat"] = _array_nonfinite_report(rhs_mat, "pulse_rhs_mat")

    try:
        solver._ensure_finite_linear_system(a_mat, rhs_mat, label="diagnostic pulse/collocation system")
        report["finite_guard"] = {"ok": True}
    except Exception as exc:
        report["finite_guard"] = {"ok": False, "error": str(exc), "traceback": traceback.format_exc()}
        return report

    try:
        prepared = solver._prepare_linear_solver(a_mat)
        sol_mat = solver._solve_with_prepared_solver(prepared, rhs_mat)
        if np.asarray(sol_mat).ndim == 1:
            sol_mat = np.asarray(sol_mat).reshape(-1, 1)
        residual = solver._residual_norm_many(a_mat, sol_mat, rhs_mat)
        report["solve"] = {
            "ok": True,
            "solution": _array_nonfinite_report(sol_mat, "pulse_solution"),
            "residual": _jsonable(np.asarray(residual)),
        }
    except Exception as exc:
        report["solve"] = {"ok": False, "error": str(exc), "traceback": traceback.format_exc()}
    return report


def _diagnose_galerkin(
    solver: Any,
    panels: list[Any],
    materials: Any,
    freq_ghz: float,
    pol_internal: str,
    k0: float,
    elev_deg: float,
) -> dict[str, Any]:
    report: dict[str, Any] = {"method": "linear/galerkin"}
    base_infos = solver._build_coupled_panel_info(panels, materials, freq_ghz, pol_internal, k0)
    report["preview_panel_infos"] = _inspect_coupled_infos(base_infos, "galerkin_preview_infos")

    mesh, mesh_stats = solver._build_linear_mesh_interface_aware(panels, base_infos)
    report["mesh_stats"] = _jsonable(mesh_stats)
    report["mesh_node_count"] = int(len(mesh.nodes))
    report["mesh_element_count"] = int(len(mesh.elements))

    infos = solver._build_linear_coupled_infos(mesh, materials, freq_ghz, pol_internal, k0)
    report["coupled_infos"] = _inspect_coupled_infos(infos, "galerkin_infos")
    node_report = solver._linear_coupled_node_report(mesh, infos)
    report["node_report"] = _jsonable(node_report)

    linear_junction_constraints, linear_junction_stats = solver._build_linear_junction_constraints(mesh, infos)
    report["junction_stats"] = _jsonable(linear_junction_stats)
    report["junction_constraints"] = _array_nonfinite_report(
        linear_junction_constraints, "linear_junction_constraints"
    )

    node_mapper = _linear_unknown_mapper(mesh)
    a_core = solver._build_coupled_matrix_linear(mesh=mesh, infos=infos, pol=pol_internal)
    report["a_core"] = _array_nonfinite_report(a_core, "galerkin_a_core", node_mapper)
    if not report["a_core"]["ok"] and report["a_core"].get("first_index"):
        idx = tuple(report["a_core"]["first_index"])
        report["a_core_excerpt"] = _excerpt_around(np.asarray(a_core), idx)

    rhs_pad_count = 0
    if np.asarray(linear_junction_constraints).size > 0:
        a_mat, rhs_seed = solver._augment_system_with_constraints(
            a_core,
            np.zeros(a_core.shape[0], dtype=np.complex128),
            linear_junction_constraints,
        )
        rhs_pad_count = int(max(0, rhs_seed.shape[0] - a_core.shape[0]))
    else:
        a_mat = a_core

    elevations_arr = np.asarray([float(elev_deg)], dtype=float)
    rhs_mat = solver._build_coupled_rhs_many_linear(
        mesh=mesh,
        infos=infos,
        k_air=k0,
        elevations_deg=elevations_arr,
    )
    if rhs_pad_count > 0:
        rhs_mat = np.vstack([rhs_mat, np.zeros((rhs_pad_count, rhs_mat.shape[1]), dtype=np.complex128)])

    report["a_mat"] = _array_nonfinite_report(a_mat, "galerkin_a_mat", node_mapper)
    if not report["a_mat"]["ok"] and report["a_mat"].get("first_index"):
        idx = tuple(report["a_mat"]["first_index"])
        report["a_mat_excerpt"] = _excerpt_around(np.asarray(a_mat), idx)
    report["rhs_mat"] = _array_nonfinite_report(rhs_mat, "galerkin_rhs_mat")

    try:
        solver._ensure_finite_linear_system(a_mat, rhs_mat, label="diagnostic linear/Galerkin system")
        report["finite_guard"] = {"ok": True}
    except Exception as exc:
        report["finite_guard"] = {"ok": False, "error": str(exc), "traceback": traceback.format_exc()}
        return report

    try:
        prepared = solver._prepare_linear_solver(a_mat)
        sol_mat = solver._solve_with_prepared_solver(prepared, rhs_mat)
        if np.asarray(sol_mat).ndim == 1:
            sol_mat = np.asarray(sol_mat).reshape(-1, 1)
        residual = solver._residual_norm_many(a_mat, sol_mat, rhs_mat)
        report["solve"] = {
            "ok": True,
            "solution": _array_nonfinite_report(sol_mat, "galerkin_solution"),
            "residual": _jsonable(np.asarray(residual)),
        }
    except Exception as exc:
        report["solve"] = {"ok": False, "error": str(exc), "traceback": traceback.format_exc()}
    return report


def _full_solver_try(
    solver: Any,
    snapshot: dict[str, Any],
    base_dir: str,
    freq_ghz: float,
    elev_deg: float,
    pol_user: str,
    units: str,
    basis_family: str,
    testing_family: str,
) -> dict[str, Any]:
    try:
        result = solver.solve_monostatic_rcs_2d(
            geometry_snapshot=snapshot,
            frequencies_ghz=[float(freq_ghz)],
            elevations_deg=[float(elev_deg)],
            polarization=pol_user,
            geometry_units=units,
            material_base_dir=base_dir,
            basis_family=basis_family,
            testing_family=testing_family,
            strict_quality_gate=False,
            compute_condition_number=False,
            parallel_elevations=False,
        )
        metadata = result.get("metadata", {}) or {}
        return {"ok": True, "metadata": _jsonable(metadata)}
    except Exception as exc:
        return {"ok": False, "error": str(exc), "traceback": traceback.format_exc()}


def main() -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Diagnose why a geometry produces 'system matrix contains NaN/Inf' in the current 2D RCS solver. "
            "This runs solver-side preflight, coupled panel/material inspection, assembly-stage finite checks, "
            "and optional full-solver confirmation for pulse and/or Galerkin."
        )
    )
    ap.add_argument("geometry", help="Path to the .geo file to diagnose")
    ap.add_argument("--freq", type=float, required=True, help="Frequency in GHz")
    ap.add_argument("--elev", type=float, default=0.0, help="Elevation / incident angle in degrees")
    ap.add_argument("--pol", default="TE", help="User polarization label: TE/TM or VV/HH")
    ap.add_argument("--units", default="inches", choices=["inches", "meters"], help="Geometry units")
    ap.add_argument("--method", default="both", choices=["pulse", "galerkin", "both"], help="Which discretization path to inspect")
    ap.add_argument("--solver", default=None, help="Optional explicit path to rcs_solver.py")
    ap.add_argument("--geometry-io", dest="geometry_io", default=None, help="Optional explicit path to geometry_io.py")
    ap.add_argument("--outdir", default="diag_naninf_out", help="Directory to store JSON output")
    args = ap.parse_args()

    geo_path = os.path.abspath(args.geometry)
    if not os.path.isfile(geo_path):
        raise FileNotFoundError(f"Geometry file not found: {geo_path}")
    search_dir = os.path.dirname(geo_path) or os.getcwd()

    solver_path = _find_candidate(args.solver, REPO_CANDIDATE_SOLVER, search_dir)
    geom_path = _find_candidate(args.geometry_io, REPO_CANDIDATE_GEOM, search_dir)

    outdir = os.path.abspath(args.outdir)
    os.makedirs(outdir, exist_ok=True)

    print(f"[diag] geometry: {geo_path}")
    print(f"[diag] solver:   {solver_path}")
    print(f"[diag] geom io:  {geom_path}")
    print(f"[diag] freq:     {args.freq} GHz")
    print(f"[diag] elev:     {args.elev} deg")
    print(f"[diag] pol:      {args.pol}")
    print(f"[diag] method:   {args.method}")

    geometry_io = _load_module("diag_geometry_io", geom_path)
    solver = _load_module("diag_rcs_solver", solver_path)

    with open(geo_path, "r", encoding="utf-8") as f:
        text = f.read()
    title, segments, ibcs_entries, dielectric_entries = geometry_io.parse_geometry(text)
    snapshot = geometry_io.build_geometry_snapshot(title, segments, ibcs_entries, dielectric_entries)
    base_dir = os.path.dirname(geo_path)

    report: dict[str, Any] = {
        "inputs": {
            "geometry": geo_path,
            "solver": solver_path,
            "geometry_io": geom_path,
            "freq_ghz": float(args.freq),
            "elev_deg": float(args.elev),
            "pol_user": args.pol,
            "units": args.units,
            "method": args.method,
        },
        "geometry": {
            "title": title,
            "segment_count": len(segments),
            "ibc_row_count": len(ibcs_entries),
            "dielectric_row_count": len(dielectric_entries),
            "segment_names": [getattr(s, "name", "") for s in segments],
        },
    }

    try:
        preflight = solver.validate_geometry_snapshot_for_solver(snapshot, base_dir)
        report["preflight"] = _jsonable(preflight)
        print("[diag] preflight: ok")
        if preflight.get("warnings"):
            for w in preflight.get("warnings", []):
                print(f"  warning: {w}")
    except Exception as exc:
        report["preflight"] = {"ok": False, "error": str(exc), "traceback": traceback.format_exc()}
        print(f"[diag] preflight failed: {exc}")
        with open(os.path.join(outdir, "summary.json"), "w", encoding="utf-8") as f:
            json.dump(_jsonable(report), f, indent=2)
        return 1

    try:
        materials = solver.MaterialLibrary.from_entries(
            ibcs_entries=snapshot.get("ibcs", []) or [],
            dielectric_entries=snapshot.get("dielectrics", []) or [],
            base_dir=base_dir,
        )
        report["material_library_init"] = {"ok": True, "warnings": _jsonable(getattr(materials, "warnings", []))}
        print("[diag] material library: ok")
    except Exception as exc:
        report["material_library_init"] = {"ok": False, "error": str(exc), "traceback": traceback.format_exc()}
        print(f"[diag] material library failed: {exc}")
        with open(os.path.join(outdir, "summary.json"), "w", encoding="utf-8") as f:
            json.dump(_jsonable(report), f, indent=2)
        return 1

    unit_scale = solver._unit_scale_to_meters(args.units)
    pol_internal = solver._normalize_polarization(args.pol)
    freq_hz = float(args.freq) * 1e9
    k0 = 2.0 * math.pi * freq_hz / float(solver.C0)
    lambda_min = float(solver.C0) / freq_hz

    try:
        panels = solver._build_panels(snapshot, unit_scale, lambda_min, max_panels=int(getattr(solver, "MAX_PANELS_DEFAULT", 20000)))
        report["panels"] = _summarize_panels(panels)
        print(f"[diag] built {len(panels)} panel(s)")
    except Exception as exc:
        report["panels"] = {"ok": False, "error": str(exc), "traceback": traceback.format_exc()}
        print(f"[diag] panel build failed: {exc}")
        with open(os.path.join(outdir, "summary.json"), "w", encoding="utf-8") as f:
            json.dump(_jsonable(report), f, indent=2)
        return 1

    methods = [args.method] if args.method != "both" else ["pulse", "galerkin"]
    report["methods"] = {}
    for method in methods:
        print(f"[diag] diagnosing {method}...")
        try:
            if method == "pulse":
                method_report = _diagnose_pulse(solver, panels, materials, float(args.freq), pol_internal, k0, float(args.elev))
                method_report["full_solver_try"] = _full_solver_try(
                    solver,
                    snapshot,
                    base_dir,
                    float(args.freq),
                    float(args.elev),
                    args.pol,
                    args.units,
                    "pulse",
                    "collocation",
                )
            else:
                method_report = _diagnose_galerkin(solver, panels, materials, float(args.freq), pol_internal, k0, float(args.elev))
                method_report["full_solver_try"] = _full_solver_try(
                    solver,
                    snapshot,
                    base_dir,
                    float(args.freq),
                    float(args.elev),
                    args.pol,
                    args.units,
                    "linear",
                    "galerkin",
                )
            report["methods"][method] = _jsonable(method_report)

            finite_guard = method_report.get("finite_guard", {})
            if finite_guard.get("ok", False):
                print(f"[diag] {method}: assembled system is finite")
            else:
                print(f"[diag] {method}: non-finite system detected -> {finite_guard.get('error')}")
        except Exception as exc:
            report["methods"][method] = {"ok": False, "error": str(exc), "traceback": traceback.format_exc()}
            print(f"[diag] {method} failed before completion: {exc}")

    summary_path = os.path.join(outdir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(_jsonable(report), f, indent=2)
    print(f"[diag] wrote {summary_path}")

    # Return nonzero if any method found a non-finite system or failed.
    exit_code = 0
    for method_name, method_report in (report.get("methods") or {}).items():
        finite_guard = (method_report or {}).get("finite_guard", {})
        if (not finite_guard.get("ok", False)) or (not (method_report or {}).get("full_solver_try", {}).get("ok", True)):
            exit_code = 2
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
