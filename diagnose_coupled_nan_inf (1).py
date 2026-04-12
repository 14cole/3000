#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import traceback
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from geometry_io import build_geometry_snapshot, parse_geometry
import rcs_solver


# -----------------------------
# basic parsing helpers
# -----------------------------
def parse_flag(token: Any, default: int = 0) -> int:
    text = str(token or "").strip().lower()
    if not text:
        return default
    if text.startswith("fort."):
        text = text.split("fort.", 1)[1]
    try:
        return int(round(float(text)))
    except Exception:
        return default


def parse_float(token: Any, default: float = 0.0) -> float:
    try:
        return float(token)
    except Exception:
        return default


def ensure_prop_len(props: Sequence[str], n: int = 6) -> List[str]:
    out = list(props)
    if len(out) < n:
        out.extend([""] * (n - len(out)))
    return out


# -----------------------------
# headless geometry validation
# -----------------------------
def segment_primitives(seg) -> List[Tuple[float, float, float, float]]:
    count = min(len(seg.x), len(seg.y))
    out: List[Tuple[float, float, float, float]] = []
    for i in range(count // 2):
        idx = 2 * i
        out.append((float(seg.x[idx]), float(seg.y[idx]), float(seg.x[idx + 1]), float(seg.y[idx + 1])))
    return out


def point_key(x: float, y: float, tol: float) -> Tuple[int, int]:
    inv = 1.0 / max(tol, 1.0e-12)
    return int(round(float(x) * inv)), int(round(float(y) * inv))


def segments_intersect(
    a1: Tuple[float, float],
    a2: Tuple[float, float],
    b1: Tuple[float, float],
    b2: Tuple[float, float],
    tol: float,
) -> bool:
    ax1, ay1 = a1
    ax2, ay2 = a2
    bx1, by1 = b1
    bx2, by2 = b2

    min_ax, max_ax = min(ax1, ax2), max(ax1, ax2)
    min_ay, max_ay = min(ay1, ay2), max(ay1, ay2)
    min_bx, max_bx = min(bx1, bx2), max(bx1, bx2)
    min_by, max_by = min(by1, by2), max(by1, by2)
    if max_ax < min_bx - tol or max_bx < min_ax - tol:
        return False
    if max_ay < min_by - tol or max_by < min_ay - tol:
        return False

    def orient(px: float, py: float, qx: float, qy: float, rx: float, ry: float) -> float:
        return (qx - px) * (ry - py) - (qy - py) * (rx - px)

    def on_seg(px: float, py: float, qx: float, qy: float, rx: float, ry: float) -> bool:
        return (
            min(px, qx) - tol <= rx <= max(px, qx) + tol
            and min(py, qy) - tol <= ry <= max(py, qy) + tol
        )

    o1 = orient(ax1, ay1, ax2, ay2, bx1, by1)
    o2 = orient(ax1, ay1, ax2, ay2, bx2, by2)
    o3 = orient(bx1, by1, bx2, by2, ax1, ay1)
    o4 = orient(bx1, by1, bx2, by2, ax2, ay2)

    if (o1 > tol and o2 < -tol or o1 < -tol and o2 > tol) and (
        o3 > tol and o4 < -tol or o3 < -tol and o4 > tol
    ):
        return True

    if abs(o1) <= tol and on_seg(ax1, ay1, ax2, ay2, bx1, by1):
        return True
    if abs(o2) <= tol and on_seg(ax1, ay1, ax2, ay2, bx2, by2):
        return True
    if abs(o3) <= tol and on_seg(bx1, by1, bx2, by2, ax1, ay1):
        return True
    if abs(o4) <= tol and on_seg(bx1, by1, bx2, by2, ax2, ay2):
        return True
    return False


def validate_dielectrics(dielectric_entries: List[List[str]]) -> List[Tuple[str, str]]:
    findings: List[Tuple[str, str]] = []
    seen_flags: set[int] = set()
    for idx, row in enumerate(dielectric_entries, start=1):
        if not row:
            continue
        flag = parse_flag(row[0], 0)
        label = f"Dielectrics row {idx}"
        if flag <= 0:
            findings.append(("ERROR", f"{label}: invalid flag '{row[0] if row else ''}'."))
            continue
        if flag in seen_flags:
            findings.append(("WARN", f"{label}: duplicate dielectric flag {flag}; last definition wins."))
        seen_flags.add(flag)

        if flag <= 50:
            if len(row) < 5:
                findings.append(("ERROR", f"{label}: expected 5 columns for constant dielectric, got {len(row)}."))
                continue
            vals = [parse_float(tok, float('nan')) for tok in row[1:5]]
            names = ["eps_real", "eps_imag", "mu_real", "mu_imag"]
            for name, val in zip(names, vals):
                if not math.isfinite(val):
                    findings.append(("ERROR", f"{label}: {name} is not finite ({row})."))
            if math.isfinite(vals[0]) and abs(vals[0]) < 1e-15:
                findings.append(("WARN", f"{label}: eps_real is extremely small; solver will effectively normalize/fallback."))
            if math.isfinite(vals[2]) and abs(vals[2]) < 1e-15:
                findings.append(("WARN", f"{label}: mu_real is extremely small; solver will effectively normalize/fallback."))
        else:
            findings.append(("INFO", f"{label}: flag {flag} expects fort.{flag}."))
    return findings


def validate_ibcs(ibcs_entries: List[List[str]]) -> List[Tuple[str, str]]:
    findings: List[Tuple[str, str]] = []
    seen_flags: set[int] = set()
    for idx, row in enumerate(ibcs_entries, start=1):
        if not row:
            continue
        flag = parse_flag(row[0], 0)
        label = f"IBCS row {idx}"
        if flag <= 0:
            findings.append(("ERROR", f"{label}: invalid flag '{row[0] if row else ''}'."))
            continue
        if flag in seen_flags:
            findings.append(("WARN", f"{label}: duplicate IBC flag {flag}; last definition wins."))
        seen_flags.add(flag)
        if flag <= 50:
            vals = [parse_float(tok, float('nan')) for tok in row[1:3]]
            names = ["Z_real", "Z_imag"]
            for name, val in zip(names, vals):
                if not math.isfinite(val):
                    findings.append(("ERROR", f"{label}: {name} is not finite ({row})."))
        else:
            findings.append(("INFO", f"{label}: flag {flag} expects fort.{flag}."))
    return findings


def validate_geometry_headless(segments, ibcs_entries, dielectric_entries) -> List[Tuple[str, str]]:
    findings: List[Tuple[str, str]] = []

    tol = 1.0e-8
    ibc_flags = {parse_flag(row[0], 0) for row in ibcs_entries if row}
    diel_flags = {parse_flag(row[0], 0) for row in dielectric_entries if row}

    findings.extend(validate_ibcs(ibcs_entries))
    findings.extend(validate_dielectrics(dielectric_entries))

    for row, seg in enumerate(segments, start=1):
        props = ensure_prop_len(seg.properties, 6)
        seg_type = parse_flag(props[0], -1)
        ibc = parse_flag(props[3], 0)
        ipn1 = parse_flag(props[4], 0)
        ipn2 = parse_flag(props[5], 0)
        label = f"Row {row} ('{seg.name}')"
        primitives = segment_primitives(seg)

        if not primitives:
            findings.append(("ERROR", f"{label}: no line primitives found."))
            continue

        for i, (x1, y1, x2, y2) in enumerate(primitives, start=1):
            length = math.hypot(x2 - x1, y2 - y1)
            if length <= tol:
                findings.append(("ERROR", f"{label}: primitive {i} has near-zero length."))

        for i in range(len(primitives) - 1):
            _, _, ex, ey = primitives[i]
            nx1, ny1, nx2, ny2 = primitives[i + 1]
            d_start = math.hypot(ex - nx1, ey - ny1)
            d_end = math.hypot(ex - nx2, ey - ny2)
            if d_start > tol:
                if d_end <= tol:
                    findings.append(("WARN", f"{label}: primitive {i + 2} appears reversed relative to previous one."))
                else:
                    findings.append(("WARN", f"{label}: primitive {i + 1} and {i + 2} are not connected."))

        sx, sy, _, _ = primitives[0]
        _, _, ex, ey = primitives[-1]
        closed = math.hypot(sx - ex, sy - ey) <= tol
        if closed:
            points = [(sx, sy)] + [(x2, y2) for _, _, x2, y2 in primitives]
            area2 = 0.0
            for i in range(len(points) - 1):
                x0, y0 = points[i]
                x1, y1 = points[i + 1]
                area2 += x0 * y1 - x1 * y0
            orient = "CCW" if area2 > 0 else "CW"
            findings.append(("INFO", f"{label}: closed chain, orientation {orient}."))
            if seg_type in {2, 3, 4, 5} and area2 > 0:
                findings.append(("WARN", f"{label}: CCW orientation may mean inward-pointing normals; verify direction."))
        else:
            findings.append(("WARN", f"{label}: open chain (start/end do not close)."))
            if seg_type in {2, 3, 4, 5}:
                findings.append(("WARN", f"{label}: type {seg_type} is open; coupled junctions are sensitive to this."))

        if ibc > 0 and ibc not in ibc_flags:
            findings.append(("ERROR", f"{label}: IBC flag {ibc} is referenced but not defined in IBCS."))
        if seg_type in {3, 4, 5} and ipn1 <= 0:
            findings.append(("ERROR", f"{label}: TYPE {seg_type} requires IPN1 > 0."))
        if ipn1 > 0 and ipn1 not in diel_flags:
            findings.append(("ERROR", f"{label}: dielectric flag IPN1={ipn1} is referenced but not defined."))
        if seg_type == 5 and ipn2 <= 0:
            findings.append(("ERROR", f"{label}: TYPE 5 requires IPN2 > 0."))
        if ipn2 > 0 and ipn2 not in diel_flags:
            findings.append(("ERROR", f"{label}: dielectric flag IPN2={ipn2} is referenced but not defined."))
        if seg_type in {1, 2, 3, 4} and ipn2 != 0:
            findings.append(("WARN", f"{label}: TYPE {seg_type} typically uses IPN2=0."))

    global_primitives: List[Tuple[int, int, Tuple[float, float, float, float], str, int]] = []
    row_type: Dict[int, int] = {}
    for row0, seg in enumerate(segments):
        props = ensure_prop_len(seg.properties, 6)
        seg_type = parse_flag(props[0], -1)
        row_type[row0] = seg_type
        for pidx, prim in enumerate(segment_primitives(seg)):
            global_primitives.append((row0, pidx, prim, seg.name, seg_type))

    endpoint_hits: Dict[Tuple[int, int], List[Tuple[int, int, int]]] = {}
    for row0, pidx, (x1, y1, x2, y2), _name, _stype in global_primitives:
        k1 = point_key(x1, y1, tol)
        k2 = point_key(x2, y2, tol)
        endpoint_hits.setdefault(k1, []).append((row0, pidx, 0))
        endpoint_hits.setdefault(k2, []).append((row0, pidx, 1))

    for _key, hits in endpoint_hits.items():
        incident_rows = sorted({h[0] for h in hits})
        if len(hits) == 1:
            row0 = hits[0][0]
            if row_type.get(row0, -1) in {2, 3, 4, 5}:
                findings.append(("WARN", f"Row {row0 + 1}: dangling endpoint not connected to any other primitive."))
        if len(hits) > 6:
            findings.append(("WARN", f"Row {incident_rows[0] + 1}: high-degree node with {len(hits)} incident primitive endpoints (possible non-manifold junction)."))

    max_intersections = 100
    found = 0
    n_prims = len(global_primitives)
    for i in range(n_prims):
        row_i, pidx_i, prim_i, name_i, _ = global_primitives[i]
        x1, y1, x2, y2 = prim_i
        k_i0 = point_key(x1, y1, tol)
        k_i1 = point_key(x2, y2, tol)
        for j in range(i + 1, n_prims):
            row_j, pidx_j, prim_j, name_j, _ = global_primitives[j]
            u1, v1, u2, v2 = prim_j
            k_j0 = point_key(u1, v1, tol)
            k_j1 = point_key(u2, v2, tol)
            shared_endpoint = k_i0 in {k_j0, k_j1} or k_i1 in {k_j0, k_j1}
            if shared_endpoint:
                continue
            if row_i == row_j and abs(pidx_i - pidx_j) <= 1:
                continue
            if not segments_intersect((x1, y1), (x2, y2), (u1, v1), (u2, v2), tol):
                continue
            findings.append(("ERROR", f"Rows {row_i + 1} ('{name_i}') and {row_j + 1} ('{name_j}') have a non-endpoint primitive intersection."))
            found += 1
            if found >= max_intersections:
                findings.append(("WARN", f"Intersection reporting truncated after {max_intersections} findings."))
                break
        if found >= max_intersections:
            break

    return findings




# -----------------------------
# solver compatibility shims
# -----------------------------
def install_compatibility_shims(default_elev_deg: float) -> None:
    """
    Make the diagnostic script tolerant of small helper-signature differences
    across solver revisions.

    In particular, some builds expose _build_compiled_system(..., elev_deg)
    while some call sites omit that argument. Since this diagnostic runs a
    single elevation at a time, we can safely inject the requested test
    elevation when it is missing.
    """

    import inspect

    if hasattr(rcs_solver, "_build_compiled_system"):
        orig = rcs_solver._build_compiled_system
        try:
            sig = inspect.signature(orig)
        except Exception:
            sig = None

        def wrapped_build_compiled_system(*args, **kwargs):
            if sig is not None and "elev_deg" in sig.parameters and "elev_deg" not in kwargs:
                try:
                    bound = sig.bind_partial(*args, **kwargs)
                except TypeError as exc:
                    msg = str(exc)
                    if "elev_deg" in msg:
                        kwargs = dict(kwargs)
                        kwargs["elev_deg"] = float(default_elev_deg)
                    else:
                        raise
                else:
                    if "elev_deg" not in bound.arguments:
                        kwargs = dict(kwargs)
                        kwargs["elev_deg"] = float(default_elev_deg)
            return orig(*args, **kwargs)

        rcs_solver._build_compiled_system = wrapped_build_compiled_system

# -----------------------------
# solver instrumentation
# -----------------------------
def first_bad_index(arr: np.ndarray) -> Optional[Tuple[int, ...]]:
    bad = np.argwhere(~np.isfinite(arr))
    if bad.size == 0:
        return None
    return tuple(int(v) for v in bad[0])


def complex_is_finite(z: complex) -> bool:
    return math.isfinite(float(np.real(z))) and math.isfinite(float(np.imag(z)))


def dump_matrix_context(label: str, arr: np.ndarray, out_dir: str) -> None:
    arr = np.asarray(arr)
    idx = first_bad_index(arr)
    print(f"[instrument] {label}: shape={arr.shape}, first_bad_index={idx}")
    if idx is None:
        return
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, f"{label.replace(' ', '_')}.npy"), arr)

    if arr.ndim == 2:
        r, c = idx
        r0 = max(0, r - 2)
        r1 = min(arr.shape[0], r + 3)
        c0 = max(0, c - 2)
        c1 = min(arr.shape[1], c + 3)
        block = arr[r0:r1, c0:c1]
        print(f"[instrument] {label}: local block rows[{r0}:{r1}] cols[{c0}:{c1}] =")
        print(block)
    elif arr.ndim == 1:
        i = idx[0]
        i0 = max(0, i - 5)
        i1 = min(arr.shape[0], i + 6)
        print(f"[instrument] {label}: local slice[{i0}:{i1}] =")
        print(arr[i0:i1])


def install_instrumentation(out_dir: str) -> None:
    # _ensure_finite_linear_system
    if hasattr(rcs_solver, "_ensure_finite_linear_system"):
        orig = rcs_solver._ensure_finite_linear_system

        def wrapped_ensure(a_mat, rhs=None, label="linear system"):
            a_eval = np.asarray(a_mat)
            if not np.all(np.isfinite(a_eval)):
                dump_matrix_context(label + "_A", a_eval, out_dir)
            if rhs is not None:
                b_eval = np.asarray(rhs)
                if not np.all(np.isfinite(b_eval)):
                    dump_matrix_context(label + "_rhs", b_eval, out_dir)
            return orig(a_mat, rhs, label)

        rcs_solver._ensure_finite_linear_system = wrapped_ensure

    # catch non-finite medium/wavenumber/admittance/robin alpha earlier
    for name in ["_medium_wavenumber", "_impedance_to_admittance", "_surface_robin_alpha"]:
        if not hasattr(rcs_solver, name):
            continue
        orig = getattr(rcs_solver, name)

        def make_wrapper(func_name, func_orig):
            def wrapper(*args, **kwargs):
                out = func_orig(*args, **kwargs)
                if isinstance(out, complex):
                    finite = complex_is_finite(out)
                else:
                    finite = math.isfinite(float(out))
                if not finite:
                    raise ValueError(f"[instrument] {func_name} returned non-finite value. args={args}, kwargs={kwargs}, out={out}")
                return out
            return wrapper

        setattr(rcs_solver, name, make_wrapper(name, orig))

    # inspect coupled panel info
    if hasattr(rcs_solver, "_build_coupled_panel_info"):
        orig = rcs_solver._build_coupled_panel_info

        def wrapped_panel_info(*args, **kwargs):
            infos = orig(*args, **kwargs)
            bad = []
            for i, info in enumerate(infos):
                fields = {
                    "eps_plus": info.eps_plus,
                    "mu_plus": info.mu_plus,
                    "eps_minus": info.eps_minus,
                    "mu_minus": info.mu_minus,
                    "k_plus": info.k_plus,
                    "k_minus": info.k_minus,
                    "q_plus_beta": info.q_plus_beta,
                    "q_plus_gamma": info.q_plus_gamma,
                    "robin_impedance": info.robin_impedance,
                }
                for key, val in fields.items():
                    if not complex_is_finite(complex(val)):
                        bad.append({
                            "panel_index": i,
                            "panel_name": getattr(args[0][i], "name", f"panel_{i}") if args and args[0] else f"panel_{i}",
                            "field": key,
                            "value": [float(np.real(val)), float(np.imag(val))],
                            "seg_type": int(info.seg_type),
                            "plus_region": int(info.plus_region),
                            "minus_region": int(info.minus_region),
                            "bc_kind": str(info.bc_kind),
                        })
            if bad:
                os.makedirs(out_dir, exist_ok=True)
                with open(os.path.join(out_dir, "bad_coupled_infos.json"), "w") as f:
                    json.dump(bad, f, indent=2)
                raise ValueError(f"[instrument] non-finite coupled panel coefficients detected. see {os.path.join(out_dir, 'bad_coupled_infos.json')}")
            return infos

        rcs_solver._build_coupled_panel_info = wrapped_panel_info

    # inspect region operators
    if hasattr(rcs_solver, "_build_coupled_region_operators"):
        orig = rcs_solver._build_coupled_region_operators

        def wrapped_region_ops(*args, **kwargs):
            ops = orig(*args, **kwargs)
            for region, pair in ops.items():
                try:
                    s_mat, k_mat = pair
                except Exception:
                    continue
                if not np.all(np.isfinite(np.asarray(s_mat))):
                    dump_matrix_context(f"region_{region}_S", np.asarray(s_mat), out_dir)
                    raise ValueError(f"[instrument] region {region} S operator contains NaN/Inf")
                if not np.all(np.isfinite(np.asarray(k_mat))):
                    dump_matrix_context(f"region_{region}_K", np.asarray(k_mat), out_dir)
                    raise ValueError(f"[instrument] region {region} K operator contains NaN/Inf")
            return ops

        rcs_solver._build_coupled_region_operators = wrapped_region_ops

    # inspect coupled assembled A/rhs before solve
    if hasattr(rcs_solver, "_build_coupled_system"):
        orig = rcs_solver._build_coupled_system

        def wrapped_coupled_system(*args, **kwargs):
            a_mat, rhs = orig(*args, **kwargs)
            if not np.all(np.isfinite(np.asarray(a_mat))):
                dump_matrix_context("coupled_system_A", np.asarray(a_mat), out_dir)
            if not np.all(np.isfinite(np.asarray(rhs))):
                dump_matrix_context("coupled_system_rhs", np.asarray(rhs), out_dir)
            return a_mat, rhs

        rcs_solver._build_coupled_system = wrapped_coupled_system


# -----------------------------
# running solves
# -----------------------------
def run_one_method(
    snapshot: Dict[str, Any],
    base_dir: str,
    freq_ghz: float,
    elev_deg: float,
    pol: str,
    units: str,
    basis_family: str,
    testing_family: str,
) -> None:
    print("\n" + "=" * 80)
    print(f"Running solve: basis={basis_family}, testing={testing_family}, freq={freq_ghz} GHz, elev={elev_deg} deg, pol={pol}, units={units}")
    print("=" * 80)

    def progress(done: int, total: int, msg: str) -> None:
        print(f"[progress] {done}/{total}: {msg}")

    result = rcs_solver.solve_monostatic_rcs_2d(
        geometry_snapshot=snapshot,
        frequencies_ghz=[float(freq_ghz)],
        elevations_deg=[float(elev_deg)],
        polarization=str(pol),
        geometry_units=str(units),
        material_base_dir=base_dir,
        progress_callback=progress,
        quality_thresholds={
            "residual_norm_max": 1.0e9,
            "condition_est_max": 1.0e30,
            "warnings_max": 10_000,
        },
        strict_quality_gate=False,
        compute_condition_number=False,
        parallel_elevations=False,
        reuse_angle_invariant_matrix=True,
        basis_family=str(basis_family),
        testing_family=str(testing_family),
    )
    print("[ok] solve completed")
    print(json.dumps(result.get("metadata", {}), indent=2, default=str))
    for row in result.get("samples", []):
        print(
            "sample:",
            {
                "frequency_ghz": row.get("frequency_ghz"),
                "theta_deg": row.get("theta_scat_deg"),
                "rcs_linear": row.get("rcs_linear"),
                "rcs_db": row.get("rcs_db"),
                "linear_residual": row.get("linear_residual"),
            },
        )


# -----------------------------
# main
# -----------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description="Diagnose NaN/Inf in coupled 2D RCS solves.")
    ap.add_argument("geo", help="Path to .geo file")
    ap.add_argument("--freq", type=float, default=3.5, help="Single test frequency in GHz")
    ap.add_argument("--elev", type=float, default=0.0, help="Single test elevation in deg")
    ap.add_argument("--pol", default="TE", help="User-facing polarization: TE/TM or VV/HH")
    ap.add_argument("--units", default="inches", help="Geometry units: inches or meters")
    ap.add_argument("--method", choices=["pulse", "galerkin", "both"], default="both")
    ap.add_argument("--out-dir", default="diag_naninf_out", help="Directory for debug dumps")
    args = ap.parse_args()

    geo_path = os.path.abspath(args.geo)
    base_dir = os.path.dirname(geo_path)
    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    print(f"geometry: {geo_path}")
    print(f"base_dir:  {base_dir}")
    print(f"out_dir:   {out_dir}")
    print(f"solver module: {getattr(rcs_solver, '__file__', '<unknown>')}")

    with open(geo_path, "r") as f:
        text = f.read()
    title, segments, ibcs_entries, dielectric_entries = parse_geometry(text)
    snapshot = build_geometry_snapshot(title, segments, ibcs_entries, dielectric_entries)
    snapshot["source_path"] = geo_path

    findings = validate_geometry_headless(segments, ibcs_entries, dielectric_entries)
    errors = [msg for level, msg in findings if level == "ERROR"]
    warns = [msg for level, msg in findings if level == "WARN"]
    infos = [msg for level, msg in findings if level == "INFO"]

    print("\n" + "#" * 80)
    print("STATIC GEOMETRY CHECK")
    print("#" * 80)
    print(f"errors={len(errors)}, warnings={len(warns)}, info={len(infos)}")
    for level, msg in findings:
        print(f"[{level}] {msg}")

    install_compatibility_shims(float(args.elev))
    install_instrumentation(out_dir)

    methods: List[Tuple[str, str]] = []
    if args.method in {"pulse", "both"}:
        methods.append(("pulse", "collocation"))
    if args.method in {"galerkin", "both"}:
        methods.append(("linear", "galerkin"))

    overall_ok = True
    for basis_family, testing_family in methods:
        try:
            run_one_method(
                snapshot=snapshot,
                base_dir=base_dir,
                freq_ghz=float(args.freq),
                elev_deg=float(args.elev),
                pol=str(args.pol),
                units=str(args.units),
                basis_family=basis_family,
                testing_family=testing_family,
            )
        except Exception as exc:
            overall_ok = False
            print(f"[FAIL] {basis_family}/{testing_family}: {exc}")
            traceback.print_exc()

    summary = {
        "geometry": geo_path,
        "freq_ghz": float(args.freq),
        "elev_deg": float(args.elev),
        "pol": str(args.pol),
        "units": str(args.units),
        "static_errors": errors,
        "static_warnings": warns,
        "static_info": infos,
        "overall_ok": overall_ok,
    }
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("\nsummary written to:", os.path.join(out_dir, "summary.json"))
    return 0 if overall_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
