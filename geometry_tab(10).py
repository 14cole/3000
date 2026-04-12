import math
import os
from typing import Any, Dict, List, Optional, Set, Tuple

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QFileDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

from geometry_io import Segment, build_geometry_snapshot, build_geometry_text, parse_geometry


class MplCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.ax = self.fig.add_subplot(111)
        super().__init__(self.fig)
        self.setParent(parent)

        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.updateGeometry()


class GeometryTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        splitter = QSplitter(Qt.Horizontal)

        plot_container = QWidget()
        plot_layout = QVBoxLayout(plot_container)
        self.canvas = MplCanvas(plot_container)

        self.toolbar = NavigationToolbar(self.canvas, plot_container)
        plot_layout.addWidget(self.toolbar)
        plot_layout.addWidget(self.canvas)
        splitter.addWidget(plot_container)

        right_container = QWidget()
        right_layout = QVBoxLayout(right_container)

        btn_row = QHBoxLayout()
        self.btn_load = QPushButton("Load")
        self.btn_save = QPushButton("Save")
        self.btn_validate = QPushButton("Validate")
        self.chk_show_normals = QCheckBox("Show Normals")
        btn_row.addWidget(self.btn_load)
        btn_row.addWidget(self.btn_save)
        btn_row.addWidget(self.btn_validate)
        btn_row.addWidget(self.chk_show_normals)
        btn_row.addStretch(1)
        right_layout.addLayout(btn_row)

        self.table = QTableWidget()
        self.table.setRowCount(0)
        self.table.setColumnCount(6)
        self.table.setHorizontalHeaderLabels(["Name", "Type", "N", "IBC", "IPN1", "IPN2"])
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SingleSelection)
        right_layout.addWidget(self.table)

        bottom_row = QHBoxLayout()

        ibc_box = QVBoxLayout()
        self.lbl_ibc = QLabel("IBCS")
        self.table_ibc = QTableWidget()
        self.table_ibc.setRowCount(0)
        self.table_ibc.setColumnCount(0)
        ibc_box.addWidget(self.lbl_ibc)
        ibc_box.addWidget(self.table_ibc)

        diel_box = QVBoxLayout()
        self.lbl_diel = QLabel("Dielectrics")
        self.table_diel = QTableWidget()
        self.table_diel.setRowCount(0)
        self.table_diel.setColumnCount(0)
        diel_box.addWidget(self.lbl_diel)
        diel_box.addWidget(self.table_diel)

        bottom_row.addLayout(ibc_box, stretch=1)
        bottom_row.addLayout(diel_box, stretch=1)
        right_layout.addLayout(bottom_row)

        splitter.addWidget(right_container)

        splitter.setSizes([700, 300])
        main_layout = QHBoxLayout(self)
        main_layout.addWidget(splitter)

        self.btn_load.clicked.connect(self.load_geo)
        self.btn_save.clicked.connect(self.save_geo)
        self.btn_validate.clicked.connect(self.validate_geometry)
        self.chk_show_normals.toggled.connect(self._on_show_normals_toggled)

        self.title: str = "Geometry"
        self.segments: List[Segment] = []
        self.ibcs_entries: List[List[str]] = []
        self.dielectric_entries: List[List[str]] = []
        self.segment_lines: List = []
        self.segment_base_colors: List[str] = []
        self._populating: bool = False
        self._syncing_selection: bool = False
        self._selected_row: Optional[int] = None
        self._last_ext: str = ".geo"
        self.loaded_path: str = ""
        self.issue_rows: Set[int] = set()
        self.normal_artists: List[Any] = []

        self.table.itemChanged.connect(self._on_main_table_item_changed)
        self.table.itemSelectionChanged.connect(self._on_table_selection_changed)
        self.canvas.mpl_connect("pick_event", self._on_plot_pick)

        self.canvas.mpl_connect("button_press_event", self._on_plot_button_press)
        self.canvas.mpl_connect("scroll_event", self._on_plot_scroll)

        self._set_equal_column_widths(self.table, enabled=True)
        self._set_equal_column_widths(self.table_ibc, enabled=True)
        self._set_equal_column_widths(self.table_diel, enabled=True)

    def load_geo(self):
        fname, _ = QFileDialog.getOpenFileName(
            self, "Open Geometry File", "", "Geometry Files (*.geo);;All Files (*)"
        )
        if not fname:
            return
        try:
            with open(fname, "r") as f:
                text = f.read()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to read file: {e}")
            return
        try:
            title, segments, ibcs_entries, dielectric_entries = parse_geometry(text)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to parse geometry: {e}")
            return

        self.title = title
        self.segments = segments
        self.ibcs_entries = ibcs_entries
        self.dielectric_entries = dielectric_entries
        self.loaded_path = os.path.abspath(fname)

        self._populating = True
        try:
            self.table.clearContents()
            self.table.setRowCount(len(self.segments))
            self.table.setColumnCount(6)
            self.table.setHorizontalHeaderLabels(["Name", "Type", "N", "IBC", "IPN1", "IPN2"])
            for row, seg in enumerate(self.segments):
                props = self._ensure_prop_len(seg.properties, 6)
                itype = props[0] if len(props) >= 1 else ""
                n_value = props[1] if len(props) >= 2 else ""
                ibc = props[3] if len(props) >= 4 else ""
                ipn1 = props[4] if len(props) >= 5 else ""
                ipn2 = props[5] if len(props) >= 6 else ""
                self.table.setItem(row, 0, QTableWidgetItem(seg.name))
                self.table.setItem(row, 1, QTableWidgetItem(itype))
                self.table.setItem(row, 2, QTableWidgetItem(n_value))
                self.table.setItem(row, 3, QTableWidgetItem(ibc))
                self.table.setItem(row, 4, QTableWidgetItem(ipn1))
                self.table.setItem(row, 5, QTableWidgetItem(ipn2))
        finally:
            self._populating = False

        ax = self.canvas.ax
        ax.clear()
        self.segment_lines = []
        self.segment_base_colors = []
        self.issue_rows.clear()
        self._clear_normals()

        plot_colors = ["orange", "green", "blue", "gray", "black", "red", "purple", "cyan"]

        for row, seg in enumerate(self.segments):
            props = seg.properties
            itype = props[0] if len(props) >= 1 else ""
            try:
                color_index = (int(itype) - 1) % len(plot_colors)
                base_color = plot_colors[color_index]
            except (ValueError, TypeError):
                base_color = plot_colors[row % len(plot_colors)]

            plot_x, plot_y = self._segment_plot_xy(seg)
            (line2d,) = ax.plot(plot_x, plot_y, color=base_color, linewidth=1.5, zorder=1)
            line2d.set_picker(True)
            line2d.set_pickradius(5)
            self.segment_lines.append(line2d)
            self.segment_base_colors.append(base_color)
        ax.set_title(self.title)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_aspect("equal", adjustable="datalim")
        ax.grid(True, alpha=0.3)

        self._populate_small_table(self.table_ibc, self.ibcs_entries, label=self.lbl_ibc, title_prefix="IBCS")
        self._populate_small_table(
            self.table_diel,
            self.dielectric_entries,
            label=self.lbl_diel,
            title_prefix="Dielectrics",
        )

        self._selected_row = None
        self._refresh_segment_styles()
        self._render_normals()
        self.canvas.draw()
        QMessageBox.information(
            self,
            "Loaded",
            f"Loaded {len(self.segments)} segments(s),"
            f"{len(self.ibcs_entries)} IBCS entry(ies),"
            f"and {len(self.dielectric_entries)} dielectric entry(ies).",
        )

    def _populate_small_table(self, table: QTableWidget, rows: List[List[str]], label: QLabel, title_prefix: str):
        col_count = max((len(r) for r in rows), default=0)
        table.clearContents()
        table.setRowCount(len(rows))
        table.setColumnCount(col_count)

        if title_prefix == "IBCS":
            headers_full = ["Flag", "Z_real", "Z_imag", "Constant"]
            headers = headers_full[:col_count] if col_count > 0 else []
        elif title_prefix == "Dielectrics":
            headers_full = ["Flag", "Ep_real", "Ep_imag", "Mu_real", "Mu_imag"]
            headers = headers_full[:col_count] if col_count > 0 else []
        else:
            headers = [f"Col {i+1}" for i in range(col_count)]
        table.setHorizontalHeaderLabels(headers)

        for r, tokens in enumerate(rows):
            for c, token in enumerate(tokens):
                table.setItem(r, c, QTableWidgetItem(token))

        label.setText(f"{title_prefix} (n={len(rows)})")

    def _ensure_prop_len(self, props: List[str], n: int) -> List[str]:
        if len(props) < n:
            props.extend([""] * (n - len(props)))
        return props

    def _on_main_table_item_changed(self, item: QTableWidgetItem):
        if self._populating:
            return
        row = item.row()
        col = item.column()
        if row < 0 or row >= len(self.segments):
            return

        seg = self.segments[row]
        text = item.text().strip()

        if col == 0:
            seg.name = text
        else:
            props = self._ensure_prop_len(seg.properties, 6)
            if col == 1:
                props[0] = text
                seg.seg_type = text or None
                plot_colors = ["orange", "green", "blue", "gray", "black", "red", "purple", "cyan"]
                try:
                    color_index = (int(text) - 1) % len(plot_colors)
                    base_color = plot_colors[color_index]
                except (ValueError, TypeError):
                    base_color = plot_colors[row % len(plot_colors)]
                self.segment_base_colors[row] = base_color
                self._refresh_segment_styles()
                self.canvas.draw_idle()
            elif col == 2:
                props[1] = text
            elif col == 3:
                props[3] = text
            elif col == 4:
                props[4] = text
            elif col == 5:
                props[5] = text

    def _on_table_selection_changed(self):
        if self._syncing_selection:
            return
        row = self.table.currentRow()
        self._apply_selection(row)

    def _apply_selection(self, row: int):
        self._selected_row = row if (row is not None and row >= 0) else None
        self._refresh_segment_styles()
        self.canvas.draw_idle()

    def _on_plot_pick(self, event):
        line = getattr(event, "artist", None)
        if not line:
            return
        try:
            row = self.segment_lines.index(line)
        except ValueError:
            return
        self._syncing_selection = True
        try:
            self.table.selectRow(row)
            self._apply_selection(row)
        finally:
            self._syncing_selection = False

    def _on_plot_button_press(self, event):
        if event.inaxes != self.canvas.ax:
            return
        modifier_select = event.button == 1 and (event.key in ("control", "shift"))
        if event.button == 3 or modifier_select:
            idx = self._hit_test(event)
            if idx is not None:
                self._syncing_selection = True
                try:
                    self.table.selectRow(idx)
                    self._apply_selection(idx)
                finally:
                    self._syncing_selection = False

    def _hit_test(self, event) -> Optional[int]:
        for i in reversed(range(len(self.segment_lines))):
            line = self.segment_lines[i]
            contains, _ = line.contains(event)
            if contains:
                return i
        return None

    def _on_plot_scroll(self, event):
        if event.inaxes != self.canvas.ax or event.xdata is None or event.ydata is None:
            return
        base_scale = 1.2 if event.button == "up" else (1 / 1.2)
        self._zoom_at(event.xdata, event.ydata, base_scale)

    def _zoom_at(self, x: float, y: float, scale: float):
        ax = self.canvas.ax
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        w = (xlim[1] - xlim[0]) / scale
        h = (ylim[1] - ylim[0]) / scale
        ax.set_xlim(x - w / 2, x + w / 2)
        ax.set_ylim(y - h / 2, y + h / 2)
        self.canvas.draw_idle()

    def _refresh_segment_styles(self):
        for i, line in enumerate(self.segment_lines):
            if self._selected_row is not None and i == self._selected_row:
                line.set_color("pink")
                line.set_linewidth(2.5)
                line.set_zorder(10)
                continue
            if i in self.issue_rows:
                line.set_color("crimson")
                line.set_linewidth(2.2)
                line.set_zorder(8)
                continue
            base = self.segment_base_colors[i] if i < len(self.segment_base_colors) else "gray"
            line.set_color(base)
            line.set_linewidth(1.5)
            line.set_zorder(1)

    def _clear_normals(self):
        for art in self.normal_artists:
            try:
                art.remove()
            except Exception:
                pass
        self.normal_artists = []

    def _segment_primitives(self, seg: Segment) -> List[Tuple[float, float, float, float]]:
        count = min(len(seg.x), len(seg.y))
        n_pairs = count // 2
        out: List[Tuple[float, float, float, float]] = []
        for i in range(n_pairs):
            idx = 2 * i
            out.append((seg.x[idx], seg.y[idx], seg.x[idx + 1], seg.y[idx + 1]))
        return out

    def _segment_arc_angle_deg(self, seg: Segment) -> float:
        props = self._ensure_prop_len(seg.properties, 6)
        try:
            return float(props[2]) if props[2] else 0.0
        except (TypeError, ValueError):
            return 0.0

    def _arc_points(
        self, x1: float, y1: float, x2: float, y2: float, ang_deg: float, samples: int = 24
    ) -> List[Tuple[float, float]]:
        if abs(ang_deg) < 1e-9:
            return [(x1, y1), (x2, y2)]

        dx = x2 - x1
        dy = y2 - y1
        chord = math.hypot(dx, dy)
        if chord <= 1e-12:
            return [(x1, y1), (x2, y2)]

        ang_rad = math.radians(ang_deg)
        abs_phi = abs(ang_rad)
        if abs_phi <= 1e-9:
            return [(x1, y1), (x2, y2)]

        sin_half = math.sin(abs_phi * 0.5)
        tan_half = math.tan(abs_phi * 0.5)
        if abs(sin_half) <= 1e-12 or abs(tan_half) <= 1e-12:
            return [(x1, y1), (x2, y2)]

        radius = chord / (2.0 * sin_half)
        h = chord / (2.0 * tan_half)

        mx = 0.5 * (x1 + x2)
        my = 0.5 * (y1 + y2)
        px = -dy / chord
        py = dx / chord

        centers = [(mx + px * h, my + py * h), (mx - px * h, my - py * h)]
        best_center = centers[0]
        best_a0 = 0.0
        best_err = float("inf")

        for cx, cy in centers:
            a0 = math.atan2(y1 - cy, x1 - cx)
            x2_pred = cx + radius * math.cos(a0 + ang_rad)
            y2_pred = cy + radius * math.sin(a0 + ang_rad)
            err = math.hypot(x2_pred - x2, y2_pred - y2)
            if err < best_err:
                best_err = err
                best_center = (cx, cy)
                best_a0 = a0

        cx, cy = best_center
        n = max(8, int(samples))
        pts: List[Tuple[float, float]] = []
        for i in range(n + 1):
            t = i / n
            a = best_a0 + ang_rad * t
            pts.append((cx + radius * math.cos(a), cy + radius * math.sin(a)))
        return pts

    def _segment_plot_xy(self, seg: Segment) -> Tuple[List[float], List[float]]:
        primitives = self._segment_primitives(seg)
        if not primitives:
            return list(seg.x), list(seg.y)

        ang_deg = self._segment_arc_angle_deg(seg)
        xs: List[float] = []
        ys: List[float] = []
        for i, (x1, y1, x2, y2) in enumerate(primitives):
            pts = self._arc_points(x1, y1, x2, y2, ang_deg)
            if i > 0 and pts:
                pts = pts[1:]
            for px, py in pts:
                xs.append(px)
                ys.append(py)

        if not xs or not ys:
            return list(seg.x), list(seg.y)
        return xs, ys

    def _render_normals(self):
        self._clear_normals()
        if not self.chk_show_normals.isChecked():
            return
        if not self.segments:
            return

        all_x = [x for seg in self.segments for x in seg.x]
        all_y = [y for seg in self.segments for y in seg.y]
        if not all_x or not all_y:
            return
        diag = max(((max(all_x) - min(all_x)) ** 2 + (max(all_y) - min(all_y)) ** 2) ** 0.5, 1.0)
        arrow_len = 0.04 * diag
        ax = self.canvas.ax

        for row, seg in enumerate(self.segments):
            color = "crimson" if row in self.issue_rows else "magenta"
            for x1, y1, x2, y2 in self._segment_primitives(seg):
                dx = x2 - x1
                dy = y2 - y1
                length = (dx * dx + dy * dy) ** 0.5
                if length <= 1e-12:
                    continue
                nx = -dy / length
                ny = dx / length
                mx = 0.5 * (x1 + x2)
                my = 0.5 * (y1 + y2)
                ann = ax.annotate(
                    "",
                    xy=(mx + nx * arrow_len, my + ny * arrow_len),
                    xytext=(mx, my),
                    arrowprops={"arrowstyle": "-|>", "color": color, "lw": 0.8, "alpha": 0.75},
                    zorder=12,
                )
                self.normal_artists.append(ann)

    def _on_show_normals_toggled(self, checked: bool):
        _ = checked
        self._render_normals()
        self.canvas.draw_idle()

    def _parse_int_token(self, token: str, default: int = 0) -> int:
        text = (token or "").strip().lower()
        if not text:
            return default
        if text.startswith("fort."):
            text = text.split("fort.", 1)[1]
        try:
            return int(float(text))
        except ValueError:
            return default

    def _parse_float_token(self, token: str, default: float = 0.0) -> float:
        text = (token or "").strip()
        if not text:
            return default
        try:
            return float(text)
        except ValueError:
            return default

    def _find_fort_file(self, flag: int) -> str:
        name = f"fort.{flag}"
        base_dir = os.path.dirname(self.loaded_path) if self.loaded_path else ""
        candidates = [os.path.join(base_dir, name), os.path.join(os.getcwd(), name)]
        for path in candidates:
            if path and os.path.isfile(path):
                return path
        return ""

    def _point_key(self, x: float, y: float, tol: float) -> Tuple[int, int]:
        inv = 1.0 / max(tol, 1e-12)
        return int(round(float(x) * inv)), int(round(float(y) * inv))

    def _segments_intersect(
        self,
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

    def validate_geometry(self):
        ibcs_rows = self._read_small_table(self.table_ibc)
        dielectric_rows = self._read_small_table(self.table_diel)
        ibc_flags = {self._parse_int_token(row[0], 0) for row in ibcs_rows if row}
        diel_flags = {self._parse_int_token(row[0], 0) for row in dielectric_rows if row}

        findings: List[Tuple[str, int, str]] = []
        issue_rows: Set[int] = set()

        for ibc_idx, row in enumerate(ibcs_rows):
            if len(row) >= 3 and self._parse_int_token(row[0], 0) <= 50:
                z_real = self._parse_float_token(row[1], 0.0)
                z_imag = self._parse_float_token(row[2], 0.0)
                if not math.isfinite(z_real) or not math.isfinite(z_imag):
                    findings.append((
                        "ERROR",
                        -1,
                        f"IBCS row {ibc_idx + 1}: impedance contains non-finite value(s) ({row[1]!r}, {row[2]!r}).",
                    ))
            if len(row) >= 1 and self._parse_int_token(row[0], 0) > 50:
                fort_path = self._find_fort_file(self._parse_int_token(row[0], 0))
                if fort_path:
                    try:
                        with open(fort_path, 'r') as f:
                            for lineno, raw in enumerate(f, start=1):
                                line = raw.split('#', 1)[0].strip()
                                if not line:
                                    continue
                                toks = line.split()
                                if len(toks) < 3:
                                    continue
                                vals = [float(toks[i]) for i in range(3)]
                                if not all(math.isfinite(v) for v in vals):
                                    findings.append((
                                        "ERROR",
                                        -1,
                                        f"fort.{self._parse_int_token(row[0], 0)} line {lineno}: impedance table contains non-finite value(s).",
                                    ))
                                    break
                    except Exception:
                        pass

        for diel_idx, row in enumerate(dielectric_rows):
            flag = self._parse_int_token(row[0] if row else '', 0)
            if flag <= 50 and len(row) >= 5:
                vals = [self._parse_float_token(tok, 0.0) for tok in row[1:5]]
                if not all(math.isfinite(v) for v in vals):
                    findings.append((
                        "ERROR",
                        -1,
                        f"Dielectrics row {diel_idx + 1}: material values contain non-finite token(s).",
                    ))
            if flag > 50:
                fort_path = self._find_fort_file(flag)
                if fort_path:
                    try:
                        with open(fort_path, 'r') as f:
                            for lineno, raw in enumerate(f, start=1):
                                line = raw.split('#', 1)[0].strip()
                                if not line:
                                    continue
                                toks = line.split()
                                if len(toks) < 5:
                                    continue
                                vals = [float(toks[i]) for i in range(5)]
                                if not all(math.isfinite(v) for v in vals):
                                    findings.append((
                                        "ERROR",
                                        -1,
                                        f"fort.{flag} line {lineno}: dielectric table contains non-finite value(s).",
                                    ))
                                    break
                    except Exception:
                        pass

        all_points = [(x, y) for seg in self.segments for x, y in zip(seg.x, seg.y)]
        if all_points:
            xs = [p[0] for p in all_points]
            ys = [p[1] for p in all_points]
            diag = max(((max(xs) - min(xs)) ** 2 + (max(ys) - min(ys)) ** 2) ** 0.5, 1.0)
        else:
            diag = 1.0
        tol = max(1e-8, 1e-6 * diag)

        for row, seg in enumerate(self.segments):
            props = self._ensure_prop_len(seg.properties, 6)
            seg_type = self._parse_int_token(props[0], -1)
            n_panels = self._parse_int_token(props[1], 0)
            ibc = self._parse_int_token(props[3], 0)
            ipn1 = self._parse_int_token(props[4], 0)
            ipn2 = self._parse_int_token(props[5], 0)
            primitives = self._segment_primitives(seg)
            label = f"Row {row + 1} '{seg.name}'"

            if seg_type < 1 or seg_type > 5:
                findings.append(("ERROR", row, f"{label}: invalid TYPE '{props[0]}', expected 1..5."))
                issue_rows.add(row)

            if n_panels == 0:
                findings.append(("WARN", row, f"{label}: N should be a non-zero integer; positive N means explicit panel count and negative N means wavelength-based meshing. Current value is '{props[1]}'."))
                issue_rows.add(row)

            if not primitives:
                findings.append(("ERROR", row, f"{label}: no line primitives found."))
                issue_rows.add(row)
                continue

            for i, (x1, y1, x2, y2) in enumerate(primitives):
                length = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
                if length <= tol:
                    findings.append(("ERROR", row, f"{label}: primitive {i + 1} has near-zero length."))
                    issue_rows.add(row)

            for i in range(len(primitives) - 1):
                _, _, ex, ey = primitives[i]
                nx1, ny1, nx2, ny2 = primitives[i + 1]
                d_start = ((ex - nx1) ** 2 + (ey - ny1) ** 2) ** 0.5
                d_end = ((ex - nx2) ** 2 + (ey - ny2) ** 2) ** 0.5
                if d_start > tol:
                    if d_end <= tol:
                        findings.append(
                            ("WARN", row, f"{label}: primitive {i + 2} appears reversed relative to previous one.")
                        )
                    else:
                        findings.append(("WARN", row, f"{label}: primitive {i + 1} and {i + 2} are not connected."))
                    issue_rows.add(row)

            sx, sy, _, _ = primitives[0]
            _, _, ex, ey = primitives[-1]
            closed = (((sx - ex) ** 2 + (sy - ey) ** 2) ** 0.5) <= tol

            if closed:
                points = [(sx, sy)] + [(x2, y2) for _, _, x2, y2 in primitives]
                area2 = 0.0
                for i in range(len(points) - 1):
                    x0, y0 = points[i]
                    x1, y1 = points[i + 1]
                    area2 += x0 * y1 - x1 * y0
                orient = "CCW" if area2 > 0 else "CW"
                findings.append(("INFO", row, f"{label}: closed chain, orientation {orient}."))
                if seg_type in {2, 3, 4, 5} and area2 > 0:
                    findings.append(
                        ("WARN", row, f"{label}: CCW orientation may mean inward-pointing normals; verify direction.")
                    )
                    issue_rows.add(row)
            else:
                findings.append(("WARN", row, f"{label}: open chain (start/end do not close)."))
                if seg_type in {2, 3, 4, 5}:
                    issue_rows.add(row)

            if ibc > 0 and ibc not in ibc_flags:
                findings.append(("ERROR", row, f"{label}: IBC flag {ibc} is referenced but not defined in IBCS."))
                issue_rows.add(row)

            if ibc > 50 and not self._find_fort_file(ibc):
                findings.append(("ERROR", row, f"{label}: IBC flag {ibc} expects missing file fort.{ibc}."))
                issue_rows.add(row)

            if seg_type in {3, 4, 5} and ipn1 <= 0:
                findings.append(("ERROR", row, f"{label}: TYPE {seg_type} requires IPN1 > 0."))
                issue_rows.add(row)
            if ipn1 > 0 and ipn1 not in diel_flags:
                findings.append(
                    ("ERROR", row, f"{label}: dielectric flag IPN1={ipn1} is referenced but not defined.")
                )
                issue_rows.add(row)
            if ipn1 > 50 and not self._find_fort_file(ipn1):
                findings.append(("ERROR", row, f"{label}: IPN1={ipn1} expects missing file fort.{ipn1}."))
                issue_rows.add(row)

            if seg_type == 5 and ipn2 <= 0:
                findings.append(("ERROR", row, f"{label}: TYPE 5 requires IPN2 > 0."))
                issue_rows.add(row)
            if ipn2 > 0 and ipn2 not in diel_flags:
                findings.append(
                    ("ERROR", row, f"{label}: dielectric flag IPN2={ipn2} is referenced but not defined.")
                )
                issue_rows.add(row)
            if ipn2 > 50 and not self._find_fort_file(ipn2):
                findings.append(("ERROR", row, f"{label}: IPN2={ipn2} expects missing file fort.{ipn2}."))
                issue_rows.add(row)
            if seg_type in {1, 2, 3, 4} and ipn2 != 0:
                findings.append(("WARN", row, f"{label}: TYPE {seg_type} typically uses IPN2=0."))
                issue_rows.add(row)

        # Global topology checks across segments (not just within each row).
        global_primitives: List[Tuple[int, int, Tuple[float, float, float, float], str]] = []
        row_type: Dict[int, int] = {}
        for row, seg in enumerate(self.segments):
            props = self._ensure_prop_len(seg.properties, 6)
            seg_type = self._parse_int_token(props[0], -1)
            row_type[row] = seg_type
            for pidx, prim in enumerate(self._segment_primitives(seg)):
                global_primitives.append((row, pidx, prim, seg.name))

        endpoint_hits: Dict[Tuple[int, int], List[Tuple[int, int, int]]] = {}
        for row, pidx, (x1, y1, x2, y2), _name in global_primitives:
            k1 = self._point_key(x1, y1, tol)
            k2 = self._point_key(x2, y2, tol)
            endpoint_hits.setdefault(k1, []).append((row, pidx, 0))
            endpoint_hits.setdefault(k2, []).append((row, pidx, 1))

        for _key, hits in endpoint_hits.items():
            incident_rows = sorted({h[0] for h in hits})
            if len(hits) == 1:
                row = hits[0][0]
                if row_type.get(row, -1) in {2, 3, 4, 5}:
                    findings.append(
                        ("WARN", row, f"Row {row + 1}: dangling endpoint not connected to any other primitive.")
                    )
                    issue_rows.add(row)
            if len(hits) > 6:
                row = incident_rows[0]
                findings.append(
                    (
                        "WARN",
                        row,
                        f"Row {row + 1}: high-degree node with {len(hits)} incident primitive endpoints "
                        "(possible non-manifold junction).",
                    )
                )
                issue_rows.add(row)

        max_intersections = 30
        found_intersections = 0
        n_prims = len(global_primitives)
        stop_intersections = False
        for i in range(n_prims):
            if stop_intersections:
                break
            row_i, pidx_i, prim_i, name_i = global_primitives[i]
            x1, y1, x2, y2 = prim_i
            k_i0 = self._point_key(x1, y1, tol)
            k_i1 = self._point_key(x2, y2, tol)
            for j in range(i + 1, n_prims):
                row_j, pidx_j, prim_j, name_j = global_primitives[j]
                u1, v1, u2, v2 = prim_j
                k_j0 = self._point_key(u1, v1, tol)
                k_j1 = self._point_key(u2, v2, tol)

                shared_endpoint = k_i0 in {k_j0, k_j1} or k_i1 in {k_j0, k_j1}
                if shared_endpoint:
                    continue
                if row_i == row_j and abs(pidx_i - pidx_j) <= 1:
                    continue

                if not self._segments_intersect((x1, y1), (x2, y2), (u1, v1), (u2, v2), tol):
                    continue

                findings.append(
                    (
                        "ERROR",
                        row_i,
                        (
                            f"Rows {row_i + 1} ('{name_i}') and {row_j + 1} ('{name_j}') have a non-endpoint "
                            "primitive intersection."
                        ),
                    )
                )
                issue_rows.add(row_i)
                issue_rows.add(row_j)
                found_intersections += 1
                if found_intersections >= max_intersections:
                    findings.append(
                        (
                            "WARN",
                            row_i,
                            f"Intersection reporting truncated after {max_intersections} findings.",
                        )
                    )
                    stop_intersections = True
                    break

        self.issue_rows = issue_rows
        self._refresh_segment_styles()
        self._render_normals()
        self.canvas.draw_idle()

        errors = [msg for level, _, msg in findings if level == "ERROR"]
        warns = [msg for level, _, msg in findings if level == "WARN"]
        infos = [msg for level, _, msg in findings if level == "INFO"]

        summary = (
            f"Validation complete: {len(errors)} error(s), {len(warns)} warning(s), {len(infos)} info message(s)."
        )
        detail_lines = errors + warns + infos
        if detail_lines:
            max_lines = 30
            shown = detail_lines[:max_lines]
            detail_text = "\n".join(shown)
            if len(detail_lines) > max_lines:
                detail_text += f"\n... ({len(detail_lines) - max_lines} additional message(s))"
            message = summary + "\n\n" + detail_text
        else:
            message = summary + "\nNo issues found."

        if errors or warns:
            QMessageBox.warning(self, "Geometry Validation", message)
        else:
            QMessageBox.information(self, "Geometry Validation", message)

    def save_geo(self):
        default_name = f"geometry_out{self._last_ext}"
        fname, selected_filter = QFileDialog.getSaveFileName(
            self, "Save Geometry File", default_name, "Geometry Files (*.geo);;All Files (*)"
        )
        if not fname:
            return
        fname = self._ensure_extension(fname, selected_filter)
        self._last_ext = os.path.splitext(fname)[1].lower()
        ibcs_rows = self._read_small_table(self.table_ibc)
        dielectric_rows = self._read_small_table(self.table_diel)
        try:
            text = build_geometry_text(self.title, self.segments, ibcs_rows, dielectric_rows)
        except ValueError as e:
            QMessageBox.warning(self, "Warning", str(e))
            return

        try:
            with open(fname, "w") as f:
                f.write(text)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save file: {e}")
            return
        self.loaded_path = os.path.abspath(fname)
        QMessageBox.information(self, "Saved", f"Geometry saved to {fname}")

    def _read_small_table(self, table: QTableWidget) -> List[List[str]]:
        rows: List[List[str]] = []
        for r in range(table.rowCount()):
            tokens: List[str] = []
            for c in range(table.columnCount()):
                item = table.item(r, c)
                val = item.text().strip() if item else ""
                tokens.append(val)
            while tokens and tokens[-1] == "":
                tokens.pop()
            if tokens:
                rows.append(tokens)
        return rows

    def _set_equal_column_widths(self, table: QTableWidget, enabled: bool = True):
        header = table.horizontalHeader()
        if not header:
            return
        if enabled:
            header.setSectionResizeMode(QHeaderView.Stretch)
        else:
            header.setSectionResizeMode(QHeaderView.Interactive)

    def _ensure_extension(self, fname: str, selected_filter: str) -> str:
        root, ext = os.path.splitext(fname)
        ext = ext.lower()
        if ext in (".geo", ".txt"):
            return fname
        filt = (selected_filter or "").lower()
        if ".geo" in filt:
            return root + ".geo"
        if ".txt" in filt:
            return root + ".txt"
        return root + ".geo"

    def get_geometry_snapshot(self) -> Dict[str, Any]:
        ibcs_rows = self._read_small_table(self.table_ibc)
        dielectric_rows = self._read_small_table(self.table_diel)
        snapshot = build_geometry_snapshot(
            self.title,
            self.segments,
            ibcs_rows,
            dielectric_rows,
        )
        snapshot["source_path"] = self.loaded_path
        return snapshot
