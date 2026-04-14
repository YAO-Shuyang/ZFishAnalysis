import sys
import os
from typing import Optional

import numpy as np
import pandas as pd

from PyQt6.QtCore import Qt, QAbstractTableModel, QModelIndex, QVariant, QTimer
from PyQt6.QtGui import QAction, QKeySequence
from PyQt6.QtWidgets import (
    QApplication, QWidget, QFileDialog, QMessageBox,
    QHBoxLayout, QVBoxLayout, QGridLayout,
    QPushButton, QLabel, QLineEdit, QSplitter,
    QTableView, QAbstractItemView, QHeaderView,
    QSpinBox, QDoubleSpinBox, QGroupBox
)

import pyqtgraph as pg
from datetime import datetime


# ============================================================
# Your import16chFlt
# ============================================================
from zfish._io import import16chFlt

# ============================================================
# Pandas table model
# ============================================================

class BoutTableModel(QAbstractTableModel):
    def __init__(self, df: Optional[pd.DataFrame] = None):
        super().__init__()
        if df is None:
            df = self.empty_df()
        self._df = df.copy()

    @staticmethod
    def empty_df() -> pd.DataFrame:
        return pd.DataFrame(
            columns=[
                "bout_id",
                "start_frame",
                "end_frame",
                "start_s",
                "end_s",
                "duration_frames",
                "duration_s",
                "note",
            ]
        )

    def dataframe(self) -> pd.DataFrame:
        return self._df

    def set_dataframe(self, df: pd.DataFrame) -> None:
        self.beginResetModel()
        self._df = df.copy()
        self.endResetModel()

    def rowCount(self, parent=QModelIndex()) -> int:
        return len(self._df)

    def columnCount(self, parent=QModelIndex()) -> int:
        return len(self._df.columns)

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        if not index.isValid():
            return QVariant()

        value = self._df.iat[index.row(), index.column()]
        col = self._df.columns[index.column()]

        if role in (Qt.ItemDataRole.DisplayRole, Qt.ItemDataRole.EditRole):
            if pd.isna(value):
                return ""
            if col in {"start_s", "end_s", "duration_s"}:
                return f"{float(value):.4f}"
            return str(value)

        return QVariant()

    def headerData(self, section, orientation, role=Qt.ItemDataRole.DisplayRole):
        if role != Qt.ItemDataRole.DisplayRole:
            return QVariant()

        if orientation == Qt.Orientation.Horizontal:
            return self._df.columns[section]
        return str(section)

    def flags(self, index):
        if not index.isValid():
            return Qt.ItemFlag.NoItemFlags

        col = self._df.columns[index.column()]
        editable_cols = {"start_frame", "end_frame", "note"}
        flags = Qt.ItemFlag.ItemIsSelectable | Qt.ItemFlag.ItemIsEnabled
        if col in editable_cols:
            flags |= Qt.ItemFlag.ItemIsEditable
        return flags

    def setData(self, index, value, role=Qt.ItemDataRole.EditRole):
        if role != Qt.ItemDataRole.EditRole or not index.isValid():
            return False

        col = self._df.columns[index.column()]
        row = index.row()

        try:
            if col in {"start_frame", "end_frame"}:
                if value == "":
                    return False
                self._df.at[row, col] = int(float(value))
            elif col == "note":
                self._df.at[row, col] = str(value)
            else:
                return False

            self._recompute_row(row)
            self.dataChanged.emit(index, index)
            self.layoutChanged.emit()
            return True

        except Exception:
            return False

    def _recompute_row(self, row: int, fs: float = 6000.0):
        start_f = int(self._df.at[row, "start_frame"])
        end_f = int(self._df.at[row, "end_frame"])

        if end_f < start_f:
            start_f, end_f = end_f, start_f
            self._df.at[row, "start_frame"] = start_f
            self._df.at[row, "end_frame"] = end_f

        self._df.at[row, "start_s"] = start_f / fs
        self._df.at[row, "end_s"] = end_f / fs
        self._df.at[row, "duration_frames"] = end_f - start_f
        self._df.at[row, "duration_s"] = (end_f - start_f) / fs

    def sort_and_reindex(self, fs: float = 6000.0):
        self.beginResetModel()
        if len(self._df) > 0:
            self._df = self._df.sort_values(["start_frame", "end_frame"]).reset_index(drop=True)
            self._df["bout_id"] = np.arange(len(self._df), dtype=int)
            for row in range(len(self._df)):
                self._recompute_row(row, fs=fs)
        self.endResetModel()

    def add_row(self, row_dict: dict, position: Optional[int] = None, fs: float = 6000.0):
        if position is None:
            position = len(self._df)
        position = max(0, min(position, len(self._df)))

        top = self._df.iloc[:position]
        bottom = self._df.iloc[position:]
        new_row = pd.DataFrame([row_dict])

        self.beginResetModel()
        self._df = pd.concat([top, new_row, bottom], axis=0).reset_index(drop=True)
        self._df["bout_id"] = np.arange(len(self._df), dtype=int)
        for row in range(len(self._df)):
            self._recompute_row(row, fs=fs)
        self.endResetModel()

    def update_row(self, row: int, row_dict: dict, fs: float = 6000.0):
        if not (0 <= row < len(self._df)):
            return
        self.beginResetModel()
        for k, v in row_dict.items():
            self._df.at[row, k] = v
        self._recompute_row(row, fs=fs)
        self._df["bout_id"] = np.arange(len(self._df), dtype=int)
        self.endResetModel()

    def delete_row(self, row: int):
        if not (0 <= row < len(self._df)):
            return
        self.beginResetModel()
        self._df = self._df.drop(index=row).reset_index(drop=True)
        if len(self._df) > 0:
            self._df["bout_id"] = np.arange(len(self._df), dtype=int)
        self.endResetModel()


# ============================================================
# Main GUI
# ============================================================

class SwimBoutLabeler(QWidget):
    FS = 6000.0

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Swim Bout Labeler (.16chFlt)")
        self.resize(1500, 900)

        self.file_path: Optional[str] = None
        self.data_dict: Optional[dict] = None
        self.t: Optional[np.ndarray] = None
        self.fltCh0: Optional[np.ndarray] = None
        self.fltCh1: Optional[np.ndarray] = None
        self.n_frames: int = 0

        self.model = BoutTableModel()

        self.view_start_frame: int = 0
        self.view_end_frame: int = 0
        self.view_active: bool = False

        self.autosave_timer = QTimer(self)
        self.autosave_timer.setInterval(3 * 60 * 1000)  # 3 minutes
        self.autosave_timer.timeout.connect(self.on_autosave)
        self.autosave_enabled = False

        self._build_ui()
        self._build_actions()
        self._connect_signals()

    # ---------------- Paths / arrays ----------------

    def autosave_xlsx_path(self) -> str:
        if self.file_path is None:
            return "swim_bouts_autosave.xlsx"
        root, _ = os.path.splitext(self.file_path)
        return root + "_swim_bouts_autosave.xlsx"

    def default_xlsx_path(self) -> str:
        if self.file_path is None:
            return "labeled_bouts.xlsx"
        root, _ = os.path.splitext(self.file_path)
        return root + "_labeled_bouts.xlsx"

    def get_bout_arrays(self) -> tuple[np.ndarray, np.ndarray]:
        df = self.model.dataframe().copy()
        if len(df) == 0:
            return np.array([], dtype=np.int64), np.array([], dtype=np.int64)

        df = df.sort_values(["start_frame", "end_frame"]).reset_index(drop=True)
        onsets = df["start_frame"].to_numpy(dtype=np.int64)
        offsets = df["end_frame"].to_numpy(dtype=np.int64)
        return onsets, offsets

    # ---------------- UI ----------------

    def _build_ui(self):
        pg.setConfigOptions(background="w", foreground="k", leftButtonPan=False, useNumba=True)

        self.path_edit = QLineEdit()
        self.path_edit.setPlaceholderText("Open a .16chFlt file...")
        self.browse_btn = QPushButton("Browse")
        self.load_xlsx_btn = QPushButton("Load Excel")
        self.save_xlsx_btn = QPushButton("Save Excel")

        top_row = QHBoxLayout()
        top_row.addWidget(self.path_edit, stretch=1)
        top_row.addWidget(self.browse_btn)
        top_row.addWidget(self.load_xlsx_btn)
        top_row.addWidget(self.save_xlsx_btn)

        # ----- Plots -----
        self.plot0 = pg.PlotWidget()
        self.plot1 = pg.PlotWidget()

        self.plot0.setLabel("left", "fltCh0")
        self.plot1.setLabel("left", "fltCh1")
        self.plot1.setLabel("bottom", "Time (s)")

        self.plot0.showGrid(x=True, y=True, alpha=0.2)
        self.plot1.showGrid(x=True, y=True, alpha=0.2)
        self.plot1.setXLink(self.plot0)

        self.curve0 = self.plot0.plot(pen=pg.mkPen("#d56e9e", width=1.2))
        self.curve1 = self.plot1.plot(pen=pg.mkPen("#3c619a", width=1.2))

        self.line_start_0 = pg.InfiniteLine(angle=90, movable=True, pen=pg.mkPen("g", width=2))
        self.line_end_0 = pg.InfiniteLine(angle=90, movable=True, pen=pg.mkPen("r", width=2))
        self.line_start_1 = pg.InfiniteLine(angle=90, movable=True, pen=pg.mkPen("g", width=2))
        self.line_end_1 = pg.InfiniteLine(angle=90, movable=True, pen=pg.mkPen("r", width=2))

        for item in [self.line_start_0, self.line_end_0]:
            self.plot0.addItem(item)
        for item in [self.line_start_1, self.line_end_1]:
            self.plot1.addItem(item)

        self.region0 = pg.LinearRegionItem(
            values=[0.0, 0.1],
            orientation="vertical",
            brush=(100, 150, 255, 40),
            pen=pg.mkPen((100, 150, 255, 100)),
            movable=False,
        )
        self.region1 = pg.LinearRegionItem(
            values=[0.0, 0.1],
            orientation="vertical",
            brush=(100, 150, 255, 40),
            pen=pg.mkPen((100, 150, 255, 100)),
            movable=False,
        )
        self.plot0.addItem(self.region0)
        self.plot1.addItem(self.region1)

        self.bout_regions0 = []
        self.bout_regions1 = []

        plot_layout = QVBoxLayout()
        plot_layout.addWidget(self.plot0, stretch=1)
        plot_layout.addWidget(self.plot1, stretch=1)

        plot_panel = QWidget()
        plot_panel.setLayout(plot_layout)

        # ----- Cursor group -----
        self.cursor_group = QGroupBox("Current Cursor Range")
        cursor_grid = QGridLayout()

        self.start_frame_spin = QSpinBox()
        self.end_frame_spin = QSpinBox()
        self.start_sec_spin = QDoubleSpinBox()
        self.end_sec_spin = QDoubleSpinBox()

        for sb in [self.start_frame_spin, self.end_frame_spin]:
            sb.setRange(0, 10**9)
            sb.setSingleStep(1)

        for dsb in [self.start_sec_spin, self.end_sec_spin]:
            dsb.setRange(0.0, 10**9)
            dsb.setDecimals(4)
            dsb.setSingleStep(0.01)

        self.duration_label = QLabel("Duration: 0 frames | 0.0000 s")

        cursor_grid.addWidget(QLabel("Start frame"), 0, 0)
        cursor_grid.addWidget(self.start_frame_spin, 0, 1)
        cursor_grid.addWidget(QLabel("End frame"), 1, 0)
        cursor_grid.addWidget(self.end_frame_spin, 1, 1)
        cursor_grid.addWidget(QLabel("Start (s)"), 2, 0)
        cursor_grid.addWidget(self.start_sec_spin, 2, 1)
        cursor_grid.addWidget(QLabel("End (s)"), 3, 0)
        cursor_grid.addWidget(self.end_sec_spin, 3, 1)
        cursor_grid.addWidget(self.duration_label, 4, 0, 1, 2)
        self.cursor_group.setLayout(cursor_grid)

        # ----- Visible range group -----
        self.view_group = QGroupBox("Visible Segment / Training Range")
        view_grid = QGridLayout()

        self.view_start_frame_spin = QSpinBox()
        self.view_end_frame_spin = QSpinBox()
        self.view_start_sec_spin = QDoubleSpinBox()
        self.view_end_sec_spin = QDoubleSpinBox()

        for sb in [self.view_start_frame_spin, self.view_end_frame_spin]:
            sb.setRange(0, 10**9)
            sb.setSingleStep(100)

        for dsb in [self.view_start_sec_spin, self.view_end_sec_spin]:
            dsb.setRange(0.0, 10**9)
            dsb.setDecimals(4)
            dsb.setSingleStep(0.1)

        self.apply_view_btn = QPushButton("Apply visible range")
        self.reset_view_btn = QPushButton("Reset to whole recording")
        self.use_cursor_as_view_btn = QPushButton("Use cursors as visible range")

        view_grid.addWidget(QLabel("View start frame"), 0, 0)
        view_grid.addWidget(self.view_start_frame_spin, 0, 1)
        view_grid.addWidget(QLabel("View end frame"), 1, 0)
        view_grid.addWidget(self.view_end_frame_spin, 1, 1)
        view_grid.addWidget(QLabel("View start (s)"), 2, 0)
        view_grid.addWidget(self.view_start_sec_spin, 2, 1)
        view_grid.addWidget(QLabel("View end (s)"), 3, 0)
        view_grid.addWidget(self.view_end_sec_spin, 3, 1)
        view_grid.addWidget(self.apply_view_btn, 4, 0, 1, 2)
        view_grid.addWidget(self.use_cursor_as_view_btn, 5, 0, 1, 2)
        view_grid.addWidget(self.reset_view_btn, 6, 0, 1, 2)
        self.view_group.setLayout(view_grid)

        # ----- Bout editing group -----
        self.ctrl_group = QGroupBox("Bout Editing")
        ctrl_layout = QGridLayout()

        self.add_btn = QPushButton("Add from cursors")
        self.update_btn = QPushButton("Update selected")
        self.delete_btn = QPushButton("Delete selected")
        self.insert_before_btn = QPushButton("Insert before selected")
        self.insert_after_btn = QPushButton("Insert after selected")
        self.sort_btn = QPushButton("Sort / Renumber")
        self.jump_btn = QPushButton("Jump to selected")
        self.zoom_btn = QPushButton("Zoom to cursors")

        ctrl_layout.addWidget(self.add_btn, 0, 0)
        ctrl_layout.addWidget(self.update_btn, 0, 1)
        ctrl_layout.addWidget(self.delete_btn, 1, 0)
        ctrl_layout.addWidget(self.jump_btn, 1, 1)
        ctrl_layout.addWidget(self.insert_before_btn, 2, 0)
        ctrl_layout.addWidget(self.insert_after_btn, 2, 1)
        ctrl_layout.addWidget(self.sort_btn, 3, 0)
        ctrl_layout.addWidget(self.zoom_btn, 3, 1)
        self.ctrl_group.setLayout(ctrl_layout)

        # ----- Table -----
        self.table = QTableView()
        self.table.setModel(self.model)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table.setEditTriggers(
            QAbstractItemView.EditTrigger.DoubleClicked
            | QAbstractItemView.EditTrigger.SelectedClicked
            | QAbstractItemView.EditTrigger.EditKeyPressed
        )
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table.verticalHeader().setVisible(False)

        # ----- Right panel -----
        right_layout = QVBoxLayout()
        right_layout.addWidget(self.cursor_group)
        right_layout.addWidget(self.view_group)
        right_layout.addWidget(self.ctrl_group)
        right_layout.addWidget(QLabel("Labeled swim bouts"))
        right_layout.addWidget(self.table, stretch=1)

        right_panel = QWidget()
        right_panel.setLayout(right_layout)

        # ----- Splitter / main -----
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(plot_panel)
        splitter.addWidget(right_panel)
        splitter.setSizes([950, 500])

        main_layout = QVBoxLayout()
        main_layout.addLayout(top_row)
        main_layout.addWidget(splitter, stretch=1)
        self.setLayout(main_layout)

        self._set_empty_state()

    def _build_actions(self):
        self.act_add = QAction("Add bout", self)
        self.act_add.setShortcut(QKeySequence("A"))
        self.addAction(self.act_add)

        self.act_delete = QAction("Delete bout", self)
        self.act_delete.setShortcut(QKeySequence(Qt.Key.Key_Delete))
        self.addAction(self.act_delete)

        self.act_save = QAction("Save Excel", self)
        self.act_save.setShortcut(QKeySequence("S"))
        self.addAction(self.act_save)

    def _connect_signals(self):
        self.browse_btn.clicked.connect(self.on_browse)
        self.load_xlsx_btn.clicked.connect(self.on_load_xlsx)
        self.save_xlsx_btn.clicked.connect(self.on_save_xlsx)

        self.add_btn.clicked.connect(self.on_add_bout)
        self.update_btn.clicked.connect(self.on_update_bout)
        self.delete_btn.clicked.connect(self.on_delete_bout)
        self.insert_before_btn.clicked.connect(lambda: self.on_insert_bout(before=True))
        self.insert_after_btn.clicked.connect(lambda: self.on_insert_bout(before=False))
        self.sort_btn.clicked.connect(self.on_sort_bouts)
        self.jump_btn.clicked.connect(self.on_jump_to_selected)
        self.zoom_btn.clicked.connect(self.zoom_to_cursors)

        self.table.selectionModel().selectionChanged.connect(self.on_table_selection_changed)
        self.table.doubleClicked.connect(self.on_table_double_clicked)

        self.line_start_0.sigPositionChanged.connect(lambda: self._sync_cursor_lines("start", source=0))
        self.line_end_0.sigPositionChanged.connect(lambda: self._sync_cursor_lines("end", source=0))
        self.line_start_1.sigPositionChanged.connect(lambda: self._sync_cursor_lines("start", source=1))
        self.line_end_1.sigPositionChanged.connect(lambda: self._sync_cursor_lines("end", source=1))

        self.start_frame_spin.valueChanged.connect(self.on_spin_frame_changed)
        self.end_frame_spin.valueChanged.connect(self.on_spin_frame_changed)
        self.start_sec_spin.valueChanged.connect(self.on_spin_sec_changed)
        self.end_sec_spin.valueChanged.connect(self.on_spin_sec_changed)

        self.view_start_frame_spin.valueChanged.connect(self.on_view_frame_changed)
        self.view_end_frame_spin.valueChanged.connect(self.on_view_frame_changed)
        self.view_start_sec_spin.valueChanged.connect(self.on_view_sec_changed)
        self.view_end_sec_spin.valueChanged.connect(self.on_view_sec_changed)

        self.apply_view_btn.clicked.connect(self.on_apply_view_range)
        self.reset_view_btn.clicked.connect(self.on_reset_view_range)
        self.use_cursor_as_view_btn.clicked.connect(self.on_use_cursors_as_view)

        self.act_add.triggered.connect(self.on_add_bout)
        self.act_delete.triggered.connect(self.on_delete_bout)
        self.act_save.triggered.connect(self.on_save_xlsx)

    # ---------------- State helpers ----------------

    def _set_empty_state(self):
        for w in [self.start_frame_spin, self.end_frame_spin, self.start_sec_spin, self.end_sec_spin]:
            w.setEnabled(False)

        for w in [
            self.view_start_frame_spin,
            self.view_end_frame_spin,
            self.view_start_sec_spin,
            self.view_end_sec_spin
        ]:
            w.setEnabled(False)

        self.view_group.setEnabled(False)

    def _enable_cursor_controls(self, enabled: bool):
        for w in [self.start_frame_spin, self.end_frame_spin, self.start_sec_spin, self.end_sec_spin]:
            w.setEnabled(enabled)

    def _enable_view_controls(self, enabled: bool):
        self.view_group.setEnabled(enabled)
        for w in [
            self.view_start_frame_spin,
            self.view_end_frame_spin,
            self.view_start_sec_spin,
            self.view_end_sec_spin
        ]:
            w.setEnabled(enabled)

    # ---------------- File loading ----------------

    def on_browse(self):
        dlg = QFileDialog(self, "Open .16chFlt file")
        dlg.setFileMode(QFileDialog.FileMode.ExistingFile)
        dlg.setNameFilters(["16chFlt files (*.16chFlt *.16chflt)", "All files (*)"])
        if dlg.exec():
            files = dlg.selectedFiles()
            if not files:
                return
            path = files[0]
            self.path_edit.setText(path)
            try:
                self.load_file(path)
            except Exception as e:
                QMessageBox.critical(self, "Load error", f"Failed to load file:\n{e}")

    def load_file(self, path: str):
        if self.autosave_timer.isActive():
            self.autosave_timer.stop()
        self.autosave_enabled = False

        data = import16chFlt(path)

        if "fltCh0" not in data or "fltCh1" not in data or "behav_time" not in data:
            raise ValueError("Expected keys 'behav_time', 'fltCh0', and 'fltCh1'.")

        self.file_path = path
        self.data_dict = data
        self.fltCh0 = np.asarray(data["fltCh0"], dtype=np.float64)
        self.fltCh1 = np.asarray(data["fltCh1"], dtype=np.float64)
        self.t = np.asarray(data["behav_time"], dtype=np.float64)

        if self.fltCh0.ndim != 1 or self.fltCh1.ndim != 1 or self.t.ndim != 1:
            raise ValueError("behav_time, fltCh0, and fltCh1 must be 1D arrays.")
        if len(self.fltCh0) != len(self.fltCh1) or len(self.fltCh0) != len(self.t):
            raise ValueError("behav_time, fltCh0, and fltCh1 must have the same length.")

        self.n_frames = len(self.fltCh0)

        self.view_start_frame = 0
        self.view_end_frame = self.n_frames
        self.view_active = True

        self._set_y_ranges()
        self._initialize_cursors()
        self._enable_cursor_controls(True)
        self._enable_view_controls(True)
        self.update_view_widgets_from_state()

        self.model.set_dataframe(BoutTableModel.empty_df())
        self.refresh_visible_data()

        self.setWindowTitle(f"Swim Bout Labeler - {os.path.basename(path)}")

        self.autosave_enabled = True
        self.autosave_timer.start()

    # ---------------- Plot ranges ----------------

    def _set_y_ranges(self):
        if self.fltCh0 is None or self.fltCh1 is None or self.t is None:
            return

        def robust_range(y):
            lim = np.percentile(np.abs(y), 99.5)
            lim = max(lim, 1e-6)
            return -1.1 * lim, 1.1 * lim

        y0min, y0max = robust_range(self.fltCh0)
        y1min, y1max = robust_range(self.fltCh1)

        self.plot0.setLimits(xMin=0, xMax=float(self.t[-1]), yMin=y0min, yMax=y0max)
        self.plot1.setLimits(xMin=0, xMax=float(self.t[-1]), yMin=y1min, yMax=y1max)

        self.plot0.setYRange(y0min, y0max)
        self.plot1.setYRange(y1min, y1max)

    def refresh_visible_data(self):
        if self.t is None or self.fltCh0 is None or self.fltCh1 is None or self.n_frames == 0:
            return

        s = int(np.clip(self.view_start_frame, 0, max(0, self.n_frames - 1)))
        e = int(np.clip(self.view_end_frame, s + 1, self.n_frames))

        t_seg = self.t[s:e]
        y0_seg = self.fltCh0[s:e]
        y1_seg = self.fltCh1[s:e]

        self.curve0.setData(t_seg, y0_seg)
        self.curve1.setData(t_seg, y1_seg)

        if len(t_seg) > 1:
            self.plot0.setLimits(xMin=float(t_seg[0]), xMax=float(t_seg[-1]))
            self.plot1.setLimits(xMin=float(t_seg[0]), xMax=float(t_seg[-1]))
            self.plot0.setXRange(float(t_seg[0]), float(t_seg[-1]), padding=0)
            self.plot1.setXRange(float(t_seg[0]), float(t_seg[-1]), padding=0)

        self.refresh_bout_overlays()

    # ---------------- Cursor / visible range sync ----------------

    def _initialize_cursors(self):
        if self.t is None or self.n_frames == 0:
            return

        start_t = float(self.t[0])
        end_t = min(float(self.t[-1]), start_t + 0.2)

        self._block_cursor_updates(True)
        self.line_start_0.setValue(start_t)
        self.line_end_0.setValue(end_t)
        self.line_start_1.setValue(start_t)
        self.line_end_1.setValue(end_t)
        self._block_cursor_updates(False)

        self.update_cursor_widgets_from_lines()
        self.update_region_from_lines()

    def _block_cursor_updates(self, block: bool):
        for item in [self.line_start_0, self.line_end_0, self.line_start_1, self.line_end_1]:
            item.blockSignals(block)

    def _block_spin_signals(self, block: bool):
        for w in [self.start_frame_spin, self.end_frame_spin, self.start_sec_spin, self.end_sec_spin]:
            w.blockSignals(block)

    def _block_view_spin_signals(self, block: bool):
        for w in [
            self.view_start_frame_spin,
            self.view_end_frame_spin,
            self.view_start_sec_spin,
            self.view_end_sec_spin,
        ]:
            w.blockSignals(block)

    def _sync_cursor_lines(self, which: str, source: int):
        if self.t is None:
            return

        if which == "start":
            val = self.line_start_0.value() if source == 0 else self.line_start_1.value()
            self._block_cursor_updates(True)
            self.line_start_0.setValue(val)
            self.line_start_1.setValue(val)
            self._block_cursor_updates(False)
        else:
            val = self.line_end_0.value() if source == 0 else self.line_end_1.value()
            self._block_cursor_updates(True)
            self.line_end_0.setValue(val)
            self.line_end_1.setValue(val)
            self._block_cursor_updates(False)

        self.update_cursor_widgets_from_lines()
        self.update_region_from_lines()

    def update_region_from_lines(self):
        s = self.line_start_0.value()
        e = self.line_end_0.value()
        lo, hi = sorted([s, e])
        self.region0.setRegion((lo, hi))
        self.region1.setRegion((lo, hi))
        self._update_duration_label()

    def update_cursor_widgets_from_lines(self):
        if self.t is None:
            return

        start_t = float(self.line_start_0.value())
        end_t = float(self.line_end_0.value())

        start_frame = self.time_to_frame(start_t)
        end_frame = self.time_to_frame(end_t)

        self._block_spin_signals(True)
        self.start_frame_spin.setValue(start_frame)
        self.end_frame_spin.setValue(end_frame)
        self.start_sec_spin.setValue(start_frame / self.FS)
        self.end_sec_spin.setValue(end_frame / self.FS)
        self._block_spin_signals(False)

        self._update_duration_label()

    def update_view_widgets_from_state(self):
        self._block_view_spin_signals(True)
        self.view_start_frame_spin.setValue(int(self.view_start_frame))
        self.view_end_frame_spin.setValue(int(self.view_end_frame))
        self.view_start_sec_spin.setValue(self.view_start_frame / self.FS)
        self.view_end_sec_spin.setValue(self.view_end_frame / self.FS)
        self._block_view_spin_signals(False)

    def _update_duration_label(self):
        s = self.start_frame_spin.value()
        e = self.end_frame_spin.value()
        lo, hi = sorted([s, e])
        dur_f = hi - lo
        dur_s = dur_f / self.FS
        self.duration_label.setText(f"Duration: {dur_f} frames | {dur_s:.4f} s")

    def on_spin_frame_changed(self):
        if self.t is None:
            return
        s = self.start_frame_spin.value() / self.FS
        e = self.end_frame_spin.value() / self.FS

        self._block_cursor_updates(True)
        self.line_start_0.setValue(s)
        self.line_start_1.setValue(s)
        self.line_end_0.setValue(e)
        self.line_end_1.setValue(e)
        self._block_cursor_updates(False)

        self._block_spin_signals(True)
        self.start_sec_spin.setValue(self.start_frame_spin.value() / self.FS)
        self.end_sec_spin.setValue(self.end_frame_spin.value() / self.FS)
        self._block_spin_signals(False)

        self.update_region_from_lines()

    def on_spin_sec_changed(self):
        if self.t is None:
            return
        s = self.start_sec_spin.value()
        e = self.end_sec_spin.value()

        self._block_cursor_updates(True)
        self.line_start_0.setValue(s)
        self.line_start_1.setValue(s)
        self.line_end_0.setValue(e)
        self.line_end_1.setValue(e)
        self._block_cursor_updates(False)

        self._block_spin_signals(True)
        self.start_frame_spin.setValue(self.time_to_frame(s))
        self.end_frame_spin.setValue(self.time_to_frame(e))
        self._block_spin_signals(False)

        self.update_region_from_lines()

    def on_view_frame_changed(self):
        if self.t is None:
            return

        self.view_start_frame = self.view_start_frame_spin.value()
        self.view_end_frame = self.view_end_frame_spin.value()

        lo, hi = sorted([self.view_start_frame, self.view_end_frame])
        hi = max(lo + 1, hi)

        self.view_start_frame = lo
        self.view_end_frame = min(hi, self.n_frames)

        self._block_view_spin_signals(True)
        self.view_start_sec_spin.setValue(self.view_start_frame / self.FS)
        self.view_end_sec_spin.setValue(self.view_end_frame / self.FS)
        self._block_view_spin_signals(False)

    def on_view_sec_changed(self):
        if self.t is None:
            return

        s = self.view_start_sec_spin.value()
        e = self.view_end_sec_spin.value()

        lo, hi = sorted([s, e])
        self.view_start_frame = self.time_to_frame(lo)
        self.view_end_frame = min(self.n_frames, self.time_to_frame(hi))
        self.view_end_frame = max(self.view_start_frame + 1, self.view_end_frame)

        self._block_view_spin_signals(True)
        self.view_start_frame_spin.setValue(self.view_start_frame)
        self.view_end_frame_spin.setValue(self.view_end_frame)
        self._block_view_spin_signals(False)

    def on_apply_view_range(self):
        if self.t is None:
            return
        self.refresh_visible_data()

    def on_reset_view_range(self):
        if self.t is None:
            return
        self.view_start_frame = 0
        self.view_end_frame = self.n_frames
        self.update_view_widgets_from_state()
        self.refresh_visible_data()

    def on_use_cursors_as_view(self):
        if self.t is None:
            return

        s = self.start_frame_spin.value()
        e = self.end_frame_spin.value()
        lo, hi = sorted([s, e])

        if hi - lo < 2:
            QMessageBox.information(self, "Range too small", "Cursor range is too small to use as visible range.")
            return

        self.view_start_frame = lo
        self.view_end_frame = hi
        self.update_view_widgets_from_state()
        self.refresh_visible_data()

    def ensure_cursors_in_view(self):
        if self.t is None:
            return

        s = self.start_frame_spin.value()
        e = self.end_frame_spin.value()
        lo, hi = sorted([s, e])

        if lo < self.view_start_frame or hi > self.view_end_frame:
            self.view_start_frame = max(0, lo - int(0.1 * self.FS))
            self.view_end_frame = min(self.n_frames, hi + int(0.1 * self.FS))
            self.update_view_widgets_from_state()
            self.refresh_visible_data()

    # ---------------- Time/frame conversion ----------------

    def time_to_frame(self, t: float) -> int:
        if self.n_frames == 0:
            return 0
        frame = int(round(t * self.FS))
        return int(np.clip(frame, 0, self.n_frames - 1))

    def frame_to_time(self, frame: int) -> float:
        if self.n_frames == 0:
            return 0.0
        frame = int(np.clip(frame, 0, self.n_frames - 1))
        return frame / self.FS

    # ---------------- Bout editing ----------------

    def current_bout_dict_from_cursors(self) -> dict:
        s = self.start_frame_spin.value()
        e = self.end_frame_spin.value()
        start_frame, end_frame = sorted([s, e])

        return {
            "bout_id": 0,
            "start_frame": int(start_frame),
            "end_frame": int(end_frame),
            "start_s": start_frame / self.FS,
            "end_s": end_frame / self.FS,
            "duration_frames": int(end_frame - start_frame),
            "duration_s": (end_frame - start_frame) / self.FS,
            "note": "",
        }

    def selected_row(self) -> Optional[int]:
        sel = self.table.selectionModel().selectedRows()
        if not sel:
            return None
        return sel[0].row()

    def on_add_bout(self):
        if self.t is None:
            return

        row_dict = self.current_bout_dict_from_cursors()
        if row_dict["duration_frames"] <= 0:
            QMessageBox.warning(self, "Invalid range", "Bout duration must be positive.")
            return

        self.model.add_row(row_dict, fs=self.FS)
        self.model.sort_and_reindex(fs=self.FS)
        self.refresh_bout_overlays()
        self.select_last_matching_row(row_dict["start_frame"], row_dict["end_frame"])

    def on_update_bout(self):
        row = self.selected_row()
        if row is None:
            QMessageBox.information(self, "No selection", "Please select a bout to update.")
            return

        row_dict = self.current_bout_dict_from_cursors()
        row_dict["note"] = self.model.dataframe().at[row, "note"]
        self.model.update_row(row, row_dict, fs=self.FS)
        self.model.sort_and_reindex(fs=self.FS)
        self.refresh_bout_overlays()

    def on_delete_bout(self):
        row = self.selected_row()
        if row is None:
            return
        self.model.delete_row(row)
        self.refresh_bout_overlays()

    def on_insert_bout(self, before: bool):
        row = self.selected_row()
        if row is None:
            QMessageBox.information(self, "No selection", "Please select a reference bout.")
            return

        row_dict = self.current_bout_dict_from_cursors()
        if row_dict["duration_frames"] <= 0:
            QMessageBox.warning(self, "Invalid range", "Bout duration must be positive.")
            return

        position = row if before else row + 1
        self.model.add_row(row_dict, position=position, fs=self.FS)
        self.model.sort_and_reindex(fs=self.FS)
        self.refresh_bout_overlays()

    def on_sort_bouts(self):
        self.model.sort_and_reindex(fs=self.FS)
        self.refresh_bout_overlays()

    def select_last_matching_row(self, start_frame: int, end_frame: int):
        df = self.model.dataframe()
        matches = df.index[
            (df["start_frame"] == start_frame) &
            (df["end_frame"] == end_frame)
        ].tolist()
        if matches:
            self.table.selectRow(matches[-1])

    # ---------------- Table interactions ----------------

    def on_table_selection_changed(self):
        row = self.selected_row()
        if row is None:
            return
        self.load_row_into_cursors(row)

    def on_table_double_clicked(self, _index):
        self.on_jump_to_selected()

    def load_row_into_cursors(self, row: int):
        df = self.model.dataframe()
        if not (0 <= row < len(df)):
            return

        s = int(df.at[row, "start_frame"])
        e = int(df.at[row, "end_frame"])

        self._block_spin_signals(True)
        self.start_frame_spin.setValue(s)
        self.end_frame_spin.setValue(e)
        self.start_sec_spin.setValue(s / self.FS)
        self.end_sec_spin.setValue(e / self.FS)
        self._block_spin_signals(False)

        self._block_cursor_updates(True)
        self.line_start_0.setValue(s / self.FS)
        self.line_start_1.setValue(s / self.FS)
        self.line_end_0.setValue(e / self.FS)
        self.line_end_1.setValue(e / self.FS)
        self._block_cursor_updates(False)

        self.update_region_from_lines()
        self.ensure_cursors_in_view()

    def on_jump_to_selected(self):
        row = self.selected_row()
        if row is None:
            QMessageBox.information(self, "No selection", "Please select a bout.")
            return
        self.load_row_into_cursors(row)
        self.zoom_to_cursors()

    def zoom_to_cursors(self):
        if self.t is None:
            return

        s = self.start_sec_spin.value()
        e = self.end_sec_spin.value()
        lo, hi = sorted([s, e])
        pad = max(0.05, 0.25 * (hi - lo))
        xmax = float(self.t[-1])

        self.plot0.setXRange(max(0.0, lo - pad), min(xmax, hi + pad), padding=0)
        self.plot1.setXRange(max(0.0, lo - pad), min(xmax, hi + pad), padding=0)

    # ---------------- Overlay drawing ----------------

    def clear_bout_overlays(self):
        for item in self.bout_regions0:
            self.plot0.removeItem(item)
        for item in self.bout_regions1:
            self.plot1.removeItem(item)

        self.bout_regions0.clear()
        self.bout_regions1.clear()

    def refresh_bout_overlays(self):
        self.clear_bout_overlays()
        df = self.model.dataframe()
        if len(df) == 0 or self.t is None:
            return

        view_lo_s = self.view_start_frame / self.FS
        view_hi_s = self.view_end_frame / self.FS

        for _, row in df.iterrows():
            lo = float(row["start_s"])
            hi = float(row["end_s"])

            if hi < view_lo_s or lo > view_hi_s:
                continue

            reg0 = pg.LinearRegionItem(
                values=[lo, hi],
                orientation="vertical",
                brush=(255, 180, 0, 25),
                pen=pg.mkPen((255, 180, 0, 60)),
                movable=False,
            )
            reg1 = pg.LinearRegionItem(
                values=[lo, hi],
                orientation="vertical",
                brush=(255, 180, 0, 25),
                pen=pg.mkPen((255, 180, 0, 60)),
                movable=False,
            )
            self.plot0.addItem(reg0)
            self.plot1.addItem(reg1)
            self.bout_regions0.append(reg0)
            self.bout_regions1.append(reg1)

    # ---------------- Excel I/O ----------------

    def save_workbook(self, path: str):
        bouts_df = self.model.dataframe().copy()

        onsets, offsets = self.get_bout_arrays()
        arrays_df = pd.DataFrame({
            "start_frame_array": pd.Series(onsets, dtype="Int64"),
            "end_frame_array": pd.Series(offsets, dtype="Int64"),
        })

        meta_df = pd.DataFrame([{
            "source_file": self.file_path if self.file_path is not None else "",
            "fs": self.FS,
            "n_frames": self.n_frames,
            "view_start_frame": self.view_start_frame,
            "view_end_frame": self.view_end_frame,
            "view_start_s": self.view_start_frame / self.FS if self.FS else 0.0,
            "view_end_s": self.view_end_frame / self.FS if self.FS else 0.0,
            "saved_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }])

        with pd.ExcelWriter(path, engine="openpyxl") as writer:
            bouts_df.to_excel(writer, index=False, sheet_name="swim_bouts")
            meta_df.to_excel(writer, index=False, sheet_name="metadata")
            arrays_df.to_excel(writer, index=False, sheet_name="bout_arrays")

    def on_save_xlsx(self):
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save labels as Excel",
            self.default_xlsx_path(),
            "Excel files (*.xlsx)"
        )
        if not path:
            return
        if not path.lower().endswith(".xlsx"):
            path += ".xlsx"

        try:
            self.save_workbook(path)
        except Exception as e:
            QMessageBox.critical(self, "Save error", f"Failed to save Excel file:\n{e}")

    def on_autosave(self):
        if not self.autosave_enabled or self.file_path is None:
            return

        path = self.autosave_xlsx_path()
        try:
            self.save_workbook(path)
            print(f"[Autosave] Saved to: {path}")
        except Exception as e:
            print(f"[Autosave] Failed: {e}")

    def on_load_xlsx(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Load labels from Excel",
            self.default_xlsx_path(),
            "Excel files (*.xlsx)"
        )
        if not path:
            return

        try:
            df = pd.read_excel(path, sheet_name="swim_bouts")
            required = {"start_frame", "end_frame"}
            if not required.issubset(df.columns):
                raise ValueError(f"Excel file must contain columns: {sorted(required)}")

            if "note" not in df.columns:
                df["note"] = ""

            out = pd.DataFrame({
                "bout_id": np.arange(len(df), dtype=int),
                "start_frame": df["start_frame"].astype(int),
                "end_frame": df["end_frame"].astype(int),
                "start_s": 0.0,
                "end_s": 0.0,
                "duration_frames": 0,
                "duration_s": 0.0,
                "note": df["note"].astype(str),
            })
            self.model.set_dataframe(out)
            self.model.sort_and_reindex(fs=self.FS)

            # Restore metadata if present
            try:
                meta = pd.read_excel(path, sheet_name="metadata")
                if len(meta) > 0 and self.n_frames > 0:
                    view_start = int(meta.at[0, "view_start_frame"])
                    view_end = int(meta.at[0, "view_end_frame"])
                    view_start = max(0, min(view_start, self.n_frames - 1))
                    view_end = max(view_start + 1, min(view_end, self.n_frames))
                    self.view_start_frame = view_start
                    self.view_end_frame = view_end
                    self.update_view_widgets_from_state()
                    self.refresh_visible_data()
                else:
                    self.refresh_bout_overlays()
            except Exception:
                self.refresh_bout_overlays()

        except Exception as e:
            QMessageBox.critical(self, "Load error", f"Failed to load Excel file:\n{e}")

    # ---------------- Close ----------------

    def closeEvent(self, event):
        try:
            if self.autosave_timer.isActive():
                self.autosave_timer.stop()

            if self.autosave_enabled and self.file_path is not None:
                self.save_workbook(self.autosave_xlsx_path())
        except Exception as e:
            print(f"[Close autosave] Failed: {e}")

        super().closeEvent(event)


def main():
    app = QApplication(sys.argv)
    w = SwimBoutLabeler()
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()