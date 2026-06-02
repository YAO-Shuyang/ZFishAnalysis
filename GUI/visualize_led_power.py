import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QPushButton, QLabel,
    QFileDialog, QVBoxLayout, QHBoxLayout, QMessageBox, QComboBox
)
from PyQt6.QtCore import Qt

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas

from zfish._io import import16chFlt


class SingleFltAnalyzer:
    """
    Analyze one .16chFlt file using the same logic as the original batch script.
    """

    def __init__(
        self,
        n_pos_bins: int = 45,
        pos_range=(0, 100),
        behav_time_min: float = 5,
        pos_y_min: float = 0,
        pos_y_max: float = 55,
        max_trials: int = 150,
    ):
        self.n_pos_bins = n_pos_bins
        self.pos_range = pos_range
        self.behav_time_min = behav_time_min
        self.pos_y_min = pos_y_min
        self.pos_y_max = pos_y_max
        self.max_trials = max_trials

        self.led_powers = [0, 0.2, 0.5, 0.8, 1.5, 3.0, 5.0]

    def analyze(self, file_path: str, mode: str = "density") -> pd.DataFrame:
        """
        Parameters
        ----------
        file_path : str
            Path to one .16chFlt file.

        mode : {"density", "counts"}
            - "density": reproduces your original np.histogram(..., density=True).
            - "counts": shows raw sample counts per position bin.

        Returns
        -------
        pd.DataFrame
            Columns: LED Power, Pos, Time, Label
        """
        res = import16chFlt(file_path)

        required_keys = ["behav_time", "behav_pos_y", "n_trials"]
        for key in required_keys:
            if key not in res:
                raise KeyError(f"Missing required key in .16chFlt file: {key}")

        idx = np.where(
            (res["behav_time"] > self.behav_time_min) &
            (res["behav_pos_y"] >= self.pos_y_min) &
            (res["behav_pos_y"] < self.pos_y_max) &
            (res["n_trials"] < self.max_trials)
        )[0]

        for key in res.keys():
            res[key] = res[key][idx]

        # Convert original 0–55 coordinate to approximately 0–110,
        # matching your original behavior: res['behav_pos_y'] = res['behav_pos_y'] * 2
        res["behav_pos_y"] = res["behav_pos_y"] * 2

        # Position bin centers
        bin_edges = np.linspace(self.pos_range[0], self.pos_range[1], self.n_pos_bins + 1)
        x = (bin_edges[:-1] + bin_edges[1:]) / 2

        data = {
            "LED Power": [],
            "Pos": [],
            "Time": [],
            "Label": [],
        }

        # Your original logic:
        # i = 0,1   -> LED 0:   stim, test
        # i = 2,3   -> LED 0.2: stim, test
        # ...
        # i = 12,13 -> LED 5.0: stim, test
        for i in range(14):
            trial_idx = np.where(
                (res["n_trials"] >= i * 5) &
                (res["n_trials"] < (i + 1) * 5)
            )[0]

            led_power = self.led_powers[i // 2]
            label = ["stim", "test"][i % 2]

            density = True if mode == "density" else False

            hist = np.histogram(
                res["behav_pos_y"][trial_idx],
                bins=bin_edges,
                density=density
            )[0]

            data["LED Power"].append(np.repeat(led_power, len(hist)))
            data["Pos"].append(x)
            data["Time"].append(hist)
            data["Label"].append(np.repeat(label, len(hist)))

        for key in data:
            data[key] = np.concatenate(data[key])

        return pd.DataFrame(data)


class FltAnalysisCanvas(FigureCanvas):
    """
    Matplotlib canvas embedded in PyQt6.
    """

    def __init__(self):
        self.fig, self.axes = plt.subplots(
            nrows=7,
            ncols=1,
            figsize=(6, 9),
            sharex=True
        )
        super().__init__(self.fig)

        self.led_powers = [0, 0.2, 0.5, 0.8, 1.5, 3.0, 5.0]

    def clear_axes_style(self, ax, bottom_visible: bool):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        if not bottom_visible:
            ax.spines["bottom"].set_visible(False)
            ax.tick_params(axis="x", bottom=False, labelbottom=False)

        ax.tick_params(axis="both", direction="out", length=3, width=0.8)

    def plot_result(self, df: pd.DataFrame, ylabel: str = "Time Spent"):
        self.fig.clear()

        self.axes = self.fig.subplots(
            nrows=7,
            ncols=1,
            sharex=True
        )

        # Similar to seaborn flare-like progression, but manually defined
        colors = [
            "#f5c6a5",
            "#eea27f",
            "#df7b66",
            "#c85a5a",
            "#a83f55",
            "#7f2f4c",
            "#54233f",
        ]

        test_color = "#c9caca"

        for i, ax in enumerate(self.axes):
            led_power = self.led_powers[i]

            stim_idx = (
                np.isclose(df["LED Power"], led_power) &
                (df["Label"] == "stim")
            )
            test_idx = (
                np.isclose(df["LED Power"], led_power) &
                (df["Label"] == "test")
            )

            ax.plot(
                df.loc[test_idx, "Pos"],
                df.loc[test_idx, "Time"],
                color=test_color,
                linewidth=1.0,
                label="test"
            )

            ax.plot(
                df.loc[stim_idx, "Pos"],
                df.loc[stim_idx, "Time"],
                color=colors[i],
                linewidth=1.2,
                label="stim"
            )

            # Same blue region as your original code:
            # 500/9 ≈ 55.56, 1300/18 ≈ 72.22
            ax.axvspan(
                500 / 9,
                1300 / 18,
                color="#a9dcf5",
                alpha=0.4,
                linewidth=0
            )

            ax.text(
                0.98,
                0.78,
                f"{led_power} mW",
                transform=ax.transAxes,
                ha="right",
                va="center",
                fontsize=9
            )

            self.clear_axes_style(ax, bottom_visible=(i == 6))

            if i == 3:
                ax.set_ylabel(ylabel)
            else:
                ax.set_ylabel("")

            if i == 0:
                ax.legend(
                    frameon=False,
                    loc="upper left",
                    fontsize=8,
                    ncol=2
                )

        self.axes[-1].set_xlabel("Position (%)")

        self.fig.tight_layout()
        self.draw()

    def save_figure(self, save_path: str):
        self.fig.savefig(save_path, dpi=300, bbox_inches="tight")


class FltAnalyzerGUI(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle(".16chFlt Single-File Analyzer")
        self.resize(900, 900)

        self.file_path = None
        self.result_df = None

        self.analyzer = SingleFltAnalyzer()

        self._init_ui()

    def _init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        main_layout = QVBoxLayout(central_widget)

        control_layout = QHBoxLayout()

        self.file_label = QLabel("No file selected")
        self.file_label.setAlignment(Qt.AlignmentFlag.AlignLeft)

        self.upload_button = QPushButton("Upload .16chFlt")
        self.upload_button.clicked.connect(self.upload_file)

        self.analyze_button = QPushButton("Analyze and Plot")
        self.analyze_button.clicked.connect(self.analyze_and_plot)
        self.analyze_button.setEnabled(False)

        self.save_table_button = QPushButton("Save Table")
        self.save_table_button.clicked.connect(self.save_table)
        self.save_table_button.setEnabled(False)

        self.save_fig_button = QPushButton("Save Figure")
        self.save_fig_button.clicked.connect(self.save_figure)
        self.save_fig_button.setEnabled(False)

        self.mode_box = QComboBox()
        self.mode_box.addItems(["density", "counts"])
        self.mode_box.setToolTip(
            "density reproduces your original code; counts shows raw sample counts."
        )

        control_layout.addWidget(self.upload_button)
        control_layout.addWidget(QLabel("Mode:"))
        control_layout.addWidget(self.mode_box)
        control_layout.addWidget(self.analyze_button)
        control_layout.addWidget(self.save_table_button)
        control_layout.addWidget(self.save_fig_button)

        main_layout.addLayout(control_layout)
        main_layout.addWidget(self.file_label)

        self.canvas = FltAnalysisCanvas()
        main_layout.addWidget(self.canvas)

    def upload_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select .16chFlt file",
            "",
            "16chFlt files (*.16chFlt);;All files (*)"
        )

        if file_path:
            self.file_path = file_path
            self.file_label.setText(f"Selected file: {file_path}")
            self.analyze_button.setEnabled(True)

    def analyze_and_plot(self):
        if self.file_path is None:
            QMessageBox.warning(self, "No file", "Please upload a .16chFlt file first.")
            return

        try:
            mode = self.mode_box.currentText()
            self.result_df = self.analyzer.analyze(self.file_path, mode=mode)

            ylabel = "Density" if mode == "density" else "Sample Count"
            self.canvas.plot_result(self.result_df, ylabel=ylabel)

            self.save_table_button.setEnabled(True)
            self.save_fig_button.setEnabled(True)

        except Exception as e:
            QMessageBox.critical(
                self,
                "Analysis failed",
                f"An error occurred during analysis:\n\n{repr(e)}"
            )

    def save_table(self):
        if self.result_df is None:
            return

        save_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save analyzed table",
            "single_file_analysis.xlsx",
            "Excel file (*.xlsx);;CSV file (*.csv);;Pickle file (*.pkl)"
        )

        if not save_path:
            return

        try:
            suffix = Path(save_path).suffix.lower()

            if suffix == ".xlsx":
                self.result_df.to_excel(save_path, index=False)
            elif suffix == ".csv":
                self.result_df.to_csv(save_path, index=False)
            elif suffix == ".pkl":
                with open(save_path, "wb") as f:
                    pickle.dump(self.result_df, f)
            else:
                raise ValueError("Unsupported file format. Use .xlsx, .csv, or .pkl.")

            QMessageBox.information(self, "Saved", f"Table saved to:\n{save_path}")

        except Exception as e:
            QMessageBox.critical(
                self,
                "Save failed",
                f"Could not save table:\n\n{repr(e)}"
            )

    def save_figure(self):
        if self.result_df is None:
            return

        save_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save figure",
            "single_file_time_spent_by_pos.png",
            "PNG file (*.png);;SVG file (*.svg);;PDF file (*.pdf)"
        )

        if not save_path:
            return

        try:
            self.canvas.save_figure(save_path)
            QMessageBox.information(self, "Saved", f"Figure saved to:\n{save_path}")

        except Exception as e:
            QMessageBox.critical(
                self,
                "Save failed",
                f"Could not save figure:\n\n{repr(e)}"
            )


if __name__ == "__main__":
    import sys

    app = QApplication(sys.argv)
    window = FltAnalyzerGUI()
    window.show()
    sys.exit(app.exec())