import sys
import os
import numpy as np

from PyQt6.QtWidgets import (
    QApplication, QWidget, QHBoxLayout, QVBoxLayout, QPushButton, QLineEdit,
    QFileDialog, QLabel, QSpinBox, QSlider, QMessageBox, QComboBox
)
from PyQt6.QtCore import Qt
import pyqtgraph as pg
        
def import10ch(filename: str) -> dict:
    """ 
    Imports *.10ch or *.10chFlt file and parses it into Excel-like
    data formats (Dictionary). Dictionary can be easily converted to
    pandas DataFrame for further analysis.

    Parameters
    ----------
    filename : str
        The name of the file to import.

    Returns
    -------
    data : dict
        The imported data.\\
        For *.10ch file, the data dictionary contains the following keys:
        - 't': time vector
        - 'ch0': channel 0 data
        - 'ch1': channel 1 data
        - 'fltCh0': filtered channel 0 data
        - 'fltCh1': filtered channel 1 data
        - 'camTrigger': camera trigger signal
        - '2pTrigger': two-photon trigger signal
        - 'drift': drift signal
        - 'speed': speed signal
        - 'gain': gain signal
        - 'temp': ?
        \\
        For *.10chFlt files, the data dictionary contains the following keys:
        - 't': time vector
        - 'ch0': channel 0 data
        - 'ch1': channel 1 data
        - 'fltCh0': filtered channel 0 data
        - 'fltCh1': filtered channel 1 data
        - 'camTrigger': camera trigger signal
        - 'drift': drift signal
        - 'gain': gain signal

    Note
    ----
    10ch files contain 10 channels of data, with each channel represented
    some specific parameters of the experiment (e.g., time, channel 0,
    channel 1, gain, drift, speed etc.)

    f = open(filename, 'rb')
    A =  np.fromfile(f, np.float32).reshape((-1,10)).T
    f.close()
    """
    with open(filename, 'rb') as f:
        A = np.fromfile(f, np.float32).reshape((-1, 10)).T
    
    if filename.endswith('.10ch'):
        data = {}
        # Create a Gaussian kernel for smoothing with sigma = 20
        ker = np.exp(-np.arange(-60, 61)**2 / (2 * 20**2.))
        ker /= np.sum(ker)
        ch1 = A[0,:]
        smch1 = np.convolve(ch1, ker, mode='same')
        pow1 = (ch1 - smch1)**2
        ch2 = A[1, :]
        smch2 = np.convolve(ch2, ker, mode='same')
        pow2 = (ch2 - smch2)**2    
        data['t'] = np.arange(1, A.shape[1] + 1) / 6000
        data['ch0'] = ch1
        data['ch1'] = ch2
        data['fltCh0'] = np.convolve(pow1, ker, mode='same')
        data['fltCh1'] = np.convolve(pow2, ker, mode='same')
        data['gain'] = A[4, :]
        data['drift'] = A[5, :]
        data['speed'] = A[6, :]
        data['camTrigger'] = A[7, :]
        data['2pTrigger'] = A[8, :]
        data['temp'] = A[9, :]
        
    elif filename.endswith('.10chFlt'):
        data = {}   
        # Create a Gaussian kernel for smoothing with sigma = 20
        ker = np.exp(-np.arange(-60, 61)**2 / (2 * 20**2.))
        ker /= np.sum(ker)
        ch1 = A[0, :]
        smch1 = np.convolve(ch1, ker, mode='same')
        pow1 = (ch1 - smch1)**2
        ch2 = A[1, :]
        smch2 = np.convolve(ch2, ker, mode='same')
        pow2 = (ch2 - smch2)**2    
        data['t'] = np.arange(1, A.shape[1] + 1) / 6000
        data['ch0'] = ch1
        data['ch1'] = ch2
        data['fltCh0'] = np.convolve(pow1, ker, mode='same')
        data['fltCh1'] = np.convolve(pow2, ker, mode='same')
        data['camTrigger'] = A[2, :]
        data['drift'] = A[6, :]
        data['gain'] = A[9, :]
        
    return data

class SimpleViewer(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Two-Channel Signal Viewer")
        self.resize(1000, 600)

        # ---------- Top bar ----------
        self.path_edit = QLineEdit()
        self.path_edit.setPlaceholderText("Select a .10ch or .10chFlt file...")
        self.browse_btn = QPushButton("Browse…")
        self.browse_btn.clicked.connect(self.on_browse)

        top_row = QHBoxLayout()
        top_row.addWidget(self.path_edit, stretch=1)
        top_row.addWidget(self.browse_btn)

        # ---------- Channel selection ----------
        self.chanA_label = QLabel("Signal A:")
        self.chanA_combo = QComboBox()
        self.chanB_label = QLabel("Signal B:")
        self.chanB_combo = QComboBox()
        self.chanC_label = QLabel("Signal C:")
        self.chanC_combo = QComboBox()

        self.chanA_combo.currentTextChanged.connect(self.update_plot)
        self.chanB_combo.currentTextChanged.connect(self.update_plot)
        self.chanC_combo.currentTextChanged.connect(self.update_plot)

        chan_row = QHBoxLayout()
        chan_row.addWidget(self.chanA_label)
        chan_row.addWidget(self.chanA_combo)
        chan_row.addSpacing(15)
        chan_row.addWidget(self.chanB_label)
        chan_row.addWidget(self.chanB_combo)
        chan_row.addSpacing(15)
        chan_row.addWidget(self.chanC_label)
        chan_row.addWidget(self.chanC_combo)
        chan_row.addStretch(1)

        # ---------- Plot ----------
        pg.setConfigOptions(background='w', useNumba=True, leftButtonPan=False)
        self.plot_widget = pg.PlotWidget()
        #self.plot_widget.showGrid(x=True, y=True, alpha=0.3)
        self.plot_widget.setLabel('bottom', 'Time (s)')
        self.plot_widget.addLegend(offset=(10, 10))

        self.curveA = self.plot_widget.plot(pen=pg.mkPen(width=1, color='#d56e9e'), name="A")
        self.curveB = self.plot_widget.plot(pen=pg.mkPen(width=1, color='#3c619a'), name="B")
        self.curveC = self.plot_widget.plot(pen=pg.mkPen(width=1, color='#6eb2a6'), name="C")
        self.plot_widget.setLimits(
            yMin=-2,          # pick sensible limits for Y
            yMax=2
        )

        # ---------- Layout ----------
        main = QVBoxLayout()
        main.addLayout(top_row)
        main.addLayout(chan_row)
        main.addWidget(self.plot_widget, stretch=1)
        self.setLayout(main)

        # Internal state
        self.data_dict = None
        self.current_ext = ""

    # ---------------- File load ----------------
    def on_browse(self):
        dlg = QFileDialog(self, "Open 10ch/10chFlt/12chFlt/16chFlt file")
        dlg.setFileMode(QFileDialog.FileMode.ExistingFile)
        dlg.setNameFilters(["10ch files (*.10ch *.10chFlt)", "12chFlt files (*.12chFlt)", "16chFlt files (*.16chFlt)"])
        # Set default directory to D:\EnData\Test\Pre-Reward
        dlg.setDirectory(r"D:/EnData/Reward")
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
        ext = os.path.splitext(path)[1].lower()
        if ext not in (".10ch", ".10chflt", ".12chflt", ".16chflt"):
            raise ValueError("Only .10ch, .10chFlt, .12chFlt, .16chFlt files are supported.")
        self.current_ext = ext

        if ext == ".12chflt":
            from zfish._io import import12chFlt
            data = import12chFlt(path)
        elif ext == ".16chflt":
            from zfish._io import import16chFlt
            data = import16chFlt(path)
        else:
            data = import10ch(path)
        if not isinstance(data, dict):
            raise ValueError("import10ch must return a dict of channel_name -> np.ndarray.")

        # Keep numeric 1D channels
        self.data_dict = {k: v for k, v in data.items()
                          if isinstance(v, np.ndarray) and v.ndim == 1 and np.issubdtype(v.dtype, np.number)}
        if len(self.data_dict) < 3:
            raise ValueError("Need at least 3 numeric 1D channels.")

        # Populate channel selectors
        names = sorted(self.data_dict.keys())
        self.chanA_combo.blockSignals(True)
        self.chanB_combo.blockSignals(True)
        self.chanC_combo.blockSignals(True)
        self.chanA_combo.clear()
        self.chanB_combo.clear()
        self.chanC_combo.clear()
        self.chanA_combo.addItems(names)
        self.chanB_combo.addItems(names)
        self.chanC_combo.addItems(names)

        # Defaults
        if self.current_ext == ".10ch":
            defA, defB, defC = "ch0", "ch1", "drift"
        else:
            defA, defB, defC = "fltCh0", "fltCh1", "drift"

        self.chanA_combo.setCurrentText(defA if defA in self.data_dict.keys() else names[0])
        self.chanB_combo.setCurrentText(defB if defB in self.data_dict.keys() else names[min(1, len(names)-1)])
        self.chanC_combo.setCurrentText(defC if defC in self.data_dict.keys() else names[min(2, len(names)-1)])
        self.chanA_combo.blockSignals(False)
        self.chanB_combo.blockSignals(False)
        self.chanC_combo.blockSignals(False)

        self.update_plot()

    def update_plot(self):
        if self.data_dict is None:
            self.curveA.setData([])
            self.curveB.setData([])
            return

        chanA = self.chanA_combo.currentText()
        chanB = self.chanB_combo.currentText()
        chanC = self.chanC_combo.currentText()
        if chanA not in self.data_dict or chanB not in self.data_dict or chanC not in self.data_dict:
            return

        try:
            t = self.data_dict['t']
        except:
            t = self.data_dict['behav_time']
        yA = self.data_dict[chanA]
        yB = self.data_dict[chanB]
        
        lim = max(np.percentile(np.abs(yB), 99), np.percentile(np.abs(yA), 99))
        
        yC = self.data_dict[chanC]
        yC = yC / np.max(np.abs(yC)) * lim

        self.curveA.setData(t, yA, name=f"A: {chanA}")
        self.curveB.setData(t, -yB, name=f"B: {chanB}")
        self.curveC.setData(t, yC, name=f"C: {chanC}")
        ymax = max(np.max(yA)*1.1, np.max(yB)*1.1, np.max(yC)*1.1)
        ymin = - ymax
        self.plot_widget.setLimits(
            xMin=-5,
            xMax=max(t)+5,        # or len of your signals
            yMin=ymin,          # pick sensible limits for Y
            yMax=ymax
        )


def main():
    app = QApplication(sys.argv)
    w = SimpleViewer()
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()