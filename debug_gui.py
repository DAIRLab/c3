import sys
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QMessageBox,
    QListWidgetItem, QVBoxLayout, QHBoxLayout, QGridLayout,
    QWidget, QLabel
)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt5 import uic

class PlotApp(QMainWindow):
    def __init__(self, data):
        super().__init__()
        uic.loadUi("debug.ui", self)

        self.data = data

        # Update button text
        self.pushButtonPlot.setText("Plot Selected")

        # Populate the list widget with variable names
        self.listWidget.addItems(data.keys())
        self.listWidget.setSelectionMode(self.listWidget.MultiSelection)
        self.listWidget.selectedItems()

        # Setup matplotlib figure
        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        self.verticalLayout.insertWidget(0, self.canvas)

        # Setup signals
        self.pushButtonPlot.clicked.connect(self.plot_selected)
        self.listWidget.itemSelectionChanged.connect(self.update_dimension_controls)
        self.listWidget.itemSelectionChanged.connect(self.update_spin_limits_from_selection)

        
    
    def update_spin_limits_from_selection(self):
        selected_items = self.listWidget.selectedItems()
        if not selected_items:
            return

        lengths = []
        for item in selected_items:
            var_name = item.text()
            array = self.data[var_name]
            if array.ndim < 1:
                QMessageBox.warning(self, "Invalid Data", f"{var_name} has no axis-0 (e.g., shape={array.shape})")
                return
            lengths.append(array.shape[0])

        if len(set(lengths)) != 1:
            QMessageBox.warning(self, "Shape Mismatch",
                                f"Selected variables have different axis-0 lengths: {lengths}")
            return

        N = lengths[0]
        self.spinStart.setMinimum(0)
        self.spinStart.setMaximum(N - 1)
        self.spinEnd.setMinimum(1)
        self.spinEnd.setMaximum(N)
        self.spinStart.setValue(0)
        self.spinEnd.setValue(N)

    def update_dimension_controls(self):
        items = self.listWidget.selectedItems()
        if not items:
            return
        name = items[0].text()
        shape = self.data[name].shape
        if len(shape) == 2:
            self.comboDimension.setEnabled(False)
            self.spinStart.setEnabled(False)
            self.spinEnd.setEnabled(False)
        elif len(shape) == 3:
            self.comboDimension.setEnabled(True)
            self.spinStart.setEnabled(True)
            self.spinEnd.setEnabled(True)
            self.comboDimension.clear()
            self.comboDimension.addItems(["0", "1", "2"])
            dim = int(self.comboDimension.currentText())
            self.spinStart.setMaximum(shape[dim] - 1)
            self.spinEnd.setMaximum(shape[dim] - 1)

    def plot_selected(self):
        selected_items = self.listWidget.selectedItems()
        print(selected_items)
        if not selected_items:
            QMessageBox.warning(self, "No Selection", "Please select at least one variable.")
            return

        start_idx = self.spinStart.value()
        end_idx = self.spinEnd.value()

        self.figure.clear()
        ax = self.figure.add_subplot(111)

        for item in selected_items:
            
            name = item.text()
            arr = self.data[name]

            # Plot flattened vectors or per-column if 2D or 3D
            sliced = arr[start_idx:end_idx]

            if sliced.ndim == 1:
                ax.plot(np.arange(start_idx, end_idx), sliced, label=name)
            elif sliced.ndim == 2:
                for i in range(sliced.shape[1]):
                    ax.plot(np.arange(start_idx, end_idx), sliced[:, i], label=f"{name}[{i}]")
            elif sliced.ndim == 3:
                flat = sliced.reshape(sliced.shape[0], -1)
                for i in range(flat.shape[1]):
                    ax.plot(np.arange(start_idx, end_idx), flat[:, i], label=f"{name}[{i}]")
            else:
                print(f"Unsupported shape for variable {name}: {arr.shape}")

        ax.legend()
        ax.set_title("Selected Variables")
        ax.set_xlabel("Time (samples)")
        ax.set_ylabel("Value")
        self.canvas.draw()

if __name__ == "__main__":
    app = QApplication(sys.argv)

    # Sample data
    N, M, H = 10, 20, 5
    x_sol = np.load("/home/yufeiyang/Documents/c3/debug_output/finger_x.npy")
    delta_sol = np.load("/home/yufeiyang/Documents/c3/debug_output/finger_delta.npy")
    u_sol = np.load("/home/yufeiyang/Documents/c3/debug_output/finger_u.npy")
    # lambda_gamma1 = np.stack([delta_sol[:, 0, 6], delta_sol[:, 0, 12]], axis=0)
    
    data = {
        "x_sol": x_sol.T,
        "delta_sol": delta_sol,
        "u_sol": u_sol,
        # "lambda_gamma1": lambda_gamma1,
    }

    win = PlotApp(data)
    win.setWindowTitle("Data Viewer")
    win.show()
    sys.exit(app.exec_())
