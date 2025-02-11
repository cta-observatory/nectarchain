import argparse
import os
import pickle
import sys
import tempfile

from matplotlib.backends.backend_qt5 import NavigationToolbar2QT as NavigationToolbar
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

# from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt5.QtCore import QProcess, QTimer
from PyQt5.QtWidgets import (
    QApplication,
    QComboBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QSpacerItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
    QWidgetItem,
)

import nectarchain.trr_test_suite.deadtime as deadtime
import nectarchain.trr_test_suite.linearity as linearity
import nectarchain.trr_test_suite.pedestal as pedestal
import nectarchain.trr_test_suite.pix_tim_uncertainty as pix_tim_uncertainty
import nectarchain.trr_test_suite.trigger_timing as trigger_timing
from nectarchain.trr_test_suite import (
    pix_couple_tim_uncertainty as pix_couple_tim_uncertainty,
)

# Ensure the src directory is in sys.path
test_dir = os.path.abspath("src")
if test_dir not in sys.path:
    sys.path.append(test_dir)

# Import test modules


class TestRunner(QWidget):
    """The ``TestRunner`` class is a GUI application that allows the\
        user to run various tests and display the results.
    The class provides the following functionality:
    - Allows the user to select a test from a dropdown menu.
    - Dynamically generates input fields based on the selected test.
    - Runs the selected test and displays the output in a text box.
    - Displays the test results in a plot canvas, with navigation buttons\
        to switch between multiple plots.
    - Provides a dark-themed UI with custom styling for various UI elements.
    The class uses the PyQt5 library for the GUI implementation and the Matplotlib\
        library for plotting the test results.
    """

    test_modules = {
        "Linearity Test": linearity,
        "Deadtime Test": deadtime,
        "Pedestal Test": pedestal,
        "Pixel Time Uncertainty Test": pix_tim_uncertainty,
        "Time Uncertainty Between Couples of Pixels": pix_couple_tim_uncertainty,
        "Trigger Timing Test": trigger_timing,
    }

    def __init__(self):
        super().__init__()
        self.params = {}
        self.process = None
        self.plot_files = []  # Store the list of plot files
        self.current_plot_index = 0  # Index to track which plot is being displayed
        self.figure = Figure(figsize=(8, 6))
        self.init_ui()

    def init_ui(self):
        # Main layout: vertical, dividing into two sections (top for controls/plot
        # , bottom for output)
        main_layout = QVBoxLayout()

        self.setStyleSheet(
            """
            QWidget {
                background-color: #2e2e2e;  /* Dark background */
                color: #ffffff;  /* Light text */
            }
            QLabel {
                font-weight: bold;
                color: #ffffff;  /* Light text */
            }
            QComboBox {
                background-color: #444444;  /* Dark combo box */
                color: #ffffff;  /* Light text */
                border: 1px solid #888888;  /* Light border */
                min-width: 200px;  /* Set a minimum width */
            }
            QLineEdit {
                background-color: #444444;  /* Dark input field */
                color: #ffffff;  /* Light text */
                border: 1px solid #888888;  /* Light border */
                padding: 5px;  /* Add padding */
                min-width: 200px;  /* Fixed width */
            }
            QTextEdit {
                background-color: #1e1e1e;  /* Dark output box */
                color: #ffffff;  /* Light text */
                border: 1px solid #888888;  /* Light border */
                padding: 5px;  /* Add padding */
                min-width: 800px;  /* Set a minimum width to match the canvas */
            }
            QTextEdit:focus {
                border: 1px solid #00ff00;  /* Green border on focus for visibility */
            }
            QPushButton {
                background-color: #4caf50;  /* Green button */
                color: white;  /* White text */
                border: none;  /* No border */
                padding: 10px;  /* Add padding */
                border-radius: 5px;  /* Rounded corners */
            }
            QPushButton:disabled {
                background-color: rgba(76, 175, 80, 0.5);  /* Transparent green when\
                    disabled */
                color: rgba(255, 255, 255, 0.5);  /* Light text when disabled */
            }
            QPushButton:hover {
                background-color: #45a049;  /* Darker green on hover */
            }
            """
        )

        # Horizontal layout for test options (left) and plot canvas (right)
        top_layout = QHBoxLayout()

        # Create a QGroupBox for controls
        controls_box = QGroupBox("Test Controls", self)
        controls_box.setFixedHeight(600)
        controls_layout = QVBoxLayout()  # Layout for the controls

        # Dropdown for selecting the test
        self.label = QLabel("Select Test:", self)
        self.label.setFixedSize(100, 20)
        controls_layout.addWidget(self.label)

        self.test_selector = QComboBox(self)
        self.test_selector.addItem("Select Test")
        self.test_selector.addItems(
            [
                "Linearity Test",
                "Deadtime Test",
                "Pedestal Test",
                "Pixel Time Uncertainty Test",
                "Time Uncertainty Between Couples of Pixels",
                "Trigger Timing Test",
            ]
        )
        self.test_selector.setFixedWidth(400)  # Fixed width for the dropdown
        self.test_selector.currentIndexChanged.connect(self.update_parameters)
        controls_layout.addWidget(self.test_selector)

        # Container for parameter fields
        self.param_widgets = QWidget(self)
        self.param_widgets.setFixedSize(400, 400)
        self.param_layout = QVBoxLayout(self.param_widgets)
        controls_layout.addWidget(self.param_widgets)

        # Button to run the test
        self.run_button = QPushButton("Run Test", self)
        # Disable the run button initially
        self.run_button.setEnabled(False)
        self.run_button.clicked.connect(self.run_test)
        controls_layout.addWidget(self.run_button)

        # Set the controls layout to the group box
        controls_box.setLayout(controls_layout)

        # Add the controls box to the top layout (left side)
        top_layout.addWidget(controls_box)

        # Add a stretchable spacer to push the canvas to the right
        top_layout.addSpacerItem(
            QSpacerItem(
                40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum
            )
        )

        # Create a vertical layout for the plot container
        self.plot_container = QWidget(self)
        self.plot_layout = QVBoxLayout(self.plot_container)

        # Fixed size for the container
        self.plot_container.setFixedSize(
            800, 650
        )  # Set desired fixed size for the container

        # Create a vertical layout for the plot (canvas and toolbar)
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setFixedSize(800, 600)  # Fixed size for the canvas

        # Add toolbar for zooming and panning
        self.toolbar = NavigationToolbar(self.canvas, self)
        self.toolbar.setFixedHeight(50)  # Fixed height for the toolbar

        # Add the toolbar and canvas to the plot layout
        self.plot_layout.addWidget(self.toolbar)  # Toolbar stays on top
        self.plot_layout.addWidget(self.canvas)  # Canvas below toolbar

        # Add the plot container to the top layout (to the right)
        top_layout.addWidget(self.plot_container)

        # Add the top layout (controls + canvas) to the main layout
        main_layout.addLayout(top_layout)

        # Navigation buttons
        nav_layout = QHBoxLayout()
        self.prev_button = QPushButton("Previous Plot", self)
        self.prev_button.clicked.connect(self.show_previous_plot)
        self.prev_button.setEnabled(False)  # Initially disabled
        nav_layout.addWidget(self.prev_button)

        self.next_button = QPushButton("Next Plot", self)
        self.next_button.clicked.connect(self.show_next_plot)
        self.next_button.setEnabled(False)  # Initially disabled
        nav_layout.addWidget(self.next_button)

        main_layout.addLayout(nav_layout)

        # Output text box (bottom section of the main layout)
        self.output_text_edit = QTextEdit(self)
        self.output_text_edit.setReadOnly(True)  # To prevent user editing
        self.output_text_edit.setFixedHeight(
            150
        )  # Set a fixed height for the output box
        self.output_text_edit.setMinimumWidth(
            800
        )  # Set a minimum width to match the canvas
        main_layout.addWidget(self.output_text_edit)

        # Set the main layout to the window
        self.setLayout(main_layout)
        self.setWindowTitle("Test Runner GUI")
        self.showFullScreen()

    def get_parameters_from_module(self, module):
        # Fetch parameters from the module
        if hasattr(module, "get_args"):
            parser = module.get_args()
            params = {}
            for arg in parser._actions:
                if isinstance(arg, argparse._StoreAction):
                    params[arg.dest] = {
                        "default": arg.default,
                        "help": arg.help,  # Store the help text
                    }
            return params
        else:
            raise RuntimeError("No get_args function found in module.")

    def debug_layout(self):
        for i in range(self.param_layout.count()):
            item = self.param_layout.itemAt(i)
            widget = item.widget()
            if widget:
                print(f"Widget in layout: {widget.objectName()}")

    def update_parameters(self):
        # Clear existing parameter fields
        for i in reversed(range(self.param_layout.count())):
            item = self.param_layout.itemAt(i)

            if isinstance(
                item, QHBoxLayout
            ):  # Check if the item is a QHBoxLayout (contains label and help button)
                for j in reversed(range(item.count())):
                    widget = item.itemAt(j).widget()
                    if widget:
                        widget.deleteLater()
            elif isinstance(item, QWidgetItem):  # For direct widgets like QLineEdit
                widget = item.widget()
                if widget:
                    widget.deleteLater()

            # Remove the item itself from the layout
            self.param_layout.removeItem(item)

        # Get the selected test and corresponding module
        selected_test = self.test_selector.currentText()

        # If the placeholder is selected, do nothing
        if selected_test == "Select Test":
            self.run_button.setEnabled(False)
            return

        module = self.test_modules.get(selected_test)
        if module:
            try:
                self.params = self.get_parameters_from_module(module)

                for param, param_info in self.params.items():
                    if param == "temp_output":  # Skip temp_output
                        continue

                    # Create a horizontal layout for the label and help button
                    param_layout = QHBoxLayout()

                    # Create label
                    label = QLabel(f"{param}:", self)
                    param_layout.addWidget(label)

                    # Create tiny grey circle help button with a white question mark
                    help_button = QPushButton("?", self)
                    help_button.setFixedSize(16, 16)  # Smaller button size
                    help_button.setStyleSheet(
                        """
                        QPushButton {
                            background-color: grey;
                            color: white;
                            border-radius: 8px;  /* Circular button */
                            font-weight: bold;
                            font-size: 10px;  /* Smaller font size */
                        }
                        QPushButton:hover {
                            background-color: darkgrey;  /* Change color on hover */
                        }
                        """
                    )
                    help_button.setToolTip(param_info["help"])

                    # # Use lambda to capture the current param's help text
                    # help_button.clicked.connect(lambda _, p=param_info["help"]:
                    # self.show_help(p))

                    # Add the help button to the layout (next to the label)
                    param_layout.addWidget(help_button)
                    param_layout.addStretch()  # Add stretch to push the help button to
                    # the right

                    # Add the horizontal layout (label + help button) to the main layout
                    self.param_layout.addLayout(param_layout)

                    # Create the input field for the parameter
                    entry = QLineEdit(self)
                    entry.setText(
                        str(param_info["default"])
                        .replace("[", "")
                        .replace("]", "")
                        .replace(",", "")
                    )
                    entry.setObjectName(param)
                    entry.setFixedWidth(400)  # Set fixed width for QLineEdit
                    self.param_layout.addWidget(entry)

                # Force update the layout
                self.param_widgets.update()
                QTimer.singleShot(
                    0, self.param_widgets.update
                )  # Ensures the layout is updated

                self.run_button.setEnabled(True)
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to fetch parameters: {e}")

        else:
            QMessageBox.critical(self, "Error", "No test selected or test not found")

    def show_help(self, help_text):
        QMessageBox.information(self, "Parameter Help", help_text)

    def run_test(self):
        # Clean up old plot files to avoid loading leftover files
        self.cleanup_tempdir()

        selected_test = self.test_selector.currentText()

        module = self.test_modules.get(selected_test)

        if module:
            params = []
            self.update()
            self.repaint()

            # Generate temporary output path
            self.temp_output = tempfile.gettempdir()
            # print(f"Temporary output dir: {self.temp_output}")  # Debug print

            for param, _ in self.params.items():
                widget_list = self.param_widgets.findChildren(QLineEdit, param)
                if widget_list:
                    widget = widget_list[0]
                    params.append(f"--{param}")
                    params.extend(widget.text().split(" "))
                    if param == "output":
                        params.append(f"--output={widget.text()}")

                    params.append(f"--temp_output={self.temp_output}")
                else:
                    print(f"Widget with name {param} not found")

            test_script_path = os.path.abspath(module.__file__)
            command = [sys.executable, test_script_path] + params
            print(f"Command to run: {command}")  # Debug print

            try:
                self.output_text_edit.clear()

                self.process = QProcess(self)
                self.process.setProcessChannelMode(
                    QProcess.ProcessChannelMode.MergedChannels
                )
                self.process.readyReadStandardOutput.connect(self.read_process_output)
                self.process.finished.connect(self.process_finished)

                QTimer.singleShot(
                    0,
                    lambda: self.process.start(
                        sys.executable, [test_script_path] + params
                    ),
                )

            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to run the test: {e}")
        else:
            QMessageBox.critical(
                self, "Error", "No parameters found for the selected test"
            )

        self.plot_files = [
            os.path.join(self.temp_output, f"plot{i}.pkl") for i in range(1, 3)
        ]

    def read_process_output(self):
        """Reads and displays the process output in real-time."""
        if self.process:
            output = self.process.readAllStandardOutput().data().decode("utf-8").strip()
            if output:
                self.output_text_edit.append(output)

    def process_finished(self):
        """Handle the process when it finishes."""
        if self.process.exitCode() == 0:
            QMessageBox.information(self, "Test Output", "Test completed successfully.")

            # Delay to ensure file creation is complete
            QTimer.singleShot(1000, self.check_and_display_plot)
        else:
            QMessageBox.critical(
                self, "Error", f"Test failed with exit code {self.process.exitCode()}"
            )

    def check_and_display_plot(self):
        plot_files = [
            os.path.join(self.temp_output, f"plot{i}.pkl") for i in range(1, 3)
        ]
        self.display_plot([f for f in plot_files if os.path.exists(f)])

    def display_plot(self, plot_files):
        """Loads the plots from the pickle files and displays them on the canvas."""
        self.plots = []
        self.current_plot_index = 0
        # Load all available plots from the pickle files
        for plot_file in plot_files:
            with open(plot_file, "rb") as f:
                self.plots.append(pickle.load(f))  # Load the plot data
        # Display the first plot if there are any loaded plots
        if self.plots:
            self.update_plot_canvas()

            # Enable the "Next" button if there is more than one plot
            if len(self.plots) > 1:
                self.next_button.setEnabled(True)
            else:
                self.next_button.setEnabled(False)

    def update_plot_canvas(self):
        """Updates the canvas with the current plot."""
        if not self.plots:
            return

        try:
            # Load the current figure

            with open(self.plot_files[self.current_plot_index], "rb") as f:
                loaded_figure = pickle.load(f)
            # loaded_figure = self.plots[self.current_plot_index]

            # Remove the old canvas and toolbar from the plot layout
            self.plot_layout.removeWidget(self.canvas)
            self.canvas.deleteLater()  # Properly delete the old canvas
            self.plot_layout.removeWidget(self.toolbar)
            self.toolbar.deleteLater()  # Properly delete the old toolbar

            # Create a new canvas with the loaded figure
            self.canvas = FigureCanvas(loaded_figure)
            self.canvas.setFixedSize(800, 600)  # Set a fixed size for the canvas

            # Adjust the plot margins to ensure the x-axis is visible
            loaded_figure.subplots_adjust(bottom=0.15)  # Increase the bottom margin
            # loaded_figure.tight_layout(pad=2.0)  # Use tight layout with padding

            self.canvas.draw()

            # Create a new toolbar with the loaded figure
            self.toolbar = NavigationToolbar(self.canvas, self)
            self.toolbar.setFixedHeight(50)

            # Clear the plot layout and re-add toolbar above the canvas
            self.plot_layout.addWidget(self.toolbar)
            self.plot_layout.addWidget(self.canvas)

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load plot: {e}")

    def show_next_plot(self):
        if self.current_plot_index < len(self.plots) - 1:
            self.current_plot_index += 1
            self.update_plot_canvas()
            self.prev_button.setEnabled(True)
            if self.current_plot_index == len(self.plots) - 1:
                self.next_button.setEnabled(False)
        else:
            self.next_button.setEnabled(False)

    def show_previous_plot(self):
        if self.current_plot_index > 0:
            self.current_plot_index -= 1
            self.update_plot_canvas()
            self.next_button.setEnabled(True)
            if self.current_plot_index == 0:
                self.prev_button.setEnabled(False)
        else:
            self.prev_button.setEnabled(False)

    def cleanup_tempdir(self):
        """Remove old plot files in temp directory."""
        for i in range(1, 3):
            plot_file = os.path.join(tempfile.gettempdir(), f"plot{i}.pkl")
            if os.path.exists(plot_file):
                os.remove(plot_file)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    ex = TestRunner()
    sys.exit(app.exec())
