import sys
import json
import traceback
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                           QHBoxLayout, QPushButton, QLabel, QFileDialog, 
                           QGridLayout, QGroupBox, QCheckBox, QMessageBox,
                           QMenuBar, QMenu, QDialog, QLineEdit, QDialogButtonBox)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor, QPalette
from grid_widget import GridWidget
from data_manager import DataManager

def log_error(message, error=None):
    """Helper function to log errors with traceback"""
    print(f"\nERROR: {message}")
    if error:
        print(f"Exception: {str(error)}")
        print("Traceback:")
        traceback.print_exc()
    print("-" * 80)

class SettingsDialog(QDialog):
    def __init__(self, current_data_dir, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Settings")
        self.setMinimumWidth(400)
        
        layout = QVBoxLayout()
        
        # Data directory settings
        data_dir_group = QGroupBox("Data Directory")
        data_dir_layout = QHBoxLayout()
        
        self.data_dir_edit = QLineEdit(current_data_dir)
        self.data_dir_edit.setReadOnly(True)
        data_dir_layout.addWidget(self.data_dir_edit)
        
        browse_btn = QPushButton("Browse...")
        browse_btn.clicked.connect(self.browse_data_dir)
        data_dir_layout.addWidget(browse_btn)
        
        data_dir_group.setLayout(data_dir_layout)
        layout.addWidget(data_dir_group)
        
        # Dialog buttons
        button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | 
            QDialogButtonBox.StandardButton.Cancel
        )
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)
        
        self.setLayout(layout)
    
    def browse_data_dir(self):
        dir_path = QFileDialog.getExistingDirectory(
            self,
            "Select Data Directory",
            self.data_dir_edit.text()
        )
        if dir_path:
            self.data_dir_edit.setText(dir_path)
    
    def get_data_dir(self):
        return self.data_dir_edit.text()

class MainWindow(QMainWindow):
    def __init__(self):
        try:
            super().__init__()
            self.setWindowTitle("Cargia - ARC-AGI Labeling Tool")
            self.setMinimumSize(1200, 800)
            
            # Initialize data manager with default directory
            self.data_manager = DataManager()
            
            # Create menu bar
            self.create_menu_bar()
            
            # Create central widget and main layout
            central_widget = QWidget()
            self.setCentralWidget(central_widget)
            main_layout = QHBoxLayout(central_widget)
            
            # Create demonstration examples view
            demo_view = QGroupBox("Task Demonstration")
            demo_layout = QVBoxLayout()
            
            # Add input and output grids for training pairs
            self.input_grid = GridWidget(3, 3)
            self.output_grid = GridWidget(3, 3)
            demo_layout.addWidget(QLabel("Input:"))
            demo_layout.addWidget(self.input_grid)
            demo_layout.addWidget(QLabel("Output:"))
            demo_layout.addWidget(self.output_grid)
            
            # Add navigation button
            self.next_pair_btn = QPushButton("Show next train pair")
            demo_layout.addWidget(self.next_pair_btn)
            
            demo_view.setLayout(demo_layout)
            
            # Create evaluation view
            evaluation_view = QGroupBox("Evaluation")
            evaluation_layout = QVBoxLayout()
            
            # Test input section
            test_input_group = QGroupBox("Test Input")
            test_input_layout = QVBoxLayout()
            self.test_input_counter = QLabel("Test input grid 0/0")
            self.next_test_btn = QPushButton("Next test input")
            test_input_layout.addWidget(self.test_input_counter)
            test_input_layout.addWidget(self.next_test_btn)
            self.test_input_grid = GridWidget(3, 3)
            test_input_layout.addWidget(self.test_input_grid)
            test_input_group.setLayout(test_input_layout)
            
            # Task controls
            task_controls = QGroupBox("Task Controls")
            task_controls_layout = QVBoxLayout()
            
            # File loading controls
            file_controls = QHBoxLayout()
            self.load_task_btn = QPushButton("Load Task JSON")
            file_controls.addWidget(self.load_task_btn)
            task_controls_layout.addLayout(file_controls)
            
            # Task name display
            self.task_name_label = QLabel("Task name: ")
            task_controls_layout.addWidget(self.task_name_label)
            
            # Symbol visibility toggle
            self.show_symbols_check = QCheckBox("Show symbol numbers")
            task_controls_layout.addWidget(self.show_symbols_check)
            
            task_controls.setLayout(task_controls_layout)
            
            # Add all components to evaluation layout
            evaluation_layout.addWidget(test_input_group)
            evaluation_layout.addWidget(task_controls)
            evaluation_view.setLayout(evaluation_layout)
            
            # Add main components to main layout
            main_layout.addWidget(demo_view, 1)
            main_layout.addWidget(evaluation_view, 2)
            
            # Connect signals
            self.load_task_btn.clicked.connect(self.load_task)
            self.next_pair_btn.clicked.connect(self.next_pair)
            self.next_test_btn.clicked.connect(self.next_test_input)
            self.show_symbols_check.stateChanged.connect(self.toggle_symbol_visibility)
            
            # Initialize state
            self.current_task = None
            self.current_train_index = 0
            self.current_test_index = 0
            self.test_inputs = []
            self.train_pairs = []
            self.showing_test = False
            
            print("MainWindow initialized successfully")
        except Exception as e:
            log_error("Failed to initialize MainWindow", e)
            raise
    
    def create_menu_bar(self):
        """Create the menu bar with settings menu."""
        menu_bar = self.menuBar()
        
        # Settings menu
        settings_menu = menu_bar.addMenu("Settings")
        
        # Data directory action
        data_dir_action = settings_menu.addAction("Data Directory...")
        data_dir_action.triggered.connect(self.show_settings_dialog)
    
    def show_settings_dialog(self):
        """Show the settings dialog and update data directory if changed."""
        dialog = SettingsDialog(self.data_manager.data_dir, self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            new_data_dir = dialog.get_data_dir()
            if new_data_dir != self.data_manager.data_dir:
                # Reinitialize data manager with new directory
                self.data_manager = DataManager(new_data_dir)
                QMessageBox.information(
                    self,
                    "Settings Updated",
                    f"Data directory updated to: {new_data_dir}"
                )
    
    def load_task(self):
        try:
            file_name, _ = QFileDialog.getOpenFileName(
                self,
                "Load Task JSON",
                "",
                "JSON Files (*.json)"
            )
            if file_name:
                print(f"\nLoading task from: {file_name}")
                with open(file_name, 'r') as f:
                    self.current_task = json.load(f)
                    print(f"Task loaded successfully: {self.current_task.get('name', 'Unnamed Task')}")
                    
                    # Load training pairs
                    if 'train' in self.current_task:
                        self.train_pairs = self.current_task['train']
                        print(f"Found {len(self.train_pairs)} training pairs")
                        self.current_train_index = 0
                        self.showing_test = False
                        self.update_train_display()
                        
                    # Load test inputs
                    if 'test' in self.current_task:
                        self.test_inputs = [test['input'] for test in self.current_task['test']]
                        print(f"Found {len(self.test_inputs)} test inputs")
                        self.current_test_index = 0
                        self.update_test_display()
        except json.JSONDecodeError as e:
            log_error("Failed to parse JSON file", e)
            QMessageBox.critical(self, "Error", f"Failed to parse JSON file: {str(e)}")
        except Exception as e:
            log_error("Failed to load task", e)
            QMessageBox.critical(self, "Error", f"Failed to load task: {str(e)}")
    
    def update_train_display(self):
        try:
            if not self.train_pairs:
                print("No training pairs available")
                return
                
            if self.current_train_index >= len(self.train_pairs):
                print(f"Invalid train index: {self.current_train_index} (max: {len(self.train_pairs)-1})")
                return
                
            pair = self.train_pairs[self.current_train_index]
            print(f"\nUpdating train display for pair {self.current_train_index + 1}/{len(self.train_pairs)}")
            
            if not pair or 'input' not in pair or 'output' not in pair:
                print("Invalid pair structure")
                return
                
            # Validate grid dimensions
            input_grid = pair['input']
            output_grid = pair['output']
            
            if not input_grid or not output_grid:
                print("Empty input or output grid")
                return
                
            if not isinstance(input_grid, list) or not isinstance(output_grid, list):
                print("Invalid grid type")
                return
                
            if not input_grid[0] or not output_grid[0]:
                print("Empty grid row")
                return
                
            # Resize grids to match the input/output dimensions
            rows = len(input_grid)
            cols = len(input_grid[0])
            print(f"Grid dimensions: {rows}x{cols}")
            
            self.input_grid.resizeGrid(rows, cols)
            self.output_grid.resizeGrid(rows, cols)
            
            # Set the grid data
            self.input_grid.setGridData(input_grid)
            self.output_grid.setGridData(output_grid)
            
            # Update button text
            if self.current_train_index < len(self.train_pairs) - 1:
                self.next_pair_btn.setText("Show next train pair")
            else:
                self.next_pair_btn.setText("Show TEST input")
                
            print("Train display updated successfully")
        except Exception as e:
            log_error("Failed to update train display", e)
    
    def update_test_display(self):
        try:
            if not self.test_inputs:
                print("No test inputs available")
                return
                
            if self.current_test_index >= len(self.test_inputs):
                print(f"Invalid test index: {self.current_test_index} (max: {len(self.test_inputs)-1})")
                return
                
            print(f"\nUpdating test display for input {self.current_test_index + 1}/{len(self.test_inputs)}")
            
            self.test_input_counter.setText(
                f"Test input grid {self.current_test_index + 1}/{len(self.test_inputs)}"
            )
            
            # Resize grid to match test input dimensions
            test_input = self.test_inputs[self.current_test_index]
            
            if not test_input or not isinstance(test_input, list) or not test_input[0]:
                print("Invalid test input")
                return
                
            rows = len(test_input)
            cols = len(test_input[0])
            print(f"Test grid dimensions: {rows}x{cols}")
            
            self.test_input_grid.resizeGrid(rows, cols)
            self.test_input_grid.setGridData(test_input)
            
            print("Test display updated successfully")
        except Exception as e:
            log_error("Failed to update test display", e)
    
    def next_pair(self):
        try:
            if not self.showing_test:
                if self.current_train_index < len(self.train_pairs) - 1:
                    self.current_train_index += 1
                    print(f"\nMoving to next train pair: {self.current_train_index + 1}")
                    self.update_train_display()
                else:
                    # Switch to showing test inputs
                    print("\nSwitching to test inputs")
                    self.showing_test = True
                    self.current_test_index = 0
                    self.update_test_display()
                    self.next_pair_btn.setText("Show next test input")
            else:
                if self.test_inputs:
                    self.current_test_index = (self.current_test_index + 1) % len(self.test_inputs)
                    print(f"\nMoving to next test input: {self.current_test_index + 1}")
                    self.update_test_display()
        except Exception as e:
            log_error("Failed to handle next pair", e)
    
    def next_test_input(self):
        try:
            if self.test_inputs:
                self.current_test_index = (self.current_test_index + 1) % len(self.test_inputs)
                print(f"\nMoving to next test input: {self.current_test_index + 1}")
                self.update_test_display()
        except Exception as e:
            log_error("Failed to handle next test input", e)
    
    def toggle_symbol_visibility(self, state):
        # TODO: Implement symbol visibility toggle
        pass

def main():
    try:
        app = QApplication(sys.argv)
        window = MainWindow()
        window.show()
        sys.exit(app.exec())
    except Exception as e:
        log_error("Application failed to start", e)
        raise

if __name__ == "__main__":
    main()
