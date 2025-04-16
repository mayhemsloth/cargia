import sys
import json
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                           QHBoxLayout, QPushButton, QLabel, QFileDialog, 
                           QGridLayout, QGroupBox, QCheckBox)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor, QPalette
from grid_widget import GridWidget

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Cargia - ARC-AGI Labeling Tool")
        self.setMinimumSize(1200, 800)
        
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
    
    def load_task(self):
        file_name, _ = QFileDialog.getOpenFileName(
            self,
            "Load Task JSON",
            "",
            "JSON Files (*.json)"
        )
        if file_name:
            try:
                with open(file_name, 'r') as f:
                    self.current_task = json.load(f)
                    self.task_name_label.setText(f"Task name: {self.current_task.get('name', 'Unnamed Task')}")
                    
                    # Load training pairs
                    if 'train' in self.current_task:
                        self.train_pairs = self.current_task['train']
                        self.current_train_index = 0
                        self.showing_test = False
                        self.update_train_display()
                        
                    # Load test inputs
                    if 'test' in self.current_task:
                        self.test_inputs = [test['input'] for test in self.current_task['test']]
                        self.current_test_index = 0
                        self.update_test_display()
            except Exception as e:
                print(f"Error loading task: {e}")
    
    def update_train_display(self):
        if self.train_pairs:
            pair = self.train_pairs[self.current_train_index]
            # Resize grids to match the input/output dimensions
            rows = len(pair['input'])
            cols = len(pair['input'][0])
            self.input_grid.resizeGrid(rows, cols)
            self.output_grid.resizeGrid(rows, cols)
            
            # Set the grid data
            self.input_grid.setGridData(pair['input'])
            self.output_grid.setGridData(pair['output'])
            
            # Update button text
            if self.current_train_index < len(self.train_pairs) - 1:
                self.next_pair_btn.setText("Show next train pair")
            else:
                self.next_pair_btn.setText("Show TEST input")
    
    def update_test_display(self):
        if self.test_inputs:
            self.test_input_counter.setText(
                f"Test input grid {self.current_test_index + 1}/{len(self.test_inputs)}"
            )
            # Resize grid to match test input dimensions
            test_input = self.test_inputs[self.current_test_index]
            rows = len(test_input)
            cols = len(test_input[0])
            self.test_input_grid.resizeGrid(rows, cols)
            self.test_input_grid.setGridData(test_input)
    
    def next_pair(self):
        if not self.showing_test:
            if self.current_train_index < len(self.train_pairs) - 1:
                self.current_train_index += 1
                self.update_train_display()
            else:
                # Switch to showing test inputs
                self.showing_test = True
                self.current_test_index = 0
                self.update_test_display()
                self.next_pair_btn.setText("Show next test input")
        else:
            if self.test_inputs:
                self.current_test_index = (self.current_test_index + 1) % len(self.test_inputs)
                self.update_test_display()
    
    def next_test_input(self):
        if self.test_inputs:
            self.current_test_index = (self.current_test_index + 1) % len(self.test_inputs)
            self.update_test_display()
    
    def toggle_symbol_visibility(self, state):
        # TODO: Implement symbol visibility toggle
        pass

def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
