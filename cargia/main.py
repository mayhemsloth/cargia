import sys
import json
import traceback
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                           QHBoxLayout, QPushButton, QLabel, QFileDialog, 
                           QGridLayout, QGroupBox, QCheckBox, QMessageBox,
                           QMenuBar, QMenu, QDialog, QLineEdit, QDialogButtonBox,
                           QTextEdit, QScrollArea)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor, QPalette, QTextCursor
from grid_widget import GridWidget
from data_manager import DataManager
import os
from transcription import TranscriptionManager

def get_repo_root():
    """Get the absolute path to the cargia directory."""
    # Get the directory containing this file
    return os.path.dirname(os.path.abspath(__file__))

def log_error(message, error=None):
    """Helper function to log errors with traceback"""
    print(f"\nERROR: {message}")
    if error:
        print(f"Exception: {str(error)}")
        print("Traceback:")
        traceback.print_exc()
    print("-" * 80)

class SettingsDialog(QDialog):
    def __init__(self, current_data_dir, current_source_folder, parent=None):
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
        
        # Source folder settings
        source_folder_group = QGroupBox("Source Folder")
        source_folder_layout = QHBoxLayout()
        
        self.source_folder_edit = QLineEdit(current_source_folder)
        self.source_folder_edit.setReadOnly(True)
        source_folder_layout.addWidget(self.source_folder_edit)
        
        browse_source_btn = QPushButton("Browse...")
        browse_source_btn.clicked.connect(self.browse_source_folder)
        source_folder_layout.addWidget(browse_source_btn)
        
        source_folder_group.setLayout(source_folder_layout)
        layout.addWidget(source_folder_group)
        
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
    
    def browse_source_folder(self):
        dir_path = QFileDialog.getExistingDirectory(
            self,
            "Select Source Folder",
            self.source_folder_edit.text()
        )
        if dir_path:
            self.source_folder_edit.setText(dir_path)
    
    def get_data_dir(self):
        return self.data_dir_edit.text()
    
    def get_source_folder(self):
        return self.source_folder_edit.text()

class SetUserDialog(QDialog):
    def __init__(self, current_user, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Set User")
        self.setMinimumWidth(300)
        
        layout = QVBoxLayout()
        
        # User input
        user_group = QGroupBox("User Name")
        user_layout = QVBoxLayout()
        
        self.user_edit = QLineEdit(current_user)
        user_layout.addWidget(self.user_edit)
        
        user_group.setLayout(user_layout)
        layout.addWidget(user_group)
        
        # Dialog buttons
        button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | 
            QDialogButtonBox.StandardButton.Cancel
        )
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)
        
        self.setLayout(layout)
    
    def get_user(self):
        return self.user_edit.text().strip()

class PairWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout()
        
        # Grids container
        grids_container = QHBoxLayout()
        grids_container.setSpacing(20)  # Add spacing between input and output grids
        
        # Input grid
        input_group = QGroupBox("Input")
        input_layout = QVBoxLayout()
        self.input_grid = GridWidget(3, 3)
        self.input_grid.setMinimumSize(300, 300)  # Set minimum size for the grid
        input_layout.addWidget(self.input_grid)
        input_group.setLayout(input_layout)
        grids_container.addWidget(input_group)
        
        # Output grid
        output_group = QGroupBox("Output")
        output_layout = QVBoxLayout()
        self.output_grid = GridWidget(3, 3)
        self.output_grid.setMinimumSize(300, 300)  # Set minimum size for the grid
        output_layout.addWidget(self.output_grid)
        output_group.setLayout(output_layout)
        grids_container.addWidget(output_group)
        
        layout.addLayout(grids_container)
        
        # Thought text box
        self.thought_text = QTextEdit()
        self.thought_text.setPlaceholderText("Enter your thoughts about this transformation... (Transcription will appear here when active)")
        self.thought_text.setMinimumHeight(100)
        layout.addWidget(self.thought_text)
        
        self.setLayout(layout)
    
    def set_grid_data(self, input_data, output_data, is_test=False):
        """Set the grid data for both input and output grids."""
        rows = len(input_data)
        cols = len(input_data[0])
        
        self.input_grid.resizeGrid(rows, cols)
        self.output_grid.resizeGrid(rows, cols)
        
        self.input_grid.setGridData(input_data)
        self.output_grid.setGridData(output_data)
        
        # Set different background color only for test pairs
        if is_test:
            self.thought_text.setStyleSheet("background-color:rgb(144, 42, 16);")  # Using a more noticeable light red
            self.thought_text.setPlaceholderText("Enter your thoughts about this test transformation...")
        else:
            self.thought_text.setStyleSheet("")  # Reset to default system style
            self.thought_text.setPlaceholderText("Enter your thoughts about this transformation...")
    
    def get_thought_text(self):
        """Get the current thought text."""
        return self.thought_text.toPlainText().strip()

class MainWindow(QMainWindow):
    def __init__(self):
        try:
            super().__init__()
            self.setWindowTitle("Cargia - ARC-AGI Labeling Tool")
            self.setMinimumSize(1200, 800)
            
            # Get cargia directory
            cargia_dir = get_repo_root()
            
            # Load settings
            settings_path = os.path.join(cargia_dir, "settings.json")
            if not os.path.exists(settings_path):
                # Create default settings if they don't exist
                default_settings = {
                    "user": "",
                    "data_dir": os.path.join(cargia_dir, "data"),
                    "source_folder": os.path.join(cargia_dir, "data", "arc_agi_2_source_data")
                }
                with open(settings_path, 'w') as f:
                    json.dump(default_settings, f, indent=4)
                settings = default_settings
            else:
                with open(settings_path, 'r') as f:
                    settings = json.load(f)
            
            # Ensure data directory exists
            data_dir = os.path.abspath(settings.get('data_dir', os.path.join(cargia_dir, "data")))
            source_folder = os.path.abspath(settings.get('source_folder', os.path.join(cargia_dir, "data", "arc_agi_2_source_data")))
            
            # Create directories if they don't exist
            os.makedirs(data_dir, exist_ok=True)
            os.makedirs(source_folder, exist_ok=True)
            
            # Initialize data manager with settings
            self.data_manager = DataManager(data_dir, source_folder)
            
            # Initialize transcription manager and model
            self.transcription_manager = TranscriptionManager(settings_path)
            self.transcription_manager.initialize_model()
            
            # Create menu bar
            self.create_menu_bar()
            
            # Create central widget and main layout
            central_widget = QWidget()
            self.setCentralWidget(central_widget)
            main_layout = QVBoxLayout(central_widget)
            
            # Create controls section
            controls_group = QGroupBox("Controls")
            controls_layout = QHBoxLayout()
            
            # Left side: Task controls
            task_controls_layout = QVBoxLayout()
            
            # Task controls row
            task_row_layout = QHBoxLayout()
            
            # New solve button
            self.new_solve_btn = QPushButton("New Solve")
            task_row_layout.addWidget(self.new_solve_btn)
            
            # Next pair button
            self.next_pair_btn = QPushButton("Show Next Pair")
            self.next_pair_btn.setEnabled(False)
            task_row_layout.addWidget(self.next_pair_btn)
            
            # Task name display
            self.task_name_label = QLabel("Task: None")
            task_row_layout.addWidget(self.task_name_label)
            
            # Add transcription toggle button
            self.transcription_toggle = QPushButton("Start Transcription")
            self.transcription_toggle.setCheckable(True)
            self.transcription_toggle.setEnabled(True)
            self.transcription_toggle.clicked.connect(self.toggle_transcription)
            task_row_layout.addWidget(self.transcription_toggle)
            
            # Add transcription status indicator
            self.transcription_status = QLabel("Transcription: Off")
            task_row_layout.addWidget(self.transcription_status)
            
            task_controls_layout.addLayout(task_row_layout)
            controls_layout.addLayout(task_controls_layout)
            
            # Add a spacer to push metadata to the right
            controls_layout.addStretch()
            
            # Right side: Metadata section
            metadata_group = QGroupBox("Metadata")
            metadata_group.setMaximumWidth(200)  # Limit the width of the metadata group
            metadata_layout = QVBoxLayout()
            metadata_layout.setSpacing(2)  # Reduce spacing between buttons
            
            # Create toggle buttons for metadata labels
            self.metadata_buttons = {}
            metadata_labels = ["Rotational", "Horizontal", "Vertical", "Translation", "Invertable"]
            for label in metadata_labels:
                btn = QPushButton(label)
                btn.setCheckable(True)
                btn.setEnabled(False)  # Disabled until task is loaded
                btn.setMaximumWidth(180)  # Limit button width
                btn.clicked.connect(lambda checked, l=label: self.update_metadata_label(l, checked))
                self.metadata_buttons[label] = btn
                metadata_layout.addWidget(btn)
            
            metadata_group.setLayout(metadata_layout)
            controls_layout.addWidget(metadata_group)
            
            controls_group.setLayout(controls_layout)
            main_layout.addWidget(controls_group)
            
            # Create scroll area for pairs
            scroll_area = QScrollArea()
            scroll_area.setWidgetResizable(True)
            scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
            scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
            
            # Create container for pairs
            self.pairs_container = QWidget()
            self.pairs_layout = QVBoxLayout(self.pairs_container)
            self.pairs_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
            self.pairs_layout.setSpacing(20)  # Add some spacing between pairs
            
            scroll_area.setWidget(self.pairs_container)
            main_layout.addWidget(scroll_area)
            
            # Connect signals
            self.new_solve_btn.clicked.connect(self.start_new_solve)
            self.next_pair_btn.clicked.connect(self.show_next_pair)
            
            # Initialize state
            self.current_task = None
            self.current_task_path = None
            self.current_train_index = 0
            self.current_test_index = 0
            self.pair_widgets = []
            self.current_solve_id = None
            self.metadata_labels = {label: False for label in metadata_labels}
            self.showing_test_pairs = False  # Track whether we're showing test pairs
            
            # Update window title with current user
            self.update_window_title()
            
            print("MainWindow initialized successfully")
        except Exception as e:
            log_error("Failed to initialize MainWindow", e)
            raise
    
    def create_menu_bar(self):
        """Create the menu bar with settings menu."""
        menu_bar = self.menuBar()
        
        # Settings menu
        settings_menu = menu_bar.addMenu("Settings")
        
        # Set User action
        set_user_action = settings_menu.addAction("Set User...")
        set_user_action.triggered.connect(self.show_set_user_dialog)
        
        # Add separator
        settings_menu.addSeparator()
        
        # Data directory action
        data_dir_action = settings_menu.addAction("Data Directory...")
        data_dir_action.triggered.connect(self.show_settings_dialog)
    
    def show_set_user_dialog(self):
        """Show the set user dialog and update user if changed."""
        dialog = SetUserDialog(self.data_manager.current_user, self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            new_user = dialog.get_user()
            if new_user and new_user != self.data_manager.current_user:
                self.data_manager.set_current_user(new_user)
                self.update_window_title()
                QMessageBox.information(
                    self,
                    "User Updated",
                    f"Current user set to: {new_user}"
                )
    
    def update_window_title(self):
        """Update the window title to include the current user."""
        user = self.data_manager.current_user
        if user:
            self.setWindowTitle(f"Cargia - ARC-AGI Labeling Tool (User: {user})")
        else:
            self.setWindowTitle("Cargia - ARC-AGI Labeling Tool")
    
    def show_settings_dialog(self):
        """Show the settings dialog and update data directory if changed."""
        dialog = SettingsDialog(self.data_manager.data_dir, self.data_manager.source_folder, self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            new_data_dir = dialog.get_data_dir()
            new_source_folder = dialog.get_source_folder()
            if new_data_dir != self.data_manager.data_dir or new_source_folder != self.data_manager.source_folder:
                # Reinitialize data manager with new directory
                self.data_manager = DataManager(new_data_dir, new_source_folder)
                QMessageBox.information(
                    self,
                    "Settings Updated",
                    f"Data directory updated to: {new_data_dir}\nSource folder updated to: {new_source_folder}"
                )
    
    def update_metadata_label(self, label, value):
        """Update a metadata label and save to database."""
        try:
            self.metadata_labels[label] = value
            if self.current_solve_id is not None:
                self.data_manager.update_metadata_labels(self.current_solve_id, self.metadata_labels)
        except Exception as e:
            log_error(f"Failed to update metadata label: {label}", e)
            QMessageBox.critical(self, "Error", f"Failed to update metadata label: {str(e)}")
    
    def enable_metadata_buttons(self, enabled):
        """Enable or disable all metadata buttons."""
        for btn in self.metadata_buttons.values():
            btn.setEnabled(enabled)
            if not enabled:
                btn.setChecked(False)
    
    def start_new_solve(self):
        """Start a new solve by finding the next unsolved task."""
        try:
            # Get next task with unique order map
            next_task = self.data_manager.get_next_task()
            if next_task is None:
                QMessageBox.information(
                    self,
                    "No Tasks Available",
                    "No unsolved tasks found. All tasks have been completed by the current user."
                )
                return
            
            # Create solve entry
            self.current_solve_id = self.data_manager.create_solve(
                task_id=next_task["task_id"],
                order_map=next_task["order_map"],
                order_map_type="default",
                color_map={},  # TODO: Add color mapping
                metadata_labels=self.metadata_labels
            )
            
            # Load the task
            self.current_task = next_task["task_data"]
            self.current_task_path = os.path.join(
                self.data_manager.source_folder,
                "training",
                f"{next_task['task_id']}.json"
            )
            
            # Clear existing pairs
            for widget in self.pair_widgets:
                widget.deleteLater()
            self.pair_widgets.clear()
            
            # Update task name
            self.task_name_label.setText(f"Task: {next_task['task_id']}")
            
            # Enable metadata buttons
            self.enable_metadata_buttons(True)
            
            # Reset state
            self.current_train_index = 0
            self.current_test_index = 0
            self.showing_test_pairs = False
            
            # Create first pair widget
            if 'train' in self.current_task and self.current_task['train']:
                self.add_pair_widget()
                self.next_pair_btn.setEnabled(True)
            
        except Exception as e:
            log_error("Failed to start new solve", e)
            QMessageBox.critical(self, "Error", f"Failed to start new solve: {str(e)}")
    
    def load_task(self, task_path):
        """Load a task from the specified path."""
        try:
            with open(task_path, 'r') as f:
                self.current_task = json.load(f)
                self.current_task_path = task_path
                
                # Clear existing pairs
                for widget in self.pair_widgets:
                    widget.deleteLater()
                self.pair_widgets.clear()
                
                # Update task name
                task_id = os.path.basename(task_path)[:-5]  # Remove .json extension
                self.task_name_label.setText(f"Task: {task_id}")
                
                # Create first pair widget
                if 'train' in self.current_task and self.current_task['train']:
                    self.current_train_index = 0
                    self.add_pair_widget()
                    self.next_pair_btn.setEnabled(True)
                    
                    # Create solve entry
                    self.current_solve_id = self.data_manager.create_solve(
                        task_id=task_id,
                        order_map={"train": [], "test": []},  # Will be updated as pairs are shown
                        color_map={},  # TODO: Add color mapping
                        metadata_labels={}  # TODO: Add metadata labels
                    )
        except Exception as e:
            log_error("Failed to load task", e)
            QMessageBox.critical(self, "Error", f"Failed to load task: {str(e)}")
    
    def add_pair_widget(self):
        """Add a new pair widget to the layout at the top."""
        pair_widget = PairWidget()
        self.pairs_layout.insertWidget(0, pair_widget)  # Insert at the top
        self.pair_widgets.insert(0, pair_widget)  # Keep track of widgets in the same order
        
        # Set grid data for the new pair
        if not self.showing_test_pairs:
            # Show training pair
            if self.current_train_index < len(self.current_task['train']):
                pair = self.current_task['train'][self.current_train_index]
                pair_widget.set_grid_data(pair['input'], pair['output'], is_test=False)
        else:
            # Show test pair
            if self.current_test_index < len(self.current_task['test']):
                pair = self.current_task['test'][self.current_test_index]
                pair_widget.set_grid_data(pair['input'], pair['output'], is_test=True)
        
        # Ensure the new pair is visible by scrolling to the top
        self.centralWidget().findChild(QScrollArea).ensureWidgetVisible(pair_widget)
    
    def show_next_pair(self):
        """Show the next pair in the current task."""
        try:
            # Save current thought
            if self.pair_widgets:
                current_widget = self.pair_widgets[-1]
                thought_text = current_widget.get_thought_text()
                if thought_text:
                    pair_type = "test" if self.showing_test_pairs else "train"
                    sequence_index = self.current_test_index if self.showing_test_pairs else self.current_train_index
                    self.data_manager.add_thought(
                        solve_id=self.current_solve_id,
                        pair_label=chr(97 + sequence_index),  # 'a', 'b', 'c', etc.
                        pair_type=pair_type,
                        sequence_index=sequence_index,
                        thought_text=thought_text
                    )
            
            if not self.showing_test_pairs:
                # Move to next training pair
                self.current_train_index += 1
                
                if self.current_train_index < len(self.current_task['train']):
                    self.add_pair_widget()
                else:
                    # Switch to test pairs
                    self.showing_test_pairs = True
                    self.current_test_index = 0
                    if self.current_task.get('test') and len(self.current_task['test']) > 0:
                        self.add_pair_widget()
                    else:
                        # No test pairs, complete the solve
                        self.next_pair_btn.setEnabled(False)
                        self.data_manager.complete_solve(self.current_solve_id)
                        QMessageBox.information(
                            self,
                            "Solve Complete",
                            "All pairs have been shown. The solve has been saved."
                        )
            else:
                # Move to next test pair
                self.current_test_index += 1
                
                if self.current_test_index < len(self.current_task['test']):
                    self.add_pair_widget()
                else:
                    # No more pairs, complete the solve
                    self.next_pair_btn.setEnabled(False)
                    self.data_manager.complete_solve(self.current_solve_id)
                    QMessageBox.information(
                        self,
                        "Solve Complete",
                        "All pairs have been shown. The solve has been saved."
                    )
        except Exception as e:
            log_error("Failed to show next pair", e)
            QMessageBox.critical(self, "Error", f"Failed to show next pair: {str(e)}")
    
    def toggle_transcription(self, checked: bool):
        """Toggle transcription on/off."""
        if checked:
            # Start transcription with callback to current pair
            success = self.transcription_manager.start_transcription(
                callback=self.handle_transcribed_text
            )
            if success:
                self.transcription_toggle.setText("Stop Transcription")
                self.transcription_status.setText("Transcription: Active 🎤")
                self.transcription_status.setStyleSheet("color: green")
            else:
                self.transcription_toggle.setChecked(False)
        else:
            # Stop transcription
            self.transcription_manager.stop_transcription()
            self.transcription_toggle.setText("Start Transcription")
            self.transcription_status.setText("Transcription: Off")
            self.transcription_status.setStyleSheet("")
    
    def handle_transcribed_text(self, text: str):
        """Handle transcribed text by adding it to the current pair's thought text."""
        if self.pair_widgets:
            current_pair = self.pair_widgets[0]  # Get the topmost pair
            current_text = current_pair.thought_text.toPlainText()
            
            # Add new text with proper spacing
            if current_text and not current_text.endswith((" ", "\n")):
                current_text += " "
            current_text += text
            
            # Update the text box
            current_pair.thought_text.setPlainText(current_text)
            
            # Move cursor to the very end
            cursor = current_pair.thought_text.textCursor()
            cursor.movePosition(QTextCursor.MoveOperation.End)
            current_pair.thought_text.setTextCursor(cursor)
    
    def cleanup(self):
        """Clean up resources before closing."""
        if hasattr(self, 'transcription_manager'):
            self.transcription_manager.stop_transcription()
            self.transcription_manager.cleanup()

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
