import sys
import json
import traceback
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                           QHBoxLayout, QPushButton, QLabel, QFileDialog, 
                           QGridLayout, QGroupBox, QCheckBox, QMessageBox,
                           QMenuBar, QMenu, QDialog, QLineEdit, QDialogButtonBox,
                           QTextEdit, QScrollArea, QFrame, QSizePolicy, QTableWidget, QTableWidgetItem,
                           QProgressBar)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor, QPalette, QTextCursor, QFont, QAction
from cargia.grid_widget import GridWidget
from cargia.data_manager import DataManager
import os
from cargia.transcription import TranscriptionManager
import sqlite3
from datetime import datetime

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

class JsonViewerDialog(QDialog):
    def __init__(self, json_str, parent=None):
        super().__init__(parent)
        self.setWindowTitle("JSON Content")
        self.setMinimumSize(600, 400)
        
        layout = QVBoxLayout()
        
        # Create text widget
        self.text_widget = QTextEdit()
        self.text_widget.setReadOnly(True)
        
        # Set monospace font
        font = QFont("Consolas", 10)
        self.text_widget.setFont(font)
        
        try:
            # Parse and format JSON
            json_obj = json.loads(json_str)
            formatted_json = json.dumps(json_obj, indent=2)
            self.text_widget.setPlainText(formatted_json)
        except:
            self.text_widget.setPlainText(str(json_str))
        
        layout.addWidget(self.text_widget)
        
        # Add close button
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.accept)
        layout.addWidget(close_btn)
        
        self.setLayout(layout)

class DatabaseViewerDialog(QDialog):
    def __init__(self, db_path, table_name, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"View {table_name} Database")
        self.setMinimumSize(1000, 800)
        
        # Store parameters
        self.db_path = db_path
        self.table_name = table_name
        self.parent = parent
        
        # Create layout
        layout = QVBoxLayout()
        
        # Create table widget
        self.table = QTableWidget()
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.setAlternatingRowColors(True)
        self.table.cellClicked.connect(self.handle_cell_click)
        layout.addWidget(self.table)
        
        # Add buttons layout
        buttons_layout = QHBoxLayout()
        
        # Add comparison button (only for thoughts table)
        if table_name == "thoughts":
            self.compare_btn = QPushButton("Compare Original vs Cleaned")
            self.compare_btn.clicked.connect(self.show_text_comparison)
            self.compare_btn.setStyleSheet("""
                QPushButton {
                    background-color: #2196F3;
                    color: white;
                    border: none;
                    border-radius: 5px;
                    font-size: 12px;
                    font-weight: bold;
                    padding: 8px 16px;
                }
                QPushButton:hover {
                    background-color: #1976D2;
                }
            """)
            buttons_layout.addWidget(self.compare_btn)
        
        buttons_layout.addStretch()  # Add spacer to push close button to the right
        
        # Add close button
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.accept)
        buttons_layout.addWidget(close_btn)
        
        layout.addLayout(buttons_layout)
        
        self.setLayout(layout)
        
        # Store JSON data
        self.json_data = {}  # Store original JSON strings
        
        # Load data
        self.load_database(db_path, table_name)
    
    def show_text_comparison(self):
        """Show the text comparison dialog."""
        try:
            # Get the data manager from the parent window
            if hasattr(self.parent, 'data_manager'):
                dialog = TextComparisonDialog(self.parent.data_manager, self)
                dialog.exec()
            else:
                QMessageBox.warning(self, "Error", "Unable to access data manager for comparison.")
        except Exception as e:
            log_error("Failed to show text comparison dialog", e)
            QMessageBox.critical(self, "Error", f"Failed to show text comparison: {str(e)}")
    
    def handle_cell_click(self, row, col):
        """Handle cell click to show JSON content in a modal dialog."""
        cell_key = (row, col)
        
        # Check if this is a JSON cell
        if cell_key in self.json_data:
            dialog = JsonViewerDialog(self.json_data[cell_key], self)
            dialog.exec()
    
    def load_database(self, db_path, table_name):
        """Load and display database contents."""
        try:
            self.db_path = db_path
            self.table_name = table_name
            
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            # Get column names
            cursor.execute(f"PRAGMA table_info({table_name})")
            columns = [col[1] for col in cursor.fetchall()]
            
            # Get data
            cursor.execute(f"SELECT * FROM {table_name}")
            rows = cursor.fetchall()
            
            # Set up table
            self.table.setColumnCount(len(columns))
            self.table.setRowCount(len(rows))
            self.table.setHorizontalHeaderLabels(columns)
            
            # Fill table with data
            for i, row in enumerate(rows):
                for j, value in enumerate(row):
                    if value is None:
                        item = QTableWidgetItem("")
                        self.table.setItem(i, j, item)
                    elif isinstance(value, str) and (value.startswith('{') or value.startswith('[')):
                        # Store original JSON data
                        self.json_data[(i, j)] = value
                        # Show first line of JSON with ellipsis
                        try:
                            json_obj = json.loads(value)
                            first_line = json.dumps(json_obj, indent=2).split('\n')[0]
                            item = QTableWidgetItem(first_line + " ...")
                        except:
                            item = QTableWidgetItem(str(value))
                        self.table.setItem(i, j, item)
                    else:
                        item = QTableWidgetItem(str(value))
                        self.table.setItem(i, j, item)
            
            # Resize columns to content
            self.table.resizeColumnsToContents()
            
            # Set minimum column width
            for i in range(self.table.columnCount()):
                self.table.setColumnWidth(i, max(100, self.table.columnWidth(i)))
            
        except Exception as e:
            log_error(f"Failed to load database {table_name}", e)
            QMessageBox.critical(self, "Error", f"Failed to load database: {str(e)}")
        finally:
            if 'conn' in locals():
                conn.close()

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

class ColorSwatch(QFrame):
    def __init__(self, color, name, number, parent=None):
        super().__init__(parent)
        self.setFixedSize(30, 30)
        self.setFrameStyle(QFrame.Shape.Box | QFrame.Shadow.Plain)
        
        # Set background color
        palette = self.palette()
        palette.setColor(QPalette.ColorRole.Window, QColor(*color))
        self.setPalette(palette)
        self.setAutoFillBackground(True)
        
        # Create tooltip with name and number
        self.setToolTip(f"{name} ({number})")

class TextCleanerDialog(QDialog):
    """Dialog for choosing text cleaning options."""
    
    def __init__(self, data_manager, parent=None):
        super().__init__(parent)
        self.data_manager = data_manager
        self.setWindowTitle("Text Cleaner")
        self.setMinimumWidth(500)
        self.setMinimumHeight(300)
        
        layout = QVBoxLayout()
        
        # Title and description
        title_label = QLabel("Text Cleaner")
        title_label.setFont(QFont('Arial', 16, QFont.Weight.Bold))
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title_label)
        
        desc_label = QLabel("Choose how you want to clean the thought text in your database:")
        desc_label.setWordWrap(True)
        layout.addWidget(desc_label)
        
        # Statistics section
        stats_group = QGroupBox("Current Status")
        stats_layout = QVBoxLayout()
        
        # Get current statistics
        stats = self.data_manager.get_text_cleaning_stats()
        
        stats_text = f"""
        Total thoughts with text: {stats['total_thoughts']}
        Already cleaned: {stats['cleaned_thoughts']}
        Need cleaning: {stats['uncleaned_thoughts']}
        """
        
        stats_label = QLabel(stats_text)
        stats_label.setStyleSheet("font-family: monospace; background-color: #1b86c1; padding: 10px;")
        stats_layout.addWidget(stats_label)
        stats_group.setLayout(stats_layout)
        layout.addWidget(stats_group)
        
        # Options section
        options_group = QGroupBox("Cleaning Options")
        options_layout = QVBoxLayout()
        
        # Option 1: Fill empty columns
        self.fill_empty_btn = QPushButton("Fill Empty Columns")
        self.fill_empty_btn.setMinimumHeight(50)
        self.fill_empty_btn.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border: none;
                border-radius: 5px;
                font-size: 14px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """)
        self.fill_empty_btn.clicked.connect(self.choose_fill_empty)
        options_layout.addWidget(self.fill_empty_btn)
        
        # Option 2: Overwrite all
        self.overwrite_btn = QPushButton("Overwrite All Rows")
        self.overwrite_btn.setMinimumHeight(50)
        self.overwrite_btn.setStyleSheet("""
            QPushButton {
                background-color: #f44336;
                color: white;
                border: none;
                border-radius: 5px;
                font-size: 14px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #da190b;
            }
        """)
        self.overwrite_btn.clicked.connect(self.choose_overwrite)
        options_layout.addWidget(self.overwrite_btn)
        
        options_group.setLayout(options_layout)
        layout.addWidget(options_group)
        
        # Cancel button
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        layout.addWidget(cancel_btn)
        
        self.setLayout(layout)
        
        # Store the chosen option
        self.chosen_option = None
    
    def choose_fill_empty(self):
        """Choose to fill empty columns only."""
        self.chosen_option = "fill_empty"
        self.accept()
    
    def choose_overwrite(self):
        """Choose to overwrite all rows."""
        self.chosen_option = "overwrite"
        self.accept()
    
    def get_chosen_option(self):
        """Get the chosen cleaning option."""
        return self.chosen_option

class TextCleanerConfirmationDialog(QDialog):
    """Confirmation dialog for text cleaning operations."""
    
    def __init__(self, option, data_manager, parent=None):
        super().__init__(parent)
        self.option = option
        self.data_manager = data_manager
        self.setWindowTitle("Confirm Text Cleaning")
        self.setMinimumWidth(600)
        self.setMinimumHeight(400)
        
        layout = QVBoxLayout()
        
        # Warning icon and title
        title_label = QLabel("⚠️ Confirm Text Cleaning Operation ⚠️")
        title_label.setFont(QFont('Arial', 16, QFont.Weight.Bold))
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title_label.setStyleSheet("color: #d32f2f;")
        layout.addWidget(title_label)
        
        # Operation description
        if option == "fill_empty":
            desc_text = """
            <b>Operation:</b> Fill Empty Columns<br><br>
            This will clean all thought text entries that currently have no cleaned version.
            Existing cleaned text will be preserved.
            """
        else:  # overwrite
            desc_text = """
            <b>Operation:</b> Overwrite All Rows<br><br>
            This will clean ALL thought text entries, overwriting any existing cleaned text.
            This operation cannot be undone!
            """
        
        desc_label = QLabel(desc_text)
        desc_label.setWordWrap(True)
        desc_label.setStyleSheet("background-color: #c1ad1b; border: 1px solid #c1ad1b; padding: 15px; border-radius: 5px;")
        layout.addWidget(desc_label)
        
        # Database path information
        db_info_group = QGroupBox("Database Information")
        db_layout = QVBoxLayout()
        
        db_path = self.data_manager.thoughts_db_path
        db_text = f"""
        <b>Database Path:</b><br>
        {db_path}<br><br>
        <b>Database Size:</b> {self._get_file_size(db_path)}
        """
        
        db_label = QLabel(db_text)
        db_label.setStyleSheet("font-family: monospace; background-color: #2d3748; color: white; padding: 10px; border-radius: 3px;")
        db_layout.addWidget(db_label)
        db_info_group.setLayout(db_layout)
        layout.addWidget(db_info_group)
        
        # Statistics
        stats_group = QGroupBox("Expected Impact")
        stats_layout = QVBoxLayout()
        
        stats = self.data_manager.get_text_cleaning_stats()
        if option == "fill_empty":
            impact_text = f"""
            <b>Thoughts to be cleaned:</b> {stats['uncleaned_thoughts']}<br>
            <b>Already cleaned (will be preserved):</b> {stats['cleaned_thoughts']}<br>
            <b>Total thoughts with text:</b> {stats['total_thoughts']}
            """
        else:  # overwrite
            impact_text = f"""
            <b>All thoughts to be cleaned:</b> {stats['total_thoughts']}<br>
            <b>Existing cleaned text will be overwritten!</b>
            """
        
        impact_label = QLabel(impact_text)
        impact_label.setStyleSheet("background-color: #1b86c1; color: white; padding: 10px; border-radius: 3px;")
        stats_layout.addWidget(impact_label)
        stats_group.setLayout(stats_layout)
        layout.addWidget(stats_group)
        
        # Confirmation checkbox
        self.confirm_checkbox = QCheckBox("I understand the implications and want to proceed")
        self.confirm_checkbox.stateChanged.connect(self.update_confirm_button)
        layout.addWidget(self.confirm_checkbox)
        
        # Buttons
        button_layout = QHBoxLayout()
        
        self.confirm_btn = QPushButton("Confirm and Proceed")
        self.confirm_btn.setEnabled(False)
        self.confirm_btn.setStyleSheet("""
            QPushButton {
                background-color: #d32f2f;
                color: white;
                border: none;
                border-radius: 5px;
                font-size: 14px;
                font-weight: bold;
                padding: 10px;
            }
            QPushButton:hover {
                background-color: #b71c1c;
            }
            QPushButton:disabled {
                background-color: #cccccc;
                color: #666666;
            }
        """)
        self.confirm_btn.clicked.connect(self.accept)
        button_layout.addWidget(self.confirm_btn)
        
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        button_layout.addWidget(cancel_btn)
        
        layout.addLayout(button_layout)
        
        self.setLayout(layout)
    
    def _get_file_size(self, file_path):
        """Get the size of a file in human-readable format."""
        try:
            if os.path.exists(file_path):
                size_bytes = os.path.getsize(file_path)
                if size_bytes < 1024:
                    return f"{size_bytes} B"
                elif size_bytes < 1024 * 1024:
                    return f"{size_bytes / 1024:.1f} KB"
                else:
                    return f"{size_bytes / (1024 * 1024):.1f} MB"
            else:
                return "File not found"
        except Exception:
            return "Unknown"
    
    def update_confirm_button(self, state):
        """Update the confirm button state based on checkbox."""
        self.confirm_btn.setEnabled(state == Qt.CheckState.Checked.value)

class ProgressDialog(QDialog):
    """Dialog to show progress during text cleaning."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Text Cleaning Progress")
        self.setMinimumWidth(400)
        self.setMinimumHeight(150)
        self.setModal(True)
        
        layout = QVBoxLayout()
        
        # Progress label
        self.progress_label = QLabel("Initializing...")
        self.progress_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.progress_label)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setMinimum(0)
        self.progress_bar.setMaximum(100)
        self.progress_bar.setValue(0)
        layout.addWidget(self.progress_bar)
        
        # Status label
        self.status_label = QLabel("")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.status_label)
        
        self.setLayout(layout)
    
    def update_progress(self, current, total):
        """Update the progress bar and labels."""
        if total > 0:
            percentage = int((current / total) * 100)
            self.progress_bar.setValue(percentage)
            self.progress_label.setText(f"Processing: {current} / {total}")
            self.status_label.setText(f"Progress: {percentage}%")
        
        # Process events to update the UI
        QApplication.processEvents()
    
    def set_status(self, status_text):
        """Set the status text."""
        self.status_label.setText(status_text)
        QApplication.processEvents()

class TextComparisonDialog(QDialog):
    """Dialog for comparing original vs cleaned thought text."""
    
    def __init__(self, data_manager, parent=None):
        super().__init__(parent)
        self.data_manager = data_manager
        self.setWindowTitle("Text Comparison - Original vs Cleaned")
        self.setMinimumSize(1200, 800)
        
        # Load all thoughts for comparison
        self.thoughts = self.data_manager.get_all_thoughts_for_comparison()
        self.current_index = 0
        
        layout = QVBoxLayout()
        
        # Header with navigation and status
        header_layout = QHBoxLayout()
        
        # Navigation buttons
        self.prev_btn = QPushButton("← Previous")
        self.prev_btn.clicked.connect(self.show_previous)
        header_layout.addWidget(self.prev_btn)
        
        # Current thought info
        self.info_label = QLabel("No thoughts available")
        self.info_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        header_layout.addWidget(self.info_label)
        
        # Navigation buttons
        self.next_btn = QPushButton("Next →")
        self.next_btn.clicked.connect(self.show_next)
        header_layout.addWidget(self.next_btn)
        
        layout.addLayout(header_layout)
        
        # Status indicator
        self.status_label = QLabel("")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.status_label.setStyleSheet("font-weight: bold; padding: 5px; border-radius: 3px;")
        layout.addWidget(self.status_label)
        
        # Text comparison area
        comparison_layout = QHBoxLayout()
        
        # Original text panel
        original_group = QGroupBox("Original Text")
        original_layout = QVBoxLayout()
        
        self.original_text = QTextEdit()
        self.original_text.setReadOnly(True)
        self.original_text.setFont(QFont("Consolas", 10))
        original_layout.addWidget(self.original_text)
        
        original_group.setLayout(original_layout)
        comparison_layout.addWidget(original_group)
        
        # Cleaned text panel
        cleaned_group = QGroupBox("Cleaned Text")
        cleaned_layout = QVBoxLayout()
        
        self.cleaned_text = QTextEdit()
        self.cleaned_text.setReadOnly(True)
        self.cleaned_text.setFont(QFont("Consolas", 10))
        cleaned_layout.addWidget(self.cleaned_text)
        
        cleaned_group.setLayout(cleaned_layout)
        comparison_layout.addWidget(cleaned_group)
        
        layout.addLayout(comparison_layout)
        
        # Close button
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.accept)
        layout.addWidget(close_btn)
        
        self.setLayout(layout)
        
        # Initialize with first thought
        if self.thoughts:
            self.show_current_thought()
        else:
            self.info_label.setText("No thoughts found in database")
            self.status_label.setText("No thoughts available")
            self.status_label.setStyleSheet("background-color: #ffebee; color: #c62828; font-weight: bold; padding: 5px; border-radius: 3px;")
    
    def show_current_thought(self):
        """Display the current thought."""
        if not self.thoughts or self.current_index >= len(self.thoughts):
            return
        
        thought = self.thoughts[self.current_index]
        
        # Update info label
        self.info_label.setText(f"Thought {self.current_index + 1} of {len(self.thoughts)} (ID: {thought['id']})")
        
        # Update original text
        self.original_text.setPlainText(thought['thought_text'] or "")
        
        # Update cleaned text
        cleaned_text = thought['cleaned_thought_text'] or ""
        self.cleaned_text.setPlainText(cleaned_text)
        
        # Update status indicator
        if cleaned_text:
            self.status_label.setText("✅ Cleaned text available")
            self.status_label.setStyleSheet("background-color: #e8f5e8; color: #2e7d32; font-weight: bold; padding: 5px; border-radius: 3px;")
        else:
            self.status_label.setText("❌ No cleaned text available")
            self.status_label.setStyleSheet("background-color: #ffebee; color: #c62828; font-weight: bold; padding: 5px; border-radius: 3px;")
        
        # Update navigation buttons
        self.prev_btn.setEnabled(self.current_index > 0)
        self.next_btn.setEnabled(self.current_index < len(self.thoughts) - 1)
    
    def show_previous(self):
        """Show the previous thought."""
        if self.current_index > 0:
            self.current_index -= 1
            self.show_current_thought()
    
    def show_next(self):
        """Show the next thought."""
        if self.current_index < len(self.thoughts) - 1:
            self.current_index += 1
            self.show_current_thought()

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
            controls_group.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
            controls_layout = QHBoxLayout()
            
            # Left side: Task controls
            task_controls_layout = QVBoxLayout()
            
            # Task controls row
            task_row_layout = QHBoxLayout()
            
            # Create larger buttons
            button_height = 50  # Make buttons taller
            
            # New solve button
            self.new_solve_btn = QPushButton("New Solve")
            self.new_solve_btn.setMinimumHeight(button_height)
            self.new_solve_btn.setMinimumWidth(150)  # Make buttons wider
            self.new_solve_btn.setFont(QFont('Arial', 12, QFont.Weight.Bold))  # Larger, bold font
            task_row_layout.addWidget(self.new_solve_btn)
            
            # Next pair button
            self.next_pair_btn = QPushButton("Show Next Pair")
            self.next_pair_btn.setMinimumHeight(button_height)
            self.next_pair_btn.setMinimumWidth(150)
            self.next_pair_btn.setFont(QFont('Arial', 12, QFont.Weight.Bold))
            self.next_pair_btn.setEnabled(False)
            task_row_layout.addWidget(self.next_pair_btn)
            
            task_controls_layout.addLayout(task_row_layout)
            controls_layout.addLayout(task_controls_layout)
            
            # Add a spacer to push right-side elements to the right
            controls_layout.addStretch()
            
            # Information section
            info_group = QGroupBox("Information")
            info_group.setMaximumWidth(300)
            info_layout = QVBoxLayout()
            info_layout.setSpacing(5)
            
            # Task name display
            self.task_name_label = QLabel("Task: None")
            info_layout.addWidget(self.task_name_label)
            
            # Last solve duration
            self.last_solve_label = QLabel("Last Solve: XXm XXs")
            info_layout.addWidget(self.last_solve_label)
            
            # Average solve duration
            self.avg_solve_label = QLabel("Avg Solve: XXm XXs")
            info_layout.addWidget(self.avg_solve_label)
            
            # Total solves count
            self.total_solves_label = QLabel("Total Solves: 0")
            info_layout.addWidget(self.total_solves_label)
            
            # Add database viewer buttons
            db_buttons_layout = QHBoxLayout()
            
            view_solves_btn = QPushButton("View Solves DB")
            view_solves_btn.clicked.connect(lambda: self.show_database_viewer("solves"))
            db_buttons_layout.addWidget(view_solves_btn)
            
            view_thoughts_btn = QPushButton("View Thoughts DB")
            view_thoughts_btn.clicked.connect(lambda: self.show_database_viewer("thoughts"))
            db_buttons_layout.addWidget(view_thoughts_btn)
            
            info_layout.addLayout(db_buttons_layout)
            
            # Add transcription toggle button
            self.transcription_toggle = QPushButton("Start Transcription")
            self.transcription_toggle.setCheckable(True)
            self.transcription_toggle.setEnabled(True)
            self.transcription_toggle.clicked.connect(self.toggle_transcription)
            info_layout.addWidget(self.transcription_toggle)
            
            # Add transcription status indicator
            self.transcription_status = QLabel("Transcription: Off")
            info_layout.addWidget(self.transcription_status)
            
            info_group.setLayout(info_layout)
            controls_layout.addWidget(info_group)
            
            # Color map display in a grid
            color_map_group = QGroupBox("Color Map Reference")
            color_map_group.setMaximumWidth(300)
            color_map_layout = QGridLayout()
            color_map_layout.setSpacing(5)
            
            # Load color configuration
            with open(os.path.join(cargia_dir, "color_config.json"), 'r') as f:
                self.color_config = json.load(f)
            
            # Create color swatches in a 3x4 grid
            row = 0
            col = 0
            for number, config in self.color_config.items():
                swatch_container = QWidget()
                swatch_layout = QVBoxLayout(swatch_container)
                swatch_layout.setContentsMargins(2, 2, 2, 2)
                
                swatch = ColorSwatch(config['color'], config['name'], number)
                label = QLabel(config['name'])
                label.setAlignment(Qt.AlignmentFlag.AlignCenter)
                
                swatch_layout.addWidget(swatch, alignment=Qt.AlignmentFlag.AlignCenter)
                swatch_layout.addWidget(label, alignment=Qt.AlignmentFlag.AlignCenter)
                
                color_map_layout.addWidget(swatch_container, row, col)
                
                col += 1
                if col >= 3:
                    col = 0
                    row += 1
            
            color_map_group.setLayout(color_map_layout)
            controls_layout.addWidget(color_map_group)
            
            # Metadata section
            metadata_group = QGroupBox("Metadata")
            metadata_group.setMaximumWidth(200)
            metadata_layout = QVBoxLayout()
            metadata_layout.setSpacing(2)
            
            # Create toggle buttons for metadata labels
            self.metadata_buttons = {}
            metadata_labels = ["Rotational", "Horizontal", "Vertical", "Translation", "Invertable"]
            for label in metadata_labels:
                btn = QPushButton(label)
                btn.setCheckable(True)
                btn.setEnabled(False)
                btn.setMaximumWidth(180)
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
            scroll_area.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)  # Allow vertical expansion
            
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
            self.pending_thoughts = {}  # Store thoughts by pair index until solve is complete
            self.solve_start_time = None  # Track when the solve started
            
            # Update window title with current user
            self.update_window_title()
            
            # Create keyboard shortcuts
            self.create_shortcuts()
            
            print("MainWindow initialized successfully")
            # Update total solves count on startup
            self.update_total_solves_count()
        except Exception as e:
            log_error("Failed to initialize MainWindow", e)
            raise
    
    def create_menu_bar(self):
        """Create the menu bar with settings and shortcuts menus."""
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
        
        # Add separator
        settings_menu.addSeparator()
        
        # Run Text Cleaner action
        run_text_cleaner_action = settings_menu.addAction("Run Text Cleaner...")
        run_text_cleaner_action.triggered.connect(self.show_text_cleaner_dialog)
        
        # Shortcuts menu
        shortcuts_menu = menu_bar.addMenu("Shortcuts")
        
        # Main actions shortcuts
        shortcuts_menu.addAction("New Solve (Ctrl+N)")
        shortcuts_menu.addAction("Show Next Pair (Ctrl+Space)")
        
        # Add separator before metadata shortcuts
        shortcuts_menu.addSeparator()
        shortcuts_menu.addAction("Metadata Toggles:")
        
        # Metadata shortcuts
        metadata_shortcuts = {
            "Rotational": "Ctrl+R",
            "Horizontal": "Ctrl+H",
            "Vertical": "Ctrl+V",
            "Translation": "Ctrl+T",
            "Invertable": "Ctrl+I"
        }
        
        for label, shortcut in metadata_shortcuts.items():
            shortcuts_menu.addAction(f"{label} ({shortcut})")
    
    def show_set_user_dialog(self):
        """Show the set user dialog and update user if changed."""
        dialog = SetUserDialog(self.data_manager.current_user, self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            new_user = dialog.get_user()
            if new_user and new_user != self.data_manager.current_user:
                self.data_manager.set_current_user(new_user)
                self.update_window_title()
                self.update_total_solves_count()  # This will also update duration statistics
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
    
    def show_text_cleaner_dialog(self):
        """Show the text cleaner dialog and handle the cleaning process."""
        try:
            # Show the main text cleaner dialog
            dialog = TextCleanerDialog(self.data_manager, self)
            if dialog.exec() == QDialog.DialogCode.Accepted:
                chosen_option = dialog.get_chosen_option()
                
                if chosen_option is None:
                    return
                
                # Show confirmation dialog
                confirm_dialog = TextCleanerConfirmationDialog(chosen_option, self.data_manager, self)
                if confirm_dialog.exec() == QDialog.DialogCode.Accepted:
                    # Start the cleaning process
                    self.run_text_cleaner(chosen_option)
        
        except Exception as e:
            log_error("Failed to show text cleaner dialog", e)
            QMessageBox.critical(self, "Error", f"Failed to show text cleaner dialog: {str(e)}")
    
    def run_text_cleaner(self, option):
        """Run the text cleaner with the specified option."""
        try:
            # Create progress dialog
            progress_dialog = ProgressDialog(self)
            progress_dialog.show()
            
            # Define progress callback
            def progress_callback(current, total):
                progress_dialog.update_progress(current, total)
            
            # Run the appropriate cleaning method
            if option == "fill_empty":
                progress_dialog.set_status("Cleaning thoughts that need cleaning...")
                results = self.data_manager.run_text_cleaner(progress_callback)
            else:  # overwrite
                progress_dialog.set_status("Cleaning all thoughts (overwrite mode)...")
                results = self.data_manager.run_text_cleaner_overwrite(progress_callback)
            
            # Close progress dialog
            progress_dialog.close()
            
            # Show results
            message = f"""
            Text cleaning completed!
            
            Total thoughts processed: {results['total_thoughts']}
            Successfully cleaned: {results['cleaned_thoughts']}
            Failed to clean: {results['failed_thoughts']}
            """
            
            QMessageBox.information(self, "Text Cleaning Complete", message)
            
        except Exception as e:
            log_error("Failed to run text cleaner", e)
            QMessageBox.critical(self, "Error", f"Failed to run text cleaner: {str(e)}")
    
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
    
    def update_total_solves_count(self):
        """Update the total solves count display."""
        count = self.data_manager.get_total_solves_count()
        self.total_solves_label.setText(f"Total Solves: {count}")
        
        # Also update the duration statistics
        self.last_solve_label.setText(f"Last Solve: {self.data_manager.get_last_solve_duration()}")
        self.avg_solve_label.setText(f"Avg Solve: {self.data_manager.get_average_solve_duration()}")
    
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
            
            # Store task info but don't create solve entry yet
            self.current_task = next_task["task_data"]
            self.current_task_path = os.path.join(
                self.data_manager.source_folder,
                "training",
                f"{next_task['task_id']}.json"
            )
            self.pending_task_id = next_task["task_id"]
            self.pending_order_map = next_task["order_map"]
            
            # Pre-initialize pending_thoughts with all expected pairs
            self.pending_thoughts = {}
            # Initialize training pairs
            for idx, pair_label in enumerate(self.pending_order_map['train']):
                self.pending_thoughts[idx] = {
                    'pair_label': pair_label,
                    'pair_type': 'train',
                    'sequence_index': idx,
                    'thought_text': ''
                }
            # Initialize test pairs
            for idx, pair_label in enumerate(self.pending_order_map['test']):
                self.pending_thoughts[len(self.pending_order_map['train']) + idx] = {
                    'pair_label': pair_label,
                    'pair_type': 'test',
                    'sequence_index': len(self.pending_order_map['train']) + idx,
                    'thought_text': ''
                }
            
            self.solve_start_time = datetime.now()  # Record start time
            
            # Clear existing pairs
            for widget in self.pair_widgets:
                widget.deleteLater()
            self.pair_widgets.clear()
            
            # Update task name
            self.task_name_label.setText(f"Task: {next_task['task_id']}")
            
            # Reset metadata labels to all False
            self.metadata_labels = {label: False for label in self.metadata_labels.keys()}
            
            # Update button states to reflect reset metadata
            for label, btn in self.metadata_buttons.items():
                btn.setChecked(False)
            
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
                self.next_pair_action.setEnabled(True)
                # Reset button appearance
                self.next_pair_btn.setText("Show Next Pair")
                self.next_pair_btn.setStyleSheet("")
            
        except Exception as e:
            log_error("Failed to start new solve", e)
            QMessageBox.critical(self, "Error", f"Failed to start new solve: {str(e)}")

    def add_pair_widget(self):
        """Add a new pair widget to the layout at the top."""
        pair_widget = PairWidget()
        self.pairs_layout.insertWidget(0, pair_widget)  # Insert at the top
        self.pair_widgets.insert(0, pair_widget)  # Keep track of widgets in the same order
        
        # Set grid data for the new pair and assign metadata
        if not self.showing_test_pairs:
            # Show training pair
            if self.current_train_index < len(self.current_task['train']):
                pair = self.current_task['train'][self.current_train_index]
                pair_widget.set_grid_data(pair['input'], pair['output'], is_test=False)
                # Store the actual pair label from the order map
                pair_widget.pair_label = self.pending_order_map['train'][self.current_train_index]
                pair_widget.pair_type = 'train'
                pair_widget.sequence_index = self.current_train_index
        else:
            # Show test pair
            if self.current_test_index < len(self.current_task['test']):
                pair = self.current_task['test'][self.current_test_index]
                pair_widget.set_grid_data(pair['input'], pair['output'], is_test=True)
                # Store the actual pair label from the order map
                pair_widget.pair_label = self.pending_order_map['test'][self.current_test_index]
                pair_widget.pair_type = 'test'
                pair_widget.sequence_index = len(self.pending_order_map['train']) + self.current_test_index
        
        # Ensure the new pair is visible by scrolling to the top
        self.centralWidget().findChild(QScrollArea).ensureWidgetVisible(pair_widget)
    
    def show_next_pair(self):
        """Show the next pair in the current task."""
        try:
            # Save current thought to pending thoughts
            if self.pair_widgets:
                current_widget = self.pair_widgets[0]  # Get the most recently added pair widget
                thought_text = current_widget.get_thought_text()
                pair_type = "test" if self.showing_test_pairs else "train"
                
                # Calculate the correct sequence index
                if self.showing_test_pairs:
                    # For test pairs, add the length of train pairs to get the correct index
                    sequence_index = len(self.pending_order_map['train']) + self.current_test_index
                else:
                    sequence_index = self.current_train_index
                
                # Update the thought in our pre-initialized dictionary
                if sequence_index in self.pending_thoughts:
                    self.pending_thoughts[sequence_index]['thought_text'] = thought_text
            
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
                        # Update button for last test pair
                        if len(self.current_task['test']) == 1:
                            self.next_pair_btn.setText("Finish Solve")
                            self.next_pair_btn.setStyleSheet("background-color: #ff4444; color: white;")
                    else:
                        # No test pairs, complete the solve
                        self.complete_solve()
            else:
                # Move to next test pair
                self.current_test_index += 1
                
                if self.current_test_index < len(self.current_task['test']):
                    self.add_pair_widget()
                    # Update button for last test pair
                    if self.current_test_index == len(self.current_task['test']) - 1:
                        self.next_pair_btn.setText("Finish Solve")
                        self.next_pair_btn.setStyleSheet("background-color: #ff4444; color: white;")
                else:
                    # No more pairs, complete the solve
                    self.complete_solve()
        except Exception as e:
            log_error("Failed to show next pair", e)
            QMessageBox.critical(self, "Error", f"Failed to show next pair: {str(e)}")
    
    def complete_solve(self):
        """Complete the current solve by saving all data to the database."""
        try:
            # Create solve entry with actual start time
            self.current_solve_id = self.data_manager.create_solve(
                task_id=self.pending_task_id,
                order_map=self.pending_order_map,
                order_map_type="default",
                color_map=self.color_config,
                metadata_labels=self.metadata_labels,
                start_time=self.solve_start_time
            )
            
            # Save the FINAL state of each pair's thought text
            # pair_widgets[0] is the most recent/top, so reverse to get original order
            for pair_widget in reversed(self.pair_widgets):
                self.data_manager.add_thought(
                    solve_id=self.current_solve_id,
                    pair_label=getattr(pair_widget, 'pair_label', ''),
                    pair_type=getattr(pair_widget, 'pair_type', ''),
                    sequence_index=getattr(pair_widget, 'sequence_index', -1),
                    thought_text=pair_widget.get_thought_text()
                )
            
            # Mark solve as complete
            self.data_manager.complete_solve(self.current_solve_id)
            
            # Update UI
            self.next_pair_btn.setText("Solve Complete")
            self.next_pair_btn.setStyleSheet("background-color: #cccccc; color: #666666;")
            self.next_pair_btn.setEnabled(False)
            self.next_pair_action.setEnabled(False)
            
            # Update statistics
            self.update_total_solves_count()
            
            QMessageBox.information(
                self,
                "Solve Complete",
                "All pairs have been shown. The solve has been saved."
            )
        except Exception as e:
            log_error("Failed to complete solve", e)
            QMessageBox.critical(self, "Error", f"Failed to complete solve: {str(e)}")
    
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
        
        # Create final backup before shutting down
        if hasattr(self, 'data_manager'):
            self.data_manager.backup_databases()
    
    def create_shortcuts(self):
        """Create keyboard shortcuts for common actions."""
        # New Solve shortcut (Ctrl+N)
        new_solve_action = QAction("New Solve", self)
        new_solve_action.setShortcut("Ctrl+N")
        new_solve_action.triggered.connect(self.start_new_solve)
        self.addAction(new_solve_action)
        
        # Show Next Pair shortcut (Ctrl+Space)
        self.next_pair_action = QAction("Show Next Pair", self)
        self.next_pair_action.setShortcut("Ctrl+Space")
        self.next_pair_action.triggered.connect(self.show_next_pair)
        self.addAction(self.next_pair_action)
        
        # Metadata shortcuts
        metadata_shortcuts = {
            "Rotational": "Ctrl+R",
            "Horizontal": "Ctrl+H",
            "Vertical": "Ctrl+V",
            "Translation": "Ctrl+T",
            "Invertable": "Ctrl+I"
        }
        
        for label, shortcut in metadata_shortcuts.items():
            action = QAction(f"Toggle {label}", self)
            action.setShortcut(shortcut)
            action.triggered.connect(lambda checked, l=label: self.toggle_metadata(l))
            self.addAction(action)
    
    def toggle_metadata(self, label):
        """Toggle a metadata label using keyboard shortcut."""
        if label in self.metadata_buttons:
            btn = self.metadata_buttons[label]
            if btn.isEnabled():
                new_state = not btn.isChecked()
                btn.setChecked(new_state)
                self.update_metadata_label(label, new_state)

    def show_database_viewer(self, db_type):
        """Show the database viewer dialog for the specified database."""
        try:
            if db_type == "solves":
                db_path = self.data_manager.solves_db_path
                table_name = "solves"
            else:  # thoughts
                db_path = self.data_manager.thoughts_db_path
                table_name = "thoughts"
            
            dialog = DatabaseViewerDialog(db_path, table_name, self)
            dialog.exec()
        except Exception as e:
            log_error(f"Failed to show {db_type} database viewer", e)
            QMessageBox.critical(self, "Error", f"Failed to show database viewer: {str(e)}")

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
