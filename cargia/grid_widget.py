from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel
from PyQt6.QtCore import Qt, QSize
from PyQt6.QtGui import QImage, QPixmap
from PIL import Image
import numpy as np
from grid_utils import GridImageBuilder, ColorConfig
import traceback
import os
from datetime import datetime

def log_error(message, error=None):
    """Helper function to log errors with traceback"""
    print(f"\nERROR: {message}")
    if error:
        print(f"Exception: {str(error)}")
        print("Traceback:")
        traceback.print_exc()
    print("-" * 80)

class GridWidget(QWidget):
    def __init__(self, rows=3, cols=3, parent=None):
        try:
            super().__init__(parent)
            self.rows = rows
            self.cols = cols
            self.grid_data = None  # Initialize as None instead of empty grid
            self.image_label = QLabel()
            self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.image_label.setMinimumSize(1, 1)  # Allow the label to shrink
            
            # Initialize the image builder with default settings
            self.image_builder = GridImageBuilder()
            
            # Cache for the PIL image
            self._cached_pil_image = None
            self._cached_grid_data = None
            
            # Debug flag to control debug image saving
            self._save_debug_images = False
            
            layout = QVBoxLayout()
            layout.addWidget(self.image_label)
            self.setLayout(layout)
            
            # Show initial placeholder message
            self.show_placeholder_message()
            print("GridWidget initialized with placeholder message")
        except Exception as e:
            log_error("Failed to initialize GridWidget", e)
            raise
        
    def show_placeholder_message(self):
        """Show a placeholder message when no task is loaded"""
        self.image_label.setText("No task loaded. Please load task to start.")
        self.image_label.setStyleSheet("color: gray; font-size: 14px;")
        
    def update_display(self):
        try:
            if self.grid_data is None:
                self.show_placeholder_message()
                return
                
            print(f"Updating display for {len(self.grid_data)}x{len(self.grid_data[0])} grid")
            
            # Only generate new PIL image if grid data has changed
            if self._cached_grid_data != self.grid_data:
                print("Grid data changed, generating new image")
                # Convert grid to PIL image using the image builder
                self._cached_pil_image = self.image_builder.build(self.grid_data)
                self._cached_grid_data = [row[:] for row in self.grid_data]  # Deep copy
                
                if self._cached_pil_image is None:
                    print("Failed to convert grid to image")
                    return
                    
                # Save debug image only when generating new image, debug flag is True, and we have actual data
                if self._save_debug_images and any(any(cell != 0 for cell in row) for row in self.grid_data):
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                    debug_path = os.path.join("debug_images", f"grid_{timestamp}.png")
                    self._cached_pil_image.save(debug_path)
                    print(f"Saved debug image to {debug_path}")
            
            # Convert PIL image to QImage
            if self._cached_pil_image.mode != 'RGB':
                self._cached_pil_image = self._cached_pil_image.convert('RGB')
            
            # Get image dimensions
            width, height = self._cached_pil_image.size
            
            # Convert to bytes
            data = self._cached_pil_image.tobytes('raw', 'RGB')
            
            # Create QImage with correct format
            qimage = QImage(data, width, height, width * 3, QImage.Format.Format_RGB888)
            
            if qimage.isNull():
                print("Failed to create QImage")
                return
            
            # Convert to QPixmap
            pixmap = QPixmap.fromImage(qimage)
            
            if pixmap.isNull():
                print("Failed to create QPixmap")
                return
            
            # Scale the pixmap to fill the available space while maintaining aspect ratio
            self._update_pixmap_scaling(pixmap)
            print(f"Display updated successfully with image size: {width}x{height}")
        except Exception as e:
            log_error("Failed to update display", e)
            raise  # Re-raise the exception to see the full traceback
        
    def _update_pixmap_scaling(self, pixmap):
        """Helper method to update pixmap scaling based on current widget size"""
        if pixmap.isNull():
            return
            
        # Get the available size of the label
        label_size = self.image_label.size()
        
        # Calculate the optimal size while maintaining aspect ratio
        pixmap_ratio = pixmap.width() / pixmap.height()
        label_ratio = label_size.width() / label_size.height()
        
        if pixmap_ratio > label_ratio:
            # Image is wider than the label, scale to label width
            target_width = label_size.width()
            target_height = int(target_width / pixmap_ratio)
        else:
            # Image is taller than the label, scale to label height
            target_height = label_size.height()
            target_width = int(target_height * pixmap_ratio)
            
        # Create target size
        target_size = QSize(target_width, target_height)
        
        # Scale the pixmap while maintaining aspect ratio
        scaled_pixmap = pixmap.scaled(
            target_size,
            Qt.AspectRatioMode.IgnoreAspectRatio,  # We already calculated the correct aspect ratio
            Qt.TransformationMode.SmoothTransformation
        )
        
        # Set the pixmap
        self.image_label.setPixmap(scaled_pixmap)
        
    def resizeEvent(self, event):
        try:
            super().resizeEvent(event)
            # Only update the display scaling if we have a valid image
            if self._cached_pil_image is not None and self.grid_data is not None:
                # Convert the cached PIL image to QPixmap
                data = self._cached_pil_image.tobytes('raw', 'RGB')
                qimage = QImage(data, self._cached_pil_image.width, self._cached_pil_image.height, 
                              self._cached_pil_image.width * 3, QImage.Format.Format_RGB888)
                pixmap = QPixmap.fromImage(qimage)
                self._update_pixmap_scaling(pixmap)
        except Exception as e:
            log_error("Failed to handle resize event", e)
        
    def resizeGrid(self, rows, cols):
        try:
            print(f"Resizing grid to {rows}x{cols}")
            self.rows = rows
            self.cols = cols
            self.grid_data = [[0 for _ in range(cols)] for _ in range(rows)]
            self._cached_pil_image = None  # Clear cache when grid size changes
            self._cached_grid_data = None
            self.update_display()
        except Exception as e:
            log_error("Failed to resize grid", e)
        
    def getGridData(self):
        return self.grid_data
        
    def setGridData(self, data):
        try:
            if not data:
                print("No data provided to setGridData")
                self.grid_data = None
                self.show_placeholder_message()
                return
                
            if not isinstance(data, list):
                print(f"Invalid data type: {type(data)}")
                self.grid_data = None
                self.show_placeholder_message()
                return
                
            if not data[0] or not isinstance(data[0], list):
                print("Invalid grid structure")
                self.grid_data = None
                self.show_placeholder_message()
                return
                
            print(f"Setting grid data with dimensions {len(data)}x{len(data[0])}")
            self.rows = len(data)
            self.cols = len(data[0]) if self.rows > 0 else 0
            self.grid_data = data
            self.update_display()
        except Exception as e:
            log_error("Failed to set grid data", e)
            self.grid_data = None
            self.show_placeholder_message()
        
    def resetGrid(self):
        try:
            print("Resetting grid")
            self.grid_data = None
            self.show_placeholder_message()
        except Exception as e:
            log_error("Failed to reset grid", e)
            self.grid_data = None
            self.show_placeholder_message() 