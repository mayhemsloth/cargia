from PyQt6.QtWidgets import QWidget, QGridLayout, QPushButton
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QColor, QPainter, QPen
import json
import os

class ColorConfig:
    _instance = None
    _config = None
    
    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    def __init__(self):
        if self._config is None:
            config_path = os.path.join(os.path.dirname(__file__), 'color_config.json')
            with open(config_path, 'r') as f:
                self._config = json.load(f)
    
    def get_color(self, symbol):
        symbol_str = str(symbol)
        if symbol_str in self._config:
            rgb = self._config[symbol_str]['color']
            return QColor(rgb[0], rgb[1], rgb[2])
        return QColor(255, 255, 255)  # Default to white
    
    def get_name(self, symbol):
        symbol_str = str(symbol)
        if symbol_str in self._config:
            return self._config[symbol_str]['name']
        return "unknown"

class GridCell(QPushButton):
    clicked = pyqtSignal(int, int)  # x, y coordinates
    
    def __init__(self, x, y, parent=None):
        super().__init__(parent)
        self.x = x
        self.y = y
        self.symbol = 0  # Default symbol (0 represents empty/background)
        self.setFixedSize(30, 30)
        self.clicked.connect(lambda: self.clicked.emit(self.x, self.y))
        self.color_config = ColorConfig.get_instance()
        
    def setSymbol(self, symbol):
        self.symbol = symbol
        self.update()
        
    def paintEvent(self, event):
        super().paintEvent(event)
        painter = QPainter(self)
        painter.setPen(QPen(Qt.GlobalColor.black))
        
        # Draw the cell background based on symbol
        color = self.color_config.get_color(self.symbol)
        painter.fillRect(self.rect(), color)
        
        # Draw border
        painter.drawRect(self.rect())
        
    def getColorName(self):
        return self.color_config.get_name(self.symbol)

class GridWidget(QWidget):
    def __init__(self, rows=3, cols=3, parent=None):
        super().__init__(parent)
        self.rows = rows
        self.cols = cols
        self.cells = []
        self.selected_tool = "edit"  # Default tool
        self.selected_symbol = 0     # Default symbol
        
        self.initUI()
        
    def initUI(self):
        layout = QGridLayout()
        layout.setSpacing(0)
        self.setLayout(layout)
        
        # Create grid cells
        for row in range(self.rows):
            for col in range(self.cols):
                cell = GridCell(row, col)
                cell.clicked.connect(self.handleCellClick)
                layout.addWidget(cell, row, col)
                self.cells.append(cell)
                
    def handleCellClick(self, x, y):
        if self.selected_tool == "edit":
            self.setCellSymbol(x, y, self.selected_symbol)
        elif self.selected_tool == "select":
            # TODO: Implement selection functionality
            pass
        elif self.selected_tool == "floodfill":
            # TODO: Implement flood fill functionality
            pass
            
    def setCellSymbol(self, x, y, symbol):
        for cell in self.cells:
            if cell.x == x and cell.y == y:
                cell.setSymbol(symbol)
                break
                
    def setTool(self, tool):
        self.selected_tool = tool
        
    def setSelectedSymbol(self, symbol):
        self.selected_symbol = symbol
        
    def resizeGrid(self, rows, cols):
        # Remove existing cells
        for cell in self.cells:
            self.layout().removeWidget(cell)
            cell.deleteLater()
        self.cells.clear()
        
        # Update dimensions
        self.rows = rows
        self.cols = cols
        
        # Create new cells
        for row in range(rows):
            for col in range(cols):
                cell = GridCell(row, col)
                cell.clicked.connect(self.handleCellClick)
                self.layout().addWidget(cell, row, col)
                self.cells.append(cell)
                
    def getGridData(self):
        data = []
        for row in range(self.rows):
            row_data = []
            for col in range(self.cols):
                for cell in self.cells:
                    if cell.x == row and cell.y == col:
                        row_data.append(cell.symbol)
                        break
            data.append(row_data)
        return data
        
    def setGridData(self, data):
        for row in range(min(self.rows, len(data))):
            for col in range(min(self.cols, len(data[row]))):
                self.setCellSymbol(row, col, data[row][col])
                
    def resetGrid(self):
        for cell in self.cells:
            cell.setSymbol(0) 