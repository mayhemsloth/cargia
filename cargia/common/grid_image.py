"""
Grid image generation utilities for both GUI and training.
"""
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from PIL import Image, ImageDraw

# ------------------------------------------------------------------------------
# ColorConfig: loads once on first use, caches globally
# ------------------------------------------------------------------------------
class ColorConfig:
    _config: Dict[str, Dict] = {}

    def __init__(self, config_path: Optional[Path] = None):
        if not ColorConfig._config:
            path = config_path or Path(__file__).parent.parent / "color_config.json"
            with path.open("r", encoding="utf-8") as f:
                ColorConfig._config = json.load(f)

    def get_color(self, symbol: int) -> Tuple[int, int, int]:
        """Return (R, G, B) for this symbol, or white if not found."""
        entry = ColorConfig._config.get(str(symbol))
        return tuple(entry["color"]) if entry and "color" in entry else (255, 255, 255)

    def get_name(self, symbol: int) -> str:
        """Return the name for this symbol, or 'unknown'."""
        entry = ColorConfig._config.get(str(symbol))
        return entry.get("name", "unknown") if entry else "unknown"


# ------------------------------------------------------------------------------
# GridImageBuilder: handles all rendering logic
# ------------------------------------------------------------------------------
class GridImageBuilder:
    def __init__(
        self,
        color_config: Optional[ColorConfig] = None,
        *,
        default_cell_size: int = 30,
        border_size: int = 2,
        small_threshold: int = 10,
        medium_threshold: int = 15,
        small_size: int = 20,
        medium_size: int = 15,
        large_size: int = 15,
    ):
        self.color_config = color_config or ColorConfig()
        self.default_cell_size = default_cell_size
        self.border_size = border_size
        self.small_threshold = small_threshold
        self.medium_threshold = medium_threshold
        self.small_size = small_size
        self.medium_size = medium_size
        self.large_size = large_size

    def _choose_cell_size(self, rows: int, cols: int) -> int:
        max_dim = max(rows, cols)
        if max_dim > self.medium_threshold:
            return self.large_size
        if max_dim > self.small_threshold:
            return self.medium_size
        return self.default_cell_size

    def build(self, grid: List[List[int]], size_px: Optional[int] = None) -> Optional[Image.Image]:
        """
        Render a grid (list of list of ints) to a PIL Image.

        Args:
            grid: rectangular list of lists of ints
            size_px: if provided, resize the output image to this size (maintaining aspect ratio)
        Returns:
            PIL.Image or None if grid is empty
        """
        if not grid:
            return None

        rows = len(grid)
        # Ensure all rows have the same length (or pad with a default)
        cols = max(len(r) for r in grid)
        for r in grid:
            if len(r) != cols:
                raise ValueError("All rows in the grid must have the same length")

        cell_size = self._choose_cell_size(rows, cols)
        w = cols * (cell_size + self.border_size) - self.border_size
        h = rows * (cell_size + self.border_size) - self.border_size

        # Create image with white background
        img = Image.new("RGB", (w, h), (255, 255, 255))
        draw = ImageDraw.Draw(img)

        # Draw grid lines
        for i in range(rows + 1):
            y = i * (cell_size + self.border_size)
            draw.line([(0, y), (w, y)], fill=(255, 255, 255), width=self.border_size)
        for j in range(cols + 1):
            x = j * (cell_size + self.border_size)
            draw.line([(x, 0), (x, h)], fill=(255, 255, 255), width=self.border_size)

        # Fill cells with colors
        for y, row in enumerate(grid):
            for x, symbol in enumerate(row):
                x1 = x * (cell_size + self.border_size)
                y1 = y * (cell_size + self.border_size)
                x2 = x1 + cell_size
                y2 = y1 + cell_size

                color = self.color_config.get_color(symbol)
                draw.rectangle([x1, y1, x2, y2], fill=color)

        # Resize if requested
        if size_px is not None:
            # Calculate aspect ratio preserving dimensions
            aspect = w / h
            if aspect > 1:
                new_w = size_px
                new_h = int(size_px / aspect)
            else:
                new_h = size_px
                new_w = int(size_px * aspect)
            img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)

        return img


# ------------------------------------------------------------------------------
# Convenience function
# ------------------------------------------------------------------------------
def grid_to_image(
    grid: List[List[int]],
    *,
    config_path: Optional[str] = None,
    cell_size: int = 30,
    border_size: int = 1,
    size_px: Optional[int] = None,
) -> Optional[Image.Image]:
    """
    Quick one-liner if you don't need full customization.

    Args:
        grid: list of lists of ints
        config_path: path to color_config.json (defaults to next to this file)
        cell_size: base size for each cell
        border_size: padding between cells
        size_px: if provided, resize the output image to this size
    """
    cc = ColorConfig(Path(config_path)) if config_path else None
    builder = GridImageBuilder(
        color_config=cc,
        default_cell_size=cell_size,
        border_size=border_size,
    )
    return builder.build(grid, size_px) 