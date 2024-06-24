from __future__ import annotations

from typing import Callable, Tuple

from rich.segment import Segment
from rich.style import Style

RGBA = Tuple[int, int, int, int]
GetPixel = Callable[[Tuple[int, int]], RGBA]


def _get_color(pixel: RGBA, default_color: str | None = None) -> str | None:
    r, g, b, a = pixel
    return f"rgb({r},{g},{b})" if a > 0 else default_color


class Renderer:
    """
    Base class for renderers.
    """

    default_color: str | None
    null_style: Style | None

    def __init__(
        self,
        *,
        default_color: str | None = None,
    ) -> None:
        self.default_color = default_color
        self.null_style = (
            None if default_color is None else Style.parse(f"on {default_color}")
        )

    def _get_range(self, height: int) -> range:
        """
        Get the range of lines to render.
        """
        raise NotImplementedError

    def _render_line(
        self, *, line_index: int, width: int, get_pixel: GetPixel
    ) -> list[Segment]:
        """
        Render a line of pixels.
        """
        raise NotImplementedError


class HalfcellRenderer(Renderer):
    """
    Render an image to half-height cells.
    """

    def _get_range(self, height: int) -> range:
        return range(0, height, 2)

    def _render_line(
        self, *, line_index: int, width: int, get_pixel: GetPixel
    ) -> list[Segment]:
        line = []
        for x in range(width):
            line.append(self._render_halfcell(x=x, y=line_index, get_pixel=get_pixel))
        return line

    def _render_halfcell(self, *, x: int, y: int, get_pixel: GetPixel) -> Segment:
        colors = []

        # get lower pixel, render lower pixel use foreground color, so it must be first
        lower_color = _get_color(
            get_pixel((x, y + 1)), default_color=self.default_color
        )
        colors.append(lower_color or "")
        # get upper pixel, render upper pixel use background color, it is optional
        upper_color = _get_color(get_pixel((x, y)), default_color=self.default_color)
        if upper_color:
            colors.append(upper_color or "")

        style = Style.parse(" on ".join(colors)) if colors else self.null_style
        # use lower halfheight block to render if lower pixel is not transparent
        return Segment("▄" if lower_color else " ", style)


class FullcellRenderer(Renderer):
    """
    Render an image to full-height cells.
    """

    def _get_range(self, height: int) -> range:
        return range(height)

    def _render_line(
        self, *, line_index: int, width: int, get_pixel: GetPixel
    ) -> list[Segment]:
        line = []
        for x in range(width):
            line.append(self._render_fullcell(x=x, y=line_index, get_pixel=get_pixel))
        return line

    def _render_fullcell(self, *, x: int, y: int, get_pixel: GetPixel) -> Segment:
        pixel = get_pixel((x, y))
        style = (
            Style.parse(f"on {_get_color(pixel, default_color=self.default_color)}")
            if pixel[3] > 0
            else self.null_style
        )
        return Segment("  ", style)
