from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from PIL import Image, ImageDraw, ImageFont

_DEFAULT_IMAGE_ROOT = "./data/path"
_PROMPT_MODES = {
    "direct",
    "cot",
    "aux",
    "gta",
    "vision_reasoning",
    "step_by_step",
    "aux_step_by_step",
}
_GRID_SIZE = 10
_MOVE_DELTAS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}
_BASE_QUESTION = (
    "In the image there are five labeled points: A, B, C, D, E.\n"
    "From the orange triangle (start) to the blue triangle (destination), "
    "determine which labeled points are passed along the valid path.\n\n"
    "Consider the simple path only: do not revisit cells or backtrack."
)


def _resolve_prompt_mode(kwargs: Dict[str, Any]) -> str:
    raw_mode = kwargs.get("path_prompt_mode", "cot")
    mode = str(raw_mode).strip().lower()
    mode_aliases = {
        "step-by-step": "step_by_step",
        "aux-step-by-step": "aux_step_by_step",
        "vision-reasoning": "vision_reasoning",
    }
    mode = mode_aliases.get(mode, mode)
    if mode not in _PROMPT_MODES:
        raise ValueError(
            f"Invalid path_prompt_mode: {raw_mode}. "
            "Expected one of: aux, cot, direct, gta, vision_reasoning, "
            "step-by-step, aux_step_by_step"
        )
    return mode


def _resolve_image_path(doc: Dict[str, Any], kwargs: Dict[str, Any]) -> Path:
    image_rel_path = str(doc.get("image_path", "")).strip()
    if not image_rel_path:
        raise FileNotFoundError("missing image_path in path sample")
    image_root = str(kwargs.get("path_image_root", _DEFAULT_IMAGE_ROOT)).strip()
    rel_path = Path(image_rel_path)
    if rel_path.is_absolute():
        return rel_path
    return Path(image_root) / rel_path


def _get_font(size: int) -> ImageFont.ImageFont:
    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    ):
        try:
            return ImageFont.truetype(path, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def _draw_dashed_axis_aligned_line(
    draw: ImageDraw.ImageDraw,
    start: Tuple[int, int],
    end: Tuple[int, int],
    *,
    fill: Tuple[int, int, int],
    width: int,
    dash_len: int = 6,
    gap_len: int = 4,
    can_draw_point: Callable[[int, int], bool] | None = None,
) -> None:
    x0, y0 = start
    x1, y1 = end
    if x0 == x1:
        y_min, y_max = sorted((y0, y1))
        y = y_min
        while y <= y_max:
            segment_end = min(y + dash_len - 1, y_max)
            run_start: Optional[int] = None
            for yy in range(y, segment_end + 1):
                allowed = can_draw_point(x0, yy) if can_draw_point else True
                if allowed and run_start is None:
                    run_start = yy
                is_last = yy == segment_end
                if run_start is None:
                    continue
                if not allowed:
                    draw.line([(x0, run_start), (x0, yy - 1)], fill=fill, width=width)
                    run_start = None
                elif is_last:
                    draw.line([(x0, run_start), (x0, yy)], fill=fill, width=width)
                    run_start = None
            y = segment_end + gap_len + 1
        return
    if y0 == y1:
        x_min, x_max = sorted((x0, x1))
        x = x_min
        while x <= x_max:
            segment_end = min(x + dash_len - 1, x_max)
            run_start: Optional[int] = None
            for xx in range(x, segment_end + 1):
                allowed = can_draw_point(xx, y0) if can_draw_point else True
                if allowed and run_start is None:
                    run_start = xx
                is_last = xx == segment_end
                if run_start is None:
                    continue
                if not allowed:
                    draw.line([(run_start, y0), (xx - 1, y0)], fill=fill, width=width)
                    run_start = None
                elif is_last:
                    draw.line([(run_start, y0), (xx, y0)], fill=fill, width=width)
                    run_start = None
            x = segment_end + gap_len + 1
        return
    if can_draw_point is None:
        draw.line([start, end], fill=fill, width=width)


def _draw_solid_axis_aligned_line(
    draw: ImageDraw.ImageDraw,
    start: Tuple[int, int],
    end: Tuple[int, int],
    *,
    fill: Tuple[int, int, int],
    width: int,
    can_draw_point: Callable[[int, int], bool] | None = None,
) -> None:
    x0, y0 = start
    x1, y1 = end
    if can_draw_point is None:
        draw.line([start, end], fill=fill, width=width)
        return

    if x0 == x1:
        y_min, y_max = sorted((y0, y1))
        run_start: Optional[int] = None
        for yy in range(y_min, y_max + 1):
            allowed = can_draw_point(x0, yy)
            if allowed and run_start is None:
                run_start = yy
            if run_start is None:
                continue
            if not allowed:
                draw.line([(x0, run_start), (x0, yy - 1)], fill=fill, width=width)
                run_start = None
            elif yy == y_max:
                draw.line([(x0, run_start), (x0, yy)], fill=fill, width=width)
        return

    if y0 == y1:
        x_min, x_max = sorted((x0, x1))
        run_start: Optional[int] = None
        for xx in range(x_min, x_max + 1):
            allowed = can_draw_point(xx, y0)
            if allowed and run_start is None:
                run_start = xx
            if run_start is None:
                continue
            if not allowed:
                draw.line([(run_start, y0), (xx - 1, y0)], fill=fill, width=width)
                run_start = None
            elif xx == x_max:
                draw.line([(run_start, y0), (xx, y0)], fill=fill, width=width)
        return

    draw.line([start, end], fill=fill, width=width)


def _is_black_wall_pixel(gray_image: Image.Image, x: int, y: int) -> bool:
    width, height = gray_image.size
    x0 = max(0, x - 1)
    x1 = min(width - 1, x + 1)
    y0 = max(0, y - 1)
    y1 = min(height - 1, y + 1)
    for yy in range(y0, y1 + 1):
        for xx in range(x0, x1 + 1):
            if gray_image.getpixel((xx, yy)) < 60:
                return True
    return False


def _is_black_wall_cell(
    gray_image: Image.Image, r: int, c: int, cell_w: float, cell_h: float
) -> bool:
    x0 = int(round(c * cell_w))
    y0 = int(round(r * cell_h))
    x1 = int(round((c + 1) * cell_w)) - 1
    y1 = int(round((r + 1) * cell_h)) - 1
    x1 = max(x0, min(gray_image.size[0] - 1, x1))
    y1 = max(y0, min(gray_image.size[1] - 1, y1))

    dark_count = 0
    total = 0
    for yy in range(y0, y1 + 1):
        for xx in range(x0, x1 + 1):
            total += 1
            if gray_image.getpixel((xx, yy)) < 60:
                dark_count += 1
    if total == 0:
        return False
    # True wall cells are almost entirely black; text strokes are sparse.
    return (dark_count / total) >= 0.70


def _build_aux_overlay(image: Image.Image) -> Image.Image:
    overlay = image.copy().convert("RGB")
    draw = ImageDraw.Draw(overlay)
    width, height = overlay.size
    cell_w = width / float(_GRID_SIZE)
    cell_h = height / float(_GRID_SIZE)
    line_color = (230, 40, 40)
    line_width = 1
    coord_font_size = max(9, int(min(cell_w, cell_h) * 0.20))
    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", size=coord_font_size
        )
    except OSError:
        font = ImageFont.load_default()
    gray = image.convert("L")

    def _can_draw_red(x: int, y: int) -> bool:
        return not _is_black_wall_pixel(gray, x, y)

    _draw_solid_axis_aligned_line(
        draw,
        (0, 0),
        (width - 1, 0),
        fill=line_color,
        width=line_width,
        can_draw_point=_can_draw_red,
    )
    _draw_solid_axis_aligned_line(
        draw,
        (0, height - 1),
        (width - 1, height - 1),
        fill=line_color,
        width=line_width,
        can_draw_point=_can_draw_red,
    )
    _draw_solid_axis_aligned_line(
        draw,
        (0, 0),
        (0, height - 1),
        fill=line_color,
        width=line_width,
        can_draw_point=_can_draw_red,
    )
    _draw_solid_axis_aligned_line(
        draw,
        (width - 1, 0),
        (width - 1, height - 1),
        fill=line_color,
        width=line_width,
        can_draw_point=_can_draw_red,
    )
    for i in range(1, _GRID_SIZE):
        x = int(round(i * cell_w))
        y = int(round(i * cell_h))
        _draw_solid_axis_aligned_line(
            draw,
            (x, 0),
            (x, height - 1),
            fill=line_color,
            width=line_width,
            can_draw_point=_can_draw_red,
        )
        _draw_solid_axis_aligned_line(
            draw,
            (0, y),
            (width - 1, y),
            fill=line_color,
            width=line_width,
            can_draw_point=_can_draw_red,
        )

    for r in range(_GRID_SIZE):
        for c in range(_GRID_SIZE):
            if _is_black_wall_cell(gray, r, c, cell_w, cell_h):
                continue
            text = f"({r},{c})"
            left = c * cell_w
            top = r * cell_h
            # Place coords near top-left to avoid overlap with center markers.
            tx = int(left + 4)
            ty = int(top + 2)
            draw.text(
                (tx, ty),
                text,
                font=font,
                fill=(10, 10, 10),
            )
    return overlay


def _parse_coord_pair(raw_value: Any) -> Optional[Tuple[int, int]]:
    if not isinstance(raw_value, (list, tuple)) or len(raw_value) != 2:
        return None
    try:
        return int(raw_value[0]), int(raw_value[1])
    except (TypeError, ValueError):
        return None


def _parse_labeled_points(doc: Dict[str, Any]) -> Dict[str, Tuple[int, int]]:
    raw_points = doc.get("labeled_points")
    if not isinstance(raw_points, dict):
        return {}
    parsed: Dict[str, Tuple[int, int]] = {}
    for raw_label, raw_coord in raw_points.items():
        label = str(raw_label).strip().upper()
        if label not in {"A", "B", "C", "D", "E"}:
            continue
        coord = _parse_coord_pair(raw_coord)
        if coord is not None:
            parsed[label] = coord
    return parsed


def _parse_answer_path(doc: Dict[str, Any]) -> List[Tuple[int, int]]:
    raw_path = doc.get("answer_path")
    if not isinstance(raw_path, list):
        return []
    parsed_path: List[Tuple[int, int]] = []
    for item in raw_path:
        coord = _parse_coord_pair(item)
        if coord is not None:
            parsed_path.append(coord)
    return parsed_path


def path_extract_walkable(
    doc: Dict[str, Any],
    lmms_eval_specific_kwargs: Dict[str, Any] | None = None,
    base_image: Image.Image | None = None,
) -> Tuple[set[Tuple[int, int]], int]:
    kwargs = lmms_eval_specific_kwargs or {}
    image = base_image
    if image is None:
        image_path = _resolve_image_path(doc, kwargs)
        if not image_path.is_file():
            raise FileNotFoundError(f"path image not found: {image_path}")
        image = Image.open(image_path).convert("RGB")
    if image is None:
        raise ValueError("path image is required")

    grid_size = (
        int(doc.get("grid_size", _GRID_SIZE))
        if str(doc.get("grid_size", "")).strip()
        else _GRID_SIZE
    )
    grid_size = max(1, grid_size)
    gray = image.convert("L")
    cell_w = image.width / float(grid_size)
    cell_h = image.height / float(grid_size)
    walkable: set[Tuple[int, int]] = set()
    for row in range(grid_size):
        for col in range(grid_size):
            if not _is_black_wall_cell(gray, row, col, cell_w, cell_h):
                walkable.add((row, col))

    # Always keep task semantics anchors walkable even if pixel heuristics wobble.
    for key in ("start", "goal"):
        parsed = _parse_coord_pair(doc.get(key))
        if parsed is not None:
            walkable.add(parsed)
    for coord in _parse_labeled_points(doc).values():
        walkable.add(coord)
    return walkable, grid_size


def path_render_state_image(
    doc: Dict[str, Any],
    current_pos: Tuple[int, int],
    lmms_eval_specific_kwargs: Dict[str, Any] | None = None,
    base_image: Image.Image | None = None,
) -> Optional[Image.Image]:
    kwargs = lmms_eval_specific_kwargs or {}
    image = base_image
    if image is None:
        image_path = _resolve_image_path(doc, kwargs)
        if not image_path.is_file():
            return None
        image = Image.open(image_path).convert("RGB")
    if image is None:
        return None

    source = image.convert("RGB")
    canvas = source.copy()
    draw = ImageDraw.Draw(canvas)
    grid_size = (
        int(doc.get("grid_size", _GRID_SIZE))
        if str(doc.get("grid_size", "")).strip()
        else _GRID_SIZE
    )
    grid_size = max(1, grid_size)
    cell_w = canvas.width / float(grid_size)
    cell_h = canvas.height / float(grid_size)

    start = _parse_coord_pair(doc.get("start"))
    if start is None:
        return canvas

    def _cell_bounds(row: int, col: int) -> Tuple[int, int, int, int]:
        left = int(round(col * cell_w))
        top = int(round(row * cell_h))
        right = int(round((col + 1) * cell_w)) - 1
        bottom = int(round((row + 1) * cell_h)) - 1
        return left, top, right, bottom

    s_left, s_top, s_right, s_bottom = _cell_bounds(start[0], start[1])
    start_colors: Dict[Tuple[int, int, int], int] = {}
    for py in range(s_top, s_bottom + 1):
        for px in range(s_left, s_right + 1):
            color = source.getpixel((px, py))
            start_colors[color] = start_colors.get(color, 0) + 1
    background_color = max(start_colors.items(), key=lambda item: item[1])[0]

    def _color_distance(c1: Tuple[int, int, int], c2: Tuple[int, int, int]) -> int:
        return abs(c1[0] - c2[0]) + abs(c1[1] - c2[1]) + abs(c1[2] - c2[2])

    marker_pixels: List[Tuple[int, int, Tuple[int, int, int]]] = []
    for py in range(s_top, s_bottom + 1):
        for px in range(s_left, s_right + 1):
            color = source.getpixel((px, py))
            if _color_distance(color, background_color) > 12:
                marker_pixels.append((px - s_left, py - s_top, color))

    for rel_x, rel_y, _ in marker_pixels:
        canvas.putpixel((s_left + rel_x, s_top + rel_y), background_color)

    t_left, t_top, t_right, t_bottom = _cell_bounds(current_pos[0], current_pos[1])
    target_text_pixels: List[Tuple[int, int, Tuple[int, int, int]]] = []
    for py in range(t_top, t_bottom + 1):
        for px in range(t_left, t_right + 1):
            color = source.getpixel((px, py))
            if (
                color[0] < 90
                and color[1] < 90
                and color[2] < 90
                and _color_distance(color, background_color) > 18
            ):
                target_text_pixels.append((px - t_left, py - t_top, color))

    if marker_pixels:
        for rel_x, rel_y, color in marker_pixels:
            tx = t_left + rel_x
            ty = t_top + rel_y
            if t_left <= tx <= t_right and t_top <= ty <= t_bottom:
                canvas.putpixel((tx, ty), color)
        for rel_x, rel_y, color in target_text_pixels:
            tx = t_left + rel_x
            ty = t_top + rel_y
            if t_left <= tx <= t_right and t_top <= ty <= t_bottom:
                canvas.putpixel((tx, ty), color)
        return canvas

    # Fallback: draw a small upright orange triangle if marker extraction fails.
    cx = int(round((current_pos[1] + 0.5) * cell_w))
    cy = int(round((current_pos[0] + 0.5) * cell_h))
    half = max(3, int(min(cell_w, cell_h) * 0.16))
    triangle = [(cx, cy - half), (cx - half, cy + half), (cx + half, cy + half)]
    draw.polygon(triangle, fill=(255, 140, 0), outline=(170, 80, 0))
    for rel_x, rel_y, color in target_text_pixels:
        tx = t_left + rel_x
        ty = t_top + rel_y
        if t_left <= tx <= t_right and t_top <= ty <= t_bottom:
            canvas.putpixel((tx, ty), color)
    return canvas


def path_doc_to_visual(
    doc: Dict[str, Any], lmms_eval_specific_kwargs: Dict[str, Any] | None = None
) -> List[Image.Image]:
    kwargs = lmms_eval_specific_kwargs or {}
    prompt_mode = _resolve_prompt_mode(kwargs)
    image_path = _resolve_image_path(doc, kwargs)
    if not image_path.is_file():
        raise FileNotFoundError(f"path image not found: {image_path}")
    image = Image.open(image_path).convert("RGB")
    if prompt_mode in {"aux", "aux_step_by_step"}:
        return [image, _build_aux_overlay(image)]
    return [image]


def path_doc_to_text(
    doc: Dict[str, Any], lmms_eval_specific_kwargs: Dict[str, Any] | None = None
) -> str:
    kwargs = lmms_eval_specific_kwargs or {}
    prompt_mode = _resolve_prompt_mode(kwargs)
    direct_prompt = """In the image there are five labeled points: A, B, C, D, E.
From the orange triangle (start) to the blue triangle (destination), determine which labeled points lie on valid simple path without entering black grids.

SEMANTICS
- Black squares: walls (impassable)
- Light-gray cells are walkable.
- Orange triangle: path start.
- Blue triangle: destination.
- A/B/C/D/E: a label point, which is walkable.
- Legal moves: up, down, left, right only. One cell per step; no diagonals, no jumps; never cross walls.
- Follow the valid simple path from start to destination: no backtracking and no revisits.

Task:
From the orange triangle (start) to the blue triangle (destination), determine which labeled points lie on the valid simple path

Answer with only the passed letters in path order, using this format:
<answer>point1,point2,...</answer>"""
    cot_prompt = """In the image there are five labeled points: A, B, C, D, E.
From the orange triangle (start) to the blue triangle (destination), determine which labeled points lie on valid simple path without entering black grids.

SEMANTICS
- Black squares: walls (impassable)
- Light-gray cells are walkable.
- Orange triangle: path start.
- Blue triangle: destination.
- A/B/C/D/E: a label point, which is walkable.
- Legal moves: up, down, left, right only. One cell per step; no diagonals, no jumps; never cross walls.
- Follow the valid simple path from start to destination: no backtracking and no revisits.

Task:
From the orange triangle (start) to the blue triangle (destination), determine which labeled points lie on the valid simple path

OUTPUT FORMAT
1) Describe your reasoning briefly: identify the start and destination, explain the planned move sequence, and confirm never enters black-wall cells.
2) Answer with only the passed letters in path order, using this format:
<answer>point1,point2,...</answer>"""
    aux_prompt = """<AUX_GRID_INTERLEAVE>
In the image there are five labeled points: A, B, C, D, E.
From the orange triangle (start) to the blue triangle (destination), determine which labeled points lie on valid simple path without entering black grids.

SEMANTICS
- Black squares: walls (impassable)
- Light-gray cells are walkable.
- Orange triangle: path start.
- Blue triangle: destination.
- A/B/C/D/E: a label point, which is walkable.
- Legal moves: up, down, left, right only. One cell per step; no diagonals, no jumps; never cross walls.
- Follow the valid simple path from start to destination: no backtracking and no revisits.

Task:
From the orange triangle (start) to the blue triangle (destination), determine which labeled points lie on the valid simple path

First, generate one auxiliary image from the original image by drawing red grid-boundary lines for a 10x10 partition and small coordinate labels '(r,c)' near each cell's top-left corner while preserving the original colors and geometry.

OUTPUT FORMAT
1) Describe your reasoning briefly: identify the start and destination, explain the planned move sequence, and confirm never enters black-wall cells.
2) Answer with only the passed letters in path order, using this format:
<answer>point1,point2,...</answer>"""
    step_prompt = """In the image there are five labeled points: A, B, C, D, E.
From the orange triangle (start) to the blue triangle (destination), determine which labeled points lie on valid simple path without entering black grids.

SEMANTICS
- Black squares: walls (impassable)
- Light-gray cells are walkable.
- Orange triangle: start/current position.
- Blue triangle: destination.
- A/B/C/D/E: labeled walkable points.
- Legal moves: up, down, left, right only. One cell per step.
- Follow the valid simple path from start to destination: no backtracking and no revisits.

OUTPUT FORMAT (STRICT)
You need to start from the start point, plan the path to the destination step by step, and output the passed labeled points.

1) MULTI-IMAGE MODE — generate a SEQUENCE OF SEPARATE IMAGES, one per move:
   - Each output image must depict the state AFTER applying exactly one legal move.
   - Do NOT include the initial (pre-move) state.
   - Keep palette/layout/scale identical to the input; only the current position marker moves.
   - The number of returned images MUST equal the number of moves in the final path.
   - Absolutely FORBIDDEN: any collage/montage/spritesheet/grid/multi-panel/side-by-side/stacked images; no arrows, captions, overlays, GIFs, animations, or video.

2) STEP DECISION FORMAT — for EACH step decision, output exactly one JSON object:
   <STEP_JSON>{"move":"up|down|left|right","hit_labels":["A"]}</STEP_JSON>
   - hit_labels must be labels newly passed on THIS move only.
   - Use uppercase letters among A/B/C/D/E only.
   - If no new label is passed this step, output an empty list [].

3) FINAL LINE — after all step images/decisions, emit exactly one line:
   <answer>point1,point2,...</answer>

NO EXTRAS
- No extra explanation text."""
    gta_prompt = """
[GEN_PROMPT]
Edit the input image to create exactly one auxiliary annotated image.

Image semantics:
- Black squares are walls and must remain unchanged.
- Light-gray cells are walkable and must remain unchanged.
- The orange triangle is the start cell and must remain unchanged.
- The blue triangle is the destination cell and must remain unchanged.
- The letters A, B, C, D, E mark walkable label points and must remain unchanged.

Editing requirements:
1. Preserve the original geometry, cell layout, wall positions, start marker, destination marker, A/B/C/D/E label markers, and original colors.
2. Overlay thin red grid-boundary lines that divide the image into 10 equal parts along the horizontal axis and 10 equal parts along the vertical axis; draw 9 evenly spaced vertical red lines at 10%, 20%, ..., 90% of the image width and 9 evenly spaced horizontal red lines at 10%, 20%, ..., 90% of the image height, forming a 10x10 partition aligned with the cells.
3. Label every cell (black squares, light-gray cells, orange triangle cell, blue triangle cell, A/B/C/D/E label cells) at its center with its coordinate in black text, using the format (r,c). The top-left cell is (0,0). The first row is labeled left-to-right as (0,0), (0,1), (0,2), ..., until the last cell in that row. In general, row index r increases by 1 when moving downward, and column index c increases by 1 when moving rightward.
4. Do not add any arrows, path lines, legend, explanation text, title, or extra decorations.
[/GEN_PROMPT]

[QUESTION]
In the image there are five labeled points: A, B, C, D, E.
From the orange triangle (start) to the blue triangle (destination), determine which labeled points lie on valid simple path without entering black grids.

SEMANTICS
- Black squares: walls (impassable)
- Light-gray cells are walkable.
- Orange triangle: path start.
- Blue triangle: destination.
- A/B/C/D/E: a label point, which is walkable.
- Legal moves: up, down, left, right only. One cell per step; no diagonals, no jumps; never cross walls.
- Follow the valid simple path from start to destination: no backtracking and no revisits.

Task:
From the orange triangle (start) to the blue triangle (destination), determine which labeled points lie on the valid simple path

You are given two images: the original image and an auxiliary image from the original image by drawing red grid-boundary lines for a 10x10 partition and small coordinate labels '(r,c)' near each cell's center while preserving the original colors and geometry.

OUTPUT FORMAT
1) Describe your reasoning briefly: identify the start and destination, explain the planned move sequence, and confirm never enters black-wall cells.
2) Answer with only the passed letters in path order, using this format:
<answer>point1,point2,...</answer>
[/QUESTION]
"""
    vision_reasoning_prompt = """
[GEN_PROMPT]
Edit the input image to create exactly one auxiliary annotated image that highlights a feasible solution path.

Image semantics:
- Black squares are walls and must remain unchanged.
- Light-gray cells are walkable.
- The orange triangle is the start cell and must remain unchanged.
- The blue triangle is the destination cell and must remain unchanged.
- The letters A, B, C, D, E mark walkable label points and must remain unchanged.

Editing requirements:
1. Preserve the original geometry, cell layout, wall positions, start marker, destination marker, A/B/C/D/E label markers, and original colors.
2. Overlay a single continuous RED polyline that traces a legal simple path from the orange-triangle start cell to the blue-triangle destination cell:
   - The polyline must follow walkable (non-wall) cells only and must NEVER cross or touch any black wall cell.
   - Movement is restricted to up / down / left / right between cell centers; no diagonals, no jumps.
   - The polyline starts exactly at the center of the orange-triangle start cell and ends exactly at the center of the blue-triangle destination cell.
   - The path must be a simple path: no revisits and no backtracking.
   - Use a clean, solid bright red stroke of moderate thickness (roughly 10-15 percent of a cell width) so the path is clearly visible against the background.
3. Do not draw arrows, dots along the way, grid lines, coordinate labels, legend, captions, titles, or any other decoration. Only the single red polyline should be added.
4. Do not modify the colors or positions of the start marker, destination marker, A/B/C/D/E label markers, walls, or walkable cells in any other way.
[/GEN_PROMPT]

[QUESTION]
In the image there are five labeled points: A, B, C, D, E.
From the orange triangle (start) to the blue triangle (destination), determine which labeled points lie on valid simple path without entering black grids.

SEMANTICS
- Black squares: walls (impassable)
- Light-gray cells are walkable.
- Orange triangle: path start.
- Blue triangle: destination.
- A/B/C/D/E: a label point, which is walkable.
- Legal moves: up, down, left, right only. One cell per step; no diagonals, no jumps; never cross walls.
- Follow the valid simple path from start to destination: no backtracking and no revisits.

You are given two images: the original image and an auxiliary image where a red polyline traces a candidate simple path from the orange-triangle start to the blue-triangle destination. Use the red polyline as a visual hint, but verify every step against the image itself and never enter black-wall cells. Walk the polyline cell by cell and collect the labels among A/B/C/D/E that lie on the path, in path order.

OUTPUT FORMAT
1) Describe your reasoning briefly: identify the start and destination, follow the red polyline cell by cell, and confirm the path never enters black-wall cells.
2) Answer with only the passed letters in path order, using this format:
<answer>point1,point2,...</answer>
[/QUESTION]
"""
    aux_step_prompt = """<AUX_GRID_INTERLEAVE>
In the image there are five labeled points: A, B, C, D, E.
From the orange triangle (start) to the blue triangle (destination), determine which labeled points lie on valid simple path without entering black grids.

SEMANTICS
- Black squares: walls (impassable)
- Light-gray cells are walkable.
- Orange triangle: start/current position.
- Blue triangle: destination.
- A/B/C/D/E: labeled walkable points.
- Legal moves: up, down, left, right only. One cell per step.
- Follow the valid simple path from start to destination: no backtracking and no revisits.


OUTPUT FORMAT (STRICT)
You need to start from the start point, plan the path to the destination step by step, and output the passed labeled points.

1) FIRST OUTPUT — generate exactly one auxiliary image from the original image:
   - Draw red grid-boundary lines for a 10x10 partition.
   - Add coordinate labels '(r,c)' near each walkable cell's top-left corner.
   - Preserve original colors/layout/geometry.

2) MULTI-IMAGE MODE — after the auxiliary image, generate a SEQUENCE OF SEPARATE IMAGES, one per move:
   - Each output image must depict the state AFTER applying exactly one legal move.
   - Do NOT include the initial (pre-move) state again.
   - Keep palette/layout/scale/grid/coordinates identical; only the current position marker moves.
   - The number of returned images MUST equal the number of moves in the final path.
   - Absolutely FORBIDDEN: any collage/montage/spritesheet/grid/multi-panel/side-by-side/stacked images; no arrows, captions, overlays, GIFs, animations, or video.

3) STEP DECISION FORMAT — for EACH step decision, output exactly one JSON object:
   <STEP_JSON>{"move":"up|down|left|right","hit_labels":["A"]}</STEP_JSON>
   - hit_labels must be labels newly passed on THIS move only.
   - Use uppercase letters among A/B/C/D/E only.
   - If no new label is passed this step, output an empty list [].

4) FINAL LINE — after all images/decisions, emit exactly one line:
   <answer>point1,point2,...</answer>

NO EXTRAS
- No extra explanation text."""

    if prompt_mode == "direct":
        return direct_prompt
    if prompt_mode == "cot":
        return cot_prompt
    if prompt_mode == "aux":
        return aux_prompt
    if prompt_mode == "gta":
        return gta_prompt
    if prompt_mode == "vision_reasoning":
        return vision_reasoning_prompt
    if prompt_mode == "step_by_step":
        return step_prompt
    return aux_step_prompt


def _normalize_result_text(result_raw: Any) -> str:
    if isinstance(result_raw, str):
        try:
            parsed = json.loads(result_raw)
        except (json.JSONDecodeError, TypeError):
            return result_raw
        if isinstance(parsed, dict) and "text" in parsed:
            return str(parsed["text"])
        return result_raw
    return str(result_raw)


def _parse_letter_sequence(text: str) -> Optional[List[str]]:
    cleaned = text.strip().upper()
    cleaned = cleaned.strip("`\"'[](){}<> \n\t")
    compact_matches = re.fullmatch(r"[A-E]{1,5}", cleaned)
    if compact_matches:
        return list(compact_matches.group(0))

    token_matches = re.findall(r"\b([A-E])\b", cleaned)
    if token_matches:
        return token_matches

    sep_matches = re.findall(r"[A-E]", cleaned)
    if sep_matches and re.fullmatch(r"[A-E,\-\s>]+", cleaned):
        return sep_matches
    return None


def _extract_predicted_labels(text: str) -> Optional[List[str]]:
    stripped = text.strip()
    answer_tag_matches = list(
        re.finditer(r"<ANSWER>(.*?)</ANSWER>", stripped, re.IGNORECASE | re.DOTALL)
    )
    if answer_tag_matches:
        parsed = _parse_letter_sequence(answer_tag_matches[-1].group(1))
        if parsed:
            return parsed

    lines = [line.strip() for line in stripped.splitlines() if line.strip()]
    for line in reversed(lines):
        parsed = _parse_letter_sequence(line)
        if parsed:
            return parsed

    return _parse_letter_sequence(stripped)


def _extract_gold_labels(doc: Dict[str, Any]) -> Optional[List[str]]:
    raw_labels = doc.get("on_path_labels")
    if isinstance(raw_labels, list):
        parsed = []
        for label in raw_labels:
            if not isinstance(label, str):
                return None
            norm = label.strip().upper()
            if norm not in {"A", "B", "C", "D", "E"}:
                return None
            parsed.append(norm)
        return parsed or None
    return _parse_letter_sequence(str(doc.get("answer", "")))


def path_process_results(
    doc: Dict[str, Any], results: List[Any]
) -> Dict[str, Dict[str, float]]:
    result_raw = results[0] if results else ""
    result_text = _normalize_result_text(result_raw)
    predicted_labels = _extract_predicted_labels(result_text)
    gold_labels = _extract_gold_labels(doc)
    exact_match = (
        1.0
        if predicted_labels is not None
        and gold_labels is not None
        and predicted_labels == gold_labels
        else 0.0
    )
    difficulty = str(doc.get("difficulty", "")).strip().lower()

    def _bucket_payload(target: str) -> Dict[str, float]:
        active = 1.0 if difficulty == target else 0.0
        return {"correct": exact_match * active, "total": active}

    return {
        "path_easy_acc": _bucket_payload("easy"),
        "path_middle_acc": _bucket_payload("middle"),
        "path_hard_acc": _bucket_payload("hard"),
        "path_overall_acc": {"correct": exact_match, "total": 1.0},
    }


def path_agg_acc(results: List[Dict[str, float]]) -> float:
    if not results:
        return 0.0
    correct = sum(float(row.get("correct", 0.0)) for row in results)
    total = sum(float(row.get("total", 0.0)) for row in results)
    if total <= 0:
        return 0.0
    return correct / total
