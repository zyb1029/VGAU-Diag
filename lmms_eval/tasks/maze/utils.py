import json
import os
import re
from collections import deque
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from PIL import Image, ImageDraw, ImageFont

_MAZE_LEVEL_ENV = "LMMS_MAZE_LEVEL"
_VALID_LEVELS = {"easy", "middle", "hard"}
_MOVE_DELTAS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}
_MAZE_GRID_SIZE = 10


def _maze_load_image(
    doc: Dict, lmms_eval_specific_kwargs: Dict | None = None
) -> Optional[Image.Image]:
    kwargs = lmms_eval_specific_kwargs or {}
    if "initial_image" not in doc or not doc["initial_image"]:
        return None

    image_field = doc["initial_image"]

    if isinstance(image_field, dict):
        if "bytes" in image_field and image_field["bytes"]:
            return Image.open(BytesIO(image_field["bytes"])).convert("RGB")
        if "path" in image_field and image_field["path"]:
            image_field = image_field["path"]
        else:
            return None

    if isinstance(image_field, (bytes, bytearray)):
        return Image.open(BytesIO(image_field)).convert("RGB")

    if isinstance(image_field, str):
        image_path = Path(image_field)
        path_candidates: List[Path] = []
        if image_path.is_absolute():
            path_candidates.append(image_path)
        else:
            image_root = kwargs.get("maze_image_root")
            if isinstance(image_root, str) and image_root.strip():
                path_candidates.append(Path(image_root) / image_path)
            path_candidates.append(image_path)

        for candidate in path_candidates:
            if candidate.is_file():
                return Image.open(candidate).convert("RGB")

    return None


def _maze_aux_grid_bbox(
    image: Image.Image,
) -> Tuple[int, int, int, int, float, float]:
    """Maze content bbox and NxN cell size (same geometry as red grid)."""
    gray = image.convert("L")
    non_white_mask = gray.point(lambda p: 255 if p < 245 else 0)
    bbox = non_white_mask.getbbox()
    if bbox is None:
        x0, y0, x1, y1 = 0, 0, image.width - 1, image.height - 1
    else:
        left, upper, right, lower = bbox
        x0, y0, x1, y1 = left, upper, right - 1, lower - 1
    cell_w = (x1 - x0 + 1) / float(_MAZE_GRID_SIZE)
    cell_h = (y1 - y0 + 1) / float(_MAZE_GRID_SIZE)
    return x0, y0, x1, y1, cell_w, cell_h


def _maze_pixel_to_aux_cell(
    px: float,
    py: float,
    x0: int,
    y0: int,
    cell_w: float,
    cell_h: float,
) -> Tuple[int, int]:
    if cell_w <= 0 or cell_h <= 0:
        return 0, 0
    col = int((px - x0) // cell_w)
    row = int((py - y0) // cell_h)
    upper = _MAZE_GRID_SIZE - 1
    return max(0, min(upper, row)), max(0, min(upper, col))


def _maze_aux_cell_is_wall(
    image: Image.Image,
    x0: int,
    y0: int,
    cell_w: float,
    cell_h: float,
    row: int,
    col: int,
) -> bool:
    """True if cell interior is mostly black (wall); skips red grid margin."""
    inset_x = max(2.0, cell_w * 0.12)
    inset_y = max(2.0, cell_h * 0.12)
    cx0 = int(x0 + col * cell_w + inset_x)
    cy0 = int(y0 + row * cell_h + inset_y)
    cx1 = int(x0 + (col + 1) * cell_w - inset_x)
    cy1 = int(y0 + (row + 1) * cell_h - inset_y)
    if cx1 <= cx0 or cy1 <= cy0:
        return True
    gray = image.convert("L")
    total, dark = 0, 0
    for yy in range(cy0, cy1 + 1):
        for xx in range(cx0, cx1 + 1):
            total += 1
            if gray.getpixel((xx, yy)) < 85:
                dark += 1
    return total == 0 or (dark / total) > 0.42


def _maze_color_centroid(
    image: Image.Image,
    x0: int,
    y0: int,
    x1: int,
    y1: int,
    pred,
) -> Optional[Tuple[float, float]]:
    px_sum, py_sum, n = 0.0, 0.0, 0
    for y in range(y0, y1 + 1):
        for x in range(x0, x1 + 1):
            r, g, b = image.getpixel((x, y))
            if pred(r, g, b):
                px_sum += x
                py_sum += y
                n += 1
    if n == 0:
        return None
    return px_sum / n, py_sum / n


def _maze_start_goal_aux_cells(
    image: Image.Image,
    x0: int,
    y0: int,
    x1: int,
    y1: int,
    cell_w: float,
    cell_h: float,
) -> Tuple[Optional[Tuple[int, int]], Optional[Tuple[int, int]]]:
    def is_blue_dot(r: int, g: int, b: int) -> bool:
        return b > 120 and b > r + 40 and b > g + 30

    def is_green_goal(r: int, g: int, b: int) -> bool:
        return g > 110 and g > r + 25 and g > b + 25 and r < 200

    start_xy = _maze_color_centroid(image, x0, y0, x1, y1, is_blue_dot)
    goal_xy = _maze_color_centroid(image, x0, y0, x1, y1, is_green_goal)
    start_cell = (
        _maze_pixel_to_aux_cell(start_xy[0], start_xy[1], x0, y0, cell_w, cell_h)
        if start_xy
        else None
    )
    goal_cell = (
        _maze_pixel_to_aux_cell(goal_xy[0], goal_xy[1], x0, y0, cell_w, cell_h)
        if goal_xy
        else None
    )
    return start_cell, goal_cell


def _aux_grid_font(size: int) -> ImageFont.ImageFont:
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


def _draw_text_cell_center(
    draw: ImageDraw.ImageDraw,
    cell_left: float,
    cell_top: float,
    cell_w: float,
    cell_h: float,
    text: str,
    font: ImageFont.ImageFont,
    fill: Tuple[int, int, int],
) -> None:
    """Draw text centered in the cell rectangle."""
    bbox = draw.textbbox((0, 0), text, font=font)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    ox, oy = bbox[0], bbox[1]
    cx = cell_left + cell_w / 2.0
    cy = cell_top + cell_h / 2.0
    tx = int(cx - tw / 2.0 - ox)
    ty = int(cy - th / 2.0 - oy)
    draw.text((tx, ty), text, font=font, fill=fill)


def _build_maze_overlay_grid(
    image: Image.Image,
    label_cell_coords: bool = True,
    label_start_goal: bool = True,
) -> Image.Image:
    """NxN red grid; optional per-cell (r,c) labels and S/G markers."""
    overlay = image.copy()
    x0, y0, x1, y1, cell_w, cell_h = _maze_aux_grid_bbox(image)

    draw = ImageDraw.Draw(overlay)
    draw.rectangle([(x0, y0), (x1, y1)], outline=(255, 0, 0), width=2)

    for i in range(1, _MAZE_GRID_SIZE):
        x = round(x0 + i * cell_w)
        y = round(y0 + i * cell_h)
        draw.line([(x, y0), (x, y1)], fill=(255, 0, 0), width=2)
        draw.line([(x0, y), (x1, y)], fill=(255, 0, 0), width=2)

    black = (0, 0, 0)
    cell_font = _aux_grid_font(max(8, int(min(cell_w, cell_h) * 0.32)))

    start_cell: Optional[Tuple[int, int]] = None
    goal_cell: Optional[Tuple[int, int]] = None
    if label_start_goal:
        start_cell, goal_cell = _maze_start_goal_aux_cells(
            image, x0, y0, x1, y1, cell_w, cell_h
        )

    for r in range(_MAZE_GRID_SIZE):
        for c in range(_MAZE_GRID_SIZE):
            if _maze_aux_cell_is_wall(image, x0, y0, cell_w, cell_h, r, c):
                continue
            if not label_cell_coords and not label_start_goal:
                continue
            cell_left = x0 + c * cell_w
            cell_top = y0 + r * cell_h
            same_sg = (
                label_start_goal
                and start_cell is not None
                and goal_cell is not None
                and start_cell == goal_cell
                and (r, c) == start_cell
            )
            parts: List[str] = []
            if label_start_goal and (r, c) == start_cell:
                parts.append("S")
                if same_sg:
                    parts.append("G")
            elif label_start_goal and (r, c) == goal_cell:
                parts.append("G")
            coord = f"({r},{c})" if label_cell_coords else ""
            if parts and coord:
                label = " ".join(parts) + " " + coord
            elif parts:
                label = " ".join(parts)
            elif coord:
                label = coord
            else:
                continue
            _draw_text_cell_center(
                draw,
                cell_left,
                cell_top,
                cell_w,
                cell_h,
                label,
                cell_font,
                black,
            )

    return overlay



def maze_doc_to_visual(
    doc: Dict, lmms_eval_specific_kwargs: Dict | None = None
) -> List[Image.Image]:
    kwargs = lmms_eval_specific_kwargs or {}
    prompt_mode = _maze_resolve_prompt_mode(kwargs)
    use_aux_grid = prompt_mode in {"aux_cot", "aux_step_by_step"}
    original_image = _maze_load_image(doc, kwargs)
    if original_image is None:
        return []

    if not use_aux_grid:
        return [original_image]

    label_cells = bool(kwargs.get("aux_grid_label_cells", True))
    label_sg = bool(kwargs.get("aux_grid_label_start_goal", False))
    overlay_image = _build_maze_overlay_grid(
        original_image,
        label_cell_coords=label_cells,
        label_start_goal=label_sg,
    )
    return [original_image, overlay_image]


def _format_maze_cell_list(cells: List[Tuple[int, int]]) -> str:
    if not cells:
        return "[]"
    return ", ".join(f"({r},{c})" for r, c in cells)


def _maze_coord_summary_from_doc(
    doc: Dict, lmms_eval_specific_kwargs: Dict | None = None
) -> str:
    image = _maze_load_image(doc, lmms_eval_specific_kwargs)
    if image is None:
        return (
            "COORDINATES\n"
            "- walls: []\n"
            "- start: unknown\n"
            "- goal: unknown\n"
            "- walkable: []"
        )
    x0, y0, x1, y1, cell_w, cell_h = _maze_aux_grid_bbox(image)
    start_cell, goal_cell = _maze_start_goal_aux_cells(
        image, x0, y0, x1, y1, cell_w, cell_h
    )

    walls: List[Tuple[int, int]] = []
    walkable: List[Tuple[int, int]] = []
    for r in range(_MAZE_GRID_SIZE):
        for c in range(_MAZE_GRID_SIZE):
            if _maze_aux_cell_is_wall(image, x0, y0, cell_w, cell_h, r, c):
                walls.append((r, c))
            else:
                walkable.append((r, c))

    start_text = f"({start_cell[0]},{start_cell[1]})" if start_cell else "unknown"
    goal_text = f"({goal_cell[0]},{goal_cell[1]})" if goal_cell else "unknown"
    return (
        "COORDINATES\n"
        f"- walls: {_format_maze_cell_list(walls)}\n"
        f"- start: {start_text}\n"
        f"- goal: {goal_text}\n"
        f"- walkable: {_format_maze_cell_list(walkable)}"
    )


def _maze_resolve_prompt_mode(kwargs: Dict) -> str:
    raw_mode = kwargs.get("maze_prompt_mode")
    if raw_mode is None:
        raise ValueError(
            "maze_prompt_mode is required and must be one of: "
            "aux, cot, direct, gta, vision_reasoning, step-by-step, aux_step_by_step"
        )
    mode = str(raw_mode).strip().lower()
    mode_aliases = {
        "aux": "aux_cot",
        "step-by-step": "step_by_step",
        "aux-step-by-step": "aux_step_by_step",
        "vision-reasoning": "vision_reasoning",
    }
    resolved_mode = mode_aliases.get(mode, mode)
    if resolved_mode not in {
        "aux_cot",
        "cot",
        "direct",
        "gta",
        "vision_reasoning",
        "step_by_step",
        "aux_step_by_step",
    }:
        raise ValueError(
            f"Invalid maze_prompt_mode: {raw_mode}. "
            "Expected one of: aux, cot, direct, gta, vision_reasoning, "
            "step-by-step, aux_step_by_step"
        )
    return resolved_mode


def maze_doc_to_text(doc: Dict, lmms_eval_specific_kwargs: Dict = None) -> str:
    kwargs = lmms_eval_specific_kwargs or {}
    selected_variant = _maze_resolve_prompt_mode(kwargs)
    coord_summary = _maze_coord_summary_from_doc(doc, kwargs)

    prompt_variants: Dict[str, str] = {
        "cot": """You are a precise maze solver.

SEMANTICS
- Black squares: walls (impassable)
- White squares: path (walkable)
- Blue dot: start (the agent)
- The green square marks the goal cell. It is passable, not a wall, and success is achieved only when the agent moves onto the cell occupied by the green square.
- Legal moves: up, down, left, right only. One cell per step; no diagonals, no jumps; never cross walls.

OUTPUT FORMAT
1) Describe your reasoning briefly: identify the start (agent) and goal, explain the planned move sequence, and confirm never enters black-wall cells.
2) Output exactly one final move list as a JSON array of lowercase strings, wrapped as:
<ANSWER_JSON>[\"move1\",\"move2\",...]</ANSWER_JSON>
""",
        "direct": """You are a precise maze solver.

SEMANTICS
- Black squares: walls (impassable)
- White squares: path (walkable)
- Blue dot: start (the agent)
- The green square marks the goal cell. It is passable, not a wall, and success is achieved only when the agent moves onto the cell occupied by the green square.
- Legal moves: up, down, left, right only. One cell per step; no diagonals, no jumps; never cross walls.

OUTPUT FORMAT
Directly output final move list as a JSON array of lowercase strings, wrapped as:
<ANSWER_JSON>[\"move1\",\"move2\",...]</ANSWER_JSON>
Do not output any other text.
""",
        "aux_cot": """<AUX_GRID_INTERLEAVE>
You are a precise maze solver.

SEMANTICS
- Black squares: walls (impassable)
- White squares: path (walkable)
- Blue dot: start (the agent)
- The green square marks the goal cell. It is passable, not a wall, and success is achieved only when the agent moves onto the cell occupied by the green square.
- Legal moves: up, down, left, right only. One cell per step; no diagonals, no jumps; never cross walls.

First, generate one auxiliary image from the original maze image by drawing red grid-boundary lines for a 10x10 partition and centered coordinate labels "(r,c)" on each walkable (white) cell while preserving the original maze colors and geometry. The top-left walkable cell is (0,0), row index r increases downward, and column index c increases rightward.

OUTPUT FORMAT
1) Describe your reasoning briefly: identify the start (agent) and goal, explain the planned move sequence, and confirm never enters black-wall cells.
2) Output exactly one final move list as a JSON array of lowercase strings, wrapped as:
<ANSWER_JSON>[\"move1\",\"move2\",...]</ANSWER_JSON>
""",

        "gta": """
[GEN_PROMPT]
Edit the input maze image to create exactly one auxiliary annotated image.

Maze semantics:
- Black squares are walls and must remain unchanged.
- White squares are walkable cells.
- The blue dot is the start cell and must remain unchanged.
- The green square is the goal cell, is walkable, and must remain unchanged.

Editing requirements:
1. Preserve the original maze geometry, cell layout, wall positions, start marker, goal marker, and original colors.
2. Overlay thin red grid-boundary lines that divide the maze image into 10 equal parts along the horizontal axis and 10 equal parts along the vertical axis; draw 9 evenly spaced vertical red lines at 10%, 20%, ..., 90% of the image width and 9 evenly spaced horizontal red lines at 10%, 20%, ..., 90% of the image height, forming a 10×10 partition aligned with the maze cells. As a placement reference, the green square should lie exactly inside one grid cell, with red grid boundaries flush with its four sides.
3. Label every cell (black squares、 white squares、 blue dot、 green square) at its center with its coordinate in black text, using the format (r,c). The top-left cell is (0,0). The first row is labeled left-to-right as (0,0), (0,1), (0,2), (0,3), ... until the last cell in that row. The second row is labeled left-to-right as (1,0), (1,1), (1,2), (1,3), ... . In general, row index r increases by 1 when moving downward, and column index c increases by 1 when moving rightward.
4. Do not add any arrows, path lines, legend, explanation text, title, or extra decorations.
[/GEN_PROMPT]

[QUESTION]
You are a precise maze solver.

SEMANTICS
- Black squares: walls (impassable)
- White squares: path (walkable)
- Blue dot: start (the agent)
- The green square marks the goal cell. It is passable, not a wall, and success is achieved only when the agent moves onto the cell occupied by the green square.
- Legal moves: up, down, left, right only. One cell per step; no diagonals, no jumps; never cross walls.

OUTPUT FORMAT
1) Describe your reasoning briefly: identify thestart (agent) and goal, explain the planned move sequence, and confirm never enters black-wall cells.
2) Output exactly one final move list as a JSON array of lowercase strings, wrapped as:
<ANSWER_JSON>[\"move1\",\"move2\",...]</ANSWER_JSON>
[/QUESTION]
""",
        "vision_reasoning": """
[GEN_PROMPT]
Edit the input maze image to create exactly one auxiliary annotated image that highlights a feasible solution path.

Maze semantics:
- Black squares are walls and must remain unchanged.
- White squares are walkable cells.
- The blue dot is the start cell and must remain unchanged.
- The green square is the goal cell, is walkable, and must remain unchanged.

Editing requirements:
1. Preserve the original maze geometry, cell layout, wall positions, start marker, goal marker, and original colors.
2. Overlay a single continuous RED polyline that traces a legal path from the blue start cell to the green goal cell:
   - The polyline must follow walkable (white) cells only and must NEVER cross or touch any black wall cell.
   - Movement is restricted to up / down / left / right between cell centers; no diagonals, no jumps.
   - The polyline starts exactly at the center of the blue start cell and ends exactly at the center of the green goal cell.
   - Use a clean, solid bright red stroke of moderate thickness (roughly 10-15 percent of a cell width) so the path is clearly visible against the maze background.
3. Do not draw arrows, dots along the way, grid lines, coordinate labels, legend, captions, titles, or any other decoration. Only the single red polyline should be added.
4. Do not modify the colors or positions of the start marker, goal marker, walls, or walkable cells in any other way.
[/GEN_PROMPT]

[QUESTION]
You are a precise maze solver.

SEMANTICS
- Black squares: walls (impassable)
- White squares: path (walkable)
- Blue dot: start (the agent)
- The green square marks the goal cell. It is passable, not a wall, and success is achieved only when the agent moves onto the cell occupied by the green square.
- Legal moves: up, down, left, right only. One cell per step; no diagonals, no jumps; never cross walls.

You are given two images: the original maze and an auxiliary image where a red polyline traces a candidate path from the start to the goal. Use the red polyline as a visual hint, but verify every step against the maze itself and never enter black-wall cells.

OUTPUT FORMAT
1) Describe your reasoning briefly: identify the start (agent) and goal, follow the red polyline cell by cell, and confirm the move sequence never enters black-wall cells.
2) Output exactly one final move list as a JSON array of lowercase strings, wrapped as:
<ANSWER_JSON>[\"move1\",\"move2\",...]</ANSWER_JSON>
[/QUESTION]
""",
        "step_by_step": """You are a precise maze solver.

SEMANTICS
- Black squares: walls (impassable)
- White squares: path (walkable)
- Blue dot: start (the agent)
- The green square marks the goal cell. It is passable, not a wall.
- Legal moves: up, down, left, right only. One cell per step.

OUTPUT FORMAT (STRICT)
1) MULTI-IMAGE MODE — generate a SEQUENCE OF SEPARATE IMAGES, one per move:
   - Each output image must depict the maze state AFTER applying exactly one legal move.
   - Do NOT include the initial (pre-move) state.
   - Keep palette/layout/scale identical to the input; only the blue dot moves.
   - The number of returned images MUST equal the number of moves in the final answer (see step 2).
   - Absolutely FORBIDDEN: any collage/montage/spritesheet/grid/multi-panel/side-by-side/stacked images; no arrows, captions, or overlays; no GIFs/animations/video.

2) After all step images, emit EXACTLY ONE LINE containing ONLY the final move list as a JSON array of lowercase strings, wrapped as:
   <ANSWER_JSON>["right","down","left"]</ANSWER_JSON>


NO EXTRAS
- No tools, no OCR, no explanations, and no text other than the single <ANSWER_JSON>…</ANSWER_JSON> line.
- Do not restate the instructions or the condition.

REMINDERS
- Decide the full path first, then emit the image sequence (one image per move), then the single <ANSWER_JSON> line.
- One move per image; images must be separate files/parts, not stitched together in any way.""",
        "aux_step_by_step": """<AUX_GRID_INTERLEAVE>
You are a precise maze solver.

SEMANTICS
- Black squares: walls (impassable)
- White squares: path (walkable)
- Blue dot: start (the agent)
- The green square marks the goal cell. It is passable, not a wall.
- Legal moves: up, down, left, right only. One cell per step.

OUTPUT FORMAT (STRICT)
1) FIRST OUTPUT — generate EXACTLY ONE auxiliary image from the original maze image:
   - Draw red grid-boundary lines to show a 10x10 partition over the maze.
   - On every walkable white cell, place a centered coordinate label in the form "(r,c)".
   - Preserve the original maze colors, layout, and geometry; only add the red grid lines and coordinate labels.
   - The top-left walkable cell is (0,0).
   - Row index r increases downward.
   - Column index c increases rightward.
   - Do NOT alter the position of the blue dot or the goal in this auxiliary image.

2) MULTI-IMAGE MODE — after the auxiliary image, generate a SEQUENCE OF SEPARATE IMAGES, one per move:
   - Each output image must be based on the auxiliary image.
   - Each output image must depict the maze state AFTER applying exactly one legal move.
   - Do NOT include the initial (pre-move) state again.
   - Keep palette, layout, scale, red grid lines, and coordinate labels identical to the auxiliary image; only the blue dot moves.
   - The number of returned move-state images MUST equal the number of moves in the final answer (see step 3).
   - Absolutely FORBIDDEN: any collage, montage, spritesheet, grid, multi-panel, side-by-side, or stacked images; no arrows, captions, or overlays; no GIFs, animations, or video.

3) After all images, emit EXACTLY ONE LINE containing ONLY the final move list as a JSON array of lowercase strings, wrapped as:
   <ANSWER_JSON>["right","down","left"]</ANSWER_JSON>

NO EXTRAS
- No tools, no OCR, no explanations, and no text other than the single <ANSWER_JSON>…</ANSWER_JSON> line.
- Do not restate the instructions or the condition.

REMINDERS
- Decide the full path first, then emit the image sequence (one image per move), then the single <ANSWER_JSON> line.
- One move per image; images must be separate files/parts, not stitched together in any way.
""",
    }
    return prompt_variants[selected_variant]



def _maze_normalize_result_text(result_raw: Any) -> str:
    if isinstance(result_raw, str):
        try:
            parsed_result = json.loads(result_raw)
            if isinstance(parsed_result, dict) and "text" in parsed_result:
                return str(parsed_result["text"])
        except (json.JSONDecodeError, TypeError):
            pass
        return result_raw
    return str(result_raw)


def _maze_extract_pred_moves(result_text: str) -> List[str]:
    matches = list(
        re.finditer(
            r"<ANSWER_JSON>\s*(\[.*?\])\s*</ANSWER_JSON>",
            result_text,
            re.DOTALL | re.IGNORECASE,
        )
    )

    candidate_json: Optional[str] = None
    if matches:
        candidate_json = matches[-1].group(1)
    else:
        stripped = result_text.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            candidate_json = stripped

    if not candidate_json:
        return []

    try:
        parsed = json.loads(candidate_json)
    except Exception:
        return []
    if not isinstance(parsed, list):
        return []
    return [str(move).strip().lower() for move in parsed]


def _maze_parse_coord(raw_coord: Any) -> Optional[Tuple[int, int]]:
    if not isinstance(raw_coord, (list, tuple)) or len(raw_coord) != 2:
        return None
    try:
        return int(raw_coord[0]), int(raw_coord[1])
    except (TypeError, ValueError):
        return None


def _maze_parse_walkable(doc: Dict) -> Set[Tuple[int, int]]:
    raw_walkable = doc.get("walkable", [])
    if not isinstance(raw_walkable, list):
        return set()
    out: Set[Tuple[int, int]] = set()
    for item in raw_walkable:
        parsed = _maze_parse_coord(item)
        if parsed is not None:
            out.add(parsed)
    return out


def _maze_cell_inner_box(
    x0: int,
    y0: int,
    cell_w: float,
    cell_h: float,
    row: int,
    col: int,
    inset_ratio: float = 0.18,
) -> Tuple[int, int, int, int]:
    inset_x = max(1.0, cell_w * inset_ratio)
    inset_y = max(1.0, cell_h * inset_ratio)
    left = int(x0 + col * cell_w + inset_x)
    top = int(y0 + row * cell_h + inset_y)
    right = int(x0 + (col + 1) * cell_w - inset_x)
    bottom = int(y0 + (row + 1) * cell_h - inset_y)
    return left, top, right, bottom


def maze_render_state_image(
    doc: Dict,
    current_pos: Tuple[int, int],
    lmms_eval_specific_kwargs: Dict | None = None,
    base_image: Image.Image | None = None,
) -> Optional[Image.Image]:
    image = (
        base_image
        if base_image is not None
        else _maze_load_image(doc, lmms_eval_specific_kwargs)
    )
    if image is None:
        return None

    canvas = image.copy()
    draw = ImageDraw.Draw(canvas)
    x0, y0, _, _, cell_w, cell_h = _maze_aux_grid_bbox(canvas)

    start = _maze_parse_coord(doc.get("start"))
    if start is None:
        return canvas

    def _cell_pixel_bounds(row: int, col: int) -> Tuple[int, int, int, int]:
        left = int(round(x0 + col * cell_w))
        top = int(round(y0 + row * cell_h))
        right = int(round(x0 + (col + 1) * cell_w)) - 1
        bottom = int(round(y0 + (row + 1) * cell_h)) - 1
        return left, top, right, bottom

    def _is_blue(r: int, g: int, b: int) -> bool:
        return b > 120 and b > r + 40 and b > g + 30

    s_left, s_top, s_right, s_bottom = _cell_pixel_bounds(start[0], start[1])
    blue_pixels: List[Tuple[int, int, Tuple[int, int, int]]] = []
    background_pixels: List[Tuple[int, int, int]] = []
    for py in range(s_top, s_bottom + 1):
        for px in range(s_left, s_right + 1):
            r, g, b = image.getpixel((px, py))
            if _is_blue(r, g, b):
                blue_pixels.append((px - s_left, py - s_top, (r, g, b)))
            else:
                background_pixels.append((r, g, b))

    if background_pixels:
        avg_r = int(sum(c[0] for c in background_pixels) / len(background_pixels))
        avg_g = int(sum(c[1] for c in background_pixels) / len(background_pixels))
        avg_b = int(sum(c[2] for c in background_pixels) / len(background_pixels))
        clear_color = (avg_r, avg_g, avg_b)
    else:
        clear_color = (255, 255, 255)

    for rel_x, rel_y, _ in blue_pixels:
        canvas.putpixel((s_left + rel_x, s_top + rel_y), clear_color)

    t_left, t_top, t_right, t_bottom = _cell_pixel_bounds(current_pos[0], current_pos[1])
    if not blue_pixels:
        c_left, c_top, c_right, c_bottom = _maze_cell_inner_box(
            x0,
            y0,
            cell_w,
            cell_h,
            current_pos[0],
            current_pos[1],
            inset_ratio=0.32,
        )
        draw.rectangle((c_left, c_top, c_right, c_bottom), fill=(0, 102, 255))
        return canvas

    for rel_x, rel_y, color in blue_pixels:
        tx = t_left + rel_x
        ty = t_top + rel_y
        if t_left <= tx <= t_right and t_top <= ty <= t_bottom:
            canvas.putpixel((tx, ty), color)
    return canvas


def _maze_simulate_moves(
    start: Tuple[int, int],
    moves: List[str],
    walkable: Set[Tuple[int, int]],
) -> Tuple[bool, Tuple[int, int]]:
    cur = start
    for move in moves:
        delta = _MOVE_DELTAS.get(move)
        if delta is None:
            return False, cur
        nxt = (cur[0] + delta[0], cur[1] + delta[1])
        if nxt not in walkable:
            return False, cur
        cur = nxt
    return True, cur


def _maze_shortest_path_len(
    start: Tuple[int, int],
    goal: Tuple[int, int],
    walkable: Set[Tuple[int, int]],
) -> Optional[int]:
    if start not in walkable or goal not in walkable:
        return None
    q: deque[Tuple[int, int]] = deque([start])
    distance = {start: 0}
    while q:
        cur = q.popleft()
        if cur == goal:
            return distance[cur]
        for dr, dc in _MOVE_DELTAS.values():
            nxt = (cur[0] + dr, cur[1] + dc)
            if nxt in walkable and nxt not in distance:
                distance[nxt] = distance[cur] + 1
                q.append(nxt)
    return None


def _maze_aggregate_reach_any(
    results: List[Dict[str, Any]], level: Optional[str] = None
) -> float:
    filtered = [
        row for row in results if level is None or row.get("difficulty") == level
    ]
    if not filtered:
        return 0.0
    return sum(float(row.get("reach_any", 0.0)) for row in filtered) / len(filtered)


def _maze_aggregate_reach_optimal(
    results: List[Dict[str, Any]], level: Optional[str] = None
) -> float:
    filtered = [
        row for row in results if level is None or row.get("difficulty") == level
    ]
    if not filtered:
        return 0.0
    return sum(float(row.get("reach_optimal", 0.0)) for row in filtered) / len(filtered)


def maze_agg_reach_any_easy(results: List[Dict[str, Any]]) -> float:
    return _maze_aggregate_reach_any(results, level="easy")


def maze_agg_reach_any_middle(results: List[Dict[str, Any]]) -> float:
    return _maze_aggregate_reach_any(results, level="middle")


def maze_agg_reach_any_hard(results: List[Dict[str, Any]]) -> float:
    return _maze_aggregate_reach_any(results, level="hard")


def maze_agg_reach_any_overall(results: List[Dict[str, Any]]) -> float:
    return _maze_aggregate_reach_any(results, level=None)


def maze_agg_reach_optimal_easy(results: List[Dict[str, Any]]) -> float:
    return _maze_aggregate_reach_optimal(results, level="easy")


def maze_agg_reach_optimal_middle(results: List[Dict[str, Any]]) -> float:
    return _maze_aggregate_reach_optimal(results, level="middle")


def maze_agg_reach_optimal_hard(results: List[Dict[str, Any]]) -> float:
    return _maze_aggregate_reach_optimal(results, level="hard")


def maze_agg_reach_optimal_overall(results: List[Dict[str, Any]]) -> float:
    return _maze_aggregate_reach_optimal(results, level=None)


def maze_process_docs(dataset):
    level = os.getenv(_MAZE_LEVEL_ENV, "").strip().lower()
    if not level:
        return dataset
    if level not in _VALID_LEVELS:
        raise ValueError(
            f"Invalid {_MAZE_LEVEL_ENV}: {level}. "
            f"Expected one of {sorted(_VALID_LEVELS)}."
        )
    return dataset.filter(lambda row: str(row.get("difficulty", "")).strip().lower() == level)


def maze_process_results(doc: Dict, results: List[str]) -> Dict[str, Any]:
    result_raw = results[0] if results else ""
    result_text = _maze_normalize_result_text(result_raw)
    pred_moves = _maze_extract_pred_moves(result_text)

    difficulty = str(doc.get("difficulty", "")).strip().lower()
    start = _maze_parse_coord(doc.get("start"))
    goal = _maze_parse_coord(doc.get("goal"))
    walkable = _maze_parse_walkable(doc)

    reach_any = 0.0
    reach_optimal = 0.0
    if start is not None and goal is not None and walkable:
        legal, end = _maze_simulate_moves(start, pred_moves, walkable)
        if legal and end == goal:
            reach_any = 1.0
            shortest_len = _maze_shortest_path_len(start, goal, walkable)
            if shortest_len is not None and len(pred_moves) == shortest_len:
                reach_optimal = 1.0

    payload = {
        "difficulty": difficulty,
        "reach_any": reach_any,
        "reach_optimal": reach_optimal,
    }
    return {
        "maze_reach_any_easy": payload,
        "maze_reach_any_middle": payload,
        "maze_reach_any_hard": payload,
        "maze_reach_any_overall": payload,
        "maze_reach_optimal_easy": payload,
        "maze_reach_optimal_middle": payload,
        "maze_reach_optimal_hard": payload,
        "maze_reach_optimal_overall": payload,
    }
