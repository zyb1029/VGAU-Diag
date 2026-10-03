from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image, ImageDraw, ImageFont

_DEFAULT_IMAGE_ROOT = "./data/parking"
_GRID_SIZE = 8
_CELL_SIZE = 72
_BOARD_MARGIN = 36
_CANVAS_PADDING = 24
_PROMPT_MODES = {
    "aux",
    "cot",
    "direct",
    "step_by_step",
    "aux_step_by_step",
    "gta",
    "vision_reasoning",
}

_CAR_COLORS: Dict[str, Tuple[int, int, int]] = {
    "R": (222, 55, 52),
    "A": (59, 110, 183),
    "B": (64, 155, 88),
    "C": (198, 86, 40),
    "D": (103, 97, 108),
    "E": (141, 95, 177),
    "F": (194, 72, 69),
    "G": (165, 145, 62),
    "H": (218, 127, 57),
    "I": (55, 166, 170),
    "J": (114, 126, 61),
    "K": (102, 149, 204),
    "L": (171, 103, 84),
    "M": (89, 140, 160),
    "N": (200, 130, 60),
}


def _resolve_prompt_mode(kwargs: Dict[str, Any]) -> str:
    raw_mode = kwargs.get("parking_prompt_mode")
    if raw_mode is None:
        raise ValueError(
            "parking_prompt_mode is required and must be one of: "
            "aux, cot, direct, step_by_step, aux_step_by_step, gta, vision_reasoning"
        )
    mode = str(raw_mode).strip().lower()
    mode = {
        "step-by-step": "step_by_step",
        "aux-step-by-step": "aux_step_by_step",
        "vision-reasoning": "vision_reasoning",
    }.get(mode, mode)
    if mode not in _PROMPT_MODES:
        raise ValueError(
            f"Invalid parking_prompt_mode: {raw_mode}. "
            "Expected one of: aux, cot, direct, step_by_step, aux_step_by_step, gta, vision_reasoning"
        )
    return mode


def _resolve_image_path(doc: Dict[str, Any], kwargs: Dict[str, Any]) -> Path:
    image_rel = str(doc.get("image_path", "")).strip()
    if not image_rel:
        raise FileNotFoundError("missing image_path in parking sample")
    image_root = str(kwargs.get("parking_image_root", _DEFAULT_IMAGE_ROOT)).strip()
    image_path = Path(image_rel)
    if image_path.is_absolute():
        return image_path
    return Path(image_root) / image_path


def parking_doc_to_visual(
    doc: Dict[str, Any], lmms_eval_specific_kwargs: Dict[str, Any] | None = None
) -> List[Image.Image]:
    kwargs = lmms_eval_specific_kwargs or {}
    prompt_mode = _resolve_prompt_mode(kwargs)
    image_path = _resolve_image_path(doc, kwargs)
    if not image_path.is_file():
        raise FileNotFoundError(f"parking image not found: {image_path}")
    image = Image.open(image_path).convert("RGB")
    if prompt_mode in {"aux", "aux_step_by_step"}:
        grid_size = int(doc.get("grid_size", _GRID_SIZE))
        return [image, _build_aux_overlay(image, grid_size=grid_size)]
    return [image]


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


def _find_board_bbox(image: Image.Image) -> Tuple[int, int, int, int]:
    width, height = image.size
    pixels = image.convert("RGB")

    def is_dark(px: Tuple[int, int, int]) -> bool:
        r, g, b = px
        return r < 120 and g < 120 and b < 120

    col_counts = [0] * width
    row_counts = [0] * height
    for y in range(height):
        for x in range(width):
            if is_dark(pixels.getpixel((x, y))):
                col_counts[x] += 1
                row_counts[y] += 1

    col_threshold = max(10, int(height * 0.35))
    row_threshold = max(10, int(width * 0.35))

    candidate_cols = [x for x, c in enumerate(col_counts) if c >= col_threshold]
    candidate_rows = [y for y, c in enumerate(row_counts) if c >= row_threshold]
    if not candidate_cols or not candidate_rows:
        return 0, 0, width - 1, height - 1

    x0 = min(candidate_cols)
    x1 = max(candidate_cols)
    y0 = min(candidate_rows)
    y1 = max(candidate_rows)
    if x1 <= x0 or y1 <= y0:
        return 0, 0, width - 1, height - 1
    return x0, y0, x1, y1


def _build_aux_overlay(image: Image.Image, grid_size: int) -> Image.Image:
    overlay = image.copy().convert("RGB")
    draw = ImageDraw.Draw(overlay)
    x0, y0, x1, y1 = _find_board_bbox(overlay)
    board_w = x1 - x0 + 1
    board_h = y1 - y0 + 1
    if grid_size <= 0:
        grid_size = _GRID_SIZE
    cell_w = board_w / float(grid_size)
    cell_h = board_h / float(grid_size)

    line_color = (230, 40, 40)
    draw.rectangle(
        [(x0, y0), (x1, y1)],
        outline=line_color,
        width=2,
    )
    for i in range(1, grid_size):
        x = int(round(x0 + i * cell_w))
        y = int(round(y0 + i * cell_h))
        draw.line([(x, y0), (x, y1)], fill=line_color, width=2)
        draw.line([(x0, y), (x1, y)], fill=line_color, width=2)

    font = _get_font(max(10, int(min(cell_w, cell_h) * 0.2)))
    for r in range(grid_size):
        for c in range(grid_size):
            text = f"({r},{c})"
            left = int(round(x0 + c * cell_w))
            top = int(round(y0 + r * cell_h))
            bbox = draw.textbbox((0, 0), text, font=font)
            tw = bbox[2] - bbox[0]
            th = bbox[3] - bbox[1]
            tx = int(left + (cell_w - tw) / 2 - bbox[0])
            ty = int(top + (cell_h - th) / 2 - bbox[1])
            draw.text((tx, ty), text, font=font, fill=(0, 0, 0))
    return overlay


def parking_doc_to_text(
    doc: Dict[str, Any], lmms_eval_specific_kwargs: Dict[str, Any] | None = None
) -> str:
    kwargs = lmms_eval_specific_kwargs or {}
    prompt_mode = _resolve_prompt_mode(kwargs)
    grid_size = int(doc.get("grid_size", _GRID_SIZE))
    exit_row = int(doc.get("exit_row", 2))
    base = f"""You are solving a parking-exit puzzle on a {grid_size}x{grid_size} board.

Rules:
1) Vehicle orientations are fixed (horizontal or vertical).
2) In one move, exactly one vehicle slides by one cell along its orientation.
3) No overlap and no leaving the board.
4) Success is achieved when the rightmost cell of the red car "R" reaches the right boundary on row index {exit_row - 1} (indices are 0-based).
"""
    direct = (
        base
        + """

Output strictly:
<ANSWER_JSON>[{"car":"A","direction":"up"}, {"car":"R","direction":"right"}]</ANSWER_JSON>

Use only directions from: up, down, left, right. Do not output any extra text.
"""
    )
    cot = (
        base
        + """

OUTPUT FORMAT

1) Describe your reasoning briefly.
2) Answer using this format:

<ANSWER_JSON>[{"car":"A","direction":"up"}, {"car":"R","direction":"right"}]</ANSWER_JSON>
"""
    )
    aux = (
        "<AUX_GRID_INTERLEAVE>\n"
        + base
        + """

First, generate one auxiliary image by drawing red grid-boundary lines for a GRID_SIZE_PLACEHOLDER partition and centered coordinate labels "(r,c)" on each cell while preserving original colors and geometry.

OUTPUT FORMAT

1) Describe your reasoning briefly.
2) Answer using this format:

<ANSWER_JSON>[{"car":"A","direction":"up"}, {"car":"R","direction":"right"}]</ANSWER_JSON>
"""
    ).replace("GRID_SIZE_PLACEHOLDER", f"{grid_size}x{grid_size}")
    step_by_step = (
        base
        + """
OUTPUT FORMAT (STRICT)
1) MULTI-IMAGE MODE — generate a SEQUENCE OF SEPARATE IMAGES, one per move:
   - Each output image must depict the board state AFTER applying exactly one legal move.
   - Do NOT include the initial (pre-move) state.
   - Keep palette/layout/scale identical to the input; only the moved vehicle changes position.
   - The number of returned images MUST equal the number of moves in the final answer (see step 2).
   - Absolutely FORBIDDEN: any collage/montage/spritesheet/grid/multi-panel/side-by-side/stacked images; no arrows, captions, or overlays; no GIFs/animations/video.

2) After all step images, emit EXACTLY ONE LINE containing ONLY the final move list as a JSON array of move objects, wrapped as:
   <ANSWER_JSON>[{"car":"A","direction":"up"}, {"car":"R","direction":"right"}]</ANSWER_JSON>

NO EXTRAS
- No tools, no OCR, no explanations, and no text other than the single <ANSWER_JSON>…</ANSWER_JSON> line.
- Do not restate the instructions or the condition.

REMINDERS
- Decide the full move sequence first, then emit the image sequence (one image per move), then the single <ANSWER_JSON> line.
- One move per image; images must be separate files/parts, not stitched together in any way.
"""
    )
    aux_step_by_step = (
        "<AUX_GRID_INTERLEAVE>\n"
        + base
        + f"""
OUTPUT FORMAT (STRICT)
1) FIRST OUTPUT — generate EXACTLY ONE auxiliary image from the original board image:
   - Draw red grid-boundary lines to show a {grid_size}x{grid_size} partition over the board.
   - On every cell, place a centered coordinate label in the form "(r,c)".
   - Preserve the original board colors, layout, and geometry; only add the red grid lines and coordinate labels.
   - The top-left cell is (0,0). Row index r increases downward. Column index c increases rightward.

2) MULTI-IMAGE MODE — after the auxiliary image, generate a SEQUENCE OF SEPARATE IMAGES, one per move:
   - Each output image must be based on the auxiliary image.
   - Each output image must depict the board state AFTER applying exactly one legal move.
   - Do NOT include the initial (pre-move) state again.
   - Keep palette, layout, scale, red grid lines, and coordinate labels identical to the auxiliary image; only the moved vehicle changes position.
   - The number of returned move-state images MUST equal the number of moves in the final answer (see step 3).
   - Absolutely FORBIDDEN: any collage, montage, spritesheet, grid, multi-panel, side-by-side, or stacked images; no arrows, captions, or overlays; no GIFs, animations, or video.

3) After all images, emit EXACTLY ONE LINE containing ONLY the final move list as a JSON array of move objects, wrapped as:
   <ANSWER_JSON>[{{"car":"A","direction":"up"}}, {{"car":"R","direction":"right"}}]</ANSWER_JSON>

NO EXTRAS
- No tools, no OCR, no explanations, and no text other than the single <ANSWER_JSON>…</ANSWER_JSON> line.
- Do not restate the instructions or the condition.

REMINDERS
- Decide the full move sequence first, then emit the image sequence (one image per move), then the single <ANSWER_JSON> line.
- One move per image; images must be separate files/parts, not stitched together in any way.
"""
    )
    gta = (
        f"""[GEN_PROMPT]
Edit the input parking-exit puzzle image to create exactly one auxiliary annotated image.

Puzzle semantics:
- The board is a {grid_size}x{grid_size} grid.
- Each vehicle has fixed orientation and length.
- Legal moves slide one vehicle by one cell along its orientation.
- The red car "R" must exit to the right boundary on row index {exit_row - 1} (0-based).

Editing requirements:
1. Preserve the original board geometry, vehicle positions, labels, and colors exactly.
2. Overlay thin red grid-boundary lines aligned to the board cell boundaries.
3. Label every cell at its center with black text in the format (r,c). The top-left cell is (0,0); r increases downward and c increases rightward.
4. Do not add arrows, path lines, legends, captions, titles, or any extra decorations.
[/GEN_PROMPT]

[QUESTION]
You are solving a parking-exit puzzle on a {grid_size}x{grid_size} board.

Rules:
1) Vehicle orientations are fixed (horizontal or vertical).
2) In one move, exactly one vehicle slides by one cell along its orientation.
3) No overlap and no leaving the board.
4) Success is achieved when the rightmost cell of the red car "R" reaches the right boundary on row index {exit_row - 1} (indices are 0-based).

You are given two images: the original board and an auxiliary image with grid lines and coordinate labels. Use the auxiliary image only as a coordinate aid and verify every move remains legal.

OUTPUT FORMAT
1) Describe your reasoning briefly.
2) Output exactly one final move list wrapped as:
<ANSWER_JSON>[{{"car":"A","direction":"up"}}, {{"car":"R","direction":"right"}}]</ANSWER_JSON>
[/QUESTION]
"""
    )
    vision_reasoning = (
        f"""[GEN_PROMPT]
Edit the input parking-exit puzzle image to create exactly one auxiliary annotated image that highlights a feasible move strategy for the red car.

Puzzle semantics:
- The board is a {grid_size}x{grid_size} grid.
- Each vehicle has fixed orientation and length.
- Legal moves slide one vehicle by one cell along its orientation.
- The red car "R" must exit to the right boundary on row index {exit_row - 1} (0-based).

Editing requirements:
1. Preserve the original board geometry, vehicle positions, labels, and colors exactly.
2. Add visual hints that indicate a feasible strategy (for example, numbered move markers or directional cues near vehicles) while keeping all hints on valid cells only.
3. Ensure the hints correspond to legal single-step moves and do not imply illegal orientation changes or overlaps.
4. Do not alter car locations in the image; only overlay hints. Do not add unrelated decorations.
[/GEN_PROMPT]

[QUESTION]
You are solving a parking-exit puzzle on a {grid_size}x{grid_size} board.

Rules:
1) Vehicle orientations are fixed (horizontal or vertical).
2) In one move, exactly one vehicle slides by one cell along its orientation.
3) No overlap and no leaving the board.
4) Success is achieved when the rightmost cell of the red car "R" reaches the right boundary on row index {exit_row - 1} (indices are 0-based).

You are given two images: the original board and an auxiliary image with visual hints. Use the hints as guidance, but verify every move is legal on the original board.

OUTPUT FORMAT
1) Describe your reasoning briefly.
2) Output exactly one final move list wrapped as:
<ANSWER_JSON>[{{"car":"A","direction":"up"}}, {{"car":"R","direction":"right"}}]</ANSWER_JSON>
[/QUESTION]
"""
    )
    if prompt_mode == "direct":
        return direct
    if prompt_mode == "cot":
        return cot
    if prompt_mode == "gta":
        return gta
    if prompt_mode == "vision_reasoning":
        return vision_reasoning
    if prompt_mode == "step_by_step":
        return step_by_step
    if prompt_mode == "aux_step_by_step":
        return aux_step_by_step
    return aux


def _darken(color: Tuple[int, int, int], amount: int) -> Tuple[int, int, int]:
    return (max(0, color[0] - amount), max(0, color[1] - amount), max(0, color[2] - amount))


def _lighten(color: Tuple[int, int, int], amount: int) -> Tuple[int, int, int]:
    return (
        min(255, color[0] + amount),
        min(255, color[1] + amount),
        min(255, color[2] + amount),
    )


def parking_render_state_image(
    cars: List[Dict[str, Any]],
    grid_size: int,
    exit_row_0: int,
    base_image: Optional[Image.Image] = None,
) -> Image.Image:
    """Render the current parking board state as a PIL image.

    Args:
        cars: List of car dicts with id, orientation, length, row, col.
        grid_size: Board side length in cells.
        exit_row_0: 0-based exit row (red car must reach here).
        base_image: Original board image for canvas sizing (copied as background).
    """
    board_px = grid_size * _CELL_SIZE
    x0 = _CANVAS_PADDING + _BOARD_MARGIN
    y0 = _CANVAS_PADDING + _BOARD_MARGIN
    x1 = x0 + board_px
    y1 = y0 + board_px
    canvas_w = x1 + _BOARD_MARGIN + _CANVAS_PADDING
    canvas_h = y1 + _BOARD_MARGIN + _CANVAS_PADDING

    _BOARD_BG = (247, 245, 237)
    _BOARD_LINE = (94, 82, 67)
    _BOARD_SHADOW = (205, 200, 191)
    _CANVAS_BG = (232, 237, 244)
    _EXIT_COLOR = (234, 245, 252)

    image = Image.new("RGB", (canvas_w, canvas_h), _CANVAS_BG)
    draw = ImageDraw.Draw(image)

    draw.rounded_rectangle([(x0 + 6, y0 + 8), (x1 + 10, y1 + 12)], radius=18, fill=_BOARD_SHADOW)
    draw.rounded_rectangle(
        [(x0, y0), (x1, y1)], radius=18, fill=_BOARD_BG, outline=_BOARD_LINE, width=5
    )

    exit_top = y0 + exit_row_0 * _CELL_SIZE + 10
    exit_bottom = exit_top + _CELL_SIZE - 20
    draw.rounded_rectangle(
        [(x1 - 2, exit_top), (x1 + 20, exit_bottom)],
        radius=8,
        fill=_EXIT_COLOR,
        outline=_darken(_EXIT_COLOR, 48),
        width=2,
    )

    font_size = max(20, int(_CELL_SIZE * 0.5))
    font = _get_font(font_size)

    for car in cars:
        color = _CAR_COLORS.get(str(car["id"]).upper(), (120, 120, 120))
        car_x0 = x0 + int(car["col"]) * _CELL_SIZE + 7
        car_y0 = y0 + int(car["row"]) * _CELL_SIZE + 7
        if str(car["orientation"]).upper() == "H":
            car_x1 = x0 + (int(car["col"]) + int(car["length"])) * _CELL_SIZE - 7
            car_y1 = car_y0 + _CELL_SIZE - 14
        else:
            car_x1 = car_x0 + _CELL_SIZE - 14
            car_y1 = y0 + (int(car["row"]) + int(car["length"])) * _CELL_SIZE - 7

        draw.rounded_rectangle(
            [(car_x0 + 2, car_y0 + 3), (car_x1 + 5, car_y1 + 6)],
            radius=10,
            fill=_darken(color, 42),
        )
        draw.rounded_rectangle(
            [(car_x0, car_y0), (car_x1, car_y1)],
            radius=10,
            fill=color,
            outline=_darken(color, 65),
            width=3,
        )
        draw.rounded_rectangle(
            [(car_x0 + 7, car_y0 + 7), (car_x1 - 7, car_y0 + 15)],
            radius=5,
            fill=_lighten(color, 40),
        )
        draw.ellipse(
            [(car_x0 + 6, car_y0 + 6), (car_x0 + 12, car_y0 + 12)],
            fill=_lighten(color, 65),
        )
        label = str(car["id"]).upper()
        text_box = draw.textbbox((0, 0), label, font=font)
        text_w = text_box[2] - text_box[0]
        text_h = text_box[3] - text_box[1]
        tx = int((car_x0 + car_x1 - text_w) / 2 - text_box[0])
        ty = int((car_y0 + car_y1 - text_h) / 2 - text_box[1])
        draw.text(
            (tx, ty),
            label,
            font=font,
            fill=(248, 248, 248),
            stroke_width=2,
            stroke_fill=_darken(color, 85),
        )

    return image


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


def _parse_solution_payload(payload: Any) -> List[Dict[str, str]]:
    if isinstance(payload, dict):
        payload = payload.get("solution", [])
    if not isinstance(payload, list):
        return []
    out: List[Dict[str, str]] = []
    for step in payload:
        if not isinstance(step, dict):
            continue
        car = str(step.get("car", "")).strip().upper()
        direction_raw = str(step.get("direction", "")).strip().lower()
        direction_alias = {
            "u": "up",
            "d": "down",
            "l": "left",
            "r": "right",
        }
        direction = direction_alias.get(direction_raw, direction_raw)
        repeats_raw = step.get("steps", step.get("distance", 1))
        try:
            repeats = int(repeats_raw)
        except (TypeError, ValueError):
            repeats = 1
        repeats = max(1, repeats)
        out.append({"car": car, "direction": direction, "steps": str(repeats)})
    return out


def _extract_pred_solution(text: str) -> List[Dict[str, str]]:
    matches = list(
        re.finditer(
            r"<ANSWER_JSON>\s*(\[.*?\]|\{.*?\})\s*</ANSWER_JSON>",
            text,
            re.DOTALL | re.IGNORECASE,
        )
    )
    candidate: Optional[str] = None
    if matches:
        candidate = matches[-1].group(1)
    else:
        stripped = text.strip()
        if (
            stripped.startswith("[")
            and stripped.endswith("]")
            or stripped.startswith("{")
            and stripped.endswith("}")
        ):
            candidate = stripped
    if candidate is None:
        return []
    try:
        parsed = json.loads(candidate)
    except Exception:
        return []
    return _parse_solution_payload(parsed)


def _state_from_doc(doc: Dict[str, Any]) -> List[Dict[str, Any]]:
    raw_cars = doc.get("cars", [])
    out: List[Dict[str, Any]] = []
    if not isinstance(raw_cars, list):
        return out
    for item in raw_cars:
        if not isinstance(item, dict):
            continue
        out.append(
            {
                "id": str(item.get("id", "")).upper(),
                "orientation": str(item.get("orientation", "")).upper(),
                "length": int(item.get("length", 0)),
                "row": int(item.get("row", -1)),
                "col": int(item.get("col", -1)),
            }
        )
    return out


def _cells(car: Dict[str, Any]) -> List[Tuple[int, int]]:
    row = int(car["row"])
    col = int(car["col"])
    length = int(car["length"])
    orientation = str(car["orientation"])
    if orientation == "H":
        return [(row, col + i) for i in range(length)]
    return [(row + i, col) for i in range(length)]


def _occupied(cars: List[Dict[str, Any]]) -> Dict[Tuple[int, int], str]:
    occ: Dict[Tuple[int, int], str] = {}
    for car in cars:
        for rc in _cells(car):
            occ[rc] = str(car["id"])
    return occ


def _apply_one_step(
    cars: List[Dict[str, Any]], step: Dict[str, Any], grid_size: int
) -> Tuple[bool, List[Dict[str, Any]]]:
    car_id = str(step.get("car", "")).upper()
    direction = str(step.get("direction", "")).lower()
    if direction not in {"up", "down", "left", "right"}:
        return False, cars

    index = None
    for i, car in enumerate(cars):
        if str(car["id"]).upper() == car_id:
            index = i
            break
    if index is None:
        return False, cars

    target = dict(cars[index])
    orientation = str(target["orientation"]).upper()
    if orientation == "H" and direction not in {"left", "right"}:
        return False, cars
    if orientation == "V" and direction not in {"up", "down"}:
        return False, cars

    next_cars = [dict(car) for car in cars]
    if direction == "left":
        target["col"] = int(target["col"]) - 1
    elif direction == "right":
        target["col"] = int(target["col"]) + 1
    elif direction == "up":
        target["row"] = int(target["row"]) - 1
    elif direction == "down":
        target["row"] = int(target["row"]) + 1
    next_cars[index] = target

    occ: Dict[Tuple[int, int], str] = {}
    for car in next_cars:
        for r, c in _cells(car):
            if not (0 <= r < grid_size and 0 <= c < grid_size):
                return False, cars
            if (r, c) in occ:
                return False, cars
            occ[(r, c)] = str(car["id"])
    return True, next_cars


def _is_solved(cars: List[Dict[str, Any]], exit_row: int, grid_size: int) -> bool:
    # Dataset stores exit_row as 1-based, while car coordinates are 0-based.
    exit_row_index = max(0, int(exit_row) - 1)
    for car in cars:
        if str(car["id"]).upper() != "R":
            continue
        if str(car["orientation"]).upper() != "H":
            return False
        row = int(car["row"])
        col = int(car["col"])
        length = int(car["length"])
        return row == exit_row_index and col + length - 1 == grid_size - 1
    return False


def _aggregate(results: List[Dict[str, Any]], key: str, level: Optional[str]) -> float:
    filtered = [
        row for row in results if level is None or str(row.get("difficulty")) == level
    ]
    if not filtered:
        return 0.0
    return sum(float(row.get(key, 0.0)) for row in filtered) / len(filtered)


def parking_agg_solved_easy(results: List[Dict[str, Any]]) -> float:
    return _aggregate(results, "solved", "easy")


def parking_agg_solved_middle(results: List[Dict[str, Any]]) -> float:
    return _aggregate(results, "solved", "middle")


def parking_agg_solved_hard(results: List[Dict[str, Any]]) -> float:
    return _aggregate(results, "solved", "hard")


def parking_agg_solved_overall(results: List[Dict[str, Any]]) -> float:
    return _aggregate(results, "solved", None)


def parking_agg_optimal_easy(results: List[Dict[str, Any]]) -> float:
    return _aggregate(results, "optimal", "easy")


def parking_agg_optimal_middle(results: List[Dict[str, Any]]) -> float:
    return _aggregate(results, "optimal", "middle")


def parking_agg_optimal_hard(results: List[Dict[str, Any]]) -> float:
    return _aggregate(results, "optimal", "hard")


def parking_agg_optimal_overall(results: List[Dict[str, Any]]) -> float:
    return _aggregate(results, "optimal", None)


def parking_process_results(
    doc: Dict[str, Any], results: List[Any]
) -> Dict[str, Dict[str, Any]]:
    result_raw = results[0] if results else ""
    result_text = _normalize_result_text(result_raw)
    pred_solution = _extract_pred_solution(result_text)

    cars = _state_from_doc(doc)
    grid_size = int(doc.get("grid_size", _GRID_SIZE))
    legal = True
    steps_used = 0
    for step in pred_solution:
        repeats = int(step.get("steps", "1"))
        for _ in range(repeats):
            legal, cars = _apply_one_step(cars, step, grid_size)
            if not legal:
                break
            steps_used += 1
        if not legal:
            break

    exit_row = int(doc.get("exit_row", 2))
    solved = 1.0 if legal and _is_solved(cars, exit_row, grid_size) else 0.0
    optimal_steps = int(doc.get("optimal_steps", -1))
    optimal = 1.0 if solved == 1.0 and steps_used == optimal_steps else 0.0

    payload = {
        "difficulty": str(doc.get("difficulty", "")).strip().lower(),
        "solved": solved,
        "optimal": optimal,
    }
    return {
        "parking_solved_easy": payload,
        "parking_solved_middle": payload,
        "parking_solved_hard": payload,
        "parking_solved_overall": payload,
        "parking_optimal_easy": payload,
        "parking_optimal_middle": payload,
        "parking_optimal_hard": payload,
        "parking_optimal_overall": payload,
    }
