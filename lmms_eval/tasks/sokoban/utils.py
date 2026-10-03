from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image, ImageDraw, ImageFont

_DEFAULT_IMAGE_ROOT = "./data/sokoban"
_PROMPT_MODES = {
    "aux",
    "cot",
    "direct",
    "gta",
    "vision_reasoning",
    "step_by_step",
    "aux_step_by_step",
}
_MOVE_DELTAS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}


def _resolve_prompt_mode(kwargs: Dict[str, Any]) -> str:
    raw_mode = kwargs.get("sokoban_prompt_mode")
    if raw_mode is None:
        raise ValueError(
            "sokoban_prompt_mode is required and must be one of: "
            "aux, cot, direct, gta, vision_reasoning, "
            "step-by-step, aux_step_by_step"
        )
    mode = str(raw_mode).strip().lower()
    mode_aliases = {
        "step-by-step": "step_by_step",
        "aux-step-by-step": "aux_step_by_step",
        "vision-reasoning": "vision_reasoning",
    }
    mode = mode_aliases.get(mode, mode)
    if mode not in _PROMPT_MODES:
        raise ValueError(
            f"Invalid sokoban_prompt_mode: {raw_mode}. "
            "Expected one of: aux, cot, direct, gta, vision_reasoning, "
            "step-by-step, aux_step_by_step"
        )
    return mode


def _resolve_image_path(doc: Dict[str, Any], kwargs: Dict[str, Any]) -> Path:
    image_rel = str(doc.get("image_path", "")).strip()
    if not image_rel:
        raise FileNotFoundError("missing image_path in sokoban sample")
    image_root = str(kwargs.get("sokoban_image_root", _DEFAULT_IMAGE_ROOT)).strip()
    image_path = Path(image_rel)
    if image_path.is_absolute():
        return image_path
    return Path(image_root) / image_path


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


def _iter_known_cells(doc: Dict[str, Any]) -> List[Tuple[int, int]]:
    cells: List[Tuple[int, int]] = []
    for key in ("player_start", "box_start", "target"):
        parsed = _parse_coord_pair(doc.get(key))
        if parsed is not None:
            cells.append(parsed)
    for wall in _parse_walls(doc):
        cells.append(wall)
    return cells


def _infer_grid_shape(image: Image.Image, doc: Dict[str, Any]) -> Tuple[int, int]:
    known_cells = _iter_known_cells(doc)
    if known_cells:
        max_row = max(cell[0] for cell in known_cells)
        max_col = max(cell[1] for cell in known_cells)
        return max_row + 1, max_col + 1
    grid_h = max(1, int(round(image.height / 64.0)))
    grid_w = max(1, int(round(image.width / 64.0)))
    return grid_h, grid_w


def _cell_bounds(
    image: Image.Image, grid_h: int, grid_w: int, row: int, col: int
) -> Tuple[int, int, int, int]:
    cell_w = image.width / float(grid_w)
    cell_h = image.height / float(grid_h)
    left = int(round(col * cell_w))
    top = int(round(row * cell_h))
    right = int(round((col + 1) * cell_w))
    bottom = int(round((row + 1) * cell_h))
    return left, top, right, bottom


def _fit_patch(patch: Image.Image, width: int, height: int) -> Image.Image:
    if patch.size == (width, height):
        return patch
    resampling = getattr(Image, "Resampling", Image)
    return patch.resize((max(1, width), max(1, height)), resampling.BILINEAR)


def sokoban_render_state_image(
    doc: Dict[str, Any],
    player_pos: Tuple[int, int],
    box_pos: Tuple[int, int],
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

    player_start = _parse_coord_pair(doc.get("player_start"))
    box_start = _parse_coord_pair(doc.get("box_start"))
    target = _parse_coord_pair(doc.get("target"))
    if player_start is None or box_start is None:
        return canvas

    walls = _parse_walls(doc)
    grid_h, grid_w = _infer_grid_shape(source, doc)

    def bounds(cell: Tuple[int, int]) -> Tuple[int, int, int, int]:
        return _cell_bounds(source, grid_h, grid_w, cell[0], cell[1])

    player_l, player_t, player_r, player_b = bounds(player_start)
    box_l, box_t, box_r, box_b = bounds(box_start)
    player_patch = source.crop((player_l, player_t, player_r, player_b))
    box_patch = source.crop((box_l, box_t, box_r, box_b))

    floor_cell: Optional[Tuple[int, int]] = None
    for r in range(grid_h):
        for c in range(grid_w):
            if (r, c) in walls:
                continue
            if (r, c) in {player_start, box_start, target}:
                continue
            floor_cell = (r, c)
            break
        if floor_cell is not None:
            break
    if floor_cell is None and target is not None and target not in {player_start, box_start}:
        floor_cell = target
    if floor_cell is None:
        floor_cell = player_start

    floor_l, floor_t, floor_r, floor_b = bounds(floor_cell)
    floor_patch = source.crop((floor_l, floor_t, floor_r, floor_b))

    for old_cell in (player_start, box_start):
        old_l, old_t, old_r, old_b = bounds(old_cell)
        canvas.paste(_fit_patch(floor_patch, old_r - old_l, old_b - old_t), (old_l, old_t))

    new_box_l, new_box_t, new_box_r, new_box_b = bounds(box_pos)
    canvas.paste(
        _fit_patch(box_patch, new_box_r - new_box_l, new_box_b - new_box_t),
        (new_box_l, new_box_t),
    )
    new_player_l, new_player_t, new_player_r, new_player_b = bounds(player_pos)
    canvas.paste(
        _fit_patch(
            player_patch,
            new_player_r - new_player_l,
            new_player_b - new_player_t,
        ),
        (new_player_l, new_player_t),
    )
    return canvas


def _build_aux_overlay(image: Image.Image, doc: Dict[str, Any]) -> Image.Image:
    overlay = image.copy().convert("RGB")
    draw = ImageDraw.Draw(overlay)
    grid_h, grid_w = _infer_grid_shape(image, doc)
    cell_w = image.width / float(grid_w)
    cell_h = image.height / float(grid_h)
    font = _get_font(max(10, int(min(cell_w, cell_h) * 0.28)))
    line_color = (230, 40, 40)
    wall_cells = _parse_walls(doc)

    # Draw outer border and inner grid boundaries.
    draw.rectangle(
        [(0, 0), (image.width - 1, image.height - 1)],
        outline=line_color,
        width=2,
    )
    for c in range(1, grid_w):
        x = int(round(c * cell_w))
        draw.line([(x, 0), (x, image.height)], fill=line_color, width=2)
    for r in range(1, grid_h):
        y = int(round(r * cell_h))
        draw.line([(0, y), (image.width, y)], fill=line_color, width=2)

    # Label coordinates in each cell (black text at center).
    for r in range(grid_h):
        for c in range(grid_w):
            if (r, c) in wall_cells:
                continue
            text = f"({r},{c})"
            left = int(c * cell_w)
            top = int(r * cell_h)
            bbox = draw.textbbox((0, 0), text, font=font)
            tw = bbox[2] - bbox[0]
            th = bbox[3] - bbox[1]
            tx = int(left + (cell_w - tw) / 2 - bbox[0])
            ty = int(top + (cell_h - th) / 2 - bbox[1])
            draw.text((tx, ty), text, font=font, fill=(0, 0, 0))
    return overlay


def sokoban_doc_to_visual(
    doc: Dict[str, Any], lmms_eval_specific_kwargs: Dict[str, Any] | None = None
) -> List[Image.Image]:
    kwargs = lmms_eval_specific_kwargs or {}
    prompt_mode = _resolve_prompt_mode(kwargs)
    image_path = _resolve_image_path(doc, kwargs)
    if not image_path.is_file():
        raise FileNotFoundError(f"sokoban image not found: {image_path}")
    image = Image.open(image_path).convert("RGB")
    if prompt_mode in {"aux", "aux_step_by_step"}:
        return [image, _build_aux_overlay(image, doc)]
    return [image]


def sokoban_doc_to_text(
    doc: Dict[str, Any], lmms_eval_specific_kwargs: Dict[str, Any] | None = None
) -> str:
    del doc
    kwargs = lmms_eval_specific_kwargs or {}
    prompt_mode = _resolve_prompt_mode(kwargs)

    base = """You are solving a single-box Sokoban puzzle.

Legend:
- Brick wall: blocked
- Sand floor: walkable
- Small person: player start
- Wooden crate with X: box
- Green X: target

Rules:
1) Allowed moves: up, down, left, right (one cell per step).
2) If the player moves into the box, the box is pushed by one cell in the same direction.
3) A push is valid only if the box destination is not a wall.
4) No diagonal movement.

Goal:
Push the box onto the target cell.
"""
    if prompt_mode == "direct":
        return (
            base
            + """
Output strictly:
<ANSWER_JSON>["move1","move2",...]</ANSWER_JSON>
Do not output any extra text.
"""
        )
    if prompt_mode == "cot":
        return (
            base
            + """

Think briefly about legality of moves and box position updates.

Then output strictly:
<ANSWER_JSON>["move1","move2",...]</ANSWER_JSON>
"""
        )
    if prompt_mode == "aux":
        return (
            "<AUX_GRID_INTERLEAVE>\n"
            + base
            + """
First, generate one auxiliary image from the original Sokoban image by drawing red grid boundaries over all cells and centered coordinate labels "(r,c)" on non-wall cells, while preserving original colors and geometry.

Think briefly about legality of moves and box position updates.

Then output strictly:
<ANSWER_JSON>["move1","move2",...]</ANSWER_JSON>
"""
        )
    if prompt_mode == "step_by_step":
        return (
            base
            + """
OUTPUT FORMAT (STRICT)
1) MULTI-IMAGE MODE — generate a SEQUENCE OF SEPARATE IMAGES, one per move:
   - Each output image must depict the Sokoban state AFTER applying exactly one legal move.
   - Do NOT include the initial (pre-move) state.
   - Keep palette/layout/scale identical to the input; update only the player and box positions.
   - The number of returned images MUST equal the number of moves in the final answer (see step 2).
   - Absolutely FORBIDDEN: any collage/montage/spritesheet/grid/multi-panel/side-by-side/stacked images; no arrows, captions, or overlays; no GIFs/animations/video.

2) After all step images, emit EXACTLY ONE LINE containing ONLY the final move list as a JSON array of lowercase strings, wrapped as:
   <ANSWER_JSON>["right","down","left"]</ANSWER_JSON>


NO EXTRAS
- No tools, no OCR, no explanations, and no text other than the single <ANSWER_JSON>…</ANSWER_JSON> line.
- Do not restate the instructions or the condition.

REMINDERS
- Decide the full path first, then emit the image sequence (one image per move), then the single <ANSWER_JSON> line.
- One move per image; images must be separate files/parts, not stitched together in any way.
"""
        )
    if prompt_mode == "gta":
        return """
[GEN_PROMPT]
Edit the input Sokoban image to create exactly one auxiliary annotated image.

Sokoban semantics:
- Brick walls remain unchanged.
- Sand floor is walkable.
- The player marker, box marker, and green target marker must remain unchanged.

Editing requirements:
1. Preserve original colors, geometry, walls, player, box, and target.
2. Overlay thin red grid-boundary lines aligned with the puzzle cells (one line per row/column boundary).
3. Label every non-wall cell at its center with its coordinate in black text using the format (r,c). Top-left cell is (0,0); r increases downward, c increases rightward.
4. Do not add arrows, path lines, captions, titles, legend, or any other decoration.
[/GEN_PROMPT]

[QUESTION]
""" + base + """

Think briefly about legality of moves and box position updates.

Then output strictly:
<ANSWER_JSON>["move1","move2",...]</ANSWER_JSON>
[/QUESTION]
"""
    if prompt_mode == "vision_reasoning":
        return base + """
[GEN_PROMPT]
Edit the input Sokoban image to create exactly one auxiliary annotated image that highlights a feasible player path that pushes the box onto the target.

Sokoban semantics:
- Brick walls remain unchanged.
- Sand floor is walkable.
- The player, box, and green target must remain unchanged.
- Pushes happen when the player steps into the box; the box moves one cell in the same direction; pushes into walls are illegal.

Editing requirements:
1. Preserve original colors, geometry, walls, player, box, and target.
2. Overlay a single continuous RED polyline that traces the player's trajectory from the player start cell, ending exactly at the cell from which the box gets pushed onto the green target. The polyline:
   - Follows non-wall cells only and never crosses a wall.
   - Moves only orthogonally (up / down / left / right) between cell centers.
   - Uses a clean solid red stroke (~10-15 percent of a cell width) so it is clearly visible.
3. Do NOT draw the box's trajectory, arrows, dots, grid lines, coordinate labels, captions, titles, or any other decoration. Only the player's path polyline is added.
[/GEN_PROMPT]

[QUESTION]
""" + base + """
You are also given an auxiliary image where a red polyline traces a candidate player trajectory. Use it as a visual hint, but verify each step against the puzzle rules (no wall crossings; pushes are valid only if the box destination is non-wall).

Think briefly about legality of moves and box position updates.

Then output strictly:
<ANSWER_JSON>["move1","move2",...]</ANSWER_JSON>
[/QUESTION]
"""
    if prompt_mode == "aux_step_by_step":
        return (
            "<AUX_GRID_INTERLEAVE>\n"
            + base
            + """
    OUTPUT FORMAT (STRICT)
    1) FIRST OUTPUT — generate EXACTLY ONE auxiliary image from the original Sokoban image:
    - Draw red grid-boundary lines to show a 10x10 partition over the board.
    - On every non-wall cell, place a centered coordinate label in the form "(r,c)".
    - Preserve the original board colors, layout, and geometry; only add the red grid lines and coordinate labels.
    - The top-left cell is (0,0).
    - Row index r increases downward.
    - Column index c increases rightward.
    - Do NOT alter the player, box, target, or walls in this auxiliary image.

    2) MULTI-IMAGE MODE — after the auxiliary image, generate a SEQUENCE OF SEPARATE IMAGES, one per move:
    - Each output image must be based on the auxiliary image.
    - Each output image must depict the Sokoban state AFTER applying exactly one legal move.
    - Do NOT include the initial (pre-move) state again.
    - Keep palette, layout, scale, red grid lines, and coordinate labels identical to the auxiliary image; update only the player and box positions.
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
    """
    )
    else:
        assert False, f"Invalid prompt mode: {prompt_mode}"


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


def _extract_pred_moves(text: str) -> List[str]:
    matches = list(
        re.finditer(
            r"<ANSWER_JSON>\s*(\[.*?\])\s*</ANSWER_JSON>",
            text,
            re.DOTALL | re.IGNORECASE,
        )
    )
    candidate = None
    if matches:
        candidate = matches[-1].group(1)
    else:
        stripped = text.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            candidate = stripped
    if candidate is None:
        return []
    try:
        parsed = json.loads(candidate)
    except Exception:
        return []
    if not isinstance(parsed, list):
        return []
    return [str(x).strip().lower() for x in parsed]


def _parse_gt_moves(doc: Dict[str, Any]) -> List[str]:
    raw = doc.get("solution", "[]")
    if isinstance(raw, list):
        return [str(x).strip().lower() for x in raw]
    if not isinstance(raw, str):
        return []
    try:
        parsed = json.loads(raw)
    except Exception:
        return []
    if not isinstance(parsed, list):
        return []
    return [str(x).strip().lower() for x in parsed]


def _parse_coord_pair(raw: Any) -> Optional[Tuple[int, int]]:
    if not isinstance(raw, (list, tuple)) or len(raw) != 2:
        return None
    try:
        return int(raw[0]), int(raw[1])
    except (TypeError, ValueError):
        return None


def _parse_walls(doc: Dict[str, Any]) -> set[Tuple[int, int]]:
    out: set[Tuple[int, int]] = set()
    raw_walls = doc.get("walls", [])
    if not isinstance(raw_walls, list):
        return out
    for item in raw_walls:
        parsed = _parse_coord_pair(item)
        if parsed is not None:
            out.add(parsed)
    return out


def _simulate(
    walls: set[Tuple[int, int]],
    player_start: Tuple[int, int],
    box_start: Tuple[int, int],
    target: Tuple[int, int],
    moves: List[str],
) -> Tuple[bool, bool]:
    player = player_start
    box = box_start
    for move in moves:
        delta = _MOVE_DELTAS.get(move)
        if delta is None:
            return False, False
        np = (player[0] + delta[0], player[1] + delta[1])
        if np in walls:
            return False, False
        if np == box:
            nb = (box[0] + delta[0], box[1] + delta[1])
            if nb in walls:
                return False, False
            box = nb
        player = np
    return True, box == target


def _aggregate(
    results: List[Dict[str, Any]], key: str, level: Optional[str] = None
) -> float:
    filtered = [
        row for row in results if level is None or str(row.get("difficulty")) == level
    ]
    if not filtered:
        return 0.0
    return sum(float(row.get(key, 0.0)) for row in filtered) / len(filtered)


def sokoban_agg_solved_easy(results: List[Dict[str, Any]]) -> float:
    return _aggregate(results, "solved", level="easy")


def sokoban_agg_solved_middle(results: List[Dict[str, Any]]) -> float:
    return _aggregate(results, "solved", level="middle")


def sokoban_agg_solved_hard(results: List[Dict[str, Any]]) -> float:
    return _aggregate(results, "solved", level="hard")


def sokoban_agg_solved_overall(results: List[Dict[str, Any]]) -> float:
    return _aggregate(results, "solved", level=None)


def sokoban_agg_optimal_easy(results: List[Dict[str, Any]]) -> float:
    return _aggregate(results, "optimal", level="easy")


def sokoban_agg_optimal_middle(results: List[Dict[str, Any]]) -> float:
    return _aggregate(results, "optimal", level="middle")


def sokoban_agg_optimal_hard(results: List[Dict[str, Any]]) -> float:
    return _aggregate(results, "optimal", level="hard")


def sokoban_agg_optimal_overall(results: List[Dict[str, Any]]) -> float:
    return _aggregate(results, "optimal", level=None)


def sokoban_process_results(doc: Dict[str, Any], results: List[Any]) -> Dict[str, Any]:
    result_raw = results[0] if results else ""
    result_text = _normalize_result_text(result_raw)
    pred_moves = _extract_pred_moves(result_text)
    gt_moves = _parse_gt_moves(doc)

    player_start = _parse_coord_pair(doc.get("player_start"))
    box_start = _parse_coord_pair(doc.get("box_start"))
    target = _parse_coord_pair(doc.get("target"))
    walls = _parse_walls(doc)

    solved = 0.0
    optimal = 0.0
    legal = False
    difficulty = str(doc.get("difficulty", "")).strip().lower()
    if player_start and box_start and target:
        legal, is_solved = _simulate(walls, player_start, box_start, target, pred_moves)
        solved = 1.0 if legal and is_solved else 0.0
        if solved == 1.0 and len(pred_moves) == int(doc.get("optimal_steps", -1)):
            optimal = 1.0

    exact = 1.0 if pred_moves == gt_moves else 0.0
    payload = {
        "difficulty": difficulty,
        "solved": solved,
        "optimal": optimal,
        "exact": exact,
    }
    return {
        "sokoban_solved_easy": payload,
        "sokoban_solved_middle": payload,
        "sokoban_solved_hard": payload,
        "sokoban_solved_overall": payload,
        "sokoban_optimal_easy": payload,
        "sokoban_optimal_middle": payload,
        "sokoban_optimal_hard": payload,
        "sokoban_optimal_overall": payload,
        "sokoban_exact": payload,
    }
