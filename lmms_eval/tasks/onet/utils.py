from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

from PIL import Image, ImageDraw, ImageFont

_DEFAULT_IMAGE_ROOT = "./data/onet"
_PROMPT_MODES = {
    "aux_cot",
    "cot",
    "direct",
    "step_by_step",
    "aux_step_by_step",
    "gta",
    "vision_reasoning",
}

_PROMPT_VARIANTS = {
    "cot": """You are solving an Onet puzzle on a 6x6 board.

# Coordinate system
- Use 0-based coordinates.
- (r, c) means row r, column c.
- (0,0) is the top-left cell.
- r increases downward, c increases to the right.

Goal:
- Decide whether the whole board can be fully cleared.
- In each step, you may remove a pair of identical fruits if they can be linked.
- A valid link uses only horizontal/vertical segments and can have at most 2 turns.
- The path must not pass through any other fruit.

Output format (strict):
1) Brief reasoning.
2) Final answer on a separate line:
<answer_json>[[[r1,c1],[r2,c2]],[[r3,c3],[r4,c4]],...]</answer_json>
""",
    "direct": """You are solving an Onet puzzle on a 6x6 board.

# Coordinate system
- Use 0-based coordinates.
- (r, c) means row r, column c.
- (0,0) is the top-left cell.
- r increases downward, c increases to the right.

Goal:
- Decide whether the whole board can be fully cleared.
- In each step, you may remove a pair of identical fruits if they can be linked.
- A valid link uses only horizontal/vertical segments and can have at most 2 turns.
- The path must not pass through any other fruit.

Output format (strict):
<answer_json>[[[r1,c1],[r2,c2]],[[r3,c3],[r4,c4]],...]</answer_json>
Do not output any extra text.
""",
    "aux_cot": """<AUX_GRID_INTERLEAVE>
You are solving an Onet puzzle on a 6x6 board.

# Coordinate system
- Use 0-based coordinates.
- (r, c) means row r, column c.
- (0,0) is the top-left cell.
- r increases downward, c increases to the right.

Goal:
- Decide whether the whole board can be fully cleared.
- In each step, you may remove a pair of identical fruits if they can be linked.
- A valid link uses only horizontal/vertical segments and can have at most 2 turns.
- The path must not pass through any other fruit.

First, generate one auxiliary image by drawing a thin red 6x6 grid with cell
coordinates '(r,c)' on top of the original image while preserving all fruit
icons and relative positions.

Output format (strict):
1) Brief reasoning.
2) Final answer on a separate line:
<answer_json>[[[r1,c1],[r2,c2]],[[r3,c3],[r4,c4]],...]</answer_json>
""",
    "step_by_step": """You are solving an Onet puzzle on a 6x6 board.

# Coordinate system
- Use 0-based coordinates.
- (r, c) means row r, column c.
- (0,0) is the top-left cell.
- r increases downward, c increases to the right.

Goal:
- Decide whether the whole board can be fully cleared.
- In each step, remove exactly one pair of identical fruits if they can be linked.
- A valid link uses only horizontal/vertical segments and can have at most 2 turns.
- The path must not pass through any other fruit.

OUTPUT FORMAT (STRICT)
1) MULTI-IMAGE MODE — generate a SEQUENCE OF SEPARATE IMAGES, one per move:
   - Each output image must depict the board state AFTER applying exactly one pair removal.
   - Do NOT include the initial (pre-move) state.
   - Keep palette/layout/scale identical to the input; empty cells must be clearly indicated (e.g., cleared/blank).
   - The number of returned images MUST equal the number of moves in the final answer (see step 2).
   - Absolutely FORBIDDEN: any collage/montage/spritesheet/grid/multi-panel/side-by-side/stacked images; no arrows, captions, or overlays; no GIFs/animations/video.

2) After all step images, emit EXACTLY ONE LINE containing ONLY the final move list as a JSON array of coordinate pairs, wrapped as:
<answer_json>[[[r1,c1],[r2,c2]],[[r3,c3],[r4,c4]],...]</answer_json>


NO EXTRAS
- No tools, no OCR, no explanations, and no text other than the single <ANSWER_JSON>…</ANSWER_JSON> line.
- Do not restate the instructions or the condition.

REMINDERS
- Determine the full sequence of moves first, then emit the image sequence (one image per move), then the single <answer_json> line.
- One move per image; images must be separate files/parts, not stitched together in any way.
""",
    "aux_step_by_step": """<AUX_GRID_INTERLEAVE>
You are solving an Onet puzzle on a 6x6 board.

# Coordinate system
- Use 0-based coordinates.
- (r, c) means row r, column c.
- (0,0) is the top-left cell.
- r increases downward, c increases to the right.

Goal:
- Decide whether the whole board can be fully cleared.
- In each step, remove exactly one pair of identical fruits if they can be linked.
- A valid link uses only horizontal/vertical segments and can have at most 2 turns.
- The path must not pass through any other fruit.

OUTPUT FORMAT (STRICT)
1) FIRST OUTPUT — generate EXACTLY ONE auxiliary image from the original onet image:
   - Draw red grid-boundary lines to show a 6x6 partition over the onet.
   - On every cell, place a centered coordinate label in the form "(r,c)".
   - Preserve the original onet colors, layout, and geometry; only add the red grid lines and coordinate labels.
   - The top-left cell is (0,0).
   - Row index r increases downward.
   - Column index c increases rightward.
   - Do NOT alter any fruit icon or empty cell in this auxiliary image.

2) MULTI-IMAGE MODE — after the auxiliary image, generate a SEQUENCE OF SEPARATE IMAGES, one per move:
   - Each output image must be based on the auxiliary image.
   - Each output image must depict the board state AFTER applying exactly one pair removal.
   - Do NOT include the initial (pre-move) state.
   - Keep palette/layout/scale identical to the auxiliary image; empty cells must be clearly indicated (e.g., cleared/blank).
   - The number of returned images MUST equal the number of moves in the final answer (see step 2).
   - Absolutely FORBIDDEN: any collage/montage/spritesheet/grid/multi-panel/side-by-side/stacked images; no arrows, captions, or overlays; no GIFs/animations/video.

3) After all step images, emit EXACTLY ONE LINE containing ONLY the final move list as a JSON array of coordinate pairs, wrapped as:
<answer_json>[[[r1,c1],[r2,c2]],[[r3,c3],[r4,c4]],...]</answer_json>


NO EXTRAS
- No tools, no OCR, no explanations, and no text other than the single <ANSWER_JSON>…</ANSWER_JSON> line.
- Do not restate the instructions or the condition.

REMINDERS
- Determine the full sequence of moves first, then emit the image sequence (one image per move), then the single <answer_json> line.
- One move per image; images must be separate files/parts, not stitched together in any way.
""",
    "gta": """
[GEN_PROMPT]
Edit the input Onet puzzle image to create exactly one auxiliary annotated image.

Onet puzzle semantics:
- The board is a 6x6 grid of cells.
- Each cell contains a fruit icon or is empty.
- Identical fruit icons can be removed as a pair if they can be connected by a path with at most 2 turns using only horizontal/vertical segments that do not pass through other fruits.

Editing requirements:
1. Preserve the original board geometry, cell layout, fruit icons, and original colors exactly.
2. Overlay thin red grid-boundary lines that divide the image into a 6x6 grid: draw 5 evenly spaced vertical red lines and 5 evenly spaced horizontal red lines, aligned with cell boundaries.
3. Label every cell at its center with its coordinate in black text, using the format (r,c). The top-left cell is (0,0). Row index r increases downward; column index c increases rightward.
4. Do not add any arrows, path lines, legend, explanation text, title, or extra decorations.
[/GEN_PROMPT]

[QUESTION]
You are solving an Onet puzzle on a 6x6 board.

# Coordinate system
- Use 0-based coordinates.
- (r, c) means row r, column c.
- (0,0) is the top-left cell.
- r increases downward, c increases to the right.

Goal:
- Decide whether the whole board can be fully cleared.
- In each step, you may remove a pair of identical fruits if they can be linked.
- A valid link uses only horizontal/vertical segments and can have at most 2 turns.
- The path must not pass through any other fruit.

You are given two images: the original board and an auxiliary image with a red 6x6 grid and coordinate labels. Use the annotated image to identify cell coordinates precisely.

Output format (strict):
1) Brief reasoning.
2) Final answer on a separate line:
<answer_json>[[[r1,c1],[r2,c2]],[[r3,c3],[r4,c4]],...]</answer_json>
[/QUESTION]
""",
    "vision_reasoning": """
[GEN_PROMPT]
Edit the input Onet puzzle image to create exactly one auxiliary annotated image that highlights all matching pairs with their connection paths.

Onet puzzle semantics:
- The board is a 6x6 grid of cells.
- Each cell contains a fruit icon or is empty.
- Identical fruit icons can be removed as a pair if they can be connected by a path with at most 2 turns using only horizontal/vertical segments that do not pass through other fruits.

Editing requirements:
1. Preserve the original board geometry, cell layout, fruit icons, and original colors exactly.
2. For each pair of identical fruits that can be validly connected (forming a complete clearing solution), draw a solid colored polyline from the center of one fruit cell to the center of the other, following the valid connection path (horizontal/vertical segments, at most 2 turns, not passing through other fruits). Use a distinct bright color per pair so all pairs are visually distinguishable.
3. Use a clean stroke of moderate thickness (roughly 10-15 percent of a cell width) so paths are clearly visible against the background.
4. Do not add grid lines, coordinate labels, arrows, legend, captions, titles, or any other decoration. Only the colored polylines should be added.
5. Do not modify the fruit icons, empty cells, or board layout in any other way.
[/GEN_PROMPT]

[QUESTION]
You are solving an Onet puzzle on a 6x6 board.

# Coordinate system
- Use 0-based coordinates.
- (r, c) means row r, column c.
- (0,0) is the top-left cell.
- r increases downward, c increases to the right.

Goal:
- Decide whether the whole board can be fully cleared.
- In each step, you may remove a pair of identical fruits if they can be linked.
- A valid link uses only horizontal/vertical segments and can have at most 2 turns.
- The path must not pass through any other fruit.

You are given two images: the original board and an auxiliary image where colored polylines trace the connection paths between matching fruit pairs. Use the visual hints to identify the pairs and their order of removal.

Output format (strict):
1) Brief reasoning.
2) Final answer on a separate line:
<answer_json>[[[r1,c1],[r2,c2]],[[r3,c3],[r4,c4]],...]</answer_json>
[/QUESTION]
""",
}


def _resolve_image_path(doc: Dict[str, Any], kwargs: Dict[str, Any]) -> Path:
    image_rel_path = str(doc.get("image_path", "")).strip()
    if not image_rel_path:
        raise FileNotFoundError("missing image_path in onet sample")
    image_root = str(kwargs.get("onet_image_root", _DEFAULT_IMAGE_ROOT)).strip()
    rel_path = Path(image_rel_path)
    if rel_path.is_absolute():
        return rel_path
    return Path(image_root) / rel_path


def _resolve_prompt_mode(kwargs: Dict[str, Any]) -> str:
    raw_mode = kwargs.get("onet_prompt_mode")
    if raw_mode is None:
        raise ValueError(
            "onet_prompt_mode is required and must be one of: "
            "aux_cot, cot, direct, step_by_step, aux_step_by_step, gta, vision_reasoning"
        )
    mode = str(raw_mode).strip().lower()
    mode = {
        "step-by-step": "step_by_step",
        "aux-step-by-step": "aux_step_by_step",
        "vision-reasoning": "vision_reasoning",
    }.get(mode, mode)
    if mode not in _PROMPT_MODES:
        raise ValueError(
            f"Invalid onet_prompt_mode: {raw_mode}. "
            "Expected one of: aux_cot, cot, direct, step_by_step, aux_step_by_step, gta, vision_reasoning"
        )
    return mode


def _parse_board_grid(doc: Dict[str, Any]) -> List[List[str | None]]:
    raw_board = doc.get("board", [])
    if not isinstance(raw_board, list):
        return []
    board: List[List[str | None]] = []
    for row in raw_board:
        if not isinstance(row, list):
            return []
        board.append([str(cell) if cell not in {"", None} else None for cell in row])
    if not board:
        return []
    row_len = len(board[0])
    if row_len <= 0:
        return []
    if any(len(row) != row_len for row in board):
        return []
    return board


def _resolve_grid_size_from_doc(doc: Dict[str, Any], board: List[List[str | None]]) -> int:
    raw_grid_size = doc.get("grid_size")
    if raw_grid_size is not None:
        try:
            parsed = int(raw_grid_size)
            if parsed > 0:
                return parsed
        except (TypeError, ValueError):
            pass
    if board:
        return len(board)
    return 6


def _load_board_image(doc: Dict[str, Any], kwargs: Dict[str, Any]) -> Image.Image:
    image_path = _resolve_image_path(doc, kwargs)
    if not image_path.is_file():
        raise FileNotFoundError(f"onet image not found: {image_path}")
    return Image.open(image_path).convert("RGB")


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


def _build_aux_grid(image: Image.Image, grid_size: int) -> Image.Image:
    canvas = image.copy()
    draw = ImageDraw.Draw(canvas)
    width, height = canvas.size
    x0, y0 = 0, 0
    x1, y1 = width - 1, height - 1
    cell_w = width / float(grid_size)
    cell_h = height / float(grid_size)

    draw.rectangle([(x0, y0), (x1, y1)], outline=(255, 0, 0), width=2)
    for i in range(1, grid_size):
        x = round(i * cell_w)
        y = round(i * cell_h)
        draw.line([(x, y0), (x, y1)], fill=(255, 0, 0), width=2)
        draw.line([(x0, y), (x1, y)], fill=(255, 0, 0), width=2)

    font_size = max(12, int(min(cell_w, cell_h) * 0.2))
    font = _aux_grid_font(font_size)
    for r in range(grid_size):
        for c in range(grid_size):
            text = f"({r},{c})"
            bbox = draw.textbbox((0, 0), text, font=font)
            text_w = bbox[2] - bbox[0]
            text_h = bbox[3] - bbox[1]
            cx = int((c + 0.5) * cell_w)
            cy = int((r + 0.5) * cell_h)
            tx = int(cx - text_w / 2)
            ty = int(cy - text_h / 2)
            draw.text((tx, ty), text, fill=(0, 0, 0), font=font)

    return canvas


def onet_doc_to_visual(
    doc: Dict[str, Any], lmms_eval_specific_kwargs: Dict[str, Any] | None = None
) -> List[Image.Image]:
    kwargs = lmms_eval_specific_kwargs or {}
    prompt_mode = _resolve_prompt_mode(kwargs)
    image = _load_board_image(doc, kwargs)
    if prompt_mode not in {"aux_cot", "aux_step_by_step"}:
        return [image]
    board = _parse_board_grid(doc)
    grid_size = _resolve_grid_size_from_doc(doc, board)
    aux_image = _build_aux_grid(image, grid_size)
    return [image, aux_image]


def onet_doc_to_text(
    doc: Dict[str, Any], lmms_eval_specific_kwargs: Dict[str, Any] | None = None
) -> str:
    del doc
    kwargs = lmms_eval_specific_kwargs or {}
    prompt_mode = _resolve_prompt_mode(kwargs)
    return _PROMPT_VARIANTS[prompt_mode]


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


def _extract_step_json(text: str) -> str | None:
    closed_matches = list(
        re.finditer(
            r"<answer_json>\s*(\[.*?\])\s*</answer_json>",
            text,
            re.IGNORECASE | re.DOTALL,
        )
    )
    if closed_matches:
        return closed_matches[-1].group(1)
    open_only = re.search(
        r"<answer_json>\s*(\[[\s\S]*\])\s*$",
        text,
        re.IGNORECASE,
    )
    if open_only:
        return open_only.group(1)
    stripped = text.strip()
    if stripped.startswith("[") and stripped.endswith("]"):
        return stripped
    return None


def _parse_coord(raw: Any) -> tuple[int, int] | None:
    if not isinstance(raw, (list, tuple)) or len(raw) != 2:
        return None
    try:
        row = int(raw[0])
        col = int(raw[1])
    except (TypeError, ValueError):
        return None
    return row, col


def _extract_steps(text: str) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    payload = _extract_step_json(text)
    if payload is None:
        return []
    try:
        parsed = json.loads(payload)
    except json.JSONDecodeError:
        return []
    if not isinstance(parsed, list):
        return []
    out: list[tuple[tuple[int, int], tuple[int, int]]] = []
    for step in parsed:
        if not isinstance(step, (list, tuple)) or len(step) != 2:
            return []
        p1 = _parse_coord(step[0])
        p2 = _parse_coord(step[1])
        if p1 is None or p2 is None:
            return []
        out.append((p1, p2))
    return out


def _in_bounds(rows: int, cols: int, cell: tuple[int, int]) -> bool:
    return 0 <= cell[0] < rows and 0 <= cell[1] < cols


def _clear_line(
    board: list[list[str | None]],
    a: tuple[int, int],
    b: tuple[int, int],
) -> bool:
    ar, ac = a
    br, bc = b
    if ar == br:
        left = min(ac, bc) + 1
        right = max(ac, bc)
        return all(board[ar][c] is None for c in range(left, right))
    if ac == bc:
        top = min(ar, br) + 1
        bottom = max(ar, br)
        return all(board[r][ac] is None for r in range(top, bottom))
    return False


def _can_connect_with_max_two_turns(
    board: list[list[str | None]],
    a: tuple[int, int],
    b: tuple[int, int],
) -> bool:
    if a == b:
        return False
    if _clear_line(board, a, b):
        return True
    ar, ac = a
    br, bc = b
    p1 = (ar, bc)
    p2 = (br, ac)

    rows = len(board)
    cols = len(board[0]) if rows > 0 else 0

    for pivot in (p1, p2):
        if not _in_bounds(rows, cols, pivot):
            continue
        if pivot != b and board[pivot[0]][pivot[1]] is not None:
            continue
        if _clear_line(board, a, pivot) and _clear_line(board, pivot, b):
            return True

    for c in range(cols):
        pivot1 = (ar, c)
        if pivot1 != a and board[pivot1[0]][pivot1[1]] is not None:
            continue
        if not _clear_line(board, a, pivot1):
            continue
        pivot2 = (br, c)
        if pivot2 != b and board[pivot2[0]][pivot2[1]] is not None:
            continue
        if _clear_line(board, pivot1, pivot2) and _clear_line(board, pivot2, b):
            return True

    for r in range(rows):
        pivot1 = (r, ac)
        if pivot1 != a and board[pivot1[0]][pivot1[1]] is not None:
            continue
        if not _clear_line(board, a, pivot1):
            continue
        pivot2 = (r, bc)
        if pivot2 != b and board[pivot2[0]][pivot2[1]] is not None:
            continue
        if _clear_line(board, pivot1, pivot2) and _clear_line(board, pivot2, b):
            return True
    return False


def _validate_and_apply_steps(
    board: list[list[str | None]],
    steps: list[tuple[tuple[int, int], tuple[int, int]]],
) -> bool:
    rows = len(board)
    cols = len(board[0]) if rows > 0 else 0
    for p1, p2 in steps:
        if not _in_bounds(rows, cols, p1) or not _in_bounds(rows, cols, p2):
            return False
        if p1 == p2:
            return False
        v1 = board[p1[0]][p1[1]]
        v2 = board[p2[0]][p2[1]]
        if v1 is None or v2 is None:
            return False
        if v1 != v2:
            return False
        if not _can_connect_with_max_two_turns(board, p1, p2):
            return False
        board[p1[0]][p1[1]] = None
        board[p2[0]][p2[1]] = None
    return True


def _is_board_cleared(board: list[list[str | None]]) -> bool:
    return all(cell is None for row in board for cell in row)


def onet_render_state_image(
    doc: Dict[str, Any],
    board: List[List[str | None]],
    initial_board: List[List[str | None]] | None = None,
    base_image: Image.Image | None = None,
) -> Image.Image | None:
    board_height = len(board)
    board_width = len(board[0]) if board_height > 0 else 0
    if board_height <= 0 or board_width <= 0:
        return None
    image = base_image.copy() if isinstance(base_image, Image.Image) else _load_board_image(doc, {})
    if image is None:
        return None

    cell_w = image.width / float(board_width)
    cell_h = image.height / float(board_height)
    draw = ImageDraw.Draw(image)
    for r in range(board_height):
        for c in range(board_width):
            if board[r][c] is not None:
                continue
            if (
                initial_board is not None
                and 0 <= r < len(initial_board)
                and 0 <= c < len(initial_board[r])
                and initial_board[r][c] is None
            ):
                continue
            left = int(round(c * cell_w))
            top = int(round(r * cell_h))
            right = int(round((c + 1) * cell_w)) - 1
            bottom = int(round((r + 1) * cell_h)) - 1
            if right < left or bottom < top:
                continue
            draw.rectangle(
                [(left, top), (right, bottom)],
                fill=(248, 248, 248),
            )
    return image


def onet_process_results(
    doc: Dict[str, Any], results: List[Any]
) -> Dict[str, Dict[str, float]]:
    result_raw = results[0] if results else ""
    result_text = _normalize_result_text(result_raw)
    steps = _extract_steps(result_text)

    board = _parse_board_grid(doc)

    valid = _validate_and_apply_steps(board, steps) if board else False
    acc = 1.0 if valid and _is_board_cleared(board) else 0.0
    difficulty = str(doc.get("difficulty", "")).strip().lower()
    payload = {"difficulty": difficulty, "acc": acc}
    return {
        "onet_acc_easy": payload,
        "onet_acc_middle": payload,
        "onet_acc_hard": payload,
        "onet_acc_overall": payload,
    }


def _onet_aggregate_acc(
    results: List[Dict[str, float]], level: str | None = None
) -> float:
    filtered = [
        row for row in results if level is None or row.get("difficulty") == level
    ]
    if not filtered:
        return 0.0
    return sum(float(row.get("acc", 0.0)) for row in filtered) / len(filtered)


def onet_agg_acc_easy(results: List[Dict[str, float]]) -> float:
    return _onet_aggregate_acc(results, level="easy")


def onet_agg_acc_middle(results: List[Dict[str, float]]) -> float:
    return _onet_aggregate_acc(results, level="middle")


def onet_agg_acc_hard(results: List[Dict[str, float]]) -> float:
    return _onet_aggregate_acc(results, level="hard")


def onet_agg_acc_overall(results: List[Dict[str, float]]) -> float:
    return _onet_aggregate_acc(results, level=None)
