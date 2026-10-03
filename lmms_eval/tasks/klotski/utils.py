import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image, ImageDraw, ImageFont

_KLOTSKI_PROMPT_MODES = {
    "aux",
    "cot",
    "direct",
    "step_by_step",
    "aux_step_by_step",
    "gta",
    "vision_reasoning",
}
_DEFAULT_IMAGE_ROOT = "./data/klotski"
_BOARD_WIDTH = 6
_BOARD_HEIGHT = 6
_GOAL_T_POS_RC = (4, 2)
_AUX_GRID_RED = (255, 0, 0)
_DELTA_TO_DIR = {
    (1, 0): "RIGHT",
    (-1, 0): "LEFT",
    (0, 1): "DOWN",
    (0, -1): "UP",
}
_KLOTSKI_COLORS = {
    "empty": (240, 240, 240),
    "T": (170, 80, 220),
    "H": (46, 204, 113),
    "V": (52, 152, 219),
    "S": (241, 196, 15),
}

SYSTEM_PROMPT = """You are an expert Klotski puzzle solver.

# Coordinate system
- Use 0-based coordinates.
- (r, c) means row r, column c.
- (0,0) is the top-left cell.
- r increases downward, c increases to the right.

# Board
- Grid size: width=6, height=6.
- There are exactly two empty cells at any time.
- Every non-empty cell belongs to exactly one block from the listed block shapes.

# Block shapes
- One 2x2 target block.
- Multiple 1x2 vertical blocks.
- Multiple 2x1 horizontal blocks.
- Multiple 1x1 single blocks.

# Block reference in actions
- In the action output, a block is identified by its CURRENT TOP-LEFT coordinate at the time of that move.

# Legal move definition
1. Each step slides ONE block by exactly ONE cell in {UP, DOWN, LEFT, RIGHT}.
2. No rotation, no diagonal move, no jump.
3. A move is legal only if EVERY destination cell of that block is currently empty.
   - Let current_cells be the cells currently occupied by the moving block.
   - Let destination_cells be the cells occupied after translating the block by one cell in the chosen direction.
   - A move is legal iff every cell in (destination_cells - current_cells) is empty before the move.
   - For 1x2 / 2x1 blocks, required empty destination cells can be 1 or 2 depending on move direction.
   - For 2x2 blocks, required empty destination cells are 2.
4. Cells occupied by OTHER blocks are never passable.

# Goal
- Make the 2x2 target block's TOP-LEFT coordinate reach (4,2).
"""

State = Tuple[
    Tuple[int, int],
    Tuple[Tuple[int, int], ...],
    Tuple[Tuple[int, int], ...],
    Tuple[Tuple[int, int], ...],
]


def deserialize_state(data: Dict[str, Any]) -> State:
    def rc_to_xy(pos: Any) -> Tuple[int, int]:
        if not isinstance(pos, (list, tuple)) or len(pos) != 2:
            raise ValueError(f"invalid coordinate: {pos!r}")
        r, c = pos
        return (int(c), int(r))

    return (
        rc_to_xy(data["T"]),
        tuple(rc_to_xy(v) for v in data["Vs"]),
        tuple(rc_to_xy(h) for h in data["Hs"]),
        tuple(rc_to_xy(s) for s in data["Ss"]),
    )


def get_user_prompt(method: str) -> str:
    base_instruction = SYSTEM_PROMPT.strip() + "\n\n"
    if method == "direct":
        return base_instruction + """Do not output any intermediate reasoning, text thinking, or visual representations. Output ONLY the complete sequence of actions required to move the 2x2 target block to the goal.

Strictly format your response as follows:
{
  "solution": [
    {"from": [<r1>, <c1>], "direction": "<DIRECTION>"},
    {"from": [<r2>, <c2>], "direction": "<DIRECTION>"}
  ]
}"""
    if method == "cot":
        return base_instruction + """

Before outputting the final sequence, briefly describe your reasoning for the solution.

Finally, output the solution in the following format:

{
  "solution": [
    {"from": [<r1>, <c1>], "direction": "<DIRECTION>"},
    {"from": [<r2>, <c2>], "direction": "<DIRECTION>"}
  ]
}"""
    if method == "aux":
        return base_instruction + """

First, generate one auxiliary image from the original board image by drawing red dashed grid-boundary lines for a 6x6 partition and placing centered coordinate labels "(r,c)" in each cell, while preserving the original board colors and geometry. The top-left cell is (0,0), r increases downward, and c increases rightward.

Before outputting the final sequence, briefly describe your reasoning for the solution.

Finally, output the solution in the following format:

{
  "solution": [
    {"from": [<r1>, <c1>], "direction": "<DIRECTION>"},
    {"from": [<r2>, <c2>], "direction": "<DIRECTION>"}
  ]
}"""
    if method == "gta":
        return (
            """[GEN_PROMPT]
Edit the input Klotski image to create exactly one auxiliary annotated image.

Klotski semantics:
- Grid is 6x6.
- Blocks keep shape and orientation; every move slides one block by one cell.
- Target is to move the 2x2 block T so its top-left reaches (4,2).

Editing requirements:
1. Preserve original image dimensions, board geometry, block positions, labels, and
   colors exactly. The input image may omit visible cell lines; infer the implied 6x6
   tiling from the layout and keep each block's cell extent unchanged (e.g. 1x2 stays
   1x2; do not shrink, split, or merge blocks).
2. Overlay thin red dashed grid-boundary lines aligned to the 6x6 cell boundaries.
3. Label every cell center with coordinate "(r,c)" in black text.
4. Do not add arrows, legends, captions, titles, or extra decorations.
[/GEN_PROMPT]

[QUESTION]
"""
            + base_instruction
            + """

Before outputting the final sequence, briefly describe your reasoning for the solution.

Finally, output the solution in the following format:

{
  "solution": [
    {"from": [<r1>, <c1>], "direction": "<DIRECTION>"},
    {"from": [<r2>, <c2>], "direction": "<DIRECTION>"}
  ]
[/QUESTION]"""
        )
    if method == "vision_reasoning":
        return (
            """[GEN_PROMPT]
Edit the input Klotski image to create exactly one auxiliary annotated image that highlights one feasible solving strategy.

Klotski semantics:
- Grid is 6x6.
- Blocks keep shape and orientation; every move slides one block by one cell.
- Target is to move the 2x2 block T so its top-left reaches (4,2).

Editing requirements:
1. Preserve original image dimensions, board geometry, block positions, labels, and
   colors exactly. The input image may omit visible cell lines; infer the implied 6x6
   tiling from the layout and keep each block's cell extent unchanged (e.g. 1x2 stays
   1x2; do not shrink, split, or merge blocks).
2. Add concise visual hints indicating a feasible move strategy (for example, ordered markers and local directional cues near moved blocks).
3. Hints must correspond to legal one-cell slides and must not imply illegal overlaps or rotations.
4. Do not actually move blocks in the edited image; only overlay hints.
5. Do not add unrelated decorations or long text.
[/GEN_PROMPT]

[QUESTION]
"""
            + base_instruction
            + """
Solve the Klotski puzzle using the hint image as guidance, but verify legality on the original board state.

Strictly format your response as:
{
  "solution": [
    {"from": [<r1>, <c1>], "direction": "<DIRECTION>"},
    {"from": [<r2>, <c2>], "direction": "<DIRECTION>"}
  ]
}
[/QUESTION]"""
        )
    if method == "step_by_step":
        return base_instruction + """OUTPUT FORMAT (STRICT)
1) MULTI-IMAGE MODE — generate a SEQUENCE OF SEPARATE IMAGES, one per move:
   - Each output image must depict the board state AFTER applying exactly one legal move.
   - Do NOT include the initial (pre-move) state.
   - Keep board layout/scale/colors consistent with the input board.
   - The number of returned images MUST equal the number of moves in your final answer.
   - Absolutely FORBIDDEN: collage/montage/spritesheet/grid/multi-panel/side-by-side/stacked images; no GIF/animation/video.

2) For each planning turn, output exactly one move as:
<STEP_JSON>{"from":[r,c],"direction":"UP|DOWN|LEFT|RIGHT"}</STEP_JSON>

3) After reaching the goal, output exactly one final JSON object:
{
  "solution": [
    {"from": [<r1>, <c1>], "direction": "<DIRECTION>"},
    {"from": [<r2>, <c2>], "direction": "<DIRECTION>"}
  ]
}
"""
    if method == "aux_step_by_step":
        return  "<AUX_GRID_INTERLEAVE>\n" + base_instruction + """OUTPUT FORMAT (STRICT)
1) FIRST OUTPUT — generate exactly one auxiliary board image:
   - Draw red dashed grid-boundary lines for a 6x6 partition.
   - Place centered coordinate labels "(r,c)" in each cell.
   - Preserve original board colors/layout/geometry.

2) MULTI-IMAGE MODE — after the auxiliary image, generate a SEQUENCE OF SEPARATE IMAGES, one per move:
   - Each output image must be based on the auxiliary image.
   - Each output image must depict the board state AFTER applying exactly one legal move.
   - Do NOT include the initial pre-move state again.
   - The number of returned move-state images MUST equal the number of moves in your final answer.
   - Absolutely FORBIDDEN: collage/montage/spritesheet/grid/multi-panel/side-by-side/stacked images; no GIF/animation/video.

3) For each planning turn, output exactly one move as:
<STEP_JSON>{"from":[r,c],"direction":"UP|DOWN|LEFT|RIGHT"}</STEP_JSON>

4) After reaching the goal, output exactly one final JSON object:
{
  "solution": [
    {"from": [<r1>, <c1>], "direction": "<DIRECTION>"},
    {"from": [<r2>, <c2>], "direction": "<DIRECTION>"}
  ]
}
"""
    raise ValueError(f"unknown method: {method}")


def extract_json_from_text(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None
    decoder = json.JSONDecoder()
    for idx, char in enumerate(text):
        if char != "{":
            continue
        try:
            obj, _ = decoder.raw_decode(text, idx)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and "solution" in obj:
            return obj
    return None


def _cells_for_block(block_type: str, x: int, y: int) -> List[Tuple[int, int]]:
    if block_type == "T":
        return [(x, y), (x + 1, y), (x, y + 1), (x + 1, y + 1)]
    if block_type == "V":
        return [(x, y), (x, y + 1)]
    if block_type == "H":
        return [(x, y), (x + 1, y)]
    return [(x, y)]


def get_valid_moves_with_details(
    state: State,
) -> List[Tuple[State, Tuple[int, int], Tuple[int, int], str]]:
    target, verticals, horizontals, singles = state
    blocks = (
        [("T", target)]
        + [("V", pos) for pos in verticals]
        + [("H", pos) for pos in horizontals]
        + [("S", pos) for pos in singles]
    )

    occupied: set[Tuple[int, int]] = set()
    for block_type, (bx, by) in blocks:
        occupied.update(_cells_for_block(block_type, bx, by))

    moves: List[Tuple[State, Tuple[int, int], Tuple[int, int], str]] = []
    for idx, (block_type, (bx, by)) in enumerate(blocks):
        old_cells = set(_cells_for_block(block_type, bx, by))
        for dx, dy in ((0, 1), (0, -1), (1, 0), (-1, 0)):
            nx, ny = bx + dx, by + dy
            target_cells = _cells_for_block(block_type, nx, ny)

            valid = True
            for cx, cy in target_cells:
                if not (0 <= cx < _BOARD_WIDTH and 0 <= cy < _BOARD_HEIGHT):
                    valid = False
                    break
                if (cx, cy) in occupied and (cx, cy) not in old_cells:
                    valid = False
                    break
            if not valid:
                continue

            new_target = target
            new_verticals = list(verticals)
            new_horizontals = list(horizontals)
            new_singles = list(singles)
            if block_type == "T":
                new_target = (nx, ny)
            elif block_type == "V":
                new_verticals[idx - 1] = (nx, ny)
            elif block_type == "H":
                new_horizontals[idx - 1 - len(verticals)] = (nx, ny)
            else:
                offset = idx - 1 - len(verticals) - len(horizontals)
                new_singles[offset] = (nx, ny)

            new_state: State = (
                new_target,
                tuple(sorted(new_verticals)),
                tuple(sorted(new_horizontals)),
                tuple(sorted(new_singles)),
            )
            moves.append((new_state, (bx, by), (nx, ny), block_type))
    return moves


def _direction_from_positions(
    old_pos: Tuple[int, int], new_pos: Tuple[int, int]
) -> Optional[str]:
    delta = (new_pos[0] - old_pos[0], new_pos[1] - old_pos[1])
    return _DELTA_TO_DIR.get(delta)


def _apply_move(
    state: State,
    from_pos: Tuple[int, int],
    direction: str,
) -> Tuple[Optional[State], Optional[str]]:
    normalized_direction = str(direction).strip().upper()
    candidates: List[State] = []
    for new_state, old_pos, new_pos, _ in get_valid_moves_with_details(state):
        if old_pos != from_pos:
            continue
        if _direction_from_positions(old_pos, new_pos) == normalized_direction:
            candidates.append(new_state)

    if not candidates:
        return None, "no_valid_move"
    if len(candidates) > 1:
        return None, "ambiguous_move"
    return candidates[0], None


def _parse_from_position(value: Any) -> Optional[Tuple[int, int]]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return None
    r, c = value
    if not isinstance(r, int) or not isinstance(c, int):
        return None
    return (c, r)


def _xy_to_rc(pos_xy: Tuple[int, int]) -> Tuple[int, int]:
    x, y = pos_xy
    return (y, x)


def evaluate_solution(
    initial_state: State, moves: Any
) -> Dict[str, Optional[Any]]:
    if moves is None:
        return {
            "solved": False,
            "steps_used": 0,
            "error": "no_moves",
            "failed_step": None,
            "message": "parsed solution missing or null",
        }
    if not isinstance(moves, list):
        return {
            "solved": False,
            "steps_used": 0,
            "error": "invalid_moves_type",
            "failed_step": None,
            "message": "solution must be a list",
        }

    state = initial_state
    for idx, step in enumerate(moves):
        if not isinstance(step, dict):
            return {
                "solved": False,
                "steps_used": idx,
                "error": "invalid_step",
                "failed_step": idx,
                "message": "each step must be an object",
            }
        from_pos = _parse_from_position(step.get("from"))
        direction = step.get("direction")
        if from_pos is None or direction is None:
            return {
                "solved": False,
                "steps_used": idx,
                "error": "missing_field",
                "failed_step": idx,
                "message": "step requires from=[r,c] and direction",
            }
        new_state, error = _apply_move(state, from_pos, str(direction))
        if error:
            return {
                "solved": False,
                "steps_used": idx,
                "error": error,
                "failed_step": idx,
                "message": f"step {idx}: {error} for from={from_pos!r} {direction!r}",
            }
        state = new_state

    solved = _xy_to_rc(state[0]) == _GOAL_T_POS_RC
    return {
        "solved": solved,
        "steps_used": len(moves),
        "error": None if solved else "not_goal",
        "failed_step": None if solved else len(moves),
        "message": (
            None
            if solved
            else (
                f"T top-left is {_xy_to_rc(state[0])} in (r,c), "
                f"expected {_GOAL_T_POS_RC}"
            )
        ),
    }


def solutions_equal(a: Any, b: Any) -> bool:
    if a is None or b is None:
        return a is b
    if len(a) != len(b):
        return False
    for left, right in zip(a, b):
        if left.get("from") != right.get("from"):
            return False
        left_direction = str(left.get("direction", "")).upper()
        right_direction = str(right.get("direction", "")).upper()
        if left_direction != right_direction:
            return False
    return True


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


def _resolve_prompt_mode(kwargs: Dict[str, Any]) -> str:
    raw_mode = kwargs.get("klotski_prompt_mode")
    if raw_mode is None:
        raise ValueError(
            "klotski_prompt_mode is required and must be one of: "
            "aux, cot, direct, step_by_step, aux_step_by_step, gta, vision_reasoning"
        )
    mode = str(raw_mode).strip().lower()
    mode = {
        "step-by-step": "step_by_step",
        "aux-step-by-step": "aux_step_by_step",
        "vision-reasoning": "vision_reasoning",
    }.get(mode, mode)
    if mode not in _KLOTSKI_PROMPT_MODES:
        raise ValueError(
            f"Invalid klotski_prompt_mode: {raw_mode}. "
            "Expected one of: aux, cot, direct, step_by_step, aux_step_by_step, gta, vision_reasoning"
        )
    return mode


def _normalize_klotski_difficulty(level: str) -> str:
    mapping = {
        "level1": "easy",
        "level2": "middle",
        "level3": "hard",
    }
    return mapping.get(level, level)


def _resolve_image_path(doc: Dict[str, Any], kwargs: Dict[str, Any]) -> Path:
    image_rel = str(doc.get("image_path", "")).strip()
    if not image_rel:
        raise FileNotFoundError("missing image_path in klotski sample")
    image_root = str(kwargs.get("klotski_image_root", _DEFAULT_IMAGE_ROOT)).strip()
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


def _draw_cell_center_text(
    draw: ImageDraw.ImageDraw,
    left: float,
    top: float,
    cell_w: float,
    cell_h: float,
    text: str,
    font: ImageFont.ImageFont,
) -> None:
    bbox = draw.textbbox((0, 0), text, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]
    offset_x, offset_y = bbox[0], bbox[1]
    center_x = left + cell_w / 2.0
    center_y = top + cell_h / 2.0
    text_x = int(center_x - text_w / 2.0 - offset_x)
    text_y = int(center_y - text_h / 2.0 - offset_y)
    draw.text((text_x, text_y), text, font=font, fill=(0, 0, 0))


def _draw_dashed_line(
    draw: ImageDraw.ImageDraw,
    start: Tuple[int, int],
    end: Tuple[int, int],
    color: Tuple[int, int, int],
    width: int = 2,
    dash_len: int = 12,
    gap_len: int = 18,
) -> None:
    x0, y0 = start
    x1, y1 = end
    if x0 == x1:
        y = min(y0, y1)
        y_end = max(y0, y1)
        while y <= y_end:
            y_dash_end = min(y + dash_len, y_end)
            draw.line([(x0, y), (x1, y_dash_end)], fill=color, width=width)
            y += dash_len + gap_len
        return
    if y0 == y1:
        x = min(x0, x1)
        x_end = max(x0, x1)
        while x <= x_end:
            x_dash_end = min(x + dash_len, x_end)
            draw.line([(x, y0), (x_dash_end, y1)], fill=color, width=width)
            x += dash_len + gap_len


def _build_aux_overlay(image: Image.Image) -> Image.Image:
    overlay = image.copy()
    draw = ImageDraw.Draw(overlay)
    width, height = overlay.size
    cell_w = width / float(_BOARD_WIDTH)
    cell_h = height / float(_BOARD_HEIGHT)
    font = _get_font(max(6, int(min(cell_w, cell_h) * 0.24)))

    _draw_dashed_line(draw, (0, 0), (width - 1, 0), color=_AUX_GRID_RED, width=2)
    _draw_dashed_line(
        draw, (0, height - 1), (width - 1, height - 1), color=_AUX_GRID_RED, width=2
    )
    _draw_dashed_line(draw, (0, 0), (0, height - 1), color=_AUX_GRID_RED, width=2)
    _draw_dashed_line(
        draw, (width - 1, 0), (width - 1, height - 1), color=_AUX_GRID_RED, width=2
    )
    for x_idx in range(1, _BOARD_WIDTH):
        x = round(x_idx * cell_w)
        _draw_dashed_line(draw, (x, 0), (x, height - 1), color=_AUX_GRID_RED, width=2)
    for y_idx in range(1, _BOARD_HEIGHT):
        y = round(y_idx * cell_h)
        _draw_dashed_line(draw, (0, y), (width - 1, y), color=_AUX_GRID_RED, width=2)

    for y_idx in range(_BOARD_HEIGHT):
        for x_idx in range(_BOARD_WIDTH):
            label = f"({y_idx},{x_idx})"
            left = x_idx * cell_w
            top = y_idx * cell_h
            _draw_cell_center_text(
                draw,
                left,
                top,
                cell_w,
                cell_h,
                label,
                font,
            )
    return overlay


def klotski_render_state_image(
    state: State, base_image: Image.Image | None = None
) -> Image.Image:
    width = int(base_image.width) if isinstance(base_image, Image.Image) else 600
    height = int(base_image.height) if isinstance(base_image, Image.Image) else 600
    width = max(120, width)
    height = max(120, height)
    cell_w = width / float(_BOARD_WIDTH)
    cell_h = height / float(_BOARD_HEIGHT)

    image = Image.new("RGB", (width, height), _KLOTSKI_COLORS["empty"])
    draw = ImageDraw.Draw(image)
    for x_idx in range(_BOARD_WIDTH + 1):
        x = int(round(x_idx * cell_w))
        x = min(width - 1, max(0, x))
        draw.line([(x, 0), (x, height - 1)], fill=(200, 200, 200), width=2)
    for y_idx in range(_BOARD_HEIGHT + 1):
        y = int(round(y_idx * cell_h))
        y = min(height - 1, max(0, y))
        draw.line([(0, y), (width - 1, y)], fill=(200, 200, 200), width=2)

    target, verticals, horizontals, singles = state
    blocks = (
        [("T", target)]
        + [("V", pos) for pos in verticals]
        + [("H", pos) for pos in horizontals]
        + [("S", pos) for pos in singles]
    )
    for block_type, (bx, by) in blocks:
        if block_type == "T":
            bw, bh = 2, 2
        elif block_type == "V":
            bw, bh = 1, 2
        elif block_type == "H":
            bw, bh = 2, 1
        else:
            bw, bh = 1, 1
        left = int(round(bx * cell_w))
        top = int(round(by * cell_h))
        right = int(round((bx + bw) * cell_w)) - 1
        bottom = int(round((by + bh) * cell_h)) - 1
        if right < left or bottom < top:
            continue
        draw.rectangle(
            [(left, top), (right, bottom)],
            fill=_KLOTSKI_COLORS.get(block_type, (160, 160, 160)),
            outline=(0, 0, 0),
            width=3,
        )
    return image


def klotski_doc_to_visual(
    doc: Dict[str, Any], lmms_eval_specific_kwargs: Dict[str, Any] | None = None
) -> List[Image.Image]:
    kwargs = lmms_eval_specific_kwargs or {}
    prompt_mode = _resolve_prompt_mode(kwargs)

    image_path = _resolve_image_path(doc, kwargs)
    if not image_path.is_file():
        raise FileNotFoundError(f"klotski image not found: {image_path}")
    image = Image.open(image_path).convert("RGB")

    if prompt_mode in {"aux", "aux_step_by_step"}:
        return [image, _build_aux_overlay(image)]
    return [image]


def klotski_doc_to_text(
    doc: Dict[str, Any], lmms_eval_specific_kwargs: Dict[str, Any] | None = None
) -> str:
    del doc
    kwargs = lmms_eval_specific_kwargs or {}
    prompt_mode = _resolve_prompt_mode(kwargs)

    if prompt_mode in {"aux", "aux_step_by_step"}:
        return "<AUX_GRID_INTERLEAVE>\n" + get_user_prompt(prompt_mode)

    return get_user_prompt(prompt_mode)


def _build_verification_record(doc: Dict[str, Any], parsed: Any) -> Dict[str, Any]:
    ref = doc.get("solution")
    optimal = doc.get("optimal_steps")
    moves = parsed.get("solution") if isinstance(parsed, dict) else None
    initial = doc.get("initial_state")
    if initial is None:
        return {
            "solved": None,
            "steps_used": None,
            "error": "missing_initial_state",
            "failed_step": None,
            "message": "manifest entry has no initial_state",
            "matches_reference": None,
            "optimal_steps": optimal,
        }

    state0 = deserialize_state(initial)
    ev = evaluate_solution(state0, moves)
    ev["matches_reference"] = solutions_equal(moves, ref) if ref is not None else None
    ev["optimal_steps"] = optimal
    return ev


def _klotski_aggregate_metric(
    results: List[Dict[str, Any]],
    metric_key: str,
    level: Optional[str] = None,
) -> float:
    filtered = [
        row for row in results if level is None or str(row.get("level")) == level
    ]
    if not filtered:
        return 0.0
    return sum(float(row.get(metric_key, 0.0)) for row in filtered) / len(filtered)


def klotski_agg_solved_easy(results: List[Dict[str, Any]]) -> float:
    return _klotski_aggregate_metric(results, metric_key="solved", level="easy")


def klotski_agg_solved_middle(results: List[Dict[str, Any]]) -> float:
    return _klotski_aggregate_metric(results, metric_key="solved", level="middle")


def klotski_agg_solved_hard(results: List[Dict[str, Any]]) -> float:
    return _klotski_aggregate_metric(results, metric_key="solved", level="hard")


def klotski_agg_solved_overall(results: List[Dict[str, Any]]) -> float:
    return _klotski_aggregate_metric(results, metric_key="solved", level=None)


def klotski_agg_optimal_easy(results: List[Dict[str, Any]]) -> float:
    return _klotski_aggregate_metric(results, metric_key="optimal", level="easy")


def klotski_agg_optimal_middle(results: List[Dict[str, Any]]) -> float:
    return _klotski_aggregate_metric(results, metric_key="optimal", level="middle")


def klotski_agg_optimal_hard(results: List[Dict[str, Any]]) -> float:
    return _klotski_aggregate_metric(results, metric_key="optimal", level="hard")


def klotski_agg_optimal_overall(results: List[Dict[str, Any]]) -> float:
    return _klotski_aggregate_metric(results, metric_key="optimal", level=None)


def klotski_process_results(doc: Dict[str, Any], results: List[Any]) -> Dict[str, Any]:
    result_raw = results[0] if results else ""
    result_text = _normalize_result_text(result_raw)
    parsed = extract_json_from_text(result_text)
    verification = _build_verification_record(doc, parsed)

    solved = verification.get("solved") is True
    parsed_ok = parsed is not None
    matches_reference = verification.get("matches_reference") is True
    optimal_steps = verification.get("optimal_steps")
    steps_used = verification.get("steps_used")
    is_optimal = solved and isinstance(optimal_steps, int) and steps_used == optimal_steps
    level = _normalize_klotski_difficulty(str(doc.get("level", "")).strip().lower())
    payload = {
        "level": level,
        "solved": 1.0 if solved else 0.0,
        "optimal": 1.0 if is_optimal else 0.0,
    }

    return {
        "klotski_solved_easy": payload,
        "klotski_solved_middle": payload,
        "klotski_solved_hard": payload,
        "klotski_solved_overall": payload,
        "klotski_optimal_easy": payload,
        "klotski_optimal_middle": payload,
        "klotski_optimal_hard": payload,
        "klotski_optimal_overall": payload,
        "klotski_solved": 1.0 if solved else 0.0,
        "klotski_parsed": 1.0 if parsed_ok else 0.0,
        "klotski_matches_reference": 1.0 if matches_reference else 0.0,
        "klotski_optimal": 1.0 if is_optimal else 0.0,
    }
