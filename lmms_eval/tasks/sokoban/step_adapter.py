from __future__ import annotations

from collections import deque
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from PIL import Image

from lmms_eval.models.model_utils.step_protocol import StepTransition
from lmms_eval.tasks.sokoban.utils import (
    _build_aux_overlay,
    _parse_coord_pair,
    _parse_walls,
    sokoban_render_state_image,
)

_MOVE_DELTAS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}


@dataclass
class SokobanStepSession:
    """Mutable sokoban step session state."""

    doc: Dict[str, Any]
    base_image: Image.Image
    player: Tuple[int, int]
    box: Tuple[int, int]
    target: Tuple[int, int]
    walls: set[Tuple[int, int]]
    grid_h: int
    grid_w: int
    use_aux_step_grid: bool
    initial_visual: Image.Image
    optimal_steps: Optional[int]
    executed_steps: int


class SokobanStepAdapter:
    """Sokoban adapter for generic step protocol."""

    def empty_answer(self) -> str:
        return "<ANSWER_JSON>[]</ANSWER_JSON>"

    def initialize_session(
        self, doc: Dict[str, Any], visuals: List[Any], gen_kwargs: Dict[str, Any]
    ) -> SokobanStepSession:
        base_image = self._ensure_image_object(visuals[0] if visuals else None)
        if base_image is None:
            raise ValueError("sokoban step protocol requires a base image")

        player_start = _parse_coord_pair(doc.get("player_start"))
        box_start = _parse_coord_pair(doc.get("box_start"))
        target = _parse_coord_pair(doc.get("target"))
        walls = _parse_walls(doc)
        if player_start is None or box_start is None or target is None:
            raise ValueError("sokoban sample missing player_start/box_start/target")

        grid_h, grid_w = self._resolve_grid_shape(doc, base_image)
        use_aux = self._is_aux_step_by_step_mode(gen_kwargs)
        initial_image = sokoban_render_state_image(
            doc,
            player_pos=player_start,
            box_pos=box_start,
            base_image=base_image,
        )
        if initial_image is None:
            raise ValueError("failed to render initial sokoban state")
        initial_visual = (
            _build_aux_overlay(initial_image, doc) if use_aux else initial_image
        )
        optimal_steps = self._parse_positive_int(doc.get("optimal_steps"))

        return SokobanStepSession(
            doc=doc,
            base_image=base_image,
            player=player_start,
            box=box_start,
            target=target,
            walls=walls,
            grid_h=grid_h,
            grid_w=grid_w,
            use_aux_step_grid=use_aux,
            initial_visual=initial_visual,
            optimal_steps=optimal_steps,
            executed_steps=0,
        )

    def resolve_max_steps(self, doc: Dict[str, Any], gen_kwargs: Dict[str, Any]) -> int:
        raw_steps = doc.get("solution")
        steps_count = 0
        if isinstance(raw_steps, str):
            try:
                parsed = json.loads(raw_steps)
            except (TypeError, json.JSONDecodeError):
                parsed = None
            if isinstance(parsed, list):
                steps_count = len(parsed)
        elif isinstance(raw_steps, list):
            steps_count = len(raw_steps)
        if steps_count > 0:
            return steps_count

        raw_max_steps = self._read_key(gen_kwargs, "sokoban_max_steps")
        if raw_max_steps is None:
            raw_max_steps = self._read_key(gen_kwargs, "step_protocol_max_steps")
        if raw_max_steps is not None:
            try:
                max_steps = int(raw_max_steps)
                if max_steps > 0:
                    return max_steps
            except (TypeError, ValueError):
                pass
        return 64

    def build_initial_user_content(
        self,
        *,
        context: str,
        session: SokobanStepSession,
        build_processed_visuals: Callable[[Optional[List[Any]]], List[Dict[str, Any]]],
    ) -> List[Dict[str, Any]]:
        if session.use_aux_step_grid:
            return (
                build_processed_visuals([session.base_image])
                + [{"type": "text", "text": context}]
                + build_processed_visuals([session.initial_visual])
            )
        return build_processed_visuals([session.initial_visual]) + [
            {"type": "text", "text": context}
        ]

    def build_step_prompt(self, step_idx: int) -> str:
        return (
            f'Now planning for step {step_idx}, Please output a sentence in the form: '
            '"Next, move one step up/down/left/right."\n'
        )

    def extract_step_move(self, raw_step_text: str) -> Optional[str]:
        if not raw_step_text:
            return None
        lowered = raw_step_text.strip().lower()
        match = re.search(r"\b(up|down|left|right)\b", lowered)
        if match:
            return match.group(1)
        try:
            parsed = json.loads(lowered)
        except (TypeError, json.JSONDecodeError):
            return None
        if isinstance(parsed, list) and parsed:
            token = str(parsed[0]).strip().lower()
            return token if token in _MOVE_DELTAS else None
        if isinstance(parsed, str) and parsed in _MOVE_DELTAS:
            return parsed
        return None

    def apply_step_move(
        self, session: SokobanStepSession, move: Optional[str]
    ) -> StepTransition:
        player_before = [session.player[0], session.player[1]]
        box_before = [session.box[0], session.box[1]]
        trace_base = {
            "player_before": player_before,
            "player_after": player_before,
            "box_before": box_before,
            "box_after": box_before,
            "executed_steps": session.executed_steps,
            "remaining_optimal_steps": None,
            "optimal_steps": session.optimal_steps,
            "bound_violated": False,
        }
        if move is None:
            return StepTransition(
                legal=False,
                reason="invalid_direction",
                move=None,
                trace=trace_base,
            )

        dr, dc = _MOVE_DELTAS[move]
        next_player = (session.player[0] + dr, session.player[1] + dc)
        if not self._in_bounds(next_player, session.grid_h, session.grid_w):
            return StepTransition(
                legal=False,
                reason="hit_wall_or_oob",
                move=move,
                trace=trace_base,
            )
        if next_player in session.walls:
            return StepTransition(
                legal=False,
                reason="hit_wall_or_oob",
                move=move,
                trace=trace_base,
            )

        next_box = session.box
        if next_player == session.box:
            pushed_box = (session.box[0] + dr, session.box[1] + dc)
            if not self._in_bounds(pushed_box, session.grid_h, session.grid_w):
                return StepTransition(
                    legal=False,
                    reason="box_blocked",
                    move=move,
                    trace=trace_base,
                )
            if pushed_box in session.walls:
                return StepTransition(
                    legal=False,
                    reason="box_blocked",
                    move=move,
                    trace=trace_base,
                )
            next_box = pushed_box

        session.player = next_player
        session.box = next_box
        session.executed_steps += 1
        remaining_optimal_steps = self._min_steps_to_target(
            start_player=session.player,
            start_box=session.box,
            target=session.target,
            walls=session.walls,
            grid_h=session.grid_h,
            grid_w=session.grid_w,
        )
        trace_base["player_after"] = [session.player[0], session.player[1]]
        trace_base["box_after"] = [session.box[0], session.box[1]]
        trace_base["executed_steps"] = session.executed_steps
        trace_base["remaining_optimal_steps"] = remaining_optimal_steps
        if session.optimal_steps is not None:
            if remaining_optimal_steps is None:
                trace_base["bound_violated"] = True
                return StepTransition(
                    legal=False,
                    reason="exceed_optimal_bound",
                    move=move,
                    trace=trace_base,
                )
            if session.executed_steps + remaining_optimal_steps > session.optimal_steps:
                trace_base["bound_violated"] = True
                return StepTransition(
                    legal=False,
                    reason="exceed_optimal_bound",
                    move=move,
                    trace=trace_base,
                )
        return StepTransition(
            legal=True,
            reason="",
            move=move,
            trace=trace_base,
        )

    def build_step_visual(
        self, session: SokobanStepSession, transition: StepTransition
    ) -> Image.Image | None:
        if transition.legal:
            player = session.player
            box = session.box
        else:
            player = (
                transition.trace["player_before"][0],
                transition.trace["player_before"][1],
            )
            box = (transition.trace["box_before"][0], transition.trace["box_before"][1])
        rendered = sokoban_render_state_image(
            session.doc,
            player_pos=player,
            box_pos=box,
            base_image=session.base_image,
        )
        if rendered is None:
            return None
        if session.use_aux_step_grid:
            return _build_aux_overlay(rendered, session.doc)
        return rendered

    def build_image_generation_prompt(self, step_idx: int) -> str:
        return f"Now, generate the image for step {step_idx}."

    def build_fallback_answer(self, moves: List[str]) -> str:
        return f"<ANSWER_JSON>{json.dumps(moves)}</ANSWER_JSON>"

    def is_goal_reached(self, session: SokobanStepSession) -> bool:
        return session.box == session.target

    @staticmethod
    def _read_key(gen_kwargs: Dict[str, Any], key: str) -> Any:
        protocol_cfg = gen_kwargs.get("step_protocol")
        if isinstance(protocol_cfg, dict) and key in protocol_cfg:
            return protocol_cfg[key]
        return gen_kwargs.get(key)

    @classmethod
    def _is_aux_step_by_step_mode(cls, gen_kwargs: Dict[str, Any]) -> bool:
        mode = str(cls._read_key(gen_kwargs, "sokoban_prompt_mode") or "").strip().lower()
        if mode in {"aux_step_by_step", "aux-step-by-step"}:
            return True
        mode = str(cls._read_key(gen_kwargs, "step_protocol_mode") or "").strip().lower()
        return mode in {"aux_step_by_step", "aux-step-by-step"}

    @staticmethod
    def _ensure_image_object(visual: Any) -> Optional[Image.Image]:
        if isinstance(visual, Image.Image):
            return visual.copy()
        if isinstance(visual, str):
            image_path = Path(visual)
            if image_path.is_file():
                return Image.open(image_path).convert("RGB")
        return None

    @staticmethod
    def _in_bounds(cell: Tuple[int, int], grid_h: int, grid_w: int) -> bool:
        return 0 <= cell[0] < grid_h and 0 <= cell[1] < grid_w

    @staticmethod
    def _resolve_grid_shape(doc: Dict[str, Any], image: Image.Image) -> Tuple[int, int]:
        known: List[Tuple[int, int]] = []
        for key in ("player_start", "box_start", "target"):
            parsed = _parse_coord_pair(doc.get(key))
            if parsed is not None:
                known.append(parsed)
        known.extend(_parse_walls(doc))
        if known:
            max_row = max(cell[0] for cell in known)
            max_col = max(cell[1] for cell in known)
            return max_row + 1, max_col + 1
        grid_h = max(1, int(round(image.height / 64.0)))
        grid_w = max(1, int(round(image.width / 64.0)))
        return grid_h, grid_w

    @staticmethod
    def _parse_positive_int(raw_value: Any) -> Optional[int]:
        try:
            parsed = int(raw_value)
        except (TypeError, ValueError):
            return None
        if parsed <= 0:
            return None
        return parsed

    @classmethod
    def _min_steps_to_target(
        cls,
        *,
        start_player: Tuple[int, int],
        start_box: Tuple[int, int],
        target: Tuple[int, int],
        walls: set[Tuple[int, int]],
        grid_h: int,
        grid_w: int,
    ) -> Optional[int]:
        if start_box == target:
            return 0

        start_state = (start_player, start_box)
        queue = deque([(start_state, 0)])
        visited = {start_state}
        while queue:
            (player, box), dist = queue.popleft()
            for dr, dc in _MOVE_DELTAS.values():
                next_player = (player[0] + dr, player[1] + dc)
                if not cls._in_bounds(next_player, grid_h, grid_w):
                    continue
                if next_player in walls:
                    continue

                next_box = box
                if next_player == box:
                    pushed_box = (box[0] + dr, box[1] + dc)
                    if not cls._in_bounds(pushed_box, grid_h, grid_w):
                        continue
                    if pushed_box in walls:
                        continue
                    next_box = pushed_box

                next_state = (next_player, next_box)
                if next_state in visited:
                    continue
                if next_box == target:
                    return dist + 1
                visited.add(next_state)
                queue.append((next_state, dist + 1))
        return None
