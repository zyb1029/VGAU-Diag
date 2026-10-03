from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from PIL import Image

from lmms_eval.models.model_utils.step_protocol import StepTransition
from lmms_eval.tasks.maze.utils import (
    _build_maze_overlay_grid,
    _maze_parse_coord,
    _maze_parse_walkable,
    maze_render_state_image,
)

_MOVE_DELTAS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}


@dataclass
class MazeStepSession:
    """Mutable maze step session state."""

    doc: Dict[str, Any]
    base_image: Image.Image
    current: Tuple[int, int]
    walkable: set[Tuple[int, int]]
    use_aux_step_grid: bool
    label_cells: bool
    label_start_goal: bool
    initial_visual: Image.Image
    gold_steps: List[str]
    step_cursor: int
    executed_steps: int
    optimal_steps: Optional[int]


class MazeStepAdapter:
    """Maze adapter for generic step protocol."""

    def empty_answer(self) -> str:
        return "<ANSWER_JSON>[]</ANSWER_JSON>"

    def initialize_session(
        self, doc: Dict[str, Any], visuals: List[Any], gen_kwargs: Dict[str, Any]
    ) -> MazeStepSession:
        base_image = self._ensure_image_object(visuals[0] if visuals else None)
        if base_image is None:
            raise ValueError("maze step protocol requires a base image")

        start = _maze_parse_coord(doc.get("start"))
        goal = _maze_parse_coord(doc.get("goal"))
        walkable = _maze_parse_walkable(doc)
        if start is None or goal is None or not walkable or start not in walkable:
            raise ValueError("maze sample missing start/goal/walkable")
        gold_steps = self._parse_gold_steps(doc)

        use_aux = self._is_aux_step_by_step_mode(gen_kwargs)
        label_cells = self._to_bool(
            self._read_key(gen_kwargs, "aux_grid_label_cells"), default=True
        )
        label_start_goal = self._to_bool(
            self._read_key(gen_kwargs, "aux_grid_label_start_goal"), default=False
        )
        initial_image = maze_render_state_image(doc, start, base_image=base_image)
        if initial_image is None:
            raise ValueError("failed to render initial maze state")

        initial_visual = self._build_step_visual(
            initial_image,
            use_aux_step_grid=use_aux,
            label_cells=label_cells,
            label_start_goal=label_start_goal,
        )
        optimal_steps = self.resolve_max_steps(doc, gen_kwargs)
        return MazeStepSession(
            doc=doc,
            base_image=base_image,
            current=start,
            walkable=walkable,
            use_aux_step_grid=use_aux,
            label_cells=label_cells,
            label_start_goal=label_start_goal,
            initial_visual=initial_visual,
            gold_steps=gold_steps,
            step_cursor=0,
            executed_steps=0,
            optimal_steps=optimal_steps if optimal_steps > 0 else None,
        )

    def resolve_max_steps(self, doc: Dict[str, Any], gen_kwargs: Dict[str, Any]) -> int:
        raw_steps = doc.get("steps")
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

        raw_max_steps = self._read_key(gen_kwargs, "maze_max_steps")
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
        session: MazeStepSession,
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

    def build_image_generation_prompt(self, step_idx: int) -> str:
        return f"Now, generate the image for step {step_idx}."

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
        self, session: MazeStepSession, move: Optional[str]
    ) -> StepTransition:
        before = [session.current[0], session.current[1]]
        expected_move: Optional[str] = None
        if 0 <= session.step_cursor < len(session.gold_steps):
            expected_move = session.gold_steps[session.step_cursor]
        trace_base = {
            "expected_move": expected_move,
            "predicted_move": move,
            "position_before": before,
            "position_after": before,
            "executed_steps": session.executed_steps,
            "remaining_optimal_steps": None,
            "optimal_steps": session.optimal_steps,
            "bound_violated": False,
        }
        session.step_cursor += 1
        if move is None:
            return StepTransition(
                legal=False,
                reason="invalid_direction",
                move=None,
                trace=trace_base,
            )

        dr, dc = _MOVE_DELTAS[move]
        candidate = (session.current[0] + dr, session.current[1] + dc)
        if candidate not in session.walkable:
            return StepTransition(
                legal=False,
                reason="hit_wall_or_oob",
                move=move,
                trace=trace_base,
            )

        session.current = candidate
        session.executed_steps += 1
        goal = _maze_parse_coord(session.doc.get("goal"))
        remaining_optimal_steps = self._min_steps_to_goal(
            start=candidate,
            goal=goal,
            walkable=session.walkable,
        )
        trace_base["position_after"] = [candidate[0], candidate[1]]
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
        self, session: MazeStepSession, transition: StepTransition
    ) -> Image.Image | None:
        current_pos = (
            session.current
            if transition.legal
            else (transition.trace["position_before"][0], transition.trace["position_before"][1])
        )
        rendered = maze_render_state_image(
            session.doc,
            current_pos=current_pos,
            base_image=session.base_image,
        )
        if rendered is None:
            return None
        return self._build_step_visual(
            rendered,
            use_aux_step_grid=session.use_aux_step_grid,
            label_cells=session.label_cells,
            label_start_goal=session.label_start_goal,
        )

    def build_fallback_answer(self, moves: List[str]) -> str:
        return f"<ANSWER_JSON>{json.dumps(moves)}</ANSWER_JSON>"

    def build_process_answer(
        self,
        *,
        doc: Dict[str, Any],
        session: MazeStepSession,
        accepted_moves: List[str],
        step_texts: List[str],
        generated_images: List[str],
    ) -> str:
        del doc, session, step_texts, generated_images
        return self.build_fallback_answer(accepted_moves)

    def is_goal_reached(self, session: MazeStepSession) -> bool:
        goal = _maze_parse_coord(session.doc.get("goal"))
        if goal is None:
            return False
        return session.current == goal

    @staticmethod
    def _read_key(gen_kwargs: Dict[str, Any], key: str) -> Any:
        protocol_cfg = gen_kwargs.get("step_protocol")
        if isinstance(protocol_cfg, dict) and key in protocol_cfg:
            return protocol_cfg[key]
        return gen_kwargs.get(key)

    @classmethod
    def _is_aux_step_by_step_mode(cls, gen_kwargs: Dict[str, Any]) -> bool:
        mode = str(cls._read_key(gen_kwargs, "maze_prompt_mode") or "").strip().lower()
        if mode in {"aux_step_by_step", "aux-step-by-step"}:
            return True
        mode = str(cls._read_key(gen_kwargs, "step_protocol_mode") or "").strip().lower()
        return mode in {"aux_step_by_step", "aux-step-by-step"}

    @staticmethod
    def _to_bool(value: Any, default: bool = False) -> bool:
        if isinstance(value, bool):
            return value
        if value is None:
            return default
        if isinstance(value, str):
            norm = value.strip().lower()
            if norm in {"1", "true", "yes", "y", "on"}:
                return True
            if norm in {"0", "false", "no", "n", "off"}:
                return False
        return bool(value)

    @staticmethod
    def _parse_gold_steps(doc: Dict[str, Any]) -> List[str]:
        raw_steps = doc.get("steps")
        steps: Any = raw_steps
        if isinstance(raw_steps, str):
            try:
                steps = json.loads(raw_steps)
            except (TypeError, json.JSONDecodeError):
                steps = []
        if not isinstance(steps, list):
            return []
        normalized: List[str] = []
        for token in steps:
            direction = str(token).strip().lower()
            if direction in _MOVE_DELTAS:
                normalized.append(direction)
        return normalized

    @staticmethod
    def _min_steps_to_goal(
        start: Tuple[int, int],
        goal: Optional[Tuple[int, int]],
        walkable: set[Tuple[int, int]],
    ) -> Optional[int]:
        if goal is None:
            return None
        if start == goal:
            return 0
        if start not in walkable or goal not in walkable:
            return None

        queue: List[Tuple[Tuple[int, int], int]] = [(start, 0)]
        visited: set[Tuple[int, int]] = {start}
        cursor = 0
        while cursor < len(queue):
            current, dist = queue[cursor]
            cursor += 1
            for dr, dc in _MOVE_DELTAS.values():
                nxt = (current[0] + dr, current[1] + dc)
                if nxt in visited or nxt not in walkable:
                    continue
                if nxt == goal:
                    return dist + 1
                visited.add(nxt)
                queue.append((nxt, dist + 1))
        return None

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
    def _build_step_visual(
        image_obj: Image.Image,
        *,
        use_aux_step_grid: bool,
        label_cells: bool,
        label_start_goal: bool,
    ) -> Image.Image:
        if not use_aux_step_grid:
            return image_obj
        return _build_maze_overlay_grid(
            image_obj,
            label_cell_coords=label_cells,
            label_start_goal=label_start_goal,
        )
