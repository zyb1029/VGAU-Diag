from __future__ import annotations

import json
import re
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from PIL import Image

from lmms_eval.models.model_utils.step_protocol import StepTransition
from lmms_eval.tasks.parking.utils import (
    _apply_one_step,
    _build_aux_overlay,
    _is_solved,
    _state_from_doc,
    parking_render_state_image,
)

_MOVE_DIRS: Dict[str, Tuple[int, int]] = {
    "left": (0, -1),
    "right": (0, 1),
    "up": (-1, 0),
    "down": (1, 0),
}


@dataclass
class ParkingExitStepSession:
    """Mutable session state for one parking-exit step protocol run."""

    doc: Dict[str, Any]
    base_image: Image.Image
    cars: List[Dict[str, Any]]
    grid_size: int
    exit_row_1based: int
    exit_row_0: int
    use_aux_step_grid: bool
    initial_visual: Image.Image
    optimal_steps: int
    executed_steps: int


class ParkingStepAdapter:
    """Parking-exit adapter for the generic step protocol."""

    def empty_answer(self) -> str:
        return "<ANSWER_JSON>[]</ANSWER_JSON>"

    def initialize_session(
        self,
        doc: Dict[str, Any],
        visuals: List[Any],
        gen_kwargs: Dict[str, Any],
    ) -> ParkingExitStepSession:
        base_image = self._ensure_image_object(visuals[0] if visuals else None)
        if base_image is None:
            raise ValueError("parking step protocol requires a base image")

        cars = _state_from_doc(doc)
        if not cars:
            raise ValueError("parking sample missing cars")

        grid_size = int(doc.get("grid_size", 8))
        exit_row_1based = int(doc.get("exit_row", 4))
        exit_row_0 = exit_row_1based - 1
        optimal_steps = int(doc.get("optimal_steps", 0))
        use_aux = self._is_aux_step_by_step_mode(gen_kwargs)

        rendered = parking_render_state_image(cars, grid_size, exit_row_0)
        if rendered is None:
            raise ValueError("failed to render initial parking state")
        initial_visual = _build_aux_overlay(rendered, grid_size) if use_aux else rendered

        return ParkingExitStepSession(
            doc=doc,
            base_image=base_image,
            cars=[dict(c) for c in cars],
            grid_size=grid_size,
            exit_row_1based=exit_row_1based,
            exit_row_0=exit_row_0,
            use_aux_step_grid=use_aux,
            initial_visual=initial_visual,
            optimal_steps=optimal_steps,
            executed_steps=0,
        )

    def resolve_max_steps(self, doc: Dict[str, Any], gen_kwargs: Dict[str, Any]) -> int:
        raw_sol = doc.get("solution")
        if isinstance(raw_sol, list) and raw_sol:
            return len(raw_sol)
        if isinstance(raw_sol, str):
            try:
                parsed = json.loads(raw_sol)
            except (TypeError, json.JSONDecodeError):
                parsed = None
            if isinstance(parsed, list) and parsed:
                return len(parsed)

        optimal = int(doc.get("optimal_steps", 0))
        if optimal > 0:
            return optimal

        raw_max = self._read_key(gen_kwargs, "parking_max_steps")
        if raw_max is None:
            raw_max = self._read_key(gen_kwargs, "step_protocol_max_steps")
        if raw_max is not None:
            try:
                v = int(raw_max)
                if v > 0:
                    return v
            except (TypeError, ValueError):
                pass
        return 128

    def build_initial_user_content(
        self,
        *,
        context: str,
        session: ParkingExitStepSession,
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
            f"Now planning for step {step_idx}. Output exactly one move in this format:\n"
            '<STEP_JSON>{"car":"A","direction":"right"}</STEP_JSON>\n'
            "No extra text."
        )

    def build_image_generation_prompt(self, step_idx: int) -> str:
        return f"Now, generate the image for step {step_idx}."

    def extract_step_move(self, raw_step_text: str) -> Optional[str]:
        move = self._parse_single_step(raw_step_text)
        if move is None:
            return None
        return json.dumps(move, separators=(",", ":"))

    def apply_step_move(
        self,
        session: ParkingExitStepSession,
        move: Optional[str],
    ) -> StepTransition:
        trace_base: Dict[str, Any] = {
            "predicted_move": move,
            "executed_steps": session.executed_steps,
            "remaining_optimal": None,
            "bound_violated": False,
        }

        if move is None:
            return StepTransition(
                legal=False,
                reason="invalid_format",
                move=None,
                trace=trace_base,
            )

        step = self._deserialize_step(move)
        if step is None:
            return StepTransition(
                legal=False,
                reason="invalid_format",
                move=None,
                trace=trace_base,
            )

        legal, new_cars = _apply_one_step(session.cars, step, session.grid_size)
        if not legal:
            return StepTransition(
                legal=False,
                reason="illegal_move",
                move=move,
                trace=trace_base,
            )

        session.cars = new_cars
        session.executed_steps += 1
        trace_base["executed_steps"] = session.executed_steps

        if session.optimal_steps > 0:
            remaining = self._bfs_remaining(session)
            trace_base["remaining_optimal"] = remaining
            if remaining is None or session.executed_steps + remaining > session.optimal_steps:
                trace_base["bound_violated"] = True
                return StepTransition(
                    legal=False,
                    reason="exceed_optimal_bound",
                    move=move,
                    trace=trace_base,
                )

        return StepTransition(legal=True, reason="", move=move, trace=trace_base)

    def build_step_visual(
        self,
        session: ParkingExitStepSession,
        transition: StepTransition,
    ) -> Image.Image | None:
        del transition
        rendered = parking_render_state_image(
            session.cars,
            session.grid_size,
            session.exit_row_0,
        )
        if rendered is None:
            return None
        if session.use_aux_step_grid:
            return _build_aux_overlay(rendered, session.grid_size)
        return rendered

    def build_fallback_answer(self, moves: List[str]) -> str:
        payload: List[Dict[str, str]] = []
        for move in moves:
            step = self._deserialize_step(move)
            if step is None:
                continue
            payload.append({"car": step["car"], "direction": step["direction"]})
        return f"<ANSWER_JSON>{json.dumps(payload)}</ANSWER_JSON>"

    def is_goal_reached(self, session: ParkingExitStepSession) -> bool:
        return _is_solved(session.cars, session.exit_row_1based, session.grid_size)

    # ── helpers ────────────────────────────────────────────────────────────

    @staticmethod
    def _read_key(gen_kwargs: Dict[str, Any], key: str) -> Any:
        protocol_cfg = gen_kwargs.get("step_protocol")
        if isinstance(protocol_cfg, dict) and key in protocol_cfg:
            return protocol_cfg[key]
        return gen_kwargs.get(key)

    @classmethod
    def _is_aux_step_by_step_mode(cls, gen_kwargs: Dict[str, Any]) -> bool:
        mode = str(cls._read_key(gen_kwargs, "parking_prompt_mode") or "").strip().lower()
        if mode in {"aux_step_by_step", "aux-step-by-step"}:
            return True
        mode = str(cls._read_key(gen_kwargs, "step_protocol_mode") or "").strip().lower()
        return mode in {"aux_step_by_step", "aux-step-by-step"}

    @staticmethod
    def _ensure_image_object(visual: Any) -> Optional[Image.Image]:
        if isinstance(visual, Image.Image):
            return visual.copy()
        if isinstance(visual, str):
            p = Path(visual)
            if p.is_file():
                return Image.open(p).convert("RGB")
        return None

    @classmethod
    def _parse_single_step(cls, text: str) -> Optional[Dict[str, str]]:
        if not text:
            return None
        # try <STEP_JSON>...</STEP_JSON>
        tagged = list(
            re.finditer(
                r"<STEP_JSON>\s*([\s\S]*?)\s*</STEP_JSON>",
                text,
                re.IGNORECASE,
            )
        )
        if tagged:
            candidate = tagged[-1].group(1)
            step = cls._try_parse_step(candidate)
            if step is not None:
                return step

        # fallback: any {"car":...,"direction":...} in text
        for m in re.finditer(r"\{[^{}]*\}", text):
            step = cls._try_parse_step(m.group(0))
            if step is not None:
                return step
        return None

    @staticmethod
    def _try_parse_step(raw: Any) -> Optional[Dict[str, str]]:
        if isinstance(raw, str):
            try:
                raw = json.loads(raw)
            except (TypeError, json.JSONDecodeError):
                return None
        if not isinstance(raw, dict):
            return None
        car = str(raw.get("car", "")).strip().upper()
        direction = str(raw.get("direction", "")).strip().lower()
        direction = {"u": "up", "d": "down", "l": "left", "r": "right"}.get(
            direction, direction
        )
        if not car or direction not in {"up", "down", "left", "right"}:
            return None
        return {"car": car, "direction": direction}

    @classmethod
    def _deserialize_step(cls, text: str) -> Optional[Dict[str, str]]:
        try:
            raw = json.loads(text)
        except (TypeError, json.JSONDecodeError):
            return None
        return cls._try_parse_step(raw)

    @staticmethod
    def _bfs_remaining(session: ParkingExitStepSession) -> Optional[int]:
        """BFS from current state; return minimum steps to reach goal, or None."""
        grid = session.grid_size
        exit_row_0 = session.exit_row_0

        def encode(cars: List[Dict[str, Any]]) -> Tuple[int, ...]:
            return tuple(
                (c["row"], c["col"])
                for c in sorted(cars, key=lambda x: x["id"])
            )

        def is_goal(cars: List[Dict[str, Any]]) -> bool:
            for c in cars:
                if c["id"] == "R":
                    return (
                        c["row"] == exit_row_0
                        and c["col"] + c["length"] - 1 == grid - 1
                    )
            return False

        if is_goal(session.cars):
            return 0

        def occ_map(cars: List[Dict[str, Any]]) -> Dict[Tuple[int, int], str]:
            m: Dict[Tuple[int, int], str] = {}
            for c in cars:
                r, col = c["row"], c["col"]
                for i in range(c["length"]):
                    if c["orientation"] == "H":
                        m[(r, col + i)] = c["id"]
                    else:
                        m[(r + i, col)] = c["id"]
            return m

        def neighbors(cars: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
            result = []
            occ = occ_map(cars)
            for idx, car in enumerate(cars):
                dirs = ["left", "right"] if car["orientation"] == "H" else ["up", "down"]
                for d in dirs:
                    dr, dc = _MOVE_DIRS[d]
                    nr, nc = car["row"] + dr, car["col"] + dc
                    length = car["length"]
                    in_bounds = (
                        0 <= nr
                        and 0 <= nc
                        and nr + (length - 1 if car["orientation"] == "V" else 0) < grid
                        and nc + (length - 1 if car["orientation"] == "H" else 0) < grid
                    )
                    if not in_bounds:
                        continue
                    if car["orientation"] == "H":
                        new_cells = [(car["row"], nc + i) for i in range(length)]
                    else:
                        new_cells = [(nr + i, car["col"]) for i in range(length)]
                    if any(occ.get(rc, car["id"]) != car["id"] for rc in new_cells):
                        continue
                    new_cars = [dict(c) for c in cars]
                    new_cars[idx] = dict(car)
                    new_cars[idx]["row"] = nr
                    new_cars[idx]["col"] = nc
                    result.append(new_cars)
            return result

        start = encode(session.cars)
        visited = {start}
        queue: deque[Tuple[List[Dict[str, Any]], int]] = deque([(session.cars, 0)])
        max_nodes = 200_000

        while queue:
            cars, depth = queue.popleft()
            if len(visited) > max_nodes:
                return None
            for nxt in neighbors(cars):
                key = encode(nxt)
                if key in visited:
                    continue
                if is_goal(nxt):
                    return depth + 1
                visited.add(key)
                queue.append((nxt, depth + 1))
        return None
