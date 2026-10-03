from __future__ import annotations

from collections import deque
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from PIL import Image

from lmms_eval.models.model_utils.step_protocol import StepTransition
from lmms_eval.tasks.klotski.utils import (
    _GOAL_T_POS_RC,
    _apply_move,
    _build_aux_overlay,
    _xy_to_rc,
    deserialize_state,
    get_valid_moves_with_details,
    klotski_render_state_image,
)

State = Tuple[
    Tuple[int, int],
    Tuple[Tuple[int, int], ...],
    Tuple[Tuple[int, int], ...],
    Tuple[Tuple[int, int], ...],
]


@dataclass
class KlotskiStepSession:
    doc: Dict[str, Any]
    base_image: Image.Image
    state: State
    use_aux_step_grid: bool
    initial_visual: Image.Image
    optimal_steps: Optional[int]
    executed_steps: int


class KlotskiStepAdapter:
    def empty_answer(self) -> str:
        return '{"solution":[]}'

    def initialize_session(
        self, doc: Dict[str, Any], visuals: List[Any], gen_kwargs: Dict[str, Any]
    ) -> KlotskiStepSession:
        base_image = self._ensure_image_object(visuals[0] if visuals else None)
        if base_image is None:
            raise ValueError("klotski step protocol requires a base image")
        initial_state_raw = doc.get("initial_state")
        if not isinstance(initial_state_raw, dict):
            raise ValueError("klotski sample missing initial_state")
        state = deserialize_state(initial_state_raw)
        use_aux = self._is_aux_step_by_step_mode(gen_kwargs)
        rendered = klotski_render_state_image(state, base_image=base_image)
        initial_visual = _build_aux_overlay(rendered) if use_aux else rendered
        optimal_steps = self._parse_positive_int(doc.get("optimal_steps"))
        return KlotskiStepSession(
            doc=doc,
            base_image=base_image,
            state=state,
            use_aux_step_grid=use_aux,
            initial_visual=initial_visual,
            optimal_steps=optimal_steps,
            executed_steps=0,
        )

    def resolve_max_steps(self, doc: Dict[str, Any], gen_kwargs: Dict[str, Any]) -> int:
        solution_raw = doc.get("solution")
        if isinstance(solution_raw, list) and solution_raw:
            return len(solution_raw)
        raw_max_steps = self._read_key(gen_kwargs, "klotski_max_steps")
        if raw_max_steps is None:
            raw_max_steps = self._read_key(gen_kwargs, "step_protocol_max_steps")
        parsed_max = self._parse_positive_int(raw_max_steps)
        if parsed_max is not None:
            return parsed_max
        parsed_optimal = self._parse_positive_int(doc.get("optimal_steps"))
        if parsed_optimal is not None:
            return parsed_optimal
        return 128

    def build_initial_user_content(
        self,
        *,
        context: str,
        session: KlotskiStepSession,
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
            '<STEP_JSON>{"from":[r,c],"direction":"UP|DOWN|LEFT|RIGHT"}</STEP_JSON>\n'
            "No extra keys and no extra text."
        )

    def extract_step_move(self, raw_step_text: str) -> Optional[str]:
        parsed = self._parse_single_step(raw_step_text)
        if parsed is None:
            return None
        return self._serialize_step(parsed)

    def apply_step_move(
        self, session: KlotskiStepSession, move: Optional[str]
    ) -> StepTransition:
        trace: Dict[str, Any] = {
            "executed_steps": session.executed_steps,
            "remaining_optimal_steps": None,
            "optimal_steps": session.optimal_steps,
            "bound_violated": False,
            "target_rc_before": list(_xy_to_rc(session.state[0])),
            "target_rc_after": list(_xy_to_rc(session.state[0])),
        }
        if move is None:
            return StepTransition(
                legal=False,
                reason="invalid_step_format",
                move=None,
                trace=trace,
            )

        parsed = self._deserialize_step(move)
        if parsed is None:
            return StepTransition(
                legal=False,
                reason="invalid_step_format",
                move=None,
                trace=trace,
            )

        from_rc, direction = parsed
        from_xy = (int(from_rc[1]), int(from_rc[0]))
        next_state, error = _apply_move(session.state, from_xy, direction)
        if error or next_state is None:
            return StepTransition(
                legal=False,
                reason=str(error or "illegal_move"),
                move=move,
                trace=trace,
            )

        session.state = next_state
        session.executed_steps += 1
        trace["executed_steps"] = session.executed_steps
        trace["target_rc_after"] = list(_xy_to_rc(session.state[0]))

        remaining_optimal_steps = self._min_steps_to_goal(session.state)
        trace["remaining_optimal_steps"] = remaining_optimal_steps

        if session.optimal_steps is not None:
            if remaining_optimal_steps is None:
                trace["bound_violated"] = True
                return StepTransition(
                    legal=False,
                    reason="exceed_optimal_bound",
                    move=move,
                    trace=trace,
                )
            if session.executed_steps + remaining_optimal_steps > session.optimal_steps:
                trace["bound_violated"] = True
                return StepTransition(
                    legal=False,
                    reason="exceed_optimal_bound",
                    move=move,
                    trace=trace,
                )

        return StepTransition(
            legal=True,
            reason="",
            move=move,
            trace=trace,
        )

    def build_step_visual(
        self, session: KlotskiStepSession, transition: StepTransition
    ) -> Image.Image | None:
        del transition
        rendered = klotski_render_state_image(session.state, base_image=session.base_image)
        if rendered is None:
            return None
        if session.use_aux_step_grid:
            return _build_aux_overlay(rendered)
        return rendered

    def build_image_generation_prompt(self, step_idx: int) -> str:
        return f"Now, generate the image for step {step_idx}."

    def build_fallback_answer(self, moves: List[str]) -> str:
        solution: List[Dict[str, Any]] = []
        for move_text in moves:
            parsed = self._deserialize_step(move_text)
            if parsed is None:
                continue
            from_rc, direction = parsed
            solution.append(
                {
                    "from": [from_rc[0], from_rc[1]],
                    "direction": str(direction).upper(),
                }
            )
        return json.dumps({"solution": solution}, ensure_ascii=False)

    def is_goal_reached(self, session: KlotskiStepSession) -> bool:
        return _xy_to_rc(session.state[0]) == _GOAL_T_POS_RC

    @staticmethod
    def _read_key(gen_kwargs: Dict[str, Any], key: str) -> Any:
        protocol_cfg = gen_kwargs.get("step_protocol")
        if isinstance(protocol_cfg, dict) and key in protocol_cfg:
            return protocol_cfg[key]
        return gen_kwargs.get(key)

    @classmethod
    def _is_aux_step_by_step_mode(cls, gen_kwargs: Dict[str, Any]) -> bool:
        mode = str(cls._read_key(gen_kwargs, "klotski_prompt_mode") or "").strip().lower()
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
    def _parse_positive_int(raw_value: Any) -> Optional[int]:
        try:
            parsed = int(raw_value)
        except (TypeError, ValueError):
            return None
        if parsed <= 0:
            return None
        return parsed

    @classmethod
    def _parse_single_step(
        cls, text: str
    ) -> Optional[Tuple[Tuple[int, int], str]]:
        if not text:
            return None
        payload_text = text.strip()
        tagged = list(
            re.finditer(
                r"<STEP_JSON>\s*([\s\S]*?)\s*</STEP_JSON>",
                payload_text,
                re.IGNORECASE,
            )
        )
        if tagged:
            payload_text = tagged[-1].group(1)

        parsed_step = cls._try_parse_step(payload_text)
        if parsed_step is not None:
            return parsed_step

        answer_tagged = list(
            re.finditer(
                r"<answer_json>\s*([\s\S]*?)\s*</answer_json>",
                text,
                re.IGNORECASE,
            )
        )
        if answer_tagged:
            answer_payload = answer_tagged[-1].group(1)
            try:
                parsed = json.loads(answer_payload)
            except (TypeError, json.JSONDecodeError):
                return None
            if isinstance(parsed, dict):
                payload = parsed.get("solution")
                if isinstance(payload, list) and payload:
                    return cls._try_parse_step(payload[0])
            if isinstance(parsed, list) and parsed:
                return cls._try_parse_step(parsed[0])

        return None

    @classmethod
    def _try_parse_step(cls, payload: Any) -> Optional[Tuple[Tuple[int, int], str]]:
        parsed = payload
        if isinstance(payload, str):
            try:
                parsed = json.loads(payload)
            except (TypeError, json.JSONDecodeError):
                return None
        if not isinstance(parsed, dict):
            return None
        from_raw = parsed.get("from")
        if not isinstance(from_raw, (list, tuple)) or len(from_raw) != 2:
            return None
        try:
            row = int(from_raw[0])
            col = int(from_raw[1])
        except (TypeError, ValueError):
            return None
        direction = str(parsed.get("direction", "")).strip().upper()
        if direction not in {"UP", "DOWN", "LEFT", "RIGHT"}:
            return None
        return (row, col), direction

    @staticmethod
    def _serialize_step(step: Tuple[Tuple[int, int], str]) -> str:
        from_rc, direction = step
        return json.dumps(
            {"from": [from_rc[0], from_rc[1]], "direction": direction},
            separators=(",", ":"),
            ensure_ascii=False,
        )

    @classmethod
    def _deserialize_step(cls, text: str) -> Optional[Tuple[Tuple[int, int], str]]:
        try:
            payload = json.loads(text)
        except (TypeError, json.JSONDecodeError):
            return None
        return cls._try_parse_step(payload)

    @staticmethod
    def _min_steps_to_goal(start_state: State) -> Optional[int]:
        if _xy_to_rc(start_state[0]) == _GOAL_T_POS_RC:
            return 0
        queue = deque([(start_state, 0)])
        visited = {start_state}
        while queue:
            state, dist = queue.popleft()
            for new_state, _, _, _ in get_valid_moves_with_details(state):
                if new_state in visited:
                    continue
                if _xy_to_rc(new_state[0]) == _GOAL_T_POS_RC:
                    return dist + 1
                visited.add(new_state)
                queue.append((new_state, dist + 1))
        return None
