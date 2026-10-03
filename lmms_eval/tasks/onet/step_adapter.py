from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from PIL import Image

from lmms_eval.models.model_utils.step_protocol import StepTransition
from lmms_eval.tasks.onet.utils import (
    _build_aux_grid,
    _is_board_cleared,
    _parse_board_grid,
    _resolve_grid_size_from_doc,
    _validate_and_apply_steps,
    onet_render_state_image,
)


Coord = Tuple[int, int]
OnetMove = Tuple[Coord, Coord]


@dataclass
class OnetStepSession:
    """Mutable ONET step session state."""

    doc: Dict[str, Any]
    base_image: Image.Image
    board: List[List[str | None]]
    initial_board: List[List[str | None]]
    use_aux_step_grid: bool
    grid_size: int
    initial_visual: Image.Image


class OnetStepAdapter:
    """ONET adapter for generic step protocol."""

    def empty_answer(self) -> str:
        return "<answer_json>[]</answer_json>"

    def initialize_session(
        self, doc: Dict[str, Any], visuals: List[Any], gen_kwargs: Dict[str, Any]
    ) -> OnetStepSession:
        base_image = self._ensure_image_object(visuals[0] if visuals else None)
        if base_image is None:
            raise ValueError("onet step protocol requires a base image")

        board = _parse_board_grid(doc)
        if not board:
            raise ValueError("onet sample missing board")

        use_aux = self._is_aux_step_by_step_mode(gen_kwargs)
        grid_size = _resolve_grid_size_from_doc(doc, board)
        rendered = onet_render_state_image(
            doc,
            board,
            initial_board=board,
            base_image=base_image,
        )
        if rendered is None:
            raise ValueError("failed to render initial onet state")
        initial_visual = _build_aux_grid(rendered, grid_size) if use_aux else rendered

        return OnetStepSession(
            doc=doc,
            base_image=base_image,
            board=[list(row) for row in board],
            initial_board=[list(row) for row in board],
            use_aux_step_grid=use_aux,
            grid_size=grid_size,
            initial_visual=initial_visual,
        )

    def resolve_max_steps(self, doc: Dict[str, Any], gen_kwargs: Dict[str, Any]) -> int:
        board = _parse_board_grid(doc)
        initial_remaining_pairs = self._remaining_cells(board) // 2 if board else 0

        raw_max_steps = self._read_key(gen_kwargs, "onet_max_steps")
        if raw_max_steps is None:
            raw_max_steps = self._read_key(gen_kwargs, "step_protocol_max_steps")
        configured_steps: Optional[int] = None
        if raw_max_steps is not None:
            try:
                max_steps = int(raw_max_steps)
                if max_steps > 0:
                    configured_steps = max_steps
            except (TypeError, ValueError):
                pass

        if configured_steps is not None:
            return max(configured_steps, initial_remaining_pairs)
        return initial_remaining_pairs

    def build_initial_user_content(
        self,
        *,
        context: str,
        session: OnetStepSession,
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
            f"Now planning for step {step_idx}. Output exactly one pair in this format:\n"
            "<STEP_JSON>[[r1,c1],[r2,c2]]</STEP_JSON>\n"
            "No extra keys and no extra text."
        )

    def build_image_generation_prompt(self, step_idx: int) -> str:
        return f"Now, generate the image for step {step_idx}."

    def extract_step_move(self, raw_step_text: str) -> Optional[str]:
        move = self._parse_single_step(raw_step_text)
        if move is None:
            return None
        return self._serialize_step(move)

    def apply_step_move(
        self, session: OnetStepSession, move: Optional[str]
    ) -> StepTransition:
        trace_base: Dict[str, Any] = {
            "predicted_move": move,
            "remaining_cells_before": self._remaining_cells(session.board),
            "remaining_cells_after": self._remaining_cells(session.board),
        }
        if move is None:
            return StepTransition(
                legal=False,
                reason="invalid_step_format",
                move=None,
                trace=trace_base,
            )
        parsed_move = self._deserialize_step(move)
        if parsed_move is None:
            return StepTransition(
                legal=False,
                reason="invalid_step_format",
                move=None,
                trace=trace_base,
            )

        legal = _validate_and_apply_steps(session.board, [parsed_move])
        trace_base["remaining_cells_after"] = self._remaining_cells(session.board)
        if not legal:
            return StepTransition(
                legal=False,
                reason="invalid_pair_or_path",
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
        self, session: OnetStepSession, transition: StepTransition
    ) -> Image.Image | None:
        del transition
        rendered = onet_render_state_image(
            session.doc,
            session.board,
            initial_board=session.initial_board,
            base_image=session.base_image,
        )
        if rendered is None:
            return None
        if session.use_aux_step_grid:
            return _build_aux_grid(rendered, session.grid_size)
        return rendered

    def build_fallback_answer(self, moves: List[str]) -> str:
        payload: List[List[List[int]]] = []
        for move in moves:
            parsed = self._deserialize_step(move)
            if parsed is None:
                continue
            payload.append(
                [
                    [parsed[0][0], parsed[0][1]],
                    [parsed[1][0], parsed[1][1]],
                ]
            )
        return f"<answer_json>{json.dumps(payload)}</answer_json>"

    def is_goal_reached(self, session: OnetStepSession) -> bool:
        return _is_board_cleared(session.board)

    @staticmethod
    def _read_key(gen_kwargs: Dict[str, Any], key: str) -> Any:
        protocol_cfg = gen_kwargs.get("step_protocol")
        if isinstance(protocol_cfg, dict) and key in protocol_cfg:
            return protocol_cfg[key]
        return gen_kwargs.get(key)

    @classmethod
    def _is_aux_step_by_step_mode(cls, gen_kwargs: Dict[str, Any]) -> bool:
        mode = str(cls._read_key(gen_kwargs, "onet_prompt_mode") or "").strip().lower()
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

    @classmethod
    def _parse_single_step(cls, text: str) -> Optional[OnetMove]:
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

        move = cls._try_parse_move(payload_text)
        if move is not None:
            return move

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
            if isinstance(parsed, list) and parsed:
                first = cls._try_parse_move(parsed[0])
                if first is not None:
                    return first

        simple = re.search(
            r"\[\s*\[\s*(-?\d+)\s*,\s*(-?\d+)\s*\]\s*,\s*"
            r"\[\s*(-?\d+)\s*,\s*(-?\d+)\s*\]\s*\]",
            text,
        )
        if not simple:
            return None
        return (
            (int(simple.group(1)), int(simple.group(2))),
            (int(simple.group(3)), int(simple.group(4))),
        )

    @classmethod
    def _try_parse_move(cls, payload: Any) -> Optional[OnetMove]:
        parsed = payload
        if isinstance(payload, str):
            try:
                parsed = json.loads(payload)
            except (TypeError, json.JSONDecodeError):
                return None
        if not isinstance(parsed, list) or len(parsed) != 2:
            return None
        p1 = cls._parse_coord(parsed[0])
        p2 = cls._parse_coord(parsed[1])
        if p1 is None or p2 is None:
            return None
        return p1, p2

    @staticmethod
    def _parse_coord(raw: Any) -> Optional[Coord]:
        if not isinstance(raw, (list, tuple)) or len(raw) != 2:
            return None
        try:
            return int(raw[0]), int(raw[1])
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _serialize_step(step: OnetMove) -> str:
        payload = [[step[0][0], step[0][1]], [step[1][0], step[1][1]]]
        return json.dumps(payload, separators=(",", ":"))

    @classmethod
    def _deserialize_step(cls, text: str) -> Optional[OnetMove]:
        try:
            payload = json.loads(text)
        except (TypeError, json.JSONDecodeError):
            return None
        return cls._try_parse_move(payload)

    @staticmethod
    def _remaining_cells(board: List[List[str | None]]) -> int:
        return sum(1 for row in board for cell in row if cell is not None)
