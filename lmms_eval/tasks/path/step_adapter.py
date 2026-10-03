from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from PIL import Image

from lmms_eval.models.model_utils.step_protocol import StepTransition
from lmms_eval.tasks.path.utils import (
    _build_aux_overlay,
    _parse_answer_path,
    _parse_coord_pair,
    _parse_labeled_points,
    path_extract_walkable,
    path_render_state_image,
)

_MOVE_DELTAS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}


@dataclass
class PathClosestStepSession:
    """Mutable state for one path step protocol run."""

    doc: Dict[str, Any]
    base_image: Image.Image
    current: Tuple[int, int]
    goal: Tuple[int, int]
    walkable: set[Tuple[int, int]]
    grid_size: int
    optimal_steps: int
    coord_to_labels: Dict[Tuple[int, int], List[str]]
    gold_labels_order: Optional[List[str]]
    hit_labels_seen: set[str] = field(default_factory=set)
    hit_labels_order: List[str] = field(default_factory=list)
    declared_hit_labels_order: List[str] = field(default_factory=list)
    use_aux_step_grid: bool = False
    initial_visual: Optional[Image.Image] = None
    executed_steps: int = 0


class PathStepAdapter:
    """Path-closest adapter for generic step protocol."""

    def __init__(self) -> None:
        self._pending_declared_hit_labels: List[str] = []
        self._current_session: Optional[PathClosestStepSession] = None

    def empty_answer(self) -> str:
        return "<answer></answer>"

    def initialize_session(
        self, doc: Dict[str, Any], visuals: List[Any], gen_kwargs: Dict[str, Any]
    ) -> PathClosestStepSession:
        base_image = self._ensure_image_object(visuals[0] if visuals else None)
        if base_image is None:
            raise ValueError("path step protocol requires a base image")

        answer_path = _parse_answer_path(doc)
        if len(answer_path) < 2:
            raise ValueError("path sample missing valid answer_path")
        start = _parse_coord_pair(doc.get("start")) or answer_path[0]
        goal = _parse_coord_pair(doc.get("goal")) or answer_path[-1]
        optimal_steps = len(answer_path) - 1
        if optimal_steps <= 0:
            raise ValueError("path sample has invalid answer_path transitions")
        walkable, grid_size = path_extract_walkable(
            doc,
            base_image=base_image,
        )
        if start not in walkable or goal not in walkable:
            raise ValueError("path start/goal not walkable")

        label_to_coord = _parse_labeled_points(doc)
        coord_to_labels: Dict[Tuple[int, int], List[str]] = {}
        for label, coord in label_to_coord.items():
            coord_to_labels.setdefault(coord, []).append(label)
        for labels in coord_to_labels.values():
            labels.sort()
        gold_labels_order = self._parse_gold_labels_order(doc)

        use_aux = self._is_aux_step_by_step_mode(gen_kwargs)
        initial_image = path_render_state_image(
            doc,
            current_pos=start,
            base_image=base_image,
        )
        if initial_image is None:
            raise ValueError("failed to render initial path state")
        initial_visual = _build_aux_overlay(initial_image) if use_aux else initial_image

        session = PathClosestStepSession(
            doc=doc,
            base_image=base_image,
            current=start,
            goal=goal,
            walkable=walkable,
            grid_size=grid_size,
            optimal_steps=optimal_steps,
            coord_to_labels=coord_to_labels,
            gold_labels_order=gold_labels_order,
            use_aux_step_grid=use_aux,
            initial_visual=initial_visual,
            executed_steps=0,
        )
        self._pending_declared_hit_labels = []
        self._current_session = session
        return session

    def resolve_max_steps(self, doc: Dict[str, Any], gen_kwargs: Dict[str, Any]) -> int:
        del gen_kwargs
        answer_path = _parse_answer_path(doc)
        if len(answer_path) >= 3:
            return len(answer_path) - 2
        return 64

    def build_initial_user_content(
        self,
        *,
        context: str,
        session: PathClosestStepSession,
        build_processed_visuals: Callable[[Optional[List[Any]]], List[Dict[str, Any]]],
    ) -> List[Dict[str, Any]]:
        if session.initial_visual is None:
            raise ValueError("path session missing initial visual")
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
            f"Now planning for step {step_idx}, Please output a sentence in the form:\n"
            '"<STEP_JSON>{\\"move\\":\\"up|down|left|right\\",\\"hit_labels\\":[\\"A\\"]}</STEP_JSON>"\n'
            "Use [] when this move passes no new label.\n"
        )

    def extract_step_move(self, raw_step_text: str) -> Optional[str]:
        self._pending_declared_hit_labels = []
        if not raw_step_text:
            return None

        payload_text = raw_step_text.strip()
        tagged_matches = list(
            re.finditer(
                r"<STEP_JSON>\s*(\{.*?\})\s*</STEP_JSON>",
                payload_text,
                re.IGNORECASE | re.DOTALL,
            )
        )
        if tagged_matches:
            payload_text = tagged_matches[-1].group(1)

        try:
            payload = json.loads(payload_text)
        except (TypeError, json.JSONDecodeError):
            payload = None

        if isinstance(payload, dict):
            move = str(payload.get("move", "")).strip().lower()
            self._pending_declared_hit_labels = self._normalize_hit_labels(
                payload.get("hit_labels")
            )
            return move if move in _MOVE_DELTAS else None

        lowered = raw_step_text.strip().lower()
        move_match = re.search(r"\b(up|down|left|right)\b", lowered)
        move = move_match.group(1) if move_match else None
        labels_match = re.search(r"hit_labels\s*[:=]\s*\[(.*?)\]", raw_step_text, re.I | re.S)
        if labels_match:
            self._pending_declared_hit_labels = self._normalize_hit_labels(
                labels_match.group(1)
            )
        return move

    def apply_step_move(
        self, session: PathClosestStepSession, move: Optional[str]
    ) -> StepTransition:
        before = [session.current[0], session.current[1]]
        declared_hit_labels = list(self._pending_declared_hit_labels)
        self._pending_declared_hit_labels = []

        trace_base = {
            "predicted_move": move,
            "position_before": before,
            "position_after": before,
            "declared_hit_labels": declared_hit_labels,
            "actual_hit_labels": [],
            "hit_labels_so_far": list(session.hit_labels_order),
            "declared_labels_cumulative_raw": list(session.declared_hit_labels_order),
            "declared_labels_cumulative_normalized": self._collapse_consecutive_duplicates(
                session.declared_hit_labels_order
            ),
            "actual_labels_cumulative": list(session.hit_labels_order),
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
        candidate = (session.current[0] + dr, session.current[1] + dc)
        if not self._in_bounds(candidate, session.grid_size):
            return StepTransition(
                legal=False,
                reason="hit_wall_or_oob",
                move=move,
                trace=trace_base,
            )
        if candidate not in session.walkable:
            return StepTransition(
                legal=False,
                reason="hit_wall_or_oob",
                move=move,
                trace=trace_base,
            )

        coord_labels = session.coord_to_labels.get(candidate, [])
        actual_hit_labels = [
            label for label in coord_labels if label not in session.hit_labels_seen
        ]
        trace_base["actual_hit_labels"] = list(actual_hit_labels)
        actual_cumulative = list(session.hit_labels_order) + list(actual_hit_labels)
        declared_cumulative_raw = list(session.declared_hit_labels_order) + list(
            declared_hit_labels
        )
        declared_cumulative_normalized = self._collapse_consecutive_duplicates(
            declared_cumulative_raw
        )
        trace_base["actual_labels_cumulative"] = actual_cumulative
        trace_base["declared_labels_cumulative_raw"] = declared_cumulative_raw
        trace_base["declared_labels_cumulative_normalized"] = (
            declared_cumulative_normalized
        )
        prefix_target = (
            session.gold_labels_order
            if session.gold_labels_order is not None
            else actual_cumulative
        )
        if not self._is_prefix_sequence(declared_cumulative_normalized, prefix_target):
            return StepTransition(
                legal=False,
                reason="hit_labels_mismatch",
                move=move,
                trace=trace_base,
            )

        session.current = candidate
        session.executed_steps += 1
        session.declared_hit_labels_order = declared_cumulative_raw
        for label in actual_hit_labels:
            session.hit_labels_seen.add(label)
            session.hit_labels_order.append(label)
        remaining_optimal_steps = self._min_steps_to_goal(
            start=candidate,
            goal=session.goal,
            walkable=session.walkable,
        )
        trace_base["position_after"] = [candidate[0], candidate[1]]
        trace_base["hit_labels_so_far"] = list(session.hit_labels_order)
        trace_base["executed_steps"] = session.executed_steps
        trace_base["remaining_optimal_steps"] = remaining_optimal_steps
        if remaining_optimal_steps is None:
            trace_base["bound_violated"] = True
            return StepTransition(
                legal=False,
                reason="goal_unreachable",
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
        self, session: PathClosestStepSession, transition: StepTransition
    ) -> Image.Image | None:
        current_pos = (
            session.current
            if transition.legal
            else (
                transition.trace["position_before"][0],
                transition.trace["position_before"][1],
            )
        )
        rendered = path_render_state_image(
            session.doc,
            current_pos=current_pos,
            base_image=session.base_image,
        )
        if rendered is None:
            return None
        if session.use_aux_step_grid:
            return _build_aux_overlay(rendered)
        return rendered

    def build_image_generation_prompt(self, step_idx: int) -> str:
        return f"Now, generate the image for step {step_idx}."

    def build_fallback_answer(self, moves: List[str]) -> str:
        del moves
        if self._current_session is None:
            return self.empty_answer()
        labels_text = ",".join(self._current_session.hit_labels_order)
        return f"<answer>{labels_text}</answer>"

    def is_goal_reached(self, session: PathClosestStepSession) -> bool:
        del session
        # For path step mode we score by ordered on-path labels,
        # so reaching the geometric goal is not required.
        return True

    @staticmethod
    def _read_key(gen_kwargs: Dict[str, Any], key: str) -> Any:
        protocol_cfg = gen_kwargs.get("step_protocol")
        if isinstance(protocol_cfg, dict) and key in protocol_cfg:
            return protocol_cfg[key]
        return gen_kwargs.get(key)

    @classmethod
    def _is_aux_step_by_step_mode(cls, gen_kwargs: Dict[str, Any]) -> bool:
        mode = str(
            cls._read_key(gen_kwargs, "path_prompt_mode") or ""
        ).strip().lower()
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
    def _normalize_hit_labels(raw_labels: Any) -> List[str]:
        if raw_labels is None:
            return []
        if isinstance(raw_labels, str):
            candidates = re.findall(r"[A-E]", raw_labels.upper())
        elif isinstance(raw_labels, list):
            candidates = [str(item).strip().upper() for item in raw_labels]
        else:
            return []
        normalized: List[str] = []
        for label in candidates:
            if label not in {"A", "B", "C", "D", "E"}:
                continue
            normalized.append(label)
        return normalized

    @staticmethod
    def _collapse_consecutive_duplicates(labels: List[str]) -> List[str]:
        collapsed: List[str] = []
        for label in labels:
            if not collapsed or collapsed[-1] != label:
                collapsed.append(label)
        return collapsed

    @staticmethod
    def _parse_gold_labels_order(doc: Dict[str, Any]) -> Optional[List[str]]:
        raw_labels = doc.get("on_path_labels")
        if isinstance(raw_labels, list):
            parsed: List[str] = []
            for label in raw_labels:
                normalized = str(label).strip().upper()
                if normalized not in {"A", "B", "C", "D", "E"}:
                    return None
                parsed.append(normalized)
            return parsed

        answer_text = str(doc.get("answer", "")).upper()
        parsed_from_answer = re.findall(r"[A-E]", answer_text)
        if parsed_from_answer:
            return parsed_from_answer
        return None

    @staticmethod
    def _is_prefix_sequence(prefix: List[str], target: List[str]) -> bool:
        if len(prefix) > len(target):
            return False
        return prefix == target[: len(prefix)]

    @staticmethod
    def _in_bounds(cell: Tuple[int, int], grid_size: int) -> bool:
        return 0 <= cell[0] < grid_size and 0 <= cell[1] < grid_size

    @staticmethod
    def _min_steps_to_goal(
        start: Tuple[int, int],
        goal: Tuple[int, int],
        walkable: set[Tuple[int, int]],
    ) -> Optional[int]:
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
