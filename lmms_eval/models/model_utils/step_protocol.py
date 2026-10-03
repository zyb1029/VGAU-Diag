from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Protocol

from loguru import logger as eval_logger


@dataclass
class StepTransition:
    """Single transition result for one step."""

    legal: bool
    reason: str
    move: Optional[str]
    trace: Dict[str, Any]


@dataclass
class StepProtocolResult:
    """Final result returned by step protocol runner."""

    answer: str
    moves: List[str]
    trace_rows: List[Dict[str, Any]]
    illegal: bool
    illegal_reason: str
    max_steps: int
    goal_checks: List[Dict[str, Any]] = field(default_factory=list)


class StepProtocolAdapter(Protocol):
    """Task adapter interface for the generic step protocol runner."""

    def empty_answer(self) -> str:
        """Return the task-specific empty answer string."""

    def initialize_session(
        self, doc: Dict[str, Any], visuals: List[Any], gen_kwargs: Dict[str, Any]
    ) -> Any:
        """Build mutable session state from one sample."""

    def resolve_max_steps(self, doc: Dict[str, Any], gen_kwargs: Dict[str, Any]) -> int:
        """Resolve maximal number of iterative planning steps."""

    def build_initial_user_content(
        self,
        *,
        context: str,
        session: Any,
        build_processed_visuals: Callable[[Optional[List[Any]]], List[Dict[str, Any]]],
    ) -> List[Dict[str, Any]]:
        """Build first user message content for step protocol."""

    def build_step_prompt(self, step_idx: int) -> str:
        """Build one-step planning prompt."""

    def extract_step_move(self, raw_step_text: str) -> Optional[str]:
        """Extract normalized move token from model step output."""

    def apply_step_move(self, session: Any, move: Optional[str]) -> StepTransition:
        """Apply one move to session state and return transition summary."""

    def build_step_visual(self, session: Any, transition: StepTransition) -> Any:
        """Build visual input for current step from updated session state."""

    def build_fallback_answer(self, moves: List[str]) -> str:
        """Build fallback final answer from accepted moves."""

    def is_goal_reached(self, session: Any) -> bool:
        """Return whether current session state already reaches goal."""


_DEFAULT_GOAL_CHECK_PROMPT = (
    'Are you currently on the destination cell? '
    'Answer strictly with "yes" or "no".'
)
_DEFAULT_NEGATED_GOAL_CHECK_PROMPT = (
    'Are you currently not on the destination cell? '
    'Answer strictly with "yes" or "no".'
)
_GOAL_CHECK_TOKEN_RE = re.compile(r"\b(yes|no)\b", re.IGNORECASE)


def _default_parse_goal_check(text: str) -> Optional[bool]:
    """Take the last yes/no token in the response. Return None if unparseable."""
    if not text:
        return None
    matches = _GOAL_CHECK_TOKEN_RE.findall(text)
    if not matches:
        return None
    return matches[-1].lower() == "yes"


_DEFAULT_STEPS_LEFT_PROMPT = (
    "How many more steps are needed to reach the destination cell from the current cell? "
    "Output exactly one non-negative integer wrapped in <answer>...</answer>, "
    "e.g. <answer>1</answer>."
)
_STEPS_LEFT_ANSWER_RE = re.compile(
    r"<answer>\s*(\d+)\s*</answer>", re.IGNORECASE | re.DOTALL
)


def _default_parse_steps_left(text: str) -> Optional[int]:
    """Parse the last <answer>N</answer> tag; return None if not found."""
    if not text:
        return None
    matches = _STEPS_LEFT_ANSWER_RE.findall(text)
    if not matches:
        return None
    return int(matches[-1])


def run_step_protocol_old(
    *,
    adapter: StepProtocolAdapter,
    context: str,
    doc: Dict[str, Any],
    visuals: List[Any],
    gen_kwargs: Dict[str, Any],
    system_prompt: str,
    generate_text_with_messages: Callable[..., str],
    build_processed_visuals: Callable[[Optional[List[Any]]], List[Dict[str, Any]]],
    final_answer_parser: Callable[[str], str],
    on_step_visual: Optional[Callable[[int, Any], None]] = None,
    generate_step_image: Optional[Callable[..., Any]] = None,
    step1_max_new_tokens: int = 8192,
    step2_max_new_tokens: int = 8192,
) -> StepProtocolResult:
    """Run task-agnostic iterative step protocol with adapter hooks."""
    del final_answer_parser, step2_max_new_tokens

    try:
        session = adapter.initialize_session(doc, visuals, gen_kwargs)
    except Exception:
        return StepProtocolResult(
            answer=adapter.empty_answer(),
            moves=[],
            trace_rows=[],
            illegal=True,
            illegal_reason="init_failed",
            max_steps=0,
        )

    max_steps = adapter.resolve_max_steps(doc, gen_kwargs)
    initial_user_content = adapter.build_initial_user_content(
        context=context.strip(),
        session=session,
        build_processed_visuals=build_processed_visuals,
    )
    history_messages: List[Dict[str, Any]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": initial_user_content},
    ]
    moves: List[str] = []
    trace_rows: List[Dict[str, Any]] = []
    illegal = False
    illegal_reason = ""

    for step_idx in range(1, max_steps + 1):
        history_messages.append(
            {
                "role": "user",
                "content": [{"type": "text", "text": adapter.build_step_prompt(step_idx)}],
            }
        )
        raw_step = generate_text_with_messages(
            messages=history_messages,
            gen_kwargs=gen_kwargs,
            max_new_tokens=step1_max_new_tokens,
        )
        history_messages.append({"role": "assistant", "content": raw_step})

        move = adapter.extract_step_move(raw_step)
        transition = adapter.apply_step_move(session, move)
        if generate_step_image is not None:
            step_visual = generate_step_image(
                step_idx=step_idx,
                adapter=adapter,
                session=session,
                transition=transition,
                history_messages=history_messages,
                gen_kwargs=gen_kwargs,
            )
        else:
            step_visual = adapter.build_step_visual(session, transition)
        if step_visual is None:
            return StepProtocolResult(
                answer=adapter.empty_answer(),
                moves=moves,
                trace_rows=trace_rows,
                illegal=True,
                illegal_reason="render_failed",
                max_steps=max_steps,
            )

        if on_step_visual is not None:
            on_step_visual(step_idx, step_visual)

        history_messages.append(
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": adapter.build_image_generation_prompt(step_idx),
                    }
                ],
            }
        )
        history_messages.append(
            {
                "role": "user",
                "content": build_processed_visuals([step_visual]),
            }
        )

        trace_rows.append(
            {
                "step": step_idx,
                "raw_output": raw_step,
                "move": transition.move,
                "legal": transition.legal,
                **transition.trace,
            }
        )

        if transition.legal and transition.move is not None:
            moves.append(transition.move)
            continue

        illegal = True
        illegal_reason = transition.reason
        break

    if illegal:
        return StepProtocolResult(
            answer=adapter.empty_answer(),
            moves=moves,
            trace_rows=trace_rows,
            illegal=True,
            illegal_reason=illegal_reason,
            max_steps=max_steps,
        )

    if not adapter.is_goal_reached(session):
        return StepProtocolResult(
            answer=adapter.empty_answer(),
            moves=moves,
            trace_rows=trace_rows,
            illegal=True,
            illegal_reason="goal_not_reached",
            max_steps=max_steps,
        )

    answer = adapter.build_fallback_answer(moves)

    return StepProtocolResult(
        answer=answer,
        moves=moves,
        trace_rows=trace_rows,
        illegal=False,
        illegal_reason="",
        max_steps=max_steps,
    )


def run_step_protocol(
    *,
    adapter: StepProtocolAdapter,
    context: str,
    doc: Dict[str, Any],
    visuals: List[Any],
    gen_kwargs: Dict[str, Any],
    system_prompt: str,
    generate_text_with_messages: Callable[..., str],
    build_processed_visuals: Callable[[Optional[List[Any]]], List[Dict[str, Any]]],
    final_answer_parser: Callable[[str], str],
    on_step_visual: Optional[Callable[[int, Any], None]] = None,
    generate_step_image: Optional[Callable[..., Any]] = None,
    step1_max_new_tokens: int = 8192,
    step2_max_new_tokens: int = 8192,
    goal_check_max_new_tokens: int = 32,
    safety_cap_factor: int = 2,
) -> StepProtocolResult:
    """Run step protocol where termination is gated by a single side-channel
    probe asking how many more steps remain to the destination.

    Each step issues exactly one probe that does NOT mutate ``history_messages``:
    the prompt requests a single non-negative integer with strict format
    (no words, punctuation, units, or explanation). The probe shares the
    current ``history_messages`` snapshot and runs as a separate
    ``generate_text_with_messages`` call. ``claims_stop`` is True iff the
    parsed integer equals 0.

    Decision rules use ``optimal_steps = adapter.resolve_max_steps(doc, gen_kwargs)``:

    - claims_stop & ``executed_steps == optimal_steps`` & goal reached -> success
    - claims_stop otherwise -> ``claim_too_early``
    - not claims_stop & ``executed_steps >= optimal_steps`` -> ``claim_too_late``
    - probe unparseable (not a bare non-negative integer) -> ``invalid_goal_check``
    - exceeded ``safety_cap_factor * optimal_steps`` -> ``safety_cap_exceeded``

    ``goal_check_max_new_tokens`` still bounds the probe generation length.
    """
    del final_answer_parser, step2_max_new_tokens

    try:
        session = adapter.initialize_session(doc, visuals, gen_kwargs)
    except Exception:
        return StepProtocolResult(
            answer=adapter.empty_answer(),
            moves=[],
            trace_rows=[],
            illegal=True,
            illegal_reason="init_failed",
            max_steps=0,
        )

    optimal_steps = adapter.resolve_max_steps(doc, gen_kwargs)
    safety_cap = max(safety_cap_factor * optimal_steps, optimal_steps + 1)

    initial_user_content = adapter.build_initial_user_content(
        context=context.strip(),
        session=session,
        build_processed_visuals=build_processed_visuals,
    )
    history_messages: List[Dict[str, Any]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": initial_user_content},
    ]
    moves: List[str] = []
    trace_rows: List[Dict[str, Any]] = []
    goal_checks: List[Dict[str, Any]] = []

    build_steps_left_prompt = getattr(adapter, "build_steps_left_prompt", None)
    parse_steps_left = getattr(adapter, "parse_steps_left", None)

    def _fail(reason: str) -> StepProtocolResult:
        return StepProtocolResult(
            answer=adapter.empty_answer(),
            moves=moves,
            trace_rows=trace_rows,
            illegal=True,
            illegal_reason=reason,
            max_steps=optimal_steps,
            goal_checks=goal_checks,
        )

    for step_idx in range(1, safety_cap + 1):
        history_messages.append(
            {
                "role": "user",
                "content": [{"type": "text", "text": adapter.build_step_prompt(step_idx)}],
            }
        )
        raw_step = generate_text_with_messages(
            messages=history_messages,
            gen_kwargs=gen_kwargs,
            max_new_tokens=step1_max_new_tokens,
        )
        history_messages.append({"role": "assistant", "content": raw_step})

        move = adapter.extract_step_move(raw_step)
        transition = adapter.apply_step_move(session, move)
        if generate_step_image is not None:
            step_visual = generate_step_image(
                step_idx=step_idx,
                adapter=adapter,
                session=session,
                transition=transition,
                history_messages=history_messages,
                gen_kwargs=gen_kwargs,
            )
        else:
            step_visual = adapter.build_step_visual(session, transition)
        if step_visual is None:
            return StepProtocolResult(
                answer=adapter.empty_answer(),
                moves=moves,
                trace_rows=trace_rows,
                illegal=True,
                illegal_reason="render_failed",
                max_steps=optimal_steps,
                goal_checks=goal_checks,
            )

        if on_step_visual is not None:
            on_step_visual(step_idx, step_visual)

        history_messages.append(
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": adapter.build_image_generation_prompt(step_idx),
                    }
                ],
            }
        )
        history_messages.append(
            {
                "role": "user",
                "content": build_processed_visuals([step_visual]),
            }
        )

        trace_rows.append(
            {
                "step": step_idx,
                "raw_output": raw_step,
                "move": transition.move,
                "legal": transition.legal,
                **transition.trace,
            }
        )

        if not (transition.legal and transition.move is not None):
            return _fail(transition.reason)

        moves.append(transition.move)
        executed_steps = len(moves)

        # Side-channel single-probe steps-left check: do NOT mutate history_messages.
        prompt_text = (
            build_steps_left_prompt(step_idx)
            if callable(build_steps_left_prompt)
            else _DEFAULT_STEPS_LEFT_PROMPT
        )
        probe_messages = history_messages + [
            {
                "role": "user",
                "content": [{"type": "text", "text": prompt_text}],
            }
        ]
        raw_steps_left = generate_text_with_messages(
            messages=probe_messages,
            gen_kwargs=gen_kwargs,
            max_new_tokens=goal_check_max_new_tokens,
        )
        parser = (
            parse_steps_left
            if callable(parse_steps_left)
            else _default_parse_steps_left
        )
        parsed_steps_left = parser(raw_steps_left)

        eval_logger.info(
            f"[step_protocol] step={step_idx} "
            f"prompt_text={prompt_text!r} "
            f"raw_steps_left={raw_steps_left!r} "
            f"parsed_steps_left={parsed_steps_left!r}"
        )

        claims_stop = parsed_steps_left == 0

        goal_checks.append(
            {
                "step": step_idx,
                "raw_steps_left": raw_steps_left,
                "parsed_steps_left": parsed_steps_left,
                "claims_stop": claims_stop,
                "executed_steps": executed_steps,
                "optimal_steps": optimal_steps,
            }
        )

        if parsed_steps_left is None:
            return _fail("invalid_goal_check")

        if claims_stop:
            if executed_steps == optimal_steps and adapter.is_goal_reached(session):
                answer = adapter.build_fallback_answer(moves)
                return StepProtocolResult(
                    answer=answer,
                    moves=moves,
                    trace_rows=trace_rows,
                    illegal=False,
                    illegal_reason="",
                    max_steps=optimal_steps,
                    goal_checks=goal_checks,
                )
            return _fail("claim_too_early")

        if executed_steps >= optimal_steps:
            return _fail("claim_too_late")

    return _fail("safety_cap_exceeded")
