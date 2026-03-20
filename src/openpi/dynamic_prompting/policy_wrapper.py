"""
Wraps a base policy with dynamic prompting: decomposes the high-level
task into subtasks and advances through them based on VLM feedback.

All Gemini API calls run in a background thread to avoid blocking the
websocket handler (which would cause keepalive ping timeouts).
"""

import logging
from concurrent.futures import Future, ThreadPoolExecutor

import numpy as np
from openpi_client import base_policy as _base_policy
from typing_extensions import override

from openpi.dynamic_prompting.progress_monitor import ProgressMonitor
from openpi.dynamic_prompting.subtask_manager import SubtaskManager, SubtaskPlan, decompose_task

logger = logging.getLogger(__name__)

# Key in the raw observation dict that contains the exterior camera image.
IMAGE_KEY = "observation/exterior_image_1_left"


class DynamicPromptingPolicy(_base_policy.BasePolicy):
    """
    Wraps a policy to replace the high-level prompt with the current
    subtask instruction, advancing subtasks based on VLM progress checks.

    The wrapper is stateful: it tracks the current subtask across calls
    to infer(). When the client sends a new high-level prompt (different
    from the previous one), the task is re-decomposed.

    Gemini calls (decomposition + completion checks) are offloaded to a
    background thread so they never block the websocket infer loop.
    """

    def __init__(
        self,
        policy: _base_policy.BasePolicy,
        check_every_n_steps: int = 15,
    ):
        self._policy = policy
        self._check_every_n_steps = check_every_n_steps

        self._monitor = ProgressMonitor(check_every_n_steps=check_every_n_steps)
        self._manager: SubtaskManager | None = None
        self._current_high_level_prompt: str | None = None

        self._executor = ThreadPoolExecutor(max_workers=1)
        self._decompose_future: Future[SubtaskPlan] | None = None
        self._check_future: Future[dict] | None = None

    @override
    def infer(self, obs: dict) -> dict:
        prompt = obs.get("prompt")
        if prompt is not None:
            if isinstance(prompt, bytes):
                prompt = prompt.decode("utf-8")
            elif isinstance(prompt, np.ndarray):
                prompt = prompt.item()

        # If the high-level prompt changed, kick off decomposition in background.
        if prompt is not None and prompt != self._current_high_level_prompt:
            self._current_high_level_prompt = prompt
            self._manager = None
            self._monitor.reset()
            logger.info("Dynamic prompting: decomposing task in background: %s", prompt)
            self._decompose_future = self._executor.submit(decompose_task, prompt)

        # Check if background decomposition has finished.
        if self._decompose_future is not None and self._decompose_future.done():
            try:
                plan = self._decompose_future.result()
                self._manager = SubtaskManager(plan)
                logger.info("Dynamic prompting: subtasks = %s", [s.instruction for s in plan.subtasks])
            except Exception:
                logger.exception("Dynamic prompting: decomposition failed")
            self._decompose_future = None

        # Feed camera frame to progress monitor.
        if IMAGE_KEY in obs:
            frame = np.asarray(obs[IMAGE_KEY])
            if np.issubdtype(frame.dtype, np.floating):
                frame = (255 * frame).astype(np.uint8)
            self._monitor.set_frame(frame)

        # Replace prompt with current subtask instruction.
        if self._manager is not None and not self._manager.is_done():
            obs["prompt"] = self._manager.current_instruction()
            self._manager.step()

        # Run the real policy.
        result = self._policy.infer(obs)

        # Check if a previous completion check has finished.
        if self._check_future is not None and self._check_future.done():
            try:
                check = self._check_future.result()
                logger.info("Dynamic prompting: %s | VLM check: %s",
                            self._manager.status() if self._manager else "?", check)
                if check["completed"] and self._manager is not None and not self._manager.is_done():
                    self._manager.advance()
                    if self._manager.is_done():
                        logger.info("Dynamic prompting: all subtasks completed")
                    else:
                        logger.info("Dynamic prompting: advancing to %s", self._manager.status())
            except Exception:
                logger.exception("Dynamic prompting: completion check failed")
            self._check_future = None

        # Fire off a new completion check in the background if it's time.
        if (
            self._manager is not None
            and not self._manager.is_done()
            and self._check_future is None
        ):
            if self._monitor.should_check():
                subtask_instruction = self._manager.current_instruction()
                self._check_future = self._executor.submit(
                    self._monitor.check_completion, subtask_instruction
                )
            elif self._manager.exceeded_subtask_limit():
                logger.info("Dynamic prompting: subtask step limit exceeded, advancing")
                self._manager.advance()

        return result

    @property
    def metadata(self) -> dict:
        base = self._policy.metadata if hasattr(self._policy, "metadata") else {}
        return {**base, "dynamic_prompting": True}
