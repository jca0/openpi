"""
Policy wrappers that add dynamic prompting and calibration on top of a base policy.

These wrap the BasePolicy.infer() call to manage subtask progression and
query Gemini at ~1Hz for completion checks. Gemini calls run in a background
thread to avoid blocking the websocket server.
"""

from __future__ import annotations

import concurrent.futures
import logging

import numpy as np
from openpi_client import base_policy as _base_policy
from typing_extensions import override

from openpi.dynamic_prompting.progress_monitor import ProgressMonitor
from openpi.dynamic_prompting.prompt_calibration import CalibrationLog
from openpi.dynamic_prompting.prompt_calibration import generate_decomposition_variations
from openpi.dynamic_prompting.subtask_manager import SubtaskManager
from openpi.dynamic_prompting.subtask_manager import decompose_task

logger = logging.getLogger(__name__)

# Shared thread pool for background Gemini calls
_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)


class DynamicPromptingPolicy(_base_policy.BasePolicy):
    """Wraps a policy to dynamically swap the prompt as subtasks complete.

    On first infer(), decomposes the high-level instruction into subtasks.
    On each subsequent infer(), checks completion at ~1Hz in a background
    thread and advances the subtask when done.
    """

    def __init__(
        self,
        policy: _base_policy.BasePolicy,
        instruction: str,
        check_interval_sec: float = 1.0,
    ):
        self._policy = policy
        self._check_interval_sec = check_interval_sec
        self._pending_check: concurrent.futures.Future | None = None

        self._instruction = instruction
        logger.info("[Dynamic Prompting] Task: \"%s\"", instruction)
        plan = decompose_task(instruction)
        logger.info("[Dynamic Prompting] Subtasks: %s", plan.subtasks)
        self._manager = SubtaskManager(plan)
        self._monitor = ProgressMonitor(check_interval_sec=check_interval_sec)

    @override
    def infer(self, obs: dict) -> dict:

        # Collect result from background check if ready
        self._process_pending_check()

        # Override the prompt with the current subtask
        if not self._manager.is_done():
            obs = {**obs, "prompt": self._manager.current_instruction()}

        result = self._policy.infer(obs)

        # Submit background completion check if it's time
        if not self._manager.is_done() and self._pending_check is None:
            frame = _extract_frame(obs)
            if frame is not None:
                self._monitor.set_frame(frame)
                if self._monitor.should_check():
                    subtask = self._manager.current_instruction()
                    self._pending_check = _executor.submit(self._monitor.check_completion, subtask)

        return result

    def _process_pending_check(self):
        if self._pending_check is None or not self._pending_check.done():
            return
        try:
            check = self._pending_check.result()
            if check["completed"]:
                logger.info("[Dynamic Prompting] ADVANCED %s", self._manager.status())
                self._manager.advance()
                if self._manager.is_done():
                    logger.info("[Dynamic Prompting] ALL SUBTASKS COMPLETED")
        except Exception:
            logger.exception("[Dynamic Prompting] Gemini check failed")
        finally:
            self._pending_check = None

    @override
    def reset(self) -> None:
        self._policy.reset()
        if self._pending_check:
            self._pending_check.cancel()
            self._pending_check = None
        self._manager.reset()
        self._monitor.reset()
        logger.info("[Dynamic Prompting] Reset")


class CalibrationPolicy(_base_policy.BasePolicy):
    """Wraps a policy to run prompt calibration.

    Generates N decomposition variations, cycles through them across resets,
    and logs per-subtask success/failure. Gemini calls run in a background thread.
    """

    def __init__(
        self,
        policy: _base_policy.BasePolicy,
        instruction: str,
        n_variations: int = 5,
        check_interval_sec: float = 1.0,
        log_dir: str = "calibration_logs",
    ):
        self._policy = policy
        self._instruction = instruction
        self._check_interval_sec = check_interval_sec
        self._cal_log = CalibrationLog(log_dir=log_dir)
        self._pending_check: concurrent.futures.Future | None = None

        logger.info("[Calibration] Task: \"%s\" (%d variations)", instruction, n_variations)
        self._plans = generate_decomposition_variations(instruction, n=n_variations)
        for i, plan in enumerate(self._plans):
            logger.info("[Calibration]   [%d] %s", i, plan.subtasks)
        self._plan_index = 0
        self._start_plan(0)
        self._cal_log.start_episode(instruction, self._plans[0].subtasks)

    @override
    def infer(self, obs: dict) -> dict:
        # Collect result from background check if ready
        self._process_pending_check()

        # Override prompt with current subtask
        if not self._manager.is_done():
            obs = {**obs, "prompt": self._manager.current_instruction()}

        result = self._policy.infer(obs)

        # Submit background completion check if it's time
        if not self._manager.is_done() and self._pending_check is None:
            frame = _extract_frame(obs)
            if frame is not None:
                self._monitor.set_frame(frame)
                if self._monitor.should_check():
                    subtask = self._manager.current_instruction()
                    self._pending_check = _executor.submit(self._monitor.check_completion, subtask)

        return result

    def _process_pending_check(self):
        if self._pending_check is None or not self._pending_check.done():
            return
        try:
            check = self._pending_check.result()
            if check["completed"]:
                logger.info("[Calibration] ADVANCED %s", self._manager.status())
                self._cal_log.log_subtask_result(
                    subtask_prompt=self._manager.current_instruction(),
                    completed=True,
                    reason=check["reason"],
                )
                self._manager.advance()
                if self._manager.is_done():
                    logger.info("[Calibration] ALL SUBTASKS COMPLETED (decomposition %d/%d)", self._plan_index + 1, len(self._plans))
        except Exception:
            logger.exception("[Calibration] Gemini check failed")
        finally:
            self._pending_check = None

    @override
    def reset(self) -> None:
        self._policy.reset()

        # Wait for any pending check before logging failures
        if self._pending_check:
            self._pending_check.cancel()
            self._pending_check = None

        # Log any remaining incomplete subtasks as failed
        if self._manager:
            while not self._manager.is_done():
                self._cal_log.log_subtask_result(
                    subtask_prompt=self._manager.current_instruction(),
                    completed=False,
                    reason="episode ended before completion",
                )
                self._manager.advance()

        # Advance to next decomposition variation
        if self._plans:
            self._plan_index += 1
            if self._plan_index < len(self._plans):
                self._start_plan(self._plan_index)
                self._cal_log.start_episode(self._instruction, self._plans[self._plan_index].subtasks)
                logger.info("[Calibration] Starting decomposition %d/%d: %s", self._plan_index + 1, len(self._plans), self._plans[self._plan_index].subtasks)
            else:
                self._cal_log.finalize()
                logger.info("[Calibration] COMPLETE\n%s", self._cal_log.summary())
                self._manager = None
                self._monitor = None

    def _start_plan(self, index: int):
        self._manager = SubtaskManager(self._plans[index])
        self._monitor = ProgressMonitor(check_interval_sec=self._check_interval_sec)


def _extract_frame(obs: dict) -> np.ndarray | None:
    """Try to extract an RGB camera frame from the observation dict.

    Searches for common image key patterns including DROID-style
    'observation/exterior_image_1_left' keys.
    """
    # Try exact keys first, then search for image-like keys
    candidates = [
        "observation/exterior_image_1_left",
        "image",
        "images",
        "external_cam",
        "agentview_rgb",
    ]
    # Also match any key containing "image" or "cam"
    for key in obs:
        if ("image" in key.lower() or "cam" in key.lower()) and key not in candidates:
            candidates.append(key)

    for key in candidates:
        if key in obs:
            frame = obs[key]
            if isinstance(frame, np.ndarray):
                if frame.ndim == 4:
                    frame = frame[0]
                return frame
    return None
