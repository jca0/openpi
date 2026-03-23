"""
Prompt calibration: generate multiple subtask decomposition variations and
track which individual subtask prompts succeed or fail.
"""

import hashlib
import json
import logging
import os
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import types

from openpi.dynamic_prompting.subtask_manager import SubtaskPlan

load_dotenv()

logger = logging.getLogger(__name__)

MODEL_ID = "gemini-2.5-flash"

DECOMPOSITION_VARIATIONS_PROMPT = """You are a robot task planner. A robot arm needs to perform this task:

"{instruction}"

Generate {n} different ways to break this task into ordered subtask sequences. Each decomposition should use different phrasing, level of detail, or action boundaries.

For example, one decomposition might say "pick up the cube" while another says "grasp the red block" or "close gripper around the cube and lift".

Each decomposition should be 2-4 subtasks. All decompositions should accomplish the same overall task.

Respond with JSON only, no markdown:
{{"decompositions": [["subtask 1a", "subtask 2a"], ["subtask 1b", "subtask 2b", "subtask 3b"], ...]}}"""


def generate_decomposition_variations(
    instruction: str,
    n: int = 5,
    model_id: str = MODEL_ID,
) -> list[SubtaskPlan]:
    """Generate N different subtask decompositions of the same high-level instruction."""
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY not found in environment")

    client = genai.Client(api_key=api_key)
    prompt = DECOMPOSITION_VARIATIONS_PROMPT.format(instruction=instruction, n=n)

    response = client.models.generate_content(
        model=model_id,
        contents=prompt,
        config=types.GenerateContentConfig(temperature=1.0),
    )

    text = response.text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1]
        text = text.rsplit("```", 1)[0]

    result = json.loads(text)
    return [
        SubtaskPlan(high_level_instruction=instruction, subtasks=subtasks)
        for subtasks in result["decompositions"]
    ]


def _instruction_slug(instruction: str) -> str:
    """Create a filesystem-safe slug from an instruction string."""
    short = instruction[:50].strip().replace(" ", "_")
    h = hashlib.md5(instruction.encode()).hexdigest()[:8]
    safe = "".join(c for c in short if c.isalnum() or c == "_")
    return f"{safe}_{h}"


class CalibrationLog:
    """Accumulates all episodes from a single calibration run into one structured log.

    Call `start_episode` at the beginning of each decomposition variation,
    `log_subtask_result` for each subtask, and `finalize` at the end of the
    full run to write the log and the success/failure prompt files.
    """

    def __init__(self, log_dir: str = "calibration_logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._instruction: str | None = None
        self._episodes: list[dict] = []
        self._current_episode: dict | None = None

    def start_episode(self, instruction: str, subtasks: list[str]):
        """Begin a new episode (decomposition variation)."""
        self._instruction = instruction
        self._current_episode = {
            "subtasks": subtasks,
            "results": [],
        }
        self._episodes.append(self._current_episode)

    def log_subtask_result(self, subtask_prompt: str, completed: bool, reason: str):
        """Record the outcome of a single subtask within the current episode."""
        self._current_episode["results"].append({
            "subtask_prompt": subtask_prompt,
            "completed": completed,
            "reason": reason,
        })

    def finalize(self):
        """Write the full run log plus separate success/failure prompt files."""
        if not self._instruction or not self._episodes:
            return

        slug = _instruction_slug(self._instruction)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = self.log_dir / f"{slug}_{timestamp}"
        run_dir.mkdir(parents=True, exist_ok=True)

        # Full structured log
        run_data = {
            "timestamp": datetime.now().isoformat(),
            "instruction": self._instruction,
            "episodes": self._episodes,
        }
        (run_dir / "run.json").write_text(json.dumps(run_data, indent=2))

        # Collect unique successful and failed subtask prompts
        successful = []
        failed = []
        for episode in self._episodes:
            for result in episode["results"]:
                if result["completed"]:
                    successful.append(result["subtask_prompt"])
                else:
                    failed.append(result["subtask_prompt"])

        (run_dir / "successful_prompts.json").write_text(json.dumps(successful, indent=2))
        (run_dir / "failed_prompts.json").write_text(json.dumps(failed, indent=2))

        logger.info("[Calibration] Logs written to %s", run_dir)

    def summary(self) -> str:
        if not self._instruction or not self._episodes:
            return "No calibration data"
        all_results = [r for ep in self._episodes for r in ep["results"]]
        successes = sum(1 for r in all_results if r["completed"])
        total = len(all_results)
        lines = [f'"{self._instruction}": {successes}/{total} subtask prompts successful']
        for i, ep in enumerate(self._episodes):
            lines.append(f"  Episode {i}: {ep['subtasks']}")
            for r in ep["results"]:
                status = "OK" if r["completed"] else "FAIL"
                lines.append(f'    [{status}] "{r["subtask_prompt"]}" — {r["reason"]}')
        return "\n".join(lines)
