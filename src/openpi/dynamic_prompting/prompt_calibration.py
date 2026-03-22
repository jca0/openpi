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
    # Append a short hash to avoid collisions
    h = hashlib.md5(instruction.encode()).hexdigest()[:8]
    safe = "".join(c for c in short if c.isalnum() or c == "_")
    return f"{safe}_{h}"


class CalibrationLog:
    """Tracks which individual subtask prompts succeeded or failed."""

    def __init__(self, log_dir: str = "calibration_logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

    def _log_path(self, instruction: str) -> Path:
        return self.log_dir / f"{_instruction_slug(instruction)}.json"

    def load(self, instruction: str) -> list[dict]:
        path = self._log_path(instruction)
        if path.exists():
            return json.loads(path.read_text())
        return []

    def log_subtask_result(
        self,
        instruction: str,
        subtask_prompt: str,
        completed: bool,
        reason: str,
    ):
        entries = self.load(instruction)
        entries.append({
            "timestamp": datetime.now().isoformat(),
            "subtask_prompt": subtask_prompt,
            "completed": completed,
            "reason": reason,
        })
        self._log_path(instruction).write_text(json.dumps(entries, indent=2))

    def get_successful_prompts(self, instruction: str) -> list[dict]:
        return [e for e in self.load(instruction) if e["completed"]]

    def get_failed_prompts(self, instruction: str) -> list[dict]:
        return [e for e in self.load(instruction) if not e["completed"]]

    def summary(self, instruction: str) -> str:
        entries = self.load(instruction)
        if not entries:
            return f'"{instruction}": no calibration data'
        successes = sum(1 for e in entries if e["completed"])
        total = len(entries)
        lines = [f'"{instruction}": {successes}/{total} subtask prompts successful']
        for e in entries:
            status = "OK" if e["completed"] else "FAIL"
            lines.append(f'  [{status}] "{e["subtask_prompt"]}"')
        return "\n".join(lines)
