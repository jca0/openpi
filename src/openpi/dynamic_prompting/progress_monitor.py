"""
Progress monitor that uses Gemini to check whether a robot subtask
has been completed, based on the current camera frame.
"""

import io
import json
import logging
import os
import time

import numpy as np
from dotenv import load_dotenv
from google import genai
from google.genai import types
from PIL import Image

load_dotenv()

logger = logging.getLogger(__name__)

MODEL_ID = "gemini-2.5-flash"

COMPLETION_PROMPT_TEMPLATE = """You are a robot task completion checker. You are given a single frame from a robot's camera.

The robot's current subtask is: "{subtask}"

Based on this frame, determine whether the subtask has been completed.

Respond with JSON only, no markdown:
{{"completed": true/false, "reason": "brief explanation"}}"""


class ProgressMonitor:
    """Calls Gemini at ~1Hz to check if the current subtask is done."""

    def __init__(
        self,
        check_interval_sec: float = 1.0,
        model_id: str = MODEL_ID,
    ):
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY not found in environment")

        self.client = genai.Client(api_key=api_key)
        self.model_id = model_id
        self.check_interval_sec = check_interval_sec
        self._last_check_time: float = 0.0
        self._latest_frame: np.ndarray | None = None

    def reset(self):
        self._last_check_time = 0.0
        self._latest_frame = None

    def set_frame(self, frame: np.ndarray):
        """Store the current camera frame (H, W, 3 uint8 RGB)."""
        self._latest_frame = frame

    def should_check(self) -> bool:
        """Returns True if enough time has passed since the last check."""
        if self._latest_frame is None:
            return False
        now = time.monotonic()
        return (now - self._last_check_time) >= self.check_interval_sec

    def check_completion(self, subtask: str) -> dict:
        """Query Gemini to determine if the subtask is complete."""
        self._last_check_time = time.monotonic()
        prompt = COMPLETION_PROMPT_TEMPLATE.format(subtask=subtask)

        img = Image.fromarray(self._latest_frame)
        img_bytes = _image_to_bytes(img)
        image_part = types.Part.from_bytes(data=img_bytes, mime_type="image/jpeg")

        response = self.client.models.generate_content(
            model=self.model_id,
            contents=[image_part, prompt],
            config=types.GenerateContentConfig(temperature=0.0),
        )

        result = _parse_response(response.text)
        status = "DONE" if result["completed"] else "NOT DONE"
        logger.info("[VLM] \"%s\" -> %s (%s)", subtask, status, result["reason"])
        return result


def _image_to_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return buf.getvalue()


def _parse_response(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1]
        text = text.rsplit("```", 1)[0]
    try:
        result = json.loads(text)
        return {
            "completed": bool(result.get("completed", False)),
            "reason": str(result.get("reason", "")),
        }
    except json.JSONDecodeError:
        return {"completed": False, "reason": f"Failed to parse VLM response: {text}"}
