from __future__ import annotations

import json
import logging
import re
from typing import Any

import httpx

from .config import Settings


logger = logging.getLogger("orisight.openrouter")


class OpenRouterClient:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def generate_structured_assessment(self, payload: dict[str, Any]) -> dict[str, Any] | None:
        if not self.settings.openrouter_api_key:
            logger.warning("OPENROUTER_API_KEY missing; skipping LLM call")
            return None

        prompt = self._build_prompt(payload)
        body = {
            "model": self.settings.openrouter_model,
            "temperature": 0.2,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are ORISIGHT, a hackathon clinical decision-support assistant. "
                        "Do not claim certainty. Return only valid JSON with no markdown."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            "response_format": {"type": "json_object"},
        }

        headers = {
            "Authorization": f"Bearer {self.settings.openrouter_api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "http://localhost:8000",
            "X-Title": "ORISIGHT-MVP",
        }

        try:
            with httpx.Client(timeout=40.0) as client:
                response = client.post(
                    f"{self.settings.openrouter_base_url}/chat/completions",
                    headers=headers,
                    json=body,
                )
                response.raise_for_status()
            content = response.json()["choices"][0]["message"]["content"]
            return _extract_json(content)
        except Exception as exc:
            logger.exception("OpenRouter request failed: %s", exc)
            return None

    def _build_prompt(self, payload: dict[str, Any]) -> str:
        return (
            "Analyze this oral lesion case for a hackathon MVP only. "
            "This is not a real diagnosis tool.\n\n"
            f"Patient symptoms:\n{payload['symptoms']}\n\n"
            f"Patient history:\n{payload['history']}\n\n"
            f"Image caption:\n{payload['image_caption']}\n\n"
            f"Extracted risk factors:\n{', '.join(payload['risk_factors']) or 'None'}\n\n"
            f"Similar lesion cases:\n{json.dumps(payload['similar_cases'], indent=2)}\n\n"
            f"Retrieved medical guidance:\n{json.dumps(payload['retrieved_knowledge'], indent=2)}\n\n"
            "Return strict JSON with keys:\n"
            "diagnosis (string), differential_diagnosis (array of strings), risk_level (string), "
            "suggested_tests (array of strings), treatment_plan (array of strings), referral (string), "
            "confidence_score (string)."
        )


def _extract_json(raw_text: str) -> dict[str, Any]:
    raw_text = raw_text.strip()
    try:
        return json.loads(raw_text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", raw_text, flags=re.DOTALL)
        if not match:
            raise
        return json.loads(match.group(0))
