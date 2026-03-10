import json
import os
import re

from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request

import google.generativeai as genai


load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    raise RuntimeError("Missing GEMINI_API_KEY in environment/.env")

genai.configure(api_key=API_KEY)

app = Flask(__name__)


SYSTEM_INSTRUCTION = (
    "You are a harsh but fair Silicon Valley Investor. "
    "You MUST evaluate the user's startup idea and provide EXACTLY THREE constructive critique points. "
    "No more, no less. Do not add introductions or conclusions. "
    "Return ONLY valid JSON in the following shape: "
    '{ "critique": ["point 1", "point 2", "point 3"] }.'
)


def _coerce_three_points(text: str) -> list[str]:
    """
    Enforce exactly 3 critique points, even if the model response is imperfect.
    """
    if not text:
        return [
            "The idea is underspecified—define the core user, pain, and why this must exist now.",
            "The differentiation is unclear—what is your unfair advantage versus existing options?",
            "The go-to-market is missing—how will you acquire customers predictably and profitably?",
        ]

    # Prefer JSON.
    try:
        data = json.loads(text)
        items = data.get("critique", [])
        if isinstance(items, list):
            items = [str(x).strip() for x in items if str(x).strip()]
            if len(items) >= 3:
                return items[:3]
    except Exception:
        pass

    # Fallback: split bullets/numbered lines.
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    bullets = []
    for ln in lines:
        m = re.match(r"^(\d+[\).\s-]+|[-*•]\s+)\s*(.+)$", ln)
        if m:
            bullets.append(m.group(2).strip())
    if len(bullets) >= 3:
        return bullets[:3]

    # Last resort: split into sentences.
    rough = re.split(r"(?<=[.!?])\s+", " ".join(lines)).strip()
    sentences = [s.strip() for s in rough if isinstance(rough, str) for s in re.split(r"(?<=[.!?])\s+", rough) if s.strip()]
    sentences = sentences[:3] if len(sentences) >= 3 else (sentences + [""] * (3 - len(sentences)))
    sentences = [s if s else "Missing critique detail—clarify the biggest risk and how you'll validate it quickly." for s in sentences]
    return sentences[:3]


@app.get("/")
def index():
    return render_template("index.html")


@app.post("/api/validate")
def validate():
    payload = request.get_json(silent=True) or {}
    idea = (payload.get("idea") or "").strip()

    if not idea:
        return jsonify({"error": "Missing 'idea'."}), 400

    model = genai.GenerativeModel(
        model_name="gemini-2.5-flash",
        system_instruction=SYSTEM_INSTRUCTION,
    )

    try:
        result = model.generate_content(
            idea,
            generation_config=genai.types.GenerationConfig(
                temperature=0.4,
                response_mime_type="application/json",
            ),
        )
        raw_text = (getattr(result, "text", None) or "").strip()
        critique = _coerce_three_points(raw_text)
        return jsonify({"critique": critique})
    except Exception as e:
        return jsonify({"error": "Gemini request failed.", "details": str(e)}), 502


if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5000)
