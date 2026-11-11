"""Give each theme in the model bundle a short, readable name with DeepSeek.

Usage:
    uv run python scripts/name_themes.py

Reads data/processed/models/model_bundle.json and writes
data/processed/models/theme_names.json:
    {"theme.00": {"name": ..., "description": ..., "members": [...]}, ...}
Names are tied to the exact member questions, so the dashboard ignores them
if the themes change after retraining. Needs DEEPSEEK_API_KEY in .env.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))  # see README "Imports on this Mac"

from linkedin_analysis.discovery import load_prompt  # noqa: E402
from linkedin_analysis.llm import LLMClient, LLMSettings  # noqa: E402
from linkedin_analysis.predict import BUNDLE  # noqa: E402

OUT = BUNDLE.parent / "theme_names.json"


def main() -> None:
    bundle = json.loads(BUNDLE.read_text())
    themes = bundle.get("themes", {})
    if not themes:
        sys.exit("The model bundle has no themes. Train with --themes N first.")

    payload = {t: v["member_questions"] for t, v in themes.items()}
    client = LLMClient(LLMSettings.from_env("deepseek"))
    print(f"Naming {len(themes)} themes with {client.settings.model} ...")
    reply = client.chat_json(load_prompt("name_themes_system").template,
                             json.dumps(payload, indent=1, ensure_ascii=False), temperature=0.0)

    names = {}
    for t, v in themes.items():
        got = reply.get("themes", {}).get(t, {})
        names[t] = {"name": str(got.get("name", "")).strip() or t,
                    "description": str(got.get("description", "")).strip(),
                    "members": v["members"]}
    OUT.write_text(json.dumps(names, indent=2, ensure_ascii=False))
    for t, v in names.items():
        print(f"{t}: {v['name']} — {v['description']}")
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()
