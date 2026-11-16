"""Bridge to the dashboard's tested tool layer (../dashboard in this repo).

We reuse its HealthTools instead of copying them, so the local model calls exactly the same
tools (and the same de-duplication logic) as the Claude-powered Coach, plus one tool of ours:
search_papers (RAG over the research corpus).

* Runtime(demo=True)  -> synthetic data generated in memory. Used to generate training data,
                         so no real health data is ever sent to the teacher model.
* Runtime(demo=False) -> the real Supabase data through the dashboard's read-only login
                         (DATABASE_URL in dashboard/.env). Used at chat time, on this Mac only.
"""

from __future__ import annotations

import os
import sys
from datetime import date, datetime
from pathlib import Path
from typing import Any

# GOTCHA: parents[3] is the repo root (bridge.py -> healthllm -> src -> health-llm -> root); moving this
# file or the health-llm folder breaks the import below unless DASHBOARD_DIR is set.
DASHBOARD = Path(os.environ.get("DASHBOARD_DIR", Path(__file__).resolve().parents[3]))
BACKEND = DASHBOARD / "dashboard" / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from app import agent_tools  # noqa: E402
from app.api import build_source  # noqa: E402
from app.catalog import load_catalog  # noqa: E402
from app.config import load_settings  # noqa: E402

from . import rag  # noqa: E402

ToolError = agent_tools.ToolError
RULES_PATH = agent_tools.RULES_PATH
DEMO_BIRTH_MONTH = "1990-06"  # a made-up age for demo conversations

# Every tool the model can call: the dashboard's ten, plus search_papers.
TOOL_SPECS: list[dict[str, Any]] = [*agent_tools.TOOL_SPECS, rag.SEARCH_SPEC]
TOOL_NAMES = [s["name"] for s in TOOL_SPECS]


class Runtime:
    """The data source and type catalog, built once and shared by many conversations."""

    def __init__(self, demo: bool, demo_now: datetime | None = None):
        if demo:
            # An explicit env dict means load_settings never reads dashboard/.env,
            # so demo runs can't even see the real DATABASE_URL.
            env = {"DEMO": "1", "TZ_NAME": "America/Los_Angeles"}
            if demo_now:
                env["DEMO_NOW"] = demo_now.isoformat()
            self.settings = load_settings(env)
            os.environ["BIRTH_MONTH"] = DEMO_BIRTH_MONTH  # HealthTools reads age from the environment
        else:
            self.settings = load_settings()
        self.demo = demo
        self.source = build_source(self.settings)
        self.catalog = load_catalog(self.settings.types_path)

    def tools(self, notes_path: Path | None = None) -> agent_tools.HealthTools:
        """A HealthTools for one conversation. Give each conversation its own notes file."""
        kwargs = {"notes_path": notes_path} if notes_path else {}
        return agent_tools.HealthTools(self.settings, self.source, self.catalog, **kwargs)

    def today(self) -> date:
        return self.tools().today()


def call_tool(tools: agent_tools.HealthTools, name: str, args: Any) -> dict[str, Any]:
    """Run any tool by name with model-supplied arguments.

    Raises ToolError with a message meant for the model, so it can correct itself.
    """
    if name == "search_papers":
        if not isinstance(args, dict) or not isinstance(args.get("query"), str):
            raise ToolError("search_papers needs a string `query`.")
        unknown = set(args) - {"query", "marker"}
        if unknown:
            raise ToolError(f"Unexpected arguments {sorted(unknown)} for search_papers.")
        marker = args.get("marker")
        if marker is not None and marker not in rag.SEARCH_SPEC["input_schema"]["properties"]["marker"]["enum"]:
            raise ToolError("marker must be one of steps, exercise, strength, sleep, resting_hr, vo2max.")
        try:
            return rag.search_papers(args["query"], marker)
        except ValueError as e:
            raise ToolError(str(e)) from e
    return agent_tools.call_tool(tools, name, args)
