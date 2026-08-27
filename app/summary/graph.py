"""
§15 AI Review Summary — an actual LangGraph pipeline (not just a single LLM
call): a theme-extraction node feeds a synthesis node. Splitting the two
steps keeps each prompt focused and makes the pipeline easy to extend later
(e.g. adding a "flag needs-human-review" branch) without touching either
existing node.

"""
import json
from typing import List, TypedDict

from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, StateGraph

from app.config import settings
from app.schemas import SummaryReviewInput
from app.summary.prompts import SYNTHESIS_PROMPT, THEME_EXTRACTION_PROMPT


class SummaryState(TypedDict):
    reviews: List[SummaryReviewInput]
    themes: List[str]
    summary: str


def _llm() -> ChatGoogleGenerativeAI:
    return ChatGoogleGenerativeAI(
        model=settings.gemini_model,
        google_api_key=settings.gemini_api_key,
        # Generous headroom: current gemini-flash-latest spends part of this
        # budget on internal reasoning before the visible JSON/text output,
        # and that reasoning cost grows with input size — 1024 was already
        # observed truncating the theme-extraction JSON (finish_reason
        # MAX_TOKENS) on a ~20-review batch, well under the 200-review cap
        # this pipeline is asked to handle.
        max_output_tokens=4096,
    )


def _strip_json_fence(text: str) -> str:
    # Gemini sometimes wraps JSON responses in ```json ... ``` despite instructions not to.
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`").removeprefix("json").strip()
    return text


def _extract_themes(state: SummaryState) -> SummaryState:
    reviews_block = "\n".join(f"- ({r.rating}/5) {r.content}" for r in state["reviews"] if r.content.strip())
    if not reviews_block:
        return {**state, "themes": []}

    response = _llm().invoke(THEME_EXTRACTION_PROMPT.format(reviews_block=reviews_block))
    try:
        themes = json.loads(_strip_json_fence(response.content))
        if not isinstance(themes, list):
            themes = []
    except (json.JSONDecodeError, TypeError):
        themes = []
    return {**state, "themes": themes}


def _synthesize_summary(state: SummaryState) -> SummaryState:
    if not state["themes"]:
        # Empty, not a placeholder sentence — the backend must treat this as
        # "nothing to save" and leave the cached row untouched, rather than
        # persisting human-readable filler text as if it were a real summary.
        return {**state, "summary": ""}

    themes_block = "\n".join(f"- {t}" for t in state["themes"])
    response = _llm().invoke(SYNTHESIS_PROMPT.format(themes_block=themes_block))
    return {**state, "summary": response.content.strip()}


def _build_graph():
    graph = StateGraph(SummaryState)
    graph.add_node("extract_themes", _extract_themes)
    graph.add_node("synthesize_summary", _synthesize_summary)
    graph.set_entry_point("extract_themes")
    graph.add_edge("extract_themes", "synthesize_summary")
    graph.add_edge("synthesize_summary", END)
    return graph.compile()


_compiled_graph = None


def _graph():
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = _build_graph()
    return _compiled_graph


def generate_summary(reviews: List[SummaryReviewInput]) -> str:
    result: SummaryState = _graph().invoke({"reviews": reviews, "themes": [], "summary": ""})
    return result["summary"]