"""
§15 AI Review Summary — an actual LangGraph pipeline (not just a single LLM
call): a theme-extraction node feeds a synthesis node. Splitting the two
steps keeps each prompt focused and makes the pipeline easy to extend later



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
        # so a Claude-sized 300-token budget was truncating responses.
        max_output_tokens=1024,
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
        return {**state, "summary": "Not enough review content yet to generate a summary."}

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
