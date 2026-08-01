"""
CONTENT_PATTERN signal (spec §14): flags generic/templated language, or
wording suspiciously similar to other reviews on the same business.
Uses multilingual embeddings since reviews are frequently Bangla/Banglish,
not an English-only assumption — explicit spec requirement.
"""
import json
from functools import lru_cache
from typing import List

import google.generativeai as genai
import numpy as np
from sentence_transformers import SentenceTransformer

from app.config import settings
from app.schemas import ReviewInput

_GENERICNESS_PROMPT = """You are checking a customer review for a local business \
directory for signs it is a generic/templated/bot-written review (as opposed to a \
genuine, specific customer experience). The review may be in English, Bangla, or \
Banglish (romanized Bangla).

Review text:
\"\"\"{content}\"\"\"

Respond with ONLY a JSON object, no other text:
{{"is_generic": true|false, "confidence_0_100": <int>, "reason": "<one short phrase>"}}"""


@lru_cache(maxsize=1)
def _embedding_model() -> SentenceTransformer:
    # Loaded once per process; downloads on first use if not already cached locally.
    return SentenceTransformer(settings.embedding_model)


@lru_cache(maxsize=1)
def _gemini_client() -> "genai.GenerativeModel":
    genai.configure(api_key=settings.gemini_api_key)
    return genai.GenerativeModel(settings.gemini_model)


def _strip_json_fence(text: str) -> str:
    # Gemini sometimes wraps JSON responses in ```json ... ``` despite instructions not to.
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`").removeprefix("json").strip()
    return text


def max_similarity_to_recent(review: ReviewInput, recent_reviews: List[ReviewInput]) -> tuple[float, str]:
    """Returns (max_similarity 0..1, id_of_most_similar_review)."""
    others = [r for r in recent_reviews if r.id != review.id and r.content.strip()]
    if not others or not review.content.strip():
        return 0.0, ""

    model = _embedding_model()
    target_vec = model.encode(review.content, normalize_embeddings=True)
    other_vecs = model.encode([r.content for r in others], normalize_embeddings=True)

    sims = other_vecs @ target_vec  # already normalized -> dot product == cosine similarity
    best_idx = int(np.argmax(sims))
    return float(sims[best_idx]), others[best_idx].id


def llm_genericness_check(content: str) -> tuple[int, str]:
    """
    Separate from the embedding comparison: an LLM judgment on whether THIS
    review reads as generic/templated on its own, with no comparison needed —
    catches templated language even the very first time it's ever used.
    Fails safe (returns 0, no-op) on any API error so ML-service hiccups never
    block review submission upstream.
    """
    if not content.strip() or not settings.gemini_api_key:
        return 0, "skipped (empty content or no API key configured)"
    try:
        resp = _gemini_client().generate_content(
            _GENERICNESS_PROMPT.format(content=content[:2000]),
            # Generous headroom: current gemini-flash-latest spends part of
            # this budget on internal reasoning before the visible JSON
            # output, so a Claude-sized 150-token budget was truncating responses.
            generation_config={"max_output_tokens": 512},
        )
        parsed = json.loads(_strip_json_fence(resp.text))
        if parsed.get("is_generic"):
            return int(parsed.get("confidence_0_100", 50)), parsed.get("reason", "flagged as generic by LLM")
        return 0, parsed.get("reason", "not flagged as generic by LLM")
    except Exception as exc:  # noqa: BLE001 - deliberate fail-safe boundary
        return 0, f"LLM genericness check unavailable ({exc.__class__.__name__})"


def score_content_pattern(review: ReviewInput, recent_reviews: List[ReviewInput]) -> tuple[int, str]:
    """
    Combines (a) embedding-similarity against recent reviews on the same
    business — catches cross-review templating/copy-paste even across
    languages — with (b) a standalone LLM genericness judgment on the review
    text itself. Returns a 0-100 score and a human-readable detail string for
    the moderation queue.
    """
    similarity, matched_id = max_similarity_to_recent(review, recent_reviews)
    similarity_score = max(0.0, (similarity - 0.75) / 0.25) * 100  # 0.75 sim -> 0, 1.0 sim -> 100
    similarity_score = min(100.0, similarity_score)

    llm_score, llm_detail = llm_genericness_check(review.content)

    score = int(round(max(similarity_score, llm_score * 0.9)))

    if matched_id and similarity_score >= llm_score:
        detail = f"{similarity:.2f} cosine similarity to review {matched_id}"
    else:
        detail = llm_detail

    return score, detail
