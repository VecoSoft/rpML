"""
Business promotion captions ("Write for me").

One Gemini call per language, then strict validation of every caption against the rules the
Java side also enforces: ≤ 220 characters, ≤ 2 emojis, ≤ 3 hashtags, no superlative/false claims
("best in Dhaka", "No.1", ...) and no number that isn't present in the business data we were given
(so no invented prices, discounts or dates). Captions that fail are dropped; if fewer than 3
survive, the endpoint returns an error and the backend falls back to its own template captions.
"""
import json
import re
from typing import Any, Dict, List, Optional

from langchain_google_genai import ChatGoogleGenerativeAI

from app.config import settings

MAX_LENGTH = 220
MAX_EMOJIS = 2
MAX_HASHTAGS = 3
FORBIDDEN_CLAIMS = [
    "best in", "no.1", "no 1", "number one", "#1", "the best", "guaranteed", "cheapest",
    "সেরা", "এক নম্বর", "১ নম্বর", "গ্যারান্টি", "সবচেয়ে সস্তা",
]
_BN_DIGITS = str.maketrans("০১২৩৪৫৬৭৮৯", "0123456789")
_DIGITS = re.compile(r"[0-9]+")
_EMOJI = re.compile("[\U0001F000-\U0001FAFF☀-➿]")

TONES = {
    "friendly": "warm, casual and welcoming",
    "premium": "polished, calm and understated",
    "urgent": "energetic, with a gentle sense of a limited time — but never pushy or fake-scarce",
}

PROMPT = """You write short social captions for a local business in Bangladesh, for the Jachai app.

Write exactly 3 different captions in {language_name}.
Tone: {tone}.
Post type: {post_type}.

Business facts (the ONLY facts you may use):
{facts}

Hard rules:
- Each caption at most 200 characters.
- At most 2 emojis per caption, at most 3 hashtags per caption.
- Never claim to be the best, No.1, cheapest, guaranteed, or anything you can't prove from the facts.
- Never invent a price, discount, percentage, date, time, phone number or any other number.
  Only use numbers that appear in the facts above, exactly as given.
- Do not mention competitors. Do not use ALL CAPS sentences.
- {language_rule}

Return ONLY a JSON array of 3 strings, no commentary, no markdown fences."""


def _facts_block(req: Dict[str, Any]) -> str:
    lines = [f"- Business name: {req['businessName']}"]
    if req.get("category"):
        lines.append(f"- Category: {req['category']}")
    if req.get("area"):
        lines.append(f"- Area: {req['area']}")
    offer = req.get("offer") or {}
    if offer:
        lines.append(f"- Offer: {offer.get('title')}")
        for key, label in (("offerType", "Offer type"), ("discountValue", "Discount value"),
                           ("originalPrice", "Original price (BDT)"), ("offerPrice", "Offer price (BDT)"),
                           ("validUntil", "Valid until")):
            if offer.get(key) is not None:
                lines.append(f"  - {label}: {offer[key]}")
    item = req.get("menuItem") or {}
    if item:
        lines.append(f"- Menu item: {item.get('name')}")
        if item.get("price") is not None:
            lines.append(f"  - Price (BDT): {item['price']}")
    event = req.get("event") or {}
    if event:
        if event.get("title"):
            lines.append(f"- Event: {event['title']}")
        if event.get("start"):
            lines.append(f"  - Starts: {event['start']}")
    return "\n".join(lines)


def allowed_numbers(req: Dict[str, Any]) -> set:
    text = json.dumps(req, ensure_ascii=False, default=str).translate(_BN_DIGITS)
    nums = set(_DIGITS.findall(text))
    # Decimal prices like "450.00" also allow the whole-number form.
    nums |= {n.lstrip("0") or "0" for n in nums}
    nums |= {"1", "2"}  # "Buy 1 Get 1", "2 for 1"
    return nums


def is_valid(caption: str, allowed: set) -> bool:
    if not caption or len(caption) > MAX_LENGTH:
        return False
    if len(_EMOJI.findall(caption)) > MAX_EMOJIS or caption.count("#") > MAX_HASHTAGS:
        return False
    lower = caption.lower()
    if any(claim in lower for claim in FORBIDDEN_CLAIMS):
        return False
    return all(n in allowed for n in _DIGITS.findall(caption.translate(_BN_DIGITS)))


def _strip_fence(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`").removeprefix("json").strip()
    return text


def generate_captions(req: Dict[str, Any], llm: Optional[ChatGoogleGenerativeAI] = None) -> List[str]:
    language = req.get("language", "en")
    prompt = PROMPT.format(
        language_name="Bangla (Bengali script)" if language == "bn" else "English",
        tone=TONES.get(req.get("tone") or "friendly", TONES["friendly"]),
        post_type=req.get("type", "GENERAL"),
        facts=_facts_block(req),
        language_rule=("Write natural, everyday Bangla as spoken in Dhaka; brand and dish names may stay in English."
                       if language == "bn" else "Write simple, clear English."),
    )
    # Interactive call: the backend gives up after 8s and falls back to templates, so fail fast
    # (no retry/backoff on 429s) instead of holding a worker for a minute.
    model = llm or ChatGoogleGenerativeAI(model=settings.gemini_model, google_api_key=settings.gemini_api_key,
                                          max_output_tokens=2048, temperature=0.8, max_retries=0, timeout=7)
    response = model.invoke(prompt)
    content = response.content if isinstance(response.content, str) else str(response.content)
    try:
        parsed = json.loads(_strip_fence(content))
    except (json.JSONDecodeError, TypeError):
        return []
    if not isinstance(parsed, list):
        return []
    allowed = allowed_numbers(req)
    out: List[str] = []
    for c in parsed:
        if isinstance(c, str):
            c = " ".join(c.split())
            if is_valid(c, allowed) and c not in out:
                out.append(c)
    return out[:3]
