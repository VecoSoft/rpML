THEME_EXTRACTION_PROMPT = """You are analyzing customer reviews for a local business \
(Bangladesh, reviews may be in English, Bangla, or Banglish). Extract the recurring \
themes — both positive and negative — mentioned across these reviews.

Reviews (rating, then text):
{reviews_block}

Respond with ONLY a JSON array of short theme strings, no other text, e.g.:
["food quality praised", "slow service mentioned by several", "friendly staff"]"""

SYNTHESIS_PROMPT = """Write a short, natural 1-2 sentence summary of a business's customer \
reviews for a review-platform profile page, based on these extracted themes:

{themes_block}

Style: neutral, factual, like "most customers are happy with X, but a few mentioned Y." \
Do not invent themes not listed above. Output ONLY the summary sentence(s), no preamble."""
