THEME_EXTRACTION_PROMPT = """You are analyzing customer reviews for a local business in \
Bangladesh. The reviews arrive in a mix of three forms, sometimes even within the same \
review, and you must read and understand ALL of them correctly:

1. English.
2. Bangla — written in Bengali script (e.g. "খাবার অনেক ভালো ছিল").
3. Banglish — Bengali words spelled out phonetically using English/Latin letters \
(e.g. "khabar valo silo", "service ta kharap silo", "dam ektu beshi mone hoyeche"). \
Treat Banglish as ordinary Bengali, just written with Latin letters — sound it out and \
translate its meaning the same way you would Bangla script.

Silently understand and translate every review's meaning (regardless of its script or \
spelling) before extracting themes. Do not let inconsistent spelling, missing vowels, or \
transliteration variants ("bhalo"/"valo"/"bhal") stop you from grouping the same theme \
together.

Extract the recurring themes — both positive and negative — mentioned across these \
reviews. A theme only counts as "recurring" if it is raised by more than one review; \
themes raised by only a single review should be skipped unless there are very few \
reviews overall.

Reviews (rating, then text):
{reviews_block}

Respond with ONLY a JSON array of short theme strings, written in English regardless of \
the reviews' original language, no other text, e.g.:
["food quality praised", "slow service mentioned by several", "friendly staff"]

If the reviews contain no usable text at all, respond with an empty JSON array: []"""

SYNTHESIS_PROMPT = """Write a short, natural 1-2 sentence summary of a business's customer \
reviews for a review-platform profile page, based on these extracted themes:

{themes_block}

Style: neutral, factual, like "most customers are happy with X, but a few mentioned Y." \
Do not invent themes not listed above. Always write the summary in clear, plain English \
— even though the original reviews may have been in Bangla or Banglish — so it reads \
consistently for every visitor of the profile page. Output ONLY the summary sentence(s), \
no preamble."""