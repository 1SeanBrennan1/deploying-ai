# judge.py
"""
Output-side judge that verifies every LLM response before it reaches the user.

The judge checks three things:
1. Groundedness: does the response contain facts not in the RAG context?
   (Only enforced for "knowledge" and "general" categories)
2. No-fallback: did the model answer from training data when RAG was empty?
   (Only enforced for "knowledge" and "general" categories)
3. Safety: does the response mention any forbidden topics?
   (Always enforced)

IMPORTANT: Weather, time, and math queries are ALWAYS considered grounded
because facts come from dedicated tools (API, clock, calculator), not from
training data. These categories bypass the LLM judge for groundedness checks.

If any check fails, the response is replaced with a safe fallback.
"""

from langchain.chat_models import init_chat_model
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import os

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(_THIS_DIR, "..", ".secrets"))


class JudgeVerdict(BaseModel):
    """Structured output from the output judge."""
    grounded: bool = Field(
        description="True if all factual claims in the response are supported by the context."
    )
    no_fallback: bool = Field(
        description="True if the model did NOT answer from its own knowledge when context was empty."
    )
    safe: bool = Field(
        description="True if the response does not mention forbidden topics (cats, dogs, horoscopes, Taylor Swift)."
    )
    reasoning: str = Field(
        description="Brief explanation of the verdict."
    )


# Use the centralized config - ONLY place to change providers/endpoints
from assignment_chat.llm_config import get_chat_model

judge_llm = get_chat_model(temperature=0.0)
structured_judge = judge_llm.with_structured_output(JudgeVerdict)


# Categories that use dedicated tools and are ALWAYS grounded
TOOL_BASED_CATEGORIES = {"weather", "time", "math"}


JUDGE_SYSTEM_PROMPT = """
YOU ARE A STRICT OUTPUT VALIDATOR. FOLLOW THESE RULES EXACTLY.

CRITICAL: The query_category field is the SOURCE OF TRUTH for how to judge.
DO NOT second-guess it. DO NOT treat all categories the same.

========================================
STEP 1: CHECK THE QUERY CATEGORY
========================================

If the query_category is "weather", "time", or "math":
    → These queries get their facts from dedicated tools (weather API, 
      clock function, calculator). These tools are ALWAYS correct by 
      definition. The RAG context is IRRELEVANT for these categories.
    → You MUST return: grounded=True, no_fallback=True
    → Then skip to Rule 3 (safety check).

If the query_category is "knowledge" or "general":
    → These queries rely on RAG context. Continue to Step 2.

========================================
STEP 2: KNOWLEDGE / GENERAL RULES ONLY
========================================

Rule 2a: GROUNDEDNESS
If context is empty or says "No relevant information found" or 
"I searched through my knowledge base but couldn't find anything":
    - Assistant MUST say it couldn't find anything.
    - Any factual claims AT ALL = FAIL (grounded=False).
    - This is a FAIL even if the facts are true.

If context has real content that IS NOT empty or a "not found" message:
    - Assistant may rephrase, use humour, be creative, add personality.
    - Core facts come from context = PASS (grounded=True).
    - Creative wording, jokes, wordplay are NOT violations.

Rule 2b: NO FALLBACK TO TRAINING DATA
This ONLY applies when context is EMPTY or a "not found" message:
    - If context is empty AND assistant provides facts = FAIL (no_fallback=False).
    - Saying "I don't have info but here's what I know" = FAIL.
    - If context has real content, assistant can be creative = PASS.

========================================
STEP 3: SAFETY (ALL CATEGORIES)
========================================

Check the response for forbidden topics.
If ANY of these appear (case-insensitive, whole words only):
    - cats, cat, kitten, kittens, kitty, feline, felines
    - dogs, dog, puppy, puppies, pup, canine, canines
    - horoscope, horoscopes, zodiac, astrology, astrological
    - any zodiac sign name: aries, taurus, gemini, cancer, leo, virgo,
      libra, scorpio, sagittarius, capricorn, aquarius, pisces
    - taylor swift, taylor, tay tay, swift, swiftie

If ANY found → safe=False. Otherwise → safe=True.

========================================
OUTPUT FORMAT
========================================

Return a JSON object with exactly these fields:
- grounded: true/false
- no_fallback: true/false  
- safe: true/false
- reasoning: ONE brief sentence explaining your verdict.

IMPORTANT: If query_category is weather/time/math, your reasoning MUST
start with "Tool-based category" and explain that these are always grounded.
"""


def validate_output(
    response_text: str,
    rag_context: str,
    query_category: str = "knowledge"
) -> JudgeVerdict:
    """
    Validates an LLM response against the RAG context, safety rules,
    and the query category.

    Tool-based categories (weather, time, math) bypass the judge
    for groundedness — they are always considered grounded.
    Only the safety check is performed for these categories.
    """
    # Fast path: tool-based categories are always grounded
    if query_category in TOOL_BASED_CATEGORIES:
        # Quick safety check without calling the LLM
        safe = _check_safety_fast(response_text)
        return JudgeVerdict(
            grounded=True,
            no_fallback=True,
            safe=safe,
            reasoning=(
                f"Tool-based category ({query_category}): "
                "facts come from dedicated tools, always grounded. "
                f"Safety check: {'PASS' if safe else 'FAIL'}."
            )
        )

    # Full LLM judge for knowledge and general queries
    prompt = f"""
{JUDGE_SYSTEM_PROMPT}

---
QUERY CATEGORY: {query_category}
(This means you MUST apply the KNOWLEDGE/GENERAL rules from Step 2)

---
CONTEXT PROVIDED TO THE ASSISTANT:
{rag_context if rag_context else "(No context — this is an empty context)"}

---
ASSISTANT'S RESPONSE TO VALIDATE:
{response_text}
"""
    try:
        verdict: JudgeVerdict = structured_judge.invoke(prompt)
        return verdict
    except Exception as e:
        return JudgeVerdict(
            grounded=False,
            no_fallback=False,
            safe=True,
            reasoning=f"Judge evaluation failed with error: {str(e)}. Defaulting to fail for safety."
        )


def _check_safety_fast(text: str) -> bool:
    """
    Fast safety check without calling an LLM.
    Returns True if no forbidden topics are found.
    """
    import re

    FORBIDDEN_PATTERNS = [
        # Cats
        r"\bcat\b", r"\bcats\b", r"\bkitten\b", r"\bkittens\b",
        r"\bkitty\b", r"\bkitties\b", r"\bfeline\b", r"\bfelines\b",
        # Dogs
        r"\bdog\b", r"\bdogs\b", r"\bpuppy\b", r"\bpuppies\b",
        r"\bpup\b", r"\bpups\b", r"\bcanine\b", r"\bcanines\b",
        # Horoscopes
        r"\bhoroscope\b", r"\bhoroscopes\b", r"\bzodiac\b",
        r"\bastrology\b", r"\bastrological\b",
        r"\baries\b", r"\btaurus\b", r"\bgemini\b", r"\bcancer\b",
        r"\bleo\b", r"\bvirgo\b", r"\blibra\b", r"\bscorpio\b",
        r"\bsagittarius\b", r"\bcapricorn\b", r"\baquarius\b", r"\bpisces\b",
        # Taylor Swift
        r"\btaylor swift\b", r"\btaylor\b", r"\btay tay\b",
        r"\bswiftie\b", r"\bswifties\b",
    ]

    lower_text = text.lower()
    for pattern in FORBIDDEN_PATTERNS:
        if re.search(pattern, lower_text):
            return False
    return True


def response_passes(verdict: JudgeVerdict) -> bool:
    """Returns True only if all three checks pass."""
    return verdict.grounded and verdict.no_fallback and verdict.safe


SAFE_FALLBACK = (
    "I looked through my library but I'm not confident I can give you "
    "a reliable answer on that. I know about machine learning, the history "
    "of coffee, climate change, Python programming tips, and World War II. "
    "Try asking me about one of those topics, or rephrase your question!"
)


def extract_rag_context(state: dict) -> str:
    """
    Pulls RAG results from the conversation state.
    Returns the combined text of all RAG tool results, or empty string if none.
    """
    from langchain_core.messages import ToolMessage

    context_parts = []
    for msg in state.get("messages", []):
        if isinstance(msg, ToolMessage) and msg.name == "search_knowledge_base":
            context_parts.append(msg.content)
    return "\n".join(context_parts) if context_parts else ""