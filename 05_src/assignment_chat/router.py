# router.py
"""
Intent router that classifies user queries by tool category.

The router considers the conversation context when classifying,
so follow-up messages like "sorry I meant PEI" after a weather query
are correctly routed to weather instead of general.
"""

from typing import Literal, Optional
from langchain.chat_models import init_chat_model
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import os

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(_THIS_DIR, "..", ".secrets"))


class RouterDecision(BaseModel):
    """Structured output from the router."""
    category: Literal["knowledge", "weather", "time", "math", "general"] = Field(
        description="The tool category that best matches the user's query."
    )
    reasoning: str = Field(
        description="Brief explanation of why this category was chosen."
    )


router_llm = init_chat_model(
    "openai:gpt-4o-mini",
    base_url="https://k7uffyg03f.execute-api.us-east-1.amazonaws.com/prod/openai/v1",
    api_key="any value",
    default_headers={"x-api-key": os.getenv("API_GATEWAY_KEY")},
    temperature=0.0,
    max_tokens=50,
)

structured_router = router_llm.with_structured_output(RouterDecision)


ROUTER_SYSTEM_PROMPT = """
You are a query classifier for an AI assistant. Your job is to read the user's
message AND the recent conversation context, then decide which category of tool
it needs.

The assistant has these capabilities:

- knowledge: Our internal knowledge base. Use for questions about facts, history,
  science, technology, definitions, concepts, how-to guides, and general information.

- weather: For questions about current weather, temperature, forecasts, or climate
  conditions in a specific city or location. ALSO use this for follow-up messages
  that correct or clarify a previous weather query (e.g., "I meant Paris" after
  asking about weather).

- time: For questions asking for the current time, date, or day.

- math: For calculations, arithmetic, or math problems.

- general: For everything else – conversation, jokes, opinions, personal advice,
  or when no other category fits. ALSO use this for follow-ups that clarify or
  correct a previous general or knowledge query.

CRITICAL RULE: 
CONTEXT MATTERS. Always check the recent conversation context to see if the user is following up on a previous query.
If the user recently asked about weather, time, or math, and the new message is a 
short correction or clarification, classify it in the same category, not general. 
For Example, if the conversation context shows the user recently asked about weather, and the
new message is a short correction or clarification (like "sorry I meant PEI" or
"what about Paris instead?"), classify it as WEATHER, not general. The same
applies for time and math corrections.

Rules:
- If the message mentions a city plus weather/temperature/forecast → weather
- If the message is a short correction to a previous query about a specific category
  (weather/time/math), use that same category
- If the user asks for time/date/day → time
- If the user asks for calculations or math → math
- If the user asks a factual question that could be in an encyclopedia → knowledge
- If unsure, default to knowledge (safest option)

Output your decision as a JSON object with 'category' and 'reasoning' fields.
"""


def classify_query(user_message: str, conversation_context: str = "") -> str:
    """
    Classifies a user message into a tool category, considering conversation context.

    Args:
        user_message: The raw user input
        conversation_context: Summary of recent conversation (previous query + response)

    Returns:
        One of: "knowledge", "weather", "time", "math", "general"
    """
    try:
        prompt = ROUTER_SYSTEM_PROMPT
        if conversation_context:
            prompt += f"\n\nRECENT CONVERSATION CONTEXT:\n{conversation_context}"
        prompt += f"\n\nNEW USER MESSAGE:\n{user_message}"

        decision: RouterDecision = structured_router.invoke(prompt)
        return decision.category
    except Exception:
        return "knowledge"


def get_conversation_context(state: dict) -> str:
    """
    Extracts a brief summary of the recent conversation from the state.
    Returns the last user query and assistant response.
    """
    from langchain_core.messages import HumanMessage, AIMessage

    messages = state.get("messages", [])
    context_parts = []

    # Get the last user message before current
    user_messages = [m for m in messages if isinstance(m, HumanMessage)]
    if len(user_messages) >= 2:
        prev_user = user_messages[-2]
        context_parts.append(f"Previous user query: {prev_user.content}")

    # Get the last assistant response
    ai_messages = [m for m in messages if isinstance(m, AIMessage) and m.content]
    if ai_messages:
        last_ai = ai_messages[-1]
        if last_ai.content:
            short_response = last_ai.content[:200] + "..." if len(last_ai.content) > 200 else last_ai.content
            context_parts.append(f"Assistant response: {short_response}")

    return "\n".join(context_parts) if context_parts else ""


# Categories that should trigger RAG before the LLM sees the query
RAG_CATEGORIES = {"knowledge", "general"}