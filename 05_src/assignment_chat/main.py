# main.py
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt.tool_node import ToolNode, tools_condition
from langchain_core.messages import SystemMessage, AIMessage, ToolMessage
from dotenv import load_dotenv
import os
from typing import Annotated, TypedDict
import operator

from assignment_chat.llm_config import get_chat_model
from assignment_chat.prompts import get_system_prompt
from assignment_chat.router import classify_query, RAG_CATEGORIES
from assignment_chat.judge import (
    validate_output,
    response_passes,
    SAFE_FALLBACK,
    extract_rag_context,
)
from assignment_chat.tools_api import get_weather
from assignment_chat.tools_rag import search_knowledge_base
from assignment_chat.tools_custom import get_current_time, calculate
from utils.logger import get_logger

_logs = get_logger(__name__)

_MAIN_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(_MAIN_DIR, "..", ".secrets"))

# Use the centralized config as the place to change providers/endpoints
chat_agent = get_chat_model(temperature=0.7)

all_tools = [search_knowledge_base, get_weather, get_current_time, calculate]
rag_tools = [search_knowledge_base]

instructions = get_system_prompt()


# ---- Custom state: adds a query_category field ----
class AgentState(TypedDict):
    messages: Annotated[list, operator.add]
    query_category: str


# ---- RAG Router ----
def rag_router(state: AgentState) -> dict:
    last_message = state["messages"][-1]
    user_text = last_message.content

    # Get context from the conversation so the router can detect follow-ups
    from assignment_chat.router import get_conversation_context
    context = get_conversation_context(state)

    category = classify_query(user_text, context)
    _logs.info(f'Router classified query as: {category}')

    if category not in RAG_CATEGORIES:
        _logs.info(f'RAG Router: skipping RAG for category "{category}"')
        return {"messages": [], "query_category": category}

    _logs.info(f'RAG Router: forcing RAG for category "{category}"')
    rag_call_message = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "search_knowledge_base",
                "args": {"query": user_text},
                "id": "forced_rag_call_001",
                "type": "tool_call",
            }
        ]
    )
    return {"messages": [rag_call_message], "query_category": category}

rag_tool_node = ToolNode(rag_tools)
all_tools_node = ToolNode(all_tools)


# ---- LLM Node ----
def call_model(state: AgentState):
    _logs.info(f'Calling model. Messages in state: {len(state["messages"])}')
    model_with_tools = chat_agent.bind_tools(all_tools)
    response = model_with_tools.invoke(
        [SystemMessage(content=instructions)] + state["messages"]
    )
    return {"messages": [response]}


# ---- Judge Node ----
def judge_node(state: AgentState) -> dict:
    _logs.info('Judge: validating output')
    messages = state["messages"]
    last_message = messages[-1]

    if not isinstance(last_message, AIMessage) or last_message.tool_calls:
        _logs.info('Judge: skipping (not a final AI message)')
        return {"messages": []}

    response_text = last_message.content
    rag_context = extract_rag_context(state)
    query_category = state.get("query_category", "knowledge")

    verdict = validate_output(response_text, rag_context, query_category)
    _logs.info(
        f'Judge verdict: grounded={verdict.grounded}, '
        f'no_fallback={verdict.no_fallback}, safe={verdict.safe}'
    )
    _logs.info(f'Judge reasoning: {verdict.reasoning}')

    if response_passes(verdict):
        _logs.info('Judge: response PASSED')
        return {"messages": []}

    _logs.warning('Judge: response FAILED, replacing with fallback')
    fallback_message = AIMessage(content=SAFE_FALLBACK)
    return {"messages": [fallback_message]}


# ---- Routing ----
def route_after_rag(state: AgentState):
    last_message = state["messages"][-1]
    if isinstance(last_message, ToolMessage) and last_message.tool_call_id == "forced_rag_call_001":
        return "call_model"
    return END


def route_after_judge(state: AgentState):
    return END


def route_rag_router(state: AgentState):
    messages = state["messages"]
    if messages and hasattr(messages[-1], "tool_calls") and messages[-1].tool_calls:
        return "rag_tools"
    return "call_model"


# ---- Build Graph ----
def get_graph():
    _logs.info('Building RAG.First agent graph with output judge')
    builder = StateGraph(AgentState)

    builder.add_node("rag_router", rag_router)
    builder.add_node("rag_tools", rag_tool_node)
    builder.add_node("call_model", call_model)
    builder.add_node("all_tools", all_tools_node)
    builder.add_node("judge", judge_node)

    builder.add_edge(START, "rag_router")

    builder.add_conditional_edges(
        "rag_router",
        route_rag_router,
        {"rag_tools": "rag_tools", "call_model": "call_model"}
    )

    builder.add_conditional_edges(
        "rag_tools",
        route_after_rag,
        {"call_model": "call_model", END: END}
    )

    builder.add_conditional_edges(
        "call_model",
        tools_condition,
        {"tools": "all_tools", END: "judge"}
    )

    builder.add_edge("all_tools", "call_model")

    builder.add_conditional_edges(
        "judge",
        route_after_judge,
        {END: END}
    )

    graph = builder.compile()
    _logs.info('Agent graph compiled successfully')
    return graph