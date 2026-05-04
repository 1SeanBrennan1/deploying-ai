# app.py
from assignment_chat.main import get_graph
from assignment_chat.guard import check_input
from langchain_core.messages import HumanMessage, AIMessage
import gradio as gr
from dotenv import load_dotenv
from utils.logger import get_logger

_logs = get_logger(__name__)

# Build the agent graph once when the app starts
llm = get_graph()

load_dotenv('.secrets')

# Store the full agent state across turns
conversation_state = {"messages": [], "query_category": ""}


def sage_chat(message: str, history: list[dict]) -> str:
    """
    The main chat function that Gradio calls for each user message.

    1. Programmatic guard check first (no LLM involved).
    2. If message is safe, appends to the persistent conversation state.
    3. Invokes the LangGraph agent with the FULL state (not a new one).
    """
    global conversation_state

    _logs.info(f'Received message: {message[:50]}...')

    # ---- CODE-BASED GUARD ----
    deflection = check_input(message)
    if deflection is not None:
        _logs.info('Input blocked by guard')
        # Still add to state so the user sees the deflection in history
        conversation_state["messages"].append(HumanMessage(content=message))
        conversation_state["messages"].append(AIMessage(content=deflection))
        return deflection

    # ---- NORMAL AGENT PROCESSING ----
    # Append the new user message to the existing state
    conversation_state["messages"].append(HumanMessage(content=message))

    # Invoke the agent with the FULL conversation state
    conversation_state = llm.invoke(conversation_state)

    # Return the last AI message
    last_message = conversation_state["messages"][-1]
    return last_message.content


# Reset state when the chat is cleared
def reset_state():
    global conversation_state
    conversation_state = {"messages": [], "query_category": ""}


chat = gr.ChatInterface(
    fn=sage_chat,
    type="messages",
    title="🤖 Sage - Your Witty AI Assistant",
    description="I can check the weather, search my knowledge base, do calculations, and tell you the time! I also have a great sense of humour (well, I think so anyway).\n\n⚠️ Warning: Do NOT ask me about cats, dogs, horoscopes, or Taylor Swift. My lawyers are very particular about this.",
    examples=[
        "What's the weather in Toronto?",
        "What time is it?",
        "Calculate the square root of 144",
        "Tell me about the French Revolution"
    ],
    theme="soft"
)

if __name__ == "__main__":
    _logs.info('Starting Sage Chat App...')
    chat.launch()