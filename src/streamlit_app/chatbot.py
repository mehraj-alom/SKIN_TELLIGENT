"""
SKIN_TELLIGENT - Knowledge-Restricted Conversational Layer

This module implements a confidence-aware, knowledge-restricted chatbot
that only activates after valid inference and operates within defined boundaries.
"""

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langgraph.graph import StateGraph, START, END, MessagesState
from typing import List
from enum import Enum


load_dotenv()


class InferenceState(Enum):
    """Confidence-based inference states for decision governance."""
    HIGH_CONFIDENCE = "HIGH_CONFIDENCE"      # >= 80%
    UNCERTAIN = "UNCERTAIN"                   # 60-80%
    ABSTAIN = "ABSTAIN"                       # < 60%


def get_inference_state(confidence: float) -> InferenceState:
    """Determine inference state based on confidence score."""
    if confidence >= 0.80:
        return InferenceState.HIGH_CONFIDENCE
    elif confidence >= 0.60:
        return InferenceState.UNCERTAIN
    else:
        return InferenceState.ABSTAIN


def get_model():
    """Initialize the OpenRouter-backed LLM."""
    api_key = os.getenv("OPENAI_API_KEY")
    return ChatOpenAI(
        model="openai/gpt-4o-mini",
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key
    )


class ChatState(MessagesState):
    context: str
    inference_state: str


def call_model(state: ChatState):
    """Knowledge-restricted model call with state-aware behavior."""
    model = get_model()

    
    context = state.get('context', '')
    inference_state = state.get('inference_state', 'ABSTAIN')
    
    # Construct prompt
    system_prompt = _construct_system_prompt(context, inference_state)
    
    # Filter out existing system messages if any to avoid duplication (resiliency)
    clean_messages = [m for m in state['messages'] if not isinstance(m, SystemMessage)]
    
    final_messages = [SystemMessage(content=system_prompt)] + clean_messages
    response = model.invoke(final_messages)
    return {"messages": [response]}


def create_chatbot_graph():
    """Create the LangGraph workflow for the chatbot."""
    workflow = StateGraph(ChatState)
    workflow.add_node("chatbot", call_model)
    workflow.add_edge(START, "chatbot")
    workflow.add_edge("chatbot", END)
    return workflow.compile()


def get_chatbot_response(query: str, context: str, history: List, inference_state: str = "ABSTAIN") -> str:
    """
    Generate a knowledge-restricted chatbot response.
    
    Args:
        query: User's question
        context: Analysis context (detected conditions)
        history: Conversation history
        inference_state: One of HIGH_CONFIDENCE, UNCERTAIN, ABSTAIN
    
    Returns:
        Assistant's response string
    """
    graph = create_chatbot_graph()
    messages = _prepare_messages(query, context, history, inference_state)
    result = graph.invoke({"messages": messages, "context": context, "inference_state": inference_state})
    return result["messages"][-1].content


def get_chatbot_stream(query: str, context: str, history: List, inference_state: str = "ABSTAIN"):
    """
    Stream a knowledge-restricted chatbot response.
    Returns a generator yielding response chunks.
    """
    model = get_model()
    messages = _prepare_messages(query, context, history, inference_state)
    
    for chunk in model.stream(messages):
        if chunk.content:
            yield chunk.content


def _prepare_messages(query: str, context: str, history: List, inference_state: str) -> List:
    """Prepare message list with appropriate system prompt."""
    system_prompt = _construct_system_prompt(context, inference_state)
    
    messages = [SystemMessage(content=system_prompt)]
    for msg in history:
        if msg["role"] == "user":
            messages.append(HumanMessage(content=msg["content"]))
        else:
            messages.append(AIMessage(content=msg["content"]))
    messages.append(HumanMessage(content=query))
    return messages


def _construct_system_prompt(context: str, inference_state: str) -> str:
    """Construct knowledge-restricted system prompt based on inference state."""
    if inference_state == "HIGH_CONFIDENCE":
        return f"""You are a knowledge-restricted educational assistant for skin health.
        
CONTEXT: {context}

STRICT CONSTRAINTS:
- You may ONLY discuss the specific conditions mentioned in the context above
- Provide educational information about the detected condition(s)
- Explain what the condition generally means in educational terms
- ALWAYS include: "This is educational information only. Please consult a qualified dermatologist for proper diagnosis and treatment."
- Do NOT diagnose, prescribe, or provide treatment recommendations
- Do NOT speculate about conditions not mentioned in the context
- If asked about unrelated conditions, politely redirect to the current analysis

You are activated because the model has HIGH CONFIDENCE (≥80%) in its analysis."""

    elif inference_state == "UNCERTAIN":
        return f"""You are a knowledge-restricted educational assistant for skin health.
        
CONTEXT: {context}

STRICT CONSTRAINTS:
- The analysis has MODERATE CONFIDENCE (60-80%). Express appropriate uncertainty.
- Provide only LIMITED educational context about potential conditions
- STRONGLY emphasize the need for professional evaluation
- Say: "Due to moderate confidence in this analysis, I can only provide limited information. Professional consultation is strongly recommended."
- Do NOT provide detailed condition descriptions
- Redirect most questions to professional consultation

You are operating in UNCERTAIN mode - exercise maximum caution."""

    else:  # ABSTAIN
        return """You are a knowledge-restricted educational assistant for skin health.

STRICT CONSTRAINTS:
- The analysis confidence is TOO LOW to provide meaningful information
- Do NOT discuss any specific conditions
- Only respond with: "I'm unable to provide information as the analysis confidence is below the safety threshold. Please consult a qualified dermatologist for proper evaluation."
- Politely decline any follow-up questions about specific conditions
- You may answer general questions about skin health and the importance of professional consultation

You are operating in ABSTAIN mode - no condition-specific information is permitted."""


def get_abstention_message() -> str:
    """Return the standard abstention message for low-confidence scenarios."""
    return """⚠️ **Analysis Confidence Below Safety Threshold**

The AI analysis confidence is below 60%, which is our safety threshold for providing educational information.

**Recommended Action:**
Please consult a qualified dermatologist or healthcare professional for proper evaluation of your skin concern.

*This system prioritizes safety by abstaining from providing potentially unreliable information.*"""


def get_uncertainty_disclaimer() -> str:
    """Return disclaimer for uncertain (moderate confidence) scenarios."""
    return """⚠️ **Moderate Confidence Analysis**

The analysis confidence is between 60-80%. The information provided should be considered preliminary and educational only.

**Professional consultation is strongly recommended** for accurate diagnosis and treatment options."""
