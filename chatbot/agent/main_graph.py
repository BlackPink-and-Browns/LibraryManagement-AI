from typing import TypedDict, List, Dict, Optional
from langgraph.graph import StateGraph, END, MessagesState, START
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.runnables import RunnableLambda
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
import sys
import os
from dotenv import load_dotenv
sys.path.insert(1, '/home/inamurahman/library-management-system/LibraryManagement-AI')
from chatbot.services.semantic_search import query_books, create_embedding_text
from chatbot.agent.recommend_graph import recommend_graph
from chatbot.agent.search_graph import search_graph
import base64
import httpx

load_dotenv()


def fetch_image_from_url(url: str) -> str:
    try:
        response = httpx.get(url)
        response.raise_for_status()
        image_data = response.content
        return base64.b64encode(image_data).decode('utf-8')
    except httpx.RequestError as e:
        print(f"Error fetching image: {e}")
        return ""


llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

system_prompt = """
    You are an intent classifier. Your task is to identify the intended action from the user query. Based on the query, classify the action into one of the following:
       - SEARCH: If the user is asking for information about books or authors.
       - RECOMMEND: If the user is asking for book recommendations based on their preferences.
       - NONE: If the query does not match any of the above.
"""
sys_msg = SystemMessage(
    content=system_prompt,
    role="system"
)

class ParentState(MessagesState):
    action: Optional[str]
    user_query: Optional[str]
    genre: Optional[str]
    rating: Optional[float]
    location: Optional[str]
    author: Optional[str]
    recommendations: Optional[str]
    available_books: Optional[List[str]]
    results: Optional[str]
    output: Optional[str]
    thread_id: Optional[str]
    books: Optional[List[dict]]

def findAction(state: ParentState):
    class Action(BaseModel):
        action: Optional[str] = Field(
            description="The action to be performed based on the user's query. Possible values are 'SEARCH', 'RECOMMEND', or 'NONE'.",
            default=None
        )
        
    result = llm.with_structured_output(Action).invoke([sys_msg] + state['messages'])
    print (state['messages'])
    print("Action:", result.action)
    return {
        "action": result.action
    }


def handle_generic_messages(state):
    system_msg = SystemMessage(
        content="Do not respond to user's queries, respond to user with a message that you are a library assistant and can help with book searches or recommendations.",
        role="system"
    )
    output = llm.invoke([system_msg] + state['messages'])
    print(state['messages'])
    return {"messages": [output], "output": output.content, "books": []}

def route_to_subagent(state):
    if state['action'] == "SEARCH":
        return "search"
    elif state['action'] == "RECOMMEND":
        return "recommend"
    else:
        return "none"

def get_values(state) -> dict:
    return {
        "messages": state["messages"],
    }

builder = StateGraph(ParentState)

builder.add_node("find_action", findAction)
builder.add_node("search_agent", get_values | search_graph)
builder.add_node("recommend_agent", get_values | recommend_graph)
builder.add_node("handle_generic_messages", handle_generic_messages)

builder.add_edge(START, "find_action")
builder.add_conditional_edges("find_action",
    route_to_subagent,
    {
        "search": "search_agent",
        "recommend": "recommend_agent",
        "none": "handle_generic_messages"
    })
builder.add_edge("search_agent", END)
builder.add_edge("recommend_agent", END)
builder.add_edge("handle_generic_messages", END)

memory = MemorySaver()
main_graph = builder.compile(checkpointer=memory)

def chat_with_agent(user_message, user_id, image_base64=None):
    config = {"configurable": {"thread_id": user_id}}
    content = [{"type": "text", "text": user_message}]
    if image_base64:
        content.append({"type": "image_url", "image_url": {
            "url": f"data:image/jpeg;base64,{image_base64}"
        }})
    user_message = HumanMessage(content=content)
    print("User Message:", user_message)
    state = main_graph.invoke({"messages": user_message}, config)
    # return state["messages"][-1].content if state["messages"] else "No response generated."
    return {
        "message" : state["output"] if "output" in state else "",
        "books": state["books"] if "books" in state else [],
    }


async def stream_chat_with_agent(user_message, user_id, image_base64=None):
    config = {"configurable": {"thread_id": user_id}}
    content = [{"type": "text", "text": user_message}]
    if image_base64:
        content.append({"type": "image_url", "image_url": {
            "url": f"data:image/jpeg;base64,{image_base64}"
        }})
    user_message = HumanMessage(content=content)
    print("User Message:", user_message)
    for event in main_graph.stream({"messages": user_message}, config, stream_mode="updates"):
        for value in event.values():
            # print(value)
            print("", value["messages"][-1].content if value.get("messages", "") else "")
            yield value


# stream_chat_with_agent("recommend me books on startup development", "user123")


# Example usage
# chat_with_agent("I want to find a book on good habits.", "user123", image_base64="asdfsd")