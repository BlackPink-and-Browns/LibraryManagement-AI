from typing import TypedDict, Optional, List
from langgraph.graph import StateGraph, START, END, MessagesState
from langchain_core.runnables import RunnableLambda
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import os

load_dotenv()

class BorrowBookState(MessagesState):
    book_title: Optional[str]
    borrow_status: Optional[str]
    output: Optional[str]

# LLM configuration
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# System prompt
system_prompt = """You are a helpful library assistant. Your task is to help users borrow books from the library.
Gather the necessary information from the user such as book title, shelf id, and copy id.
"""

sys_msg = SystemMessage(content=system_prompt)

# Step 1: Extract user requirements
def retrieveBorrowRequirement(state: BorrowBookState):
    class BorrowDetails(BaseModel):
        book_title: Optional[str] = Field(description="The title of the book the user wants to borrow.", default=None)
        shelf_id: Optional[str] = Field(description="The shelf ID of the shelf the book is present.", default=None)
        copy_id: Optional[str] = Field(description="The copy ID of the book the user wants to borrow.", default=None)

    result = llm.with_structured_output(BorrowDetails).invoke([sys_msg] + state['messages'])

    print("📥 Input messages:", [sys_msg] + state['messages'])
    print("✅ Extracted details:", result)

    return {
        "book_title": result.book_title,
        "user_id": result.user_id
    }

# Step 2: Placeholder for tool calling (e.g., check book, update database)
def borrowBook(state: BorrowBookState):
    book_title = state.get("book_title")
    user_id = state.get("user_id")

    # Placeholder logic: replace with actual tool call
    # Example: status = borrow_book_tool(book_title, user_id)
    if book_title and user_id:
        status = f"✅ Book '{book_title}' has been successfully borrowed by user ID {user_id}."
    else:
        status = "❌ Missing information: book title or user ID."

    return {
        "borrow_status": status,
        "output": status,
        "messages": [AIMessage(content=status)]
    }

# Build the graph
builder = StateGraph(BorrowBookState)
builder.add_node("retrieveBorrowRequirement", retrieveBorrowRequirement)
builder.add_node("borrowBook", borrowBook)

builder.set_entry_point("retrieveBorrowRequirement")
builder.add_edge("retrieveBorrowRequirement", "borrowBook")
builder.add_edge("borrowBook", END)

memory = MemorySaver()
borrow_graph = builder.compile(checkpointer=memory)
