from typing import TypedDict, List, Dict, Optional
from langgraph.graph import StateGraph, END, MessagesState, START
from langchain_core.runnables import RunnableLambda
from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.checkpoint.memory import MemorySaver
from pydantic import BaseModel, Field
import os
from dotenv import load_dotenv
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from chatbot.services.semantic_search import query_books, create_embedding_text
import base64
import httpx

load_dotenv()

image_url = "https://res.cloudinary.com/jerrick/image/upload/d_642250b563292b35f27461a7.png,f_jpg,fl_progressive,q_auto,w_1024/6478a69e5695fb001d1e1969.jpg"
def fetch_image_from_url(url: str) -> str:
    try:    
        response = httpx.get(url)
        response.raise_for_status()
        image_data = response.content
        return base64.b64encode(image_data).decode('utf-8')
    except httpx.RequestError as e:
        print(f"Error fetching image: {e}")
        return ""
image_base64 = fetch_image_from_url(image_url)

class BookState(MessagesState):
  title: Optional[str]
  author: Optional[str]
  genre: Optional[str]
  description: Optional[str]
  rating: Optional[float]
  location: Optional[str]
  results: Optional[str]
  output: Optional[str]

sys_msg = """You are a helpful library assistant. Your task is to help users find information about books in the library.
You will be provided with a series of messages from the user. Based on these messages, you need to extract the following information about a book (only if mentioned by the user):
(if Image is provided, you can extract the information from the image)
1. Title: The title of the book.
2. Author: The author of the book.
3. Genre: The genre of the book.
4. Description: A brief description of the book.
5. Location: The location of the book in the library."""

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

def retrieveBookToSearch(state: BookState):

    class Book(BaseModel):
        title: Optional[str] = Field(
            description="The title of the book. if not mentioned by the user, it should be None",
            default=None
        )
        author: Optional[str] = Field(
            description="The author of the book. if not mentioned by the user, it should be None",
            default=None
        )
        genre: Optional[str] = Field(
            description="The genre of the book. if not mentioned by the user, it should be None",
            default=None
        )
        description: Optional[str] = Field(
            description="A brief description of the book. if not mentioned by the user, it should be None",
            default=None
        )
        location: Optional[str] = Field(
            description="The location of the book in the library. if not mentioned by the user, it should be None",
            default=None
        )
        rating: Optional[float] = Field(
            description="The rating of the book. if not mentioned by the user, it should be None",
            default=None
        )

    result = llm.with_structured_output(Book).invoke([sys_msg] + state['messages'])
    print(result)
    return {
        "title": result.title,
        "author": result.author,
        "genre": result.genre,
        "description": result.description,
        "location": result.location,
        "rating": result.rating
    }

def handleSearch(state: BookState):
    query_text = create_embedding_text(state)
    results = query_books(query_text, top_k=1)
    if not results:
        return {"error": "No books found matching the search criteria."}
    
    search_result = {
        "title": results[0]['title'],
        "author": results[0]['author'],
        "genre": results[0]['genre'],
        "description": results[0]['description'],
        "location": results[0]['location'],
        "rating": results[0]['rating'],
        "results": " ".join([f"{book['title']} by {book['author']} (📍 {book['location']}) ⭐ {book['rating']}" for book in results]),
        "output": " ".join([f"{book['title']} by {book['author']} (📍 {book['location']}) ⭐ {book['rating']}" for book in results])
    }

    return search_result

builder = StateGraph(BookState)
builder.add_node("retrieveBookToSearch",retrieveBookToSearch)
builder.add_node("handleSearch",handleSearch)
# builder.add_node("END",END)

builder.add_edge(START, "retrieveBookToSearch")
builder.add_edge("retrieveBookToSearch","handleSearch")
builder.add_edge("handleSearch",END)

memory = MemorySaver()
# thread = {"configurable": {"thread_id": "user_id"}}

search_graph = builder.compile(checkpointer=memory)


# user_message = HumanMessage(
#     content=[
#         {"type": "text", "text": "I have an image of a book I want to search for."},
#         {"type": "image_url", "image_url": {
#             "url": f"data:image/jpeg;base64,{image_base64}"
#         }}
#     ]
# )
# state = search_graph.invoke({"messages": ["suggest me a book for good habits"]}, thread)
# state = search_graph.invoke({"messages": [user_message]})
# print("Search Result:")
# print(state["results"])