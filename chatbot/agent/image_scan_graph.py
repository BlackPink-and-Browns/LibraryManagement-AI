from typing import TypedDict, List, Dict, Optional
from langgraph.graph import StateGraph, END, MessagesState, START
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableLambda
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
import os
from dotenv import load_dotenv
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from services.semantic_search import query_books, create_embedding_text
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

class ImageScanState(MessagesState):
    user_query: Optional[str]
    image_description: Optional[str]
    title: Optional[str]
    author: Optional[str]
    genre: Optional[str]
    description: Optional[str]
    location: Optional[str]
    results: Optional[str]
    output: Optional[str]

system_prompt = """You are a helpful library assistant. Your task is to help users find information about books in the library.
You will be provided with a series of messages from the user. Based on the messages, you need to meet the requirements of user."""

sys_msg = SystemMessage(
    content=system_prompt,
    role="system"
)
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

def getBookInfoFromImage(state: ImageScanState):
    class ImageScan(BaseModel):
        user_query: Optional[str] = Field(
            description="The user's query for book information. If not mentioned, it should be None.",
            default=None
        )
        image_description: Optional[str] = Field(
            description="A description of the image provided by the user. If no image given, it should be None.",
            default=None
        )
        title: Optional[str] = Field(
            description="The title of the book. If not in the image, it should be None.",
            default=None
        )
        author: Optional[str] = Field(
            description="The author of the book. If not in the image, it should be None.",
            default=None
        )
        genre: Optional[str] = Field(
            description="The genre of the book. If not in the image, it should be None.",
            default=None
        )
        description: Optional[str] = Field(
            description="A brief description of the book. If not in the image, it should be None.",
            default=None
        )
        location: Optional[str] = Field(
            description="The location of the book in the library. If not mentioned, it should be None.",
            default=None
        )

    result = llm.with_structured_output(ImageScan).invoke([sys_msg] + state['messages'])

    if not result:
        return {"error": "No book information found in the image."}
    return {
        "user_query": result.user_query,
        "image_description": result.image_description,
        "title": result.title,
        "author": result.author,
        "genre": result.genre,
        "description": result.description,
        "location": result.location
    }

def handleSearch(state: ImageScanState):
    query_text = create_embedding_text(state)
    results = query_books(query_text, top_k=1)
    print(f"🔍 Searching for books with query: {query_text}")
    if not results:
        return {"error": "No books found matching the search criteria."}
    print("Search Results:")
    search_result = {
        "title": results[0]['title'],
        "author": results[0]['author'],
        "genre": results[0]['genre'],
        "description": results[0]['description'],
        "location": results[0]['location'],
        "rating": results[0]['rating'],
        "results": " ".join([f"{book['title']} by {book['author']} (📍 {book['location']}) ⭐ {book['rating']}" for book in results])
        "output" :  " ".join([f"{book['title']} by {book['author']} (📍 {book['location']}) ⭐ {book['rating']}" for book in results])
    }

    return search_result

builder = StateGraph(ImageScanState)
builder.add_node("getBookInfoFromImage",getBookInfoFromImage)
builder.add_node("handleSearch",handleSearch)
# builder.add_node("END",END)

builder.add_edge(START, "getBookInfoFromImage")
builder.add_edge("getBookInfoFromImage","handleSearch")
builder.add_edge("handleSearch",END)

image_scan_graph = builder.compile()

user_message = HumanMessage(
    content=[
        {"type": "text", "text": "I have an image of a book I want to search for."},
        {"type": "image_url", "image_url": {
            "url": f"data:image/jpeg;base64,{image_base64}"
        }}
    ]
)

# state = image_scan_graph.invoke({"messages": user_message})
# print("Search Result:")
# print(state["results"])