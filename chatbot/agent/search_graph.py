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

def fetch_image_from_url(url: str) -> str:
    try:    
        response = httpx.get(url)
        response.raise_for_status()
        image_data = response.content
        return base64.b64encode(image_data).decode('utf-8')
    except httpx.RequestError as e:
        print(f"Error fetching image: {e}")
        return ""


class BookState(MessagesState):
  title: Optional[str]
  author: Optional[str]
  genre: Optional[str]
  description: Optional[str]
  rating: Optional[float]
  location: Optional[str]
  results: Optional[str]
  output: Optional[str]
  books: Optional[List[dict]]

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
        "rating": result.rating,
        "messages": AIMessage(content=f"Searching for book: {result.title} by {result.author} in genre {result.genre} with rating {result.rating} and location {result.location}."),
    }

def handleSearch(state: BookState):
    query_text = create_embedding_text(state)
    results = query_books(query_text, top_k=1)
    if not results:
        return {"error": "No books found matching the search criteria."}
    
    search_result = {
        # "title": results[0]['title'],
        # "author": results[0]['author'],
        # "genre": results[0]['genre'],
        # "description": results[0]['description'],
        # "location": results[0]['location'],
        # "rating": results[0]['rating'],
        "results": str(results[0]),
        "output": " ".join([f"{key}: {value}" for key, value in results[0].items() if value is not None]),
        "messages": AIMessage(content=f"Found book: {[f"{key}: {str(value)}" for key, value in results[0].items() if value is not None]} ")
    }

    return search_result

def getBookList(state: BookState):
    system_prompt = """from the messages, extract the list of books in the given format.
    Each book should be represented as a dictionary.
    if there are no books, return an empty list.
    Also, provide a beautiful description about each books, why it is relevant, and why it is a good read.

    return the list of books in the following format:
    message: A paragraph description about each books. If there is no book, return an explanation that no books were found.
    books: A list of books, each represented as a dictionary:
    """

    class Book(BaseModel):
        title: Optional[str] = Field(
            description="The title of the book.",
            default=None
        )
        author: Optional[str] = Field(
            description="The author of the book.",
            default=None
        )
        description: Optional[str] = Field(
            description="A brief description of the book.",
            default=None
        )
        genre: Optional[str] = Field(
            description="The genre of the book.",
            default=None
        )
        rating: Optional[float] = Field(
            description="The rating of the book.",
            default=None
        )
        location: Optional[str] = Field(
            description="The location of the book in the library.",
            default=None
        )
        cover_image: Optional[str] = Field(
            description="The cover image of the book (url).",
            default=None
        )

    class BookList(BaseModel):
        description: str = Field(
            description = "A paragraph description about each books, Why it is relevant. If there is no book, return an explanation that no books were found.",
        )
        books: List[Book] = Field(
            description="A list of books.",
            default_factory=list
        )

    system_msg = AIMessage(
        content=system_prompt,
        role="system"
    )

    messages = [system_msg] + [AIMessage(content=state["results"])]
    llm_response = llm.with_structured_output(BookList).invoke(messages)

    # state['recommendations'] = recommendations
    # print(f"📚 Recommendations: {recommendations}")
    print("llm_response in getBookList Node")
    print(llm_response)
    return { "books": llm_response if llm_response else [] , "output": llm_response.description if llm_response else "" }

builder = StateGraph(BookState)
builder.add_node("retrieveBookToSearch",retrieveBookToSearch)
builder.add_node("handleSearch",handleSearch)
# builder.add_node("END",END)
builder.add_node("getBookList", getBookList)

builder.add_edge(START, "retrieveBookToSearch")
builder.add_edge("retrieveBookToSearch","handleSearch")
builder.add_edge("handleSearch","getBookList")
builder.add_edge("getBookList",END)
# builder.add_edge("handleSearch",END)

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