from typing import TypedDict, List, Dict, Optional
from langgraph.graph import StateGraph, END, MessagesState
from langchain_core.runnables import RunnableLambda
from langchain.chat_models import ChatOpenAI
from IPython.display import Image, display
from pydantic import BaseModel

class BookState(MessagesState):
  title: Optional[str]
  author: Optional[str]
  genre: Optional[str]
  description: Optional[str]
  rating: Optional[float]
  location: Optional[str]

sys_msg = """You are a helpful library assistant. Your task is to help users find information about books in the library.
You will be provided with a series of messages from the user. Based on these messages, you need to extract the following information about a book:
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
    return {
        "title": result.title,
        "author": result.author,
        "genre": result.genre,
        "description": result.description,
        "location": result.location,
        "rating": result.rating
    }

def handleSearch(state: BookState):
    #mock search result
    search_result = {
        "title": "The Great Gatsby",
        "author": "F. Scott Fitzgerald",
        "genre": "Fiction",
        "description": "A novel set in the 1920s that explores themes of decadence, idealism, resistance to change, social upheaval, and excess.",
        "location": "Shelf A3",
        "rating": 4.5
    }
   return search_result

builder = StateGraph(BookState)
builder.add_node("retrieveBookToSearch",retrieveBookToSearch)
builder.add_node("handleSearch",handleSearch)
# builder.add_node("END",END)

builder.add_edge(START, "retrieveBookToSearch")
builder.add_edge("retrieveBookToSearch","handleSearch")
builder.add_edge("handleSearch","END")

search_graph = builder.compile()
    